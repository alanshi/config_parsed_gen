import os
import json
import uuid
import shutil
from pathlib import Path
from typing import Optional, Dict, Any

from workers import WorkerEntrypoint
from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import asgi

# diagrams imports
# NOTE: diagrams and graphviz must be installed in the runtime environment
from diagrams import Diagram, Cluster, Edge
from diagrams.generic.network import Switch
# we use a generic Router-like node; some diagrams versions expose different providers.
# Try fallback by importing one of them; prefer diagrams.onprem.network if available.
try:
    from diagrams.gcp.network import Router as RouterNode
except Exception:
    try:
        from diagrams.onprem.network import Router as RouterNode
    except Exception:
        # fallback to generic blank provider if Router class not available
        from diagrams.generic.blank import Blank as RouterNode

# app paths
BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
OUTPUT_DIR = STATIC_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class Default(WorkerEntrypoint):
    async def fetch(self, request, env):
        return await asgi.fetch(app, request, env)

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

# ================= Default style config (can be overridden via form or file) ==============
DEFAULT_STYLE_CONFIG = {
    "nodes": {
        "core_router": {"shape": "box", "style": "filled,rounded", "fillcolor": "#5d391a", "fontcolor": "white", "fontsize": "9", "width": "1.2", "height": "1.1", "fixedsize": "true"},
        "edge_router": {"shape": "box", "style": "filled,rounded", "fillcolor": "#2c5282", "fontcolor": "white", "fontsize": "9", "width": "1.2", "height": "1.1"},
        "switch": {"shape": "box", "style": "filled,rounded", "fillcolor": "#3182ce", "fontcolor": "white", "fontsize": "9", "width": "1.2", "height": "1.1"},
        "other": {"shape": "ellipse", "style": "filled", "fillcolor": "#718096", "fontcolor": "white", "fontsize": "7"}
    },
    "edges": {
        "physical": {"fontsize": "9", "color": "#2d3748", "penwidth": "1.2", "labelloc": "c"},
        "bgp": {"fontsize": "9", "color": "#e6194b", "penwidth": "1.0", "style": "dashed", "labelloc": "c"}
    },
    "clusters": {
        "core_router": {"fillcolor": "#e6f7ff", "color": "#1890ff", "style": "filled,rounded"},
        "edge_router": {"fillcolor": "#fff2e6", "color": "#fa8c16", "style": "filled,rounded"},
        "switch": {"fillcolor": "#f0f9eb", "color": "#52c41a", "style": "filled,rounded"}
    }
}

# ================ helpers: load and preprocess =======================================
def load_network_config_from_text(text: str) -> dict:
    """
    Parse JSON text into expected internal structure.
    Raise HTTPException on invalid JSON.
    """
    try:
        config = json.loads(text)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"JSON decode error: {e}")

    # Basic validation
    if "network_devices" not in config or "network_connections" not in config:
        raise HTTPException(status_code=400, detail="JSON must contain 'network_devices' and 'network_connections'")

    # preprocess: build processed_devices, ip_to_hostname, as_info_map
    processed_devices = []
    ip_to_hostname = {}
    as_info_map = {}

    for device in config["network_devices"]:
        hostname = device.get("hostname")
        if not hostname:
            continue

        # determine role by leading char
        if hostname.startswith("C"):
            role, color, dtype = "core_router", "#5d391a", "router"
        elif hostname.startswith("R"):
            role, color, dtype = "edge_router", "#2c5282", "router"
        elif hostname.startswith("S"):
            role, color, dtype = "switch", "#3182ce", "switch"
        else:
            role, color, dtype = "other", "#718096", "router"

        # AS number: prefer an explicit 'as_number' at device level; else try bgp_neighbors' remote_as
        as_number = device.get("as_number")
        if as_number is None:
            bgp_neighbors = device.get("bgp_neighbors", [])
            if isinstance(bgp_neighbors, list) and bgp_neighbors:
                # take first neighbor.remote_as if exists
                first = bgp_neighbors[0]
                as_number = first.get("remote_as")

        # collect IPs & find Loopback0
        all_ips = []
        core_ip = "unknown"
        for iface in device.get("interfaces", []):
            ip = iface.get("ip_address")
            if ip:
                all_ips.append(ip)
                ip_to_hostname[ip] = hostname
            if iface.get("name") == "Loopback0" and iface.get("ip_address"):
                core_ip = iface.get("ip_address")

        processed_devices.append({
            "hostname": hostname,
            "role": role,
            "color": color,
            "type": dtype,
            "core_ip": core_ip,
            "all_ips": all_ips,
            "as_number": as_number,
            "interfaces": device.get("interfaces", []),
            "bgp_neighbors": device.get("bgp_neighbors", [])
        })
        as_info_map[hostname] = as_number

    config["processed_devices"] = processed_devices
    config["ip_to_hostname"] = ip_to_hostname
    config["as_info_map"] = as_info_map
    return config

# ================ create node =======================================================
def is_dark_color(hexcolor: str) -> bool:
    if not hexcolor or not hexcolor.startswith("#") or len(hexcolor.lstrip("#")) != 6:
        return False
    r = int(hexcolor[1:3], 16); g = int(hexcolor[3:5], 16); b = int(hexcolor[5:7], 16)
    brightness = (r * 299 + g * 587 + b * 114) / 1000
    return brightness < 128

def create_network_node(device: dict, style_config: dict):
    """create a diagrams node using style_config (RouterNode or Switch)"""
    label_lines = [device["hostname"], f"Loopback:{device['core_ip']}"]
    if device.get("as_number"):
        label_lines.append(f"AS{device['as_number']}")
    if device.get("bgp_neighbors"):
        neighbors = device["bgp_neighbors"][:2]
        s = ", ".join([f"AS{n.get('remote_as')}:{n.get('neighbor_ip')}" for n in neighbors])
        label_lines.append("BGP: " + s)
    label_lines.append(f"角色: {device['role']}")
    full_label = "\n".join(label_lines)

    style = style_config["nodes"].get(device["role"], style_config["nodes"]["other"]).copy()

    # diagrams node constructors accept label and keyword args that map to graphviz attributes
    if device["type"] == "switch":
        try:
            return Switch(full_label, **style)
        except TypeError:
            # fallback
            from diagrams.generic.blank import Blank
            return Blank(full_label)
    else:
        try:
            return RouterNode(full_label, **style)
        except TypeError:
            from diagrams.generic.blank import Blank
            return Blank(full_label)

# ================ diagram generation ================================================
def generate_diagram_from_config(config: dict, style_config: dict, output_basename: str, line_style: str = "ortho"):
    """
    Generate a diagram SVG using diagrams library.
    output_basename (no extension) inside static/outputs
    Returns output SVG relative path for embedding (e.g. "outputs/<name>.svg")
    """
    # prepare output paths
    uid = uuid.uuid4().hex[:8]
    safe_basename = f"{output_basename}_{uid}"
    out_path = OUTPUT_DIR / safe_basename
    out_path.parent.mkdir(parents=True, exist_ok=True)

    layout_dir = config.get("layout", {}).get("direction", "LR").upper()
    graph_attr = {
        "fontsize": "10",
        "fontname": "Arial",
        "bgcolor": "white",
        "rankdir": layout_dir,
        "ranksep": str(config.get("layout", {}).get("rank_sep", 3.0)),
        "nodesep": str(config.get("layout", {}).get("node_sep", 1.0)),
        "splines": "spline"
    }

    # create diagram context
    filename_noext = str(OUTPUT_DIR / safe_basename)
    # outformat svg if available
    try:
        diag = Diagram("Network Topology", filename=filename_noext, show=False, outformat="svg", graph_attr=graph_attr)
        ctx = diag
    except TypeError:
        diag = Diagram("Network Topology", filename=filename_noext, show=False, graph_attr=graph_attr)
        ctx = diag

    # build nodes and edges inside context
    with ctx:
        device_node_map = {}
        # clusters
        with Cluster("核心路由器区", graph_attr=style_config["clusters"]["core_router"]):
            for dev in config["processed_devices"]:
                if dev["role"] == "core_router":
                    node = create_network_node(dev, style_config)
                    device_node_map[dev["hostname"]] = node
        with Cluster("边缘路由器区", graph_attr=style_config["clusters"]["edge_router"]):
            for dev in config["processed_devices"]:
                if dev["role"] == "edge_router":
                    node = create_network_node(dev, style_config)
                    device_node_map[dev["hostname"]] = node
        with Cluster("汇聚交换机区", graph_attr=style_config["clusters"]["switch"]):
            for dev in config["processed_devices"]:
                if dev["role"] == "switch":
                    node = create_network_node(dev, style_config)
                    device_node_map[dev["hostname"]] = node

        # physical edges
        for conn in config["network_connections"]:
            s = conn.get("source_device"); t = conn.get("destination_device")
            si = conn.get("source_interface", ""); ti = conn.get("destination_interface", "")
            if not s or not t:
                continue
            if s not in device_node_map or t not in device_node_map:
                # skip or log
                continue
            e_style = style_config["edges"]["physical"].copy()
            e_style["label"] = f"{si} → {ti}"
            device_node_map[s] >> Edge(**e_style) >> device_node_map[t]

        # bgp edges (use ip mapping)
        ip_map = config.get("ip_to_hostname", {})
        drawn = set()
        for dev in config["processed_devices"]:
            for neigh in dev.get("bgp_neighbors", []):
                nip = neigh.get("neighbor_ip")
                if not nip:
                    continue
                target = ip_map.get(nip)
                if not target:
                    continue
                pair = tuple(sorted([dev["hostname"], target]))
                if pair in drawn:
                    continue
                drawn.add(pair)
                if target not in device_node_map:
                    continue
                e_style = style_config["edges"]["bgp"].copy()
                e_style["label"] = f"BGP\nAS{dev.get('as_number')} → AS{neigh.get('remote_as')}\n{dev.get('core_ip')} → {nip}"
                device_node_map[dev["hostname"]] >> Edge(**e_style) >> device_node_map[target]

    # ensure svg produced
    svg_path = Path(filename_noext + ".svg")
    if not svg_path.exists():
        # try .gv/.dot fallback to render via graphviz (optional)
        raise RuntimeError("SVG not produced; check diagrams/graphviz installation and file permissions")

    # copy/move produced svg into outputs directory with stable name
    final_name = f"{safe_basename}.svg"
    dest_rel = f"outputs/{final_name}"
    # already placed in outputs by diagrams (filename pointed there) — ensure file exists
    return dest_rel

# ===================== fastapi endpoints ============================================
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    """
    Serve the simple UI
    """
    return templates.TemplateResponse("index.html", {"request": request, "example_json": example_json_text(), "default_style": json.dumps(DEFAULT_STYLE_CONFIG, indent=2)})

def example_json_text():
    # a short example to prefill the textarea (keeps the big original out of template)
    sample = {
        "network_devices": [
            {"hostname": "C1-GSR12016", "interfaces": [{"name":"Loopback0","ip_address":"222.7.0.1"}], "bgp_neighbors": []}
        ],
        "network_connections": []
    }
    return json.dumps(sample, indent=2)

@app.post("/generate", response_class=HTMLResponse)
async def generate(request: Request,
                   json_text: Optional[str] = Form(None),
                   json_file: Optional[UploadFile] = File(None),
                   style_text: Optional[str] = Form(None),
                   output_name: Optional[str] = Form("topology")):
    """
    Accept JSON (text or uploaded file) and optional style JSON (text)
    Generate diagram and show embedded SVG result
    """
    # get JSON input
    if json_file and json_file.filename:
        data = await json_file.read()
        try:
            config_text = data.decode("utf-8")
        except Exception:
            raise HTTPException(status_code=400, detail="上传文件需为 UTF-8 编码的 JSON")
    elif json_text:
        config_text = json_text
    else:
        raise HTTPException(status_code=400, detail="请通过文本或文件上传网络 JSON 配置")

    # load config
    config = load_network_config_from_text(config_text)

    # merge style config
    style_config = DEFAULT_STYLE_CONFIG.copy()
    if style_text:
        try:
            ext = json.loads(style_text)
            # shallow merge: nodes/edges/clusters
            for k in ["nodes", "edges", "clusters"]:
                if k in ext and isinstance(ext[k], dict):
                    style_config.setdefault(k, {}).update(ext[k])
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"样式 JSON 解析失败：{e}")

    # set layout defaults if not set
    config.setdefault("layout", {"direction": "LR", "rank_sep": 3.0, "node_sep": 1.0})

    # generate diagram
    try:
        # generate unique basename
        basename = output_name.strip() or "topology"
        rel_svg = generate_diagram_from_config(config, style_config, basename, line_style="ortho")
        svg_url = f"/static/{rel_svg}"
    except Exception as e:
        # return template with error
        return templates.TemplateResponse("index.html", {"request": request, "error": str(e), "example_json": config_text, "default_style": json.dumps(style_config, indent=2)})

    # return page showing embedded svg
    return templates.TemplateResponse("index.html", {"request": request, "svg_url": svg_url, "example_json": config_text, "default_style": json.dumps(style_config, indent=2)})

# run via: uvicorn app.main:app --reload
