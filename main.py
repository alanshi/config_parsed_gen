import os
import json
import uuid
import shutil
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import Body, FastAPI, Request, Form, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# diagrams imports
# NOTE: diagrams and graphviz must be installed in the runtime environment
from diagrams import Diagram, Cluster, Edge
from diagrams.generic.network import Switch
from pydantic import BaseModel
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


app = FastAPI()

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# 定义接收 JSON 的 Pydantic 模型
class GenerateRequest(BaseModel):
    config: dict        # 网络拓扑 JSON
    style: dict = None  # 样式 JSON（可选）
    output_name: str = "network_topology"  # 输出文件名（可选）


class GenerateRequestV2(BaseModel):
    config: dict        # 新格式 JSON
    style: dict = None
    output_name: str = "network_topology_v2"


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
            device_node_map[s] >> Edge(**e_style, dir="back") >> device_node_map[t]

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


def convert_new_config_to_old(new_config: dict) -> dict:
    """
    将新格式配置转换成旧格式，以便复用原生成逻辑
    同时保证 BGP AS 之间的连接能被正确解析
    """
    network_devices = []
    network_connections = []

    for dev in new_config.get("devices", []):
        hostname = dev["name"]
        loopback = dev.get("loopback", "").split("/")[0]

        interfaces = []
        # 添加 Loopback0 接口（保证 core_ip 能取到）
        if loopback:
            interfaces.append({
                "name": "Loopback0",
                "ip_address": loopback,
                "description": "Loopback"
            })

        # 物理接口
        for iface in dev.get("interfaces", []):
            interface_name = iface["interface"]
            ip_address = iface["ip"].split("/")[0]
            peer_device, peer_interface = iface["peer"].split(":")
            interfaces.append({
                "name": interface_name,
                "ip_address": ip_address,
                "description": f"TO-{peer_device}"
            })

            # 添加物理连接
            network_connections.append({
                "source_device": hostname,
                "source_interface": interface_name,
                "destination_device": peer_device,
                "destination_interface": peer_interface
            })

        # BGP 邻居
        bgp_neighbors = []
        for neigh in dev.get("bgp", {}).get("neighbors", []):
            bgp_neighbors.append({
                "neighbor_ip": neigh["ip"].split("/")[0],  # 去掩码
                "remote_as": neigh["as"]
            })

        network_devices.append({
            "hostname": hostname,
            "interfaces": interfaces,
            "bgp_neighbors": bgp_neighbors,
            "as_number": dev.get("bgp", {}).get("as")  # 新增 AS 号字段
        })

    return {
        "network_devices": network_devices,
        "network_connections": network_connections
    }

@app.post("/generate", response_class=JSONResponse)
async def generate(data: GenerateRequest = Body(...)):
    try:
        # 网络配置
        config_text = json.dumps(data.config)
        config = load_network_config_from_text(config_text)
        # 样式配置（如果不传则用默认）
        style_config = data.style or DEFAULT_STYLE_CONFIG.copy()

        # 生成 SVG
        rel_svg = generate_diagram_from_config(config, style_config, data.output_name)

        # 替换 SVG 中的图标路径
        with open(STATIC_DIR / rel_svg, "r", encoding="utf-8") as f:
            svg = f.read()
        svg = svg.replace(
            "/home/www/config_parsed_gen/venv/lib/python3.10/site-packages/resources/",
            "/icons/"
        )
        with open(STATIC_DIR / rel_svg, "w", encoding="utf-8") as f:
            f.write(svg)

        return {"svg_url": f"/static/{rel_svg}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate_v2", response_class=JSONResponse)
async def generate_v2(data: GenerateRequestV2 = Body(...)):
    try:
        # 将新格式转成旧格式
        old_config = convert_new_config_to_old(data.config)
        config_text = json.dumps(old_config)
        config = load_network_config_from_text(config_text)

        style_config = data.style or DEFAULT_STYLE_CONFIG.copy()

        rel_svg = generate_diagram_from_config(config, style_config, data.output_name)

        # 替换 SVG 图标路径
        with open(STATIC_DIR / rel_svg, "r", encoding="utf-8") as f:
            svg = f.read()
        svg = svg.replace(
            "/home/www/config_parsed_gen/venv/lib/python3.10/site-packages/resources/",
            "/icons/"
        )
        with open(STATIC_DIR / rel_svg, "w", encoding="utf-8") as f:
            f.write(svg)

        return {"svg_url": f"/static/{rel_svg}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# run via: uvicorn app.main:app --reload
