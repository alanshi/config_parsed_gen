import os
import json
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, Body, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# diagrams 导入（兼容不同版本）
from diagrams import Diagram, Cluster, Edge
from diagrams.generic.network import Switch
try:
    from diagrams.onprem.network import Router as RouterNode
except Exception:
    try:
        from diagrams.gcp.network import Router as RouterNode
    except Exception:
        from diagrams.generic.blank import Blank as RouterNode

# ================= 路径配置 =================
BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
OUTPUT_DIR = STATIC_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ================= 应用初始化 =================
app = FastAPI(title="BTD 网络拓扑生成器", version="1.0")

# 跨域配置
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 静态文件和模板
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

# ================= 默认样式配置 =================
DEFAULT_STYLE_CONFIG = {
    "nodes": {
        "core_router": {
            "shape": "box", "style": "filled,rounded",
            "fillcolor": "#1a365d", "fontcolor": "white",
            "fontsize": "8", "width": "1.8", "height": "1.5", "fixedsize": "true"
        },
        "edge_router": {
            "shape": "box", "style": "filled,rounded",
            "fillcolor": "#2c5282", "fontcolor": "white",
            "fontsize": "8", "width": "1.8", "height": "1.5", "fixedsize": "true"
        },
        "agg_switch": {
            "shape": "box", "style": "filled,rounded",
            "fillcolor": "#3182ce", "fontcolor": "white",
            "fontsize": "8", "width": "1.8", "height": "1.5", "fixedsize": "true"
        },
        "other": {
            "shape": "ellipse", "style": "filled",
            "fillcolor": "#718096", "fontcolor": "white",
            "fontsize": "8", "width": "1.5", "height": "1.2", "fixedsize": "true"
        }
    },
    "edges": {
        "physical": {
            "fontsize": "7", "color": "#2d3748",
            "penwidth": "1.2", "labelloc": "c", "splines": "ortho"
        },
        "bgp": {
            "fontsize": "7", "color": "#e6194b",
            "penwidth": "1.0", "style": "dashed", "labelloc": "c", "dir": "back"
        }
    },
    "clusters": {
        "core": {
            "fillcolor": "#e6f7ff", "color": "#1890ff",
            "style": "filled,rounded", "penwidth": "1.5", "margin": "0.8"
        },
        "edge": {
            "fillcolor": "#fff2e6", "color": "#fa8c16",
            "style": "filled,rounded", "penwidth": "1.5", "margin": "0.8"
        },
        "agg": {
            "fillcolor": "#f0f9eb", "color": "#52c41a",
            "style": "filled,rounded", "penwidth": "1.5", "margin": "0.8"
        }
    },
    "graph": {
        "rankdir": "LR",
        "ranksep": "6.0",
        "nodesep": "2.5",
        "splines": "spline",
        "fontsize": "10", "fontname": "Arial", "bgcolor": "white"
    }
}

# ================= Pydantic 模型 =================
class GenerateRequest(BaseModel):
    config: dict
    style: Optional[Dict[str, Any]] = None
    output_name: str = "btd_network_topology"

# ================= 核心工具函数 =================
def parse_network_config(config: dict) -> dict:
    """解析新格式数据，生成绘制所需的结构化数据"""
    required_keys = ["network_devices", "network_connections"]
    if not all(key in config for key in required_keys):
        raise HTTPException(
            status_code=400,
            detail="数据必须包含 'network_devices' 和 'network_connections' 字段"
        )

    processed_devices = []
    ip_to_hostname = {}

    for device in config["network_devices"]:
        hostname = device.get("hostname")
        if not hostname:
            raise HTTPException(status_code=400, detail=f"设备缺少 hostname 字段")

        if hostname.startswith("C"):
            role = "core_router"
            cluster_type = "core"
        elif hostname.startswith("R"):
            role = "edge_router"
            cluster_type = "edge"
        elif hostname.startswith("S"):
            role = "agg_switch"
            cluster_type = "agg"
        else:
            role = "other"
            cluster_type = "other"

        loopback_info = device.get("loopback_interface", {})
        core_ip = loopback_info.get("ip_address", "").split("/")[0]
        if not core_ip:
            raise HTTPException(status_code=400, detail=f"设备 {hostname} 缺少 loopback_interface.ip_address")
        ip_to_hostname[core_ip] = hostname

        interfaces = device.get("interfaces", [])
        for iface in interfaces:
            if_ip = iface.get("ip_address", "").split("/")[0]
            if if_ip:
                ip_to_hostname[if_ip] = hostname

        bgp_info = device.get("routing_protocols", {}).get("bgp", {})
        as_number = bgp_info.get("asn")
        bgp_neighbors = [
            {
                "neighbor_ip": neigh.get("neighbor_ip", "").split("/")[0],
                "remote_as": neigh.get("remote_as"),
                "description": neigh.get("description", "")
            }
            for neigh in bgp_info.get("neighbors", [])
            if neigh.get("neighbor_ip")
        ]

        processed_devices.append({
            "hostname": hostname,
            "role": role,
            "cluster_type": cluster_type,
            "core_ip": core_ip,
            "as_number": as_number,
            "bgp_neighbors": bgp_neighbors,
            "interfaces": interfaces,
            "original_data": device  # 保留原始数据用于节点标签
        })

    return {
        "processed_devices": processed_devices,
        "network_connections": config["network_connections"],
        "ip_to_hostname": ip_to_hostname
    }


def create_device_node(device: dict, style_config: dict):
    """创建设备节点（展示更多属性）"""
    hostname = device.get("hostname", "未知设备")
    core_ip = device.get("core_ip", "无Loopback")
    as_number = device.get("as_number")
    as_text = f"AS:{as_number}" if as_number is not None else ""

    orig_device = device.get("original_data", {})
    device_model = orig_device.get("model", "")
    device_type = orig_device.get("device_type", "")
    ospf_process = orig_device.get("routing_protocols", {}).get("ospf", {}).get("process_id", "")
    ospf_text = f"OSPF:{ospf_process}" if ospf_process else ""

    label_lines = [
        hostname,
        f"Loopback:{core_ip}",
        as_text,
        f"{device_type}/{device_model}".strip("/"),
        ospf_text
    ]
    valid_label_lines = [line for line in label_lines if line.strip()]

    if len(valid_label_lines) > 5:
        valid_label_lines = valid_label_lines[:4] + ["..."]

    final_label = "\n".join(valid_label_lines)

    node_style = style_config["nodes"].get(
        device["role"],
        style_config["nodes"]["other"]
    ).copy()

    if device["role"] == "agg_switch":
        try:
            return Switch(final_label, **node_style)
        except TypeError:
            from diagrams.generic.blank import Blank
            return Blank(final_label, **node_style)
    else:
        try:
            return RouterNode(final_label, **node_style)
        except TypeError:
            from diagrams.generic.blank import Blank
            return Blank(final_label, **node_style)


def generate_topology_svg(parsed_config: dict, style_config: dict, output_name: str) -> str:
    """生成拓扑SVG"""
    processed_devices = parsed_config["processed_devices"]
    network_connections = parsed_config["network_connections"]
    ip_to_hostname = parsed_config["ip_to_hostname"]
    graph_style = style_config["graph"].copy()

    uid = uuid.uuid4().hex[:8]
    safe_output_name = f"{output_name}_{uid}"
    output_path = OUTPUT_DIR / safe_output_name

    try:
        diag = Diagram(
            name="BTD 网络拓扑图",
            filename=str(output_path),
            show=False,
            outformat="svg",
            graph_attr=graph_style
        )
    except TypeError:
        diag = Diagram(
            name="BTD 网络拓扑图",
            filename=str(output_path),
            show=False,
            graph_attr=graph_style
        )

    device_node_map = {}
    with diag:
        with Cluster("核心路由器区", graph_attr=style_config["clusters"]["core"]):
            for dev in processed_devices:
                if dev["cluster_type"] == "core":
                    node = create_device_node(dev, style_config)
                    device_node_map[dev["hostname"]] = node

        with Cluster("边缘路由器区", graph_attr=style_config["clusters"]["edge"]):
            for dev in processed_devices:
                if dev["cluster_type"] == "edge":
                    node = create_device_node(dev, style_config)
                    device_node_map[dev["hostname"]] = node

        with Cluster("汇聚交换机区", graph_attr=style_config["clusters"]["agg"]):
            for dev in processed_devices:
                if dev["cluster_type"] == "agg":
                    node = create_device_node(dev, style_config)
                    device_node_map[dev["hostname"]] = node

        physical_edge_style = style_config["edges"]["physical"].copy()
        drawn_physical_links = set()
        for conn in network_connections:
            src_dev = conn.get("source_device")
            dst_dev = conn.get("destination_device")
            src_if = conn.get("source_interface", "")
            dst_if = conn.get("destination_interface", "")
            if not (src_dev and dst_dev and src_dev in device_node_map and dst_dev in device_node_map):
                continue

            link_key = tuple(sorted([src_dev, dst_dev, src_if, dst_if]))
            if link_key in drawn_physical_links:
                continue
            drawn_physical_links.add(link_key)

            edge_label = f"{src_if}\n↓\n{dst_if}" if src_if and dst_if else ""
            physical_edge_style["label"] = edge_label

            device_node_map[src_dev] >> Edge(**physical_edge_style) >> device_node_map[dst_dev]

        bgp_edge_style = style_config["edges"]["bgp"].copy()
        drawn_bgp_links = set()
        bgp_allowed_roles = {"core_router", "edge_router"}

        for dev in processed_devices:
            if dev["role"] not in bgp_allowed_roles or not dev.get("bgp_neighbors"):
                continue

            src_dev_name = dev["hostname"]
            src_as = dev.get("as_number", "未知")

            for neigh in dev["bgp_neighbors"]:
                neigh_ip = neigh.get("neighbor_ip")
                neigh_as = neigh.get("remote_as", "未知")
                if not neigh_ip:
                    continue

                dst_dev_name = parsed_config["ip_to_hostname"].get(neigh_ip)
                if not (dst_dev_name and dst_dev_name in device_node_map):
                    continue

                dst_dev = next((d for d in processed_devices if d["hostname"] == dst_dev_name), None)
                if not dst_dev or dst_dev["role"] not in bgp_allowed_roles:
                    continue

                link_key = tuple(sorted([src_dev_name, dst_dev_name]))
                if link_key in drawn_bgp_links:
                    continue
                drawn_bgp_links.add(link_key)

                bgp_edge_style["label"] = f"BGP\nAS{src_as} ↔ AS{neigh_as}"

                device_node_map[src_dev_name] >> Edge(**bgp_edge_style) >> device_node_map[dst_dev_name]

    svg_file_path = output_path.with_suffix(".svg")
    if not svg_file_path.exists():
        raise RuntimeError("SVG生成失败，请检查 diagrams/graphviz 安装")

    with open(svg_file_path, "r", encoding="utf-8") as f:
        svg_content = f.read()
    svg_content = svg_content.replace(
        "/home/www/config_parsed_gen/venv/lib/python3.10/site-packages/resources/",
        "/icons/"
    )
    with open(svg_file_path, "w", encoding="utf-8") as f:
        f.write(svg_content)

    return f"/static/outputs/{svg_file_path.name}"

# ================= API 接口 =================
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    """默认前端页面"""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "default_style": json.dumps(DEFAULT_STYLE_CONFIG, indent=2),
            "example_config": json.dumps({
                "network_devices": [{"hostname": "C1-Test", "loopback_interface": {"name": "Loopback0", "ip_address": "10.0.0.1/32"}, "interfaces": [], "routing_protocols": {"bgp": {"asn": 65000, "neighbors": []}}}],
                "network_connections": []
            }, indent=2)
        }
    )


@app.post("/generate", response_class=JSONResponse)
async def generate_topology(req: GenerateRequest = Body(...)):
    """生成拓扑SVG接口"""
    try:
        style_config = DEFAULT_STYLE_CONFIG.copy()
        if req.style:
            def merge_dict(target, source):
                for k, v in source.items():
                    if isinstance(v, dict) and k in target and isinstance(target[k], dict):
                        merge_dict(target[k], v)
                    else:
                        target[k] = v
            merge_dict(style_config, req.style)

        parsed_config = parse_network_config(req.config)

        svg_url = generate_topology_svg(
            parsed_config=parsed_config,
            style_config=style_config,
            output_name=req.output_name
        )

        return {"svg_url": svg_url}

    except HTTPException as e:
        return {"code": e.status_code, "message": e.detail}
    except Exception as e:
        return {"code": 500, "message": f"生成失败：{str(e)}"}

# ================= 运行入口 =================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app="__main__:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1
    )