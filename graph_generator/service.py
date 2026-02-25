import math
import random
from typing import Optional

import networkx as nx
import numpy as np
import json
import os


def _optimize_layout(G, side, iterations=150, initial_pos=None, fixed=None):
    """planar_layout を正方形に引き伸ばし、4種の力で最適化する（交差0保証）。"""
    nodes = sorted(G.nodes())
    n = len(nodes)
    edges = list(G.edges())
    E = len(edges)
    edge_arr = np.array(edges, dtype=int)
    adj = [list(G.neighbors(v)) for v in range(n)]
    margin = side * 0.04
    fixed = set() if fixed is None else set(fixed)

    # 初期配置: planar_layout → 正方形全体にスケーリング（アフィン変換は交差0を保持）
    planar_pos = initial_pos if initial_pos is not None else nx.planar_layout(G)
    pos = np.zeros((n, 2))
    for v in range(n):
        pos[v] = planar_pos[v]

    for dim in range(2):
        mn, mx = pos[:, dim].min(), pos[:, dim].max()
        rng = mx - mn
        if rng > 1e-10:
            pos[:, dim] = (pos[:, dim] - mn) / rng * (side - 2 * margin) + margin
        else:
            pos[:, dim] = side / 2

    def seg_cross(a, b, c, d):
        d1 = (d[0]-c[0])*(a[1]-c[1]) - (d[1]-c[1])*(a[0]-c[0])
        d2 = (d[0]-c[0])*(b[1]-c[1]) - (d[1]-c[1])*(b[0]-c[0])
        d3 = (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
        d4 = (b[0]-a[0])*(d[1]-a[1]) - (b[1]-a[1])*(d[0]-a[0])
        return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))

    def has_crossing(v):
        pv = pos[v]
        for u in adj[v]:
            pu = pos[u]
            for ei in range(E):
                e0, e1 = edge_arr[ei]
                if e0 == v or e1 == v or e0 == u or e1 == u:
                    continue
                if seg_cross(pv, pu, pos[e0], pos[e1]):
                    return True
        return False

    for it in range(iterations):
        t = 1.0 - it / iterations
        step = side * 0.07 * t
        if step < 1e-8:
            break

        disp = np.zeros((n, 2))

        # 力1: 全頂点ペア間の反発（位置分散の最大化）
        for i in range(n):
            diff = pos[i] - pos
            dist = np.linalg.norm(diff, axis=1)
            dist = np.maximum(dist, 1e-6)
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            f = diff[mask] / dist[mask, None] * (step / dist[mask, None])
            disp[i] += f.sum(axis=0)

        # 力2: エッジ長の総和を最大化
        for ei in range(E):
            u, v = edge_arr[ei]
            diff = pos[u] - pos[v]
            dist = max(np.linalg.norm(diff), 1e-6)
            f = diff / dist * (step * 3.0 / dist)
            disp[u] += f
            disp[v] -= f

        # 力3: 隣接辺間の距離を最大化
        for vn in range(n):
            nbrs = adj[vn]
            for i in range(len(nbrs)):
                for j in range(i + 1, len(nbrs)):
                    u, w = nbrs[i], nbrs[j]
                    diff = pos[u] - pos[w]
                    dist = max(np.linalg.norm(diff), 1e-6)
                    f = diff / dist * (step * 2.0 / dist)
                    disp[u] += f
                    disp[w] -= f

        # 力4: 角度均等化
        for vn in range(n):
            nbrs = adj[vn]
            k = len(nbrs)
            if k < 2:
                continue
            angle_list = []
            for u in nbrs:
                d = pos[u] - pos[vn]
                angle_list.append((math.atan2(d[1], d[0]), u))
            angle_list.sort()
            for idx in range(k):
                _, uc = angle_list[idx]
                ac = angle_list[idx][0]
                an = angle_list[(idx + 1) % k][0]
                ap = angle_list[(idx - 1) % k][0]
                ga = an - ac
                if ga <= 0: ga += 2.0 * math.pi
                gb = ac - ap
                if gb <= 0: gb += 2.0 * math.pi
                dev = (ga - gb) * 0.5
                du = np.linalg.norm(pos[uc] - pos[vn])
                if du < 1e-6:
                    continue
                dv = (pos[uc] - pos[vn]) / du
                perp = np.array([-dv[1], dv[0]])
                disp[uc] += perp * dev * step * 1.5

        # 力を適用（交差が生じる移動はリバート）
        order = list(range(n))
        random.shuffle(order)
        for vn in order:
            if vn in fixed:
                continue
            d = np.linalg.norm(disp[vn])
            if d < 1e-10:
                continue
            if d > step:
                disp[vn] = disp[vn] / d * step
            new_pos = np.clip(pos[vn] + disp[vn], margin, side - margin)
            old_pos = pos[vn].copy()
            pos[vn] = new_pos
            if has_crossing(vn):
                pos[vn] = old_pos

    return {v: pos[v] for v in range(n)}


def _tutte_layout(G, outer_face, side, iterations=600):
    """外面を凸多角形に固定し、内部を重心で緩和する。"""
    n = G.number_of_nodes()
    pos = np.zeros((n, 2))
    margin = side * 0.08
    radius = (side - 2 * margin) * 0.5
    center = np.array([side / 2, side / 2])

    boundary = list(outer_face)
    m = len(boundary)
    for i, v in enumerate(boundary):
        theta = 2.0 * math.pi * i / m
        pos[v] = center + radius * np.array([math.cos(theta), math.sin(theta)])

    interior = [v for v in range(n) if v not in boundary]
    if interior:
        init = nx.planar_layout(G)
        for v in interior:
            pos[v] = init[v]

        # scale interior into box
        for dim in range(2):
            mn, mx = pos[interior, dim].min(), pos[interior, dim].max()
            rng = mx - mn
            if rng > 1e-10:
                pos[interior, dim] = (pos[interior, dim] - mn) / rng * (side - 2 * margin) + margin
            else:
                pos[interior, dim] = side / 2

        # barycentric relaxation
        for _ in range(iterations):
            max_delta = 0.0
            for v in interior:
                nbrs = list(G.neighbors(v))
                if not nbrs:
                    continue
                avg = pos[nbrs].mean(axis=0)
                delta = np.linalg.norm(avg - pos[v])
                pos[v] = avg
                if delta > max_delta:
                    max_delta = delta
            if max_delta < 1e-4:
                break

    return {v: pos[v] for v in range(n)}, set(boundary)


def generate_connected_graph(
    num_vertices: int, num_edges: Optional[int] = None
) -> dict:
    """葉のない連結な平面グラフをランダム生成する（エッジ交差なし）。

    アルゴリズム:
    1. 全辺をランダム順に試し、平面性を保つ辺のみ追加して極大平面グラフを構築
    2. 連結性と最小次数2を維持しながらランダムに辺を削除して目標辺数に近づける
    3. planar_layout を正方形にスケーリング → 4種の力で最適化（交差0保証）

    任意の連結・葉なし・平面グラフを生成するポテンシャルを持つ。
    """
    n = num_vertices
    if n < 3:
        raise ValueError("葉なし連結グラフには頂点数3以上が必要です")

    min_edges = n
    max_edges = 3 * n - 6

    if num_edges is None:
        num_edges = random.randint(min_edges, max_edges)
    else:
        if num_edges < min_edges:
            raise ValueError(
                f"葉なし連結グラフには最低 {min_edges} 辺が必要です"
            )
        if num_edges > max_edges:
            raise ValueError(
                f"平面グラフの場合、頂点数 {num_vertices} の最大辺数は {max_edges} です"
            )

    def has_bridge(graph: nx.Graph) -> bool:
        return any(True for _ in nx.bridges(graph))

    # 2-edge-connected (no bridges), min degree >=2, planar, connected
    max_attempts = 30
    for _ in range(max_attempts):
        G = nx.Graph()
        G.add_nodes_from(range(n))

        # ステップ1: ランダムな極大平面グラフを構築
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_edges)

        planar_max = 3 * n - 6
        for u, v in all_edges:
            if G.number_of_edges() >= planar_max:
                break
            G.add_edge(u, v)
            if not nx.check_planarity(G)[0]:
                G.remove_edge(u, v)

        # 極大平面グラフは橋を持たないはずだが、安全確認
        if has_bridge(G):
            continue

        # ステップ2: 辺を削除して目標辺数に近づける（橋を作らない）
        edges_list = list(G.edges())
        random.shuffle(edges_list)

        for u, v in edges_list:
            if G.number_of_edges() <= num_edges:
                break
            if G.degree(u) <= 2 or G.degree(v) <= 2:
                continue
            G.remove_edge(u, v)
            if (not nx.is_connected(G)) or has_bridge(G):
                G.add_edge(u, v)

        if G.number_of_edges() == num_edges and nx.is_connected(G) and not has_bridge(G):
            break
    else:
        raise ValueError("条件を満たすグラフ生成に失敗しました（橋なし条件）")

    # ステップ3: レイアウト最適化
    side = math.sqrt(n)
    planar, embedding = nx.check_planarity(G)
    if not planar:
        raise ValueError("グラフが平面ではありません")

    # 外面を推定（面積最大）
    init_pos = nx.planar_layout(G)
    faces = []
    seen = set()
    for u in embedding:
        for v in embedding.neighbors_cw_order(u):
            if (u, v) in seen:
                continue
            face = embedding.traverse_face(u, v)
            if len(face) < 3:
                continue
            for i in range(len(face)):
                a = face[i]
                b = face[(i + 1) % len(face)]
                seen.add((a, b))
            faces.append(face)

    def face_area(face_nodes):
        area = 0.0
        for i in range(len(face_nodes)):
            a = init_pos[face_nodes[i]]
            b = init_pos[face_nodes[(i + 1) % len(face_nodes)]]
            area += a[0] * b[1] - b[0] * a[1]
        return abs(area) * 0.5

    outer_face = max(faces, key=face_area) if faces else list(G.nodes())
    tut_pos, fixed = _tutte_layout(G, outer_face, side, iterations=600)
    pos = _optimize_layout(G, side, iterations=150, initial_pos=tut_pos, fixed=fixed)

    positions = {
        int(v): {"x": float(pos[v][0]), "y": float(pos[v][1])}
        for v in G.nodes()
    }

    edges_out = [
        {"source": int(u), "target": int(v)} for u, v in sorted(G.edges())
    ]

    return {
        "num_vertices": n,
        "num_edges": G.number_of_edges(),
        "edges": edges_out,
        "positions": positions,
    }


def compute_planar_faces(num_vertices: int, edges: list, positions: dict) -> dict:
    """平面埋め込みの面（閉領域）を列挙して返す。

    positions は outer face を判定するために使用する。
    """
    n = num_vertices
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for e in edges:
        G.add_edge(int(e["source"]), int(e["target"]))

    planar, embedding = nx.check_planarity(G)
    if not planar:
        raise ValueError("グラフが平面ではありません")

    # directed edge ごとに面をたどって列挙
    seen = set()
    faces = []
    for u in embedding:
        for v in embedding.neighbors_cw_order(u):
            if (u, v) in seen:
                continue
            face = embedding.traverse_face(u, v)
            if len(face) < 3:
                # 面として無効
                continue
            # mark directed edges along this face
            for i in range(len(face)):
                a = face[i]
                b = face[(i + 1) % len(face)]
                seen.add((a, b))
            faces.append(face)

    if not faces:
        return {"faces": [], "outer_face_index": None}

    # outer face を面積最大の多角形として推定
    def get_pos(node_id):
        if node_id in positions:
            return positions[node_id]
        return positions.get(str(node_id))

    def polygon_area(face_nodes):
        area = 0.0
        for i in range(len(face_nodes)):
            a = get_pos(face_nodes[i])
            b = get_pos(face_nodes[(i + 1) % len(face_nodes)])
            area += a["x"] * b["y"] - b["x"] * a["y"]
        return area * 0.5

    areas = [abs(polygon_area(face)) for face in faces]
    outer_idx = int(np.argmax(areas)) if areas else None

    return {"faces": faces, "outer_face_index": outer_idx}


_SAVED_DIR = os.path.join(os.path.dirname(__file__), "saved")


def _sanitize_name(name: str) -> str:
    name = name.strip()
    if not name:
        raise ValueError("保存名が空です")
    safe = "".join(ch for ch in name if ch.isalnum() or ch in ("-", "_"))
    if not safe:
        raise ValueError("保存名に使える文字がありません")
    return safe


def save_graph_json(name: str, data: dict) -> dict:
    os.makedirs(_SAVED_DIR, exist_ok=True)
    safe = _sanitize_name(name)
    path = os.path.join(_SAVED_DIR, f"{safe}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return {"name": safe, "path": f"{safe}.json"}


def list_saved_graphs() -> list:
    if not os.path.isdir(_SAVED_DIR):
        return []
    files = []
    for fn in os.listdir(_SAVED_DIR):
        if fn.endswith(".json"):
            files.append(fn[:-5])
    files.sort()
    return files


def load_graph_json(name: str) -> dict:
    safe = _sanitize_name(name)
    path = os.path.join(_SAVED_DIR, f"{safe}.json")
    if not os.path.isfile(path):
        raise ValueError("保存されたグラフが見つかりません")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
