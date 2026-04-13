import math
import random
from typing import Optional

import networkx as nx
import numpy as np
import json
import os

try:
    from dwave.samplers import SimulatedAnnealingSampler
except Exception:  # pragma: no cover
    SimulatedAnnealingSampler = None


def _optimize_layout(G, side, iterations=300, initial_pos=None, fixed=None):
    """Fruchterman-Reingold 型の力学モデルでレイアウト最適化（交差0保証）。

    力: (1) 全ペア反発 k²/d, (2) 辺引力 d²/k, (3) 頂点-辺反発,
        (4) 角度均等化, (5) ソフト境界引力。交差が生じる移動はリバート。
    """
    nodes = sorted(G.nodes())
    n = len(nodes)
    edges = list(G.edges())
    E = len(edges)
    edge_arr = np.array(edges, dtype=int) if E > 0 else np.zeros((0, 2), dtype=int)
    adj = [list(G.neighbors(v)) for v in range(n)]
    margin = side * 0.04
    fixed = set() if fixed is None else set(fixed)

    # 初期配置
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

    # 理想辺長 k = sqrt(area / n)（F-R の正規化定数）
    k = math.sqrt((side * side) / max(n, 2))
    k2 = k * k
    vertex_edge_radius = k * 0.6  # 頂点-辺反発の有効距離
    center = np.array([side / 2, side / 2])

    # 交差チェックで使う全辺端点の先行取得（pos のビューなので最新）
    e0_all = edge_arr[:, 0] if E > 0 else np.zeros(0, dtype=int)
    e1_all = edge_arr[:, 1] if E > 0 else np.zeros(0, dtype=int)
    adj_arr = [np.asarray(nbrs, dtype=int) for nbrs in adj]

    def has_crossing(v):
        """頂点 v の入射辺が非隣接辺と交差するかをベクトル化して判定。"""
        nbrs = adj_arr[v]
        if E == 0 or nbrs.size == 0:
            return False
        # 入射辺: (v, u) for u in nbrs
        a = pos[v]                      # (2,)
        b = pos[nbrs]                   # (M, 2)
        c = pos[e0_all]                 # (E, 2)
        d = pos[e1_all]                 # (E, 2)

        # (M, E) テンソルで符号付き外積を一括計算
        dcx = d[:, 0] - c[:, 0]
        dcy = d[:, 1] - c[:, 1]
        acy = a[1] - c[:, 1]            # (E,)
        acx = a[0] - c[:, 0]
        bcy = b[:, None, 1] - c[None, :, 1]   # (M, E)
        bcx = b[:, None, 0] - c[None, :, 0]

        d1 = dcx * acy - dcy * acx           # (E,)
        d2 = dcx[None, :] * bcy - dcy[None, :] * bcx   # (M, E)

        bax = b[:, None, 0] - a[0]           # (M, 1) broadcasted
        bay = b[:, None, 1] - a[1]
        cay = c[None, :, 1] - a[1]
        cax = c[None, :, 0] - a[0]
        day = d[None, :, 1] - a[1]
        dax = d[None, :, 0] - a[0]

        d3 = bax * cay - bay * cax           # (M, E)
        d4 = bax * day - bay * dax           # (M, E)

        cross = ((d1[None, :] > 0) != (d2 > 0)) & ((d3 > 0) != (d4 > 0))

        # 端点共有ペアを除外
        shares_v = (e0_all == v) | (e1_all == v)                         # (E,)
        shares_u = (e0_all[None, :] == nbrs[:, None]) | (e1_all[None, :] == nbrs[:, None])
        cross &= ~(shares_v[None, :] | shares_u)
        return bool(cross.any())

    for it in range(iterations):
        t = 1.0 - it / iterations
        step = k * 0.20 * t  # 最大変位（冷却スケジュール）
        if step < 1e-8:
            break

        disp = np.zeros((n, 2))

        # 力1: F-R 反発 k²/d（全頂点ペア）
        for i in range(n):
            diff = pos[i] - pos
            dist = np.linalg.norm(diff, axis=1)
            dist = np.maximum(dist, 1e-6)
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            f = diff[mask] / dist[mask, None] * (k2 / dist[mask, None])
            disp[i] += f.sum(axis=0)

        # 力2: F-R 引力 d²/k（辺端点間、ばね）
        for ei in range(E):
            u, v = edge_arr[ei]
            diff = pos[u] - pos[v]
            dist = max(np.linalg.norm(diff), 1e-6)
            f = diff / dist * (dist * dist / k)
            disp[u] -= f
            disp[v] += f

        # 力3: 頂点-辺反発（ベクトル化、非端点頂点が辺に近づきすぎるのを防ぐ）
        if E > 0:
            e0_arr = edge_arr[:, 0]
            e1_arr = edge_arr[:, 1]
            a_arr = pos[e0_arr]                         # (E, 2)
            ab = pos[e1_arr] - a_arr                    # (E, 2)
            ab2 = (ab * ab).sum(axis=1)                 # (E,)
            ab2_safe = np.maximum(ab2, 1e-12)

            rel = pos[:, None, :] - a_arr[None, :, :]   # (n, E, 2)
            tp = (rel * ab[None, :, :]).sum(axis=-1) / ab2_safe[None, :]
            tp = np.clip(tp, 0.0, 1.0)
            closest = a_arr[None, :, :] + tp[:, :, None] * ab[None, :, :]
            diff_ve = pos[:, None, :] - closest         # (n, E, 2)
            d_ve = np.linalg.norm(diff_ve, axis=-1)     # (n, E)

            vn_idx = np.arange(n)[:, None]
            endpoint = (vn_idx == e0_arr[None, :]) | (vn_idx == e1_arr[None, :])
            active = (~endpoint) & (d_ve > 1e-6) & (d_ve < vertex_edge_radius)
            d_safe = np.where(active, d_ve, 1.0)
            mag = (k2 * 0.35) * (1.0 / d_safe - 1.0 / vertex_edge_radius) / d_safe
            mag = np.where(active, mag, 0.0)
            force = diff_ve * (mag / d_safe)[:, :, None]   # (n, E, 2)

            disp += force.sum(axis=1)
            edge_force_sum = force.sum(axis=0)          # (E, 2)
            np.add.at(disp, e0_arr, -edge_force_sum * 0.4)
            np.add.at(disp, e1_arr, -edge_force_sum * 0.4)

        # 力4: 角度均等化（各頂点の周囲エッジを等角配置）
        for vn in range(n):
            nbrs = adj[vn]
            deg = len(nbrs)
            if deg < 2:
                continue
            angle_list = []
            for u in nbrs:
                d = pos[u] - pos[vn]
                angle_list.append((math.atan2(d[1], d[0]), u))
            angle_list.sort()
            for idx in range(deg):
                ac, uc = angle_list[idx]
                an = angle_list[(idx + 1) % deg][0]
                ap = angle_list[(idx - 1) % deg][0]
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
                disp[uc] += perp * dev * k * 0.4

        # 力5: ソフト中心引力（境界にべたつかせない）
        for vn in range(n):
            to_c = center - pos[vn]
            d = np.linalg.norm(to_c)
            if d > 1e-6:
                disp[vn] += to_c / d * (k * 0.05)

        # 適用（step でクランプ、交差したらリバート）
        order = list(range(n))
        random.shuffle(order)
        max_disp = 0.0
        for vn in order:
            if vn in fixed:
                continue
            d = np.linalg.norm(disp[vn])
            if d < 1e-10:
                continue
            if d > step:
                disp[vn] = disp[vn] / d * step
            new_pos = pos[vn] + disp[vn]
            # ソフト境界：枠外に出た分を減衰
            for dim in range(2):
                if new_pos[dim] < margin:
                    new_pos[dim] = margin + (new_pos[dim] - margin) * 0.3
                elif new_pos[dim] > side - margin:
                    new_pos[dim] = (side - margin) + (new_pos[dim] - (side - margin)) * 0.3
            new_pos = np.clip(new_pos, margin * 0.3, side - margin * 0.3)
            old_pos = pos[vn].copy()
            pos[vn] = new_pos
            if has_crossing(vn):
                pos[vn] = old_pos
            else:
                moved = np.linalg.norm(new_pos - old_pos)
                if moved > max_disp:
                    max_disp = moved

        # 収束判定
        if max_disp < k * 1e-3:
            break

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
    num_vertices: int,
    num_edges: Optional[int] = None,
    seed: Optional[int] = None,
) -> dict:
    """葉のない連結な平面グラフをランダム生成する（エッジ交差なし）。

    アルゴリズム:
    1. 全辺をランダム順に試し、平面性を保つ辺のみ追加して極大平面グラフを構築
    2. 連結性と最小次数2を維持しながらランダムに辺を削除して目標辺数に近づける
       （複数回シャッフル再試行）
    3. planar_layout を正方形にスケーリング → F-R 型の力で最適化（交差0保証）

    seed を指定すると同じグラフを再生成できる。
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

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

        # ステップ2: 辺を削除して目標辺数に近づける（橋を作らない、複数回シャッフル）
        removal_rounds = 5
        for _ in range(removal_rounds):
            if G.number_of_edges() <= num_edges:
                break
            edges_list = list(G.edges())
            random.shuffle(edges_list)
            removed_any = False
            for u, v in edges_list:
                if G.number_of_edges() <= num_edges:
                    break
                if G.degree(u) <= 2 or G.degree(v) <= 2:
                    continue
                G.remove_edge(u, v)
                if (not nx.is_connected(G)) or has_bridge(G):
                    G.add_edge(u, v)
                else:
                    removed_any = True
            if not removed_any:
                break

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
    pos = _optimize_layout(G, side, iterations=300, initial_pos=tut_pos, fixed=fixed)

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


def compute_planar_faces(
    num_vertices: int,
    edges: list,
    positions: dict,
    seed_face_index: Optional[int] = None,
    seed_orientation: Optional[str] = None,
) -> dict:
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

    # Face orientation propagation
    def face_area_signed(face_nodes):
        area = 0.0
        for i in range(len(face_nodes)):
            a = get_pos(face_nodes[i])
            b = get_pos(face_nodes[(i + 1) % len(face_nodes)])
            area += a["x"] * b["y"] - b["x"] * a["y"]
        return area * 0.5

    def is_cw(face_nodes):
        return face_area_signed(face_nodes) < 0

    def dir_in_face(face_nodes, u, v):
        for i in range(len(face_nodes)):
            a = face_nodes[i]
            b = face_nodes[(i + 1) % len(face_nodes)]
            if a == u and b == v:
                return 1
            if a == v and b == u:
                return -1
        return 0

    oriented_faces = [list(face) for face in faces]
    orientations = [None for _ in faces]

    if faces:
        # Build adjacency via shared edges
        edge_to_faces = {}
        for i, face in enumerate(faces):
            for j in range(len(face)):
                a = face[j]
                b = face[(j + 1) % len(face)]
                key = (min(a, b), max(a, b))
                edge_to_faces.setdefault(key, []).append(i)

        adj = [[] for _ in faces]
        for (u, v), flist in edge_to_faces.items():
            if len(flist) < 2:
                continue
            for i in range(len(flist)):
                for j in range(i + 1, len(flist)):
                    adj[flist[i]].append((flist[j], u, v))
                    adj[flist[j]].append((flist[i], u, v))

        orientation_flag = [None for _ in faces]  # 1: as-is, -1: reversed

        # Choose seed (outer face by default) and propagate alternation
        if seed_face_index is None or not (0 <= seed_face_index < len(faces)):
            seed_face_index = outer_idx if outer_idx is not None else 0

        desired = seed_orientation.lower() if seed_orientation else "cw"
        seed_is_cw = is_cw(faces[seed_face_index])
        if desired == "cw":
            orientation_flag[seed_face_index] = 1 if seed_is_cw else -1
        elif desired == "ccw":
            orientation_flag[seed_face_index] = 1 if not seed_is_cw else -1
        else:
            orientation_flag[seed_face_index] = 1

        queue = [seed_face_index]
        while queue:
            f = queue.pop(0)
            for nb, u, v in adj[f]:
                if orientation_flag[nb] is not None:
                    continue
                d_f = dir_in_face(faces[f], u, v)
                d_nb = dir_in_face(faces[nb], u, v)
                if d_f == 0 or d_nb == 0:
                    continue
                same_dir = d_f == d_nb
                orientation_flag[nb] = -orientation_flag[f] if same_dir else orientation_flag[f]
                queue.append(nb)

        # Fill any unassigned with their current orientation
        for i in range(len(faces)):
            if orientation_flag[i] is None:
                orientation_flag[i] = 1

        for i in range(len(faces)):
            if orientation_flag[i] == -1:
                oriented_faces[i] = list(reversed(oriented_faces[i]))
            orientations[i] = "cw" if is_cw(oriented_faces[i]) else "ccw"

    return {
        "faces": faces,
        "outer_face_index": outer_idx,
        "oriented_faces": oriented_faces,
        "orientations": orientations,
    }


def optimize_edge_orientations(
    num_vertices: int,
    edges: list,
    fixed_edges: Optional[list] = None,
    max_iterations: int = 1200,
    random_seed: Optional[int] = None,
) -> dict:
    """QUBO近似で辺向きを最適化して、混雑集中を抑えた有向化を返す。"""
    n = int(num_vertices)
    if n < 2:
        raise ValueError("頂点数は2以上が必要です")

    undirected_edges = []
    seen_undirected = set()
    for e in edges:
        u = int(e["source"])
        v = int(e["target"])
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        if key in seen_undirected:
            continue
        seen_undirected.add(key)
        undirected_edges.append(key)

    if not undirected_edges:
        raise ValueError("エッジがありません")

    rng = random.Random(random_seed)

    fixed_map = {}
    fixed_edges = fixed_edges or []
    for e in fixed_edges:
        u = int(e["source"])
        v = int(e["target"])
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        if key not in seen_undirected:
            continue
        direction = (u, v) if u < v else (v, u)
        # keyに対して向き+1: min->max, -1: max->min
        sign = 1 if (u == key[0] and v == key[1]) else -1
        if key in fixed_map and fixed_map[key] != sign:
            raise ValueError("固定辺の向きに矛盾があります")
        fixed_map[key] = sign

    UG = nx.Graph()
    UG.add_nodes_from(range(n))
    UG.add_edges_from(undirected_edges)

    all_od = [(s, t) for s in range(n) for t in range(n) if s != t]
    max_od_pairs = min(len(all_od), max(60, n * 6))
    if len(all_od) > max_od_pairs:
        od_pairs = rng.sample(all_od, max_od_pairs)
    else:
        od_pairs = all_od

    # 1) 各ODで候補経路を作る（上位K本）
    k_paths = 3
    path_pool = []
    for s, t in od_pairs:
        candidates = []
        try:
            gen = nx.shortest_simple_paths(UG, s, t)
            for _ in range(k_paths):
                p = next(gen)
                candidates.append(p)
        except (nx.NetworkXNoPath, StopIteration):
            continue
        if not candidates:
            continue
        path_pool.append({"od": (s, t), "paths": candidates})

    if not path_pool:
        raise ValueError("候補経路を作成できませんでした")

    # 2) QUBOを構築: 1変数=ODの候補経路選択
    #    E = A*one-hot + B*length + C*congestion^2 + D*fixed方向違反
    A = 30.0
    B = 1.0
    C = 0.12
    D = 40.0

    var_meta = []  # {"k": od_index, "p": path_index, "path": [...]}
    od_to_vars = [[] for _ in range(len(path_pool))]
    edge_to_vars = {e: [] for e in undirected_edges}

    for k_idx, bundle in enumerate(path_pool):
        for p_idx, path in enumerate(bundle["paths"]):
            v_idx = len(var_meta)
            var_meta.append({"k": k_idx, "p": p_idx, "path": path})
            od_to_vars[k_idx].append(v_idx)
            for i in range(len(path) - 1):
                key = (min(path[i], path[i + 1]), max(path[i], path[i + 1]))
                if key in edge_to_vars:
                    edge_to_vars[key].append(v_idx)

    m = len(var_meta)
    linear = [0.0 for _ in range(m)]
    quad = {}

    def add_quad(i, j, c):
        if i == j:
            linear[i] += c
            return
        a, b = (i, j) if i < j else (j, i)
        quad[(a, b)] = quad.get((a, b), 0.0) + c

    # one-hot penalty
    for vars_k in od_to_vars:
        for i in vars_k:
            linear[i] += -A
        for i in range(len(vars_k)):
            for j in range(i + 1, len(vars_k)):
                add_quad(vars_k[i], vars_k[j], 2.0 * A)

    # path length term + fixed方向違反
    for i, meta in enumerate(var_meta):
        path = meta["path"]
        linear[i] += B * float(len(path) - 1)
        mismatch = 0
        for idx in range(len(path) - 1):
            u = path[idx]
            v = path[idx + 1]
            key = (min(u, v), max(u, v))
            if key not in fixed_map:
                continue
            sign = 1 if (u == key[0] and v == key[1]) else -1
            if sign != fixed_map[key]:
                mismatch += 1
        if mismatch > 0:
            linear[i] += D * float(mismatch)

    # congestion square term
    for key, vars_e in edge_to_vars.items():
        for i in vars_e:
            linear[i] += C
        for i in range(len(vars_e)):
            vi = vars_e[i]
            for j in range(i + 1, len(vars_e)):
                vj = vars_e[j]
                add_quad(vi, vj, 2.0 * C)

    # 3) QUBOを解く（D-Wave SimulatedAnnealingSampler）
    def energy(x):
        e = 0.0
        for i in range(m):
            if x[i]:
                e += linear[i]
        for (i, j), c in quad.items():
            if x[i] and x[j]:
                e += c
        return e

    if SimulatedAnnealingSampler is None:
        raise ValueError("dwave.samplers が見つかりません。.venv に dwave-samplers をインストールしてください")

    qubo = {}
    for i, c in enumerate(linear):
        if abs(c) > 1e-12:
            qubo[(i, i)] = c
    for (i, j), c in quad.items():
        if abs(c) > 1e-12:
            qubo[(i, j)] = c

    sampler = SimulatedAnnealingSampler()
    num_reads = max(30, min(300, int(max_iterations // 4)))
    sampleset = sampler.sample_qubo(qubo, num_reads=num_reads)
    raw = sampleset.first.sample
    best_state = [1 if raw.get(i, 0) else 0 for i in range(m)]

    # one-hot違反を局所修復
    for vars_k in od_to_vars:
        chosen = [i for i in vars_k if best_state[i] == 1]
        if len(chosen) == 1:
            continue
        best_i = vars_k[0]
        best_e_local = None
        for cand in vars_k:
            prev = [best_state[i] for i in vars_k]
            for i in vars_k:
                best_state[i] = 1 if i == cand else 0
            e_val = energy(best_state)
            if best_e_local is None or e_val < best_e_local:
                best_e_local = e_val
                best_i = cand
            for idx, i in enumerate(vars_k):
                best_state[i] = prev[idx]
        for i in vars_k:
            best_state[i] = 1 if i == best_i else 0

    best_e = energy(best_state)

    # 4) 選択経路から辺向きを決める（固定辺優先、残りは通過向き多数決）
    flow_sign_sum = {e: 0 for e in undirected_edges}
    for i, meta in enumerate(var_meta):
        if best_state[i] != 1:
            continue
        path = meta["path"]
        for idx in range(len(path) - 1):
            u = path[idx]
            v = path[idx + 1]
            key = (min(u, v), max(u, v))
            flow_sign_sum[key] += 1 if (u == key[0] and v == key[1]) else -1

    best_orient = {}
    for key in undirected_edges:
        if key in fixed_map:
            best_orient[key] = fixed_map[key]
        else:
            ssum = flow_sign_sum.get(key, 0)
            if ssum == 0:
                best_orient[key] = 1 if rng.random() < 0.5 else -1
            else:
                best_orient[key] = 1 if ssum > 0 else -1

    directed_edges = []
    for a, b in undirected_edges:
        sign = best_orient[(a, b)]
        if sign == 1:
            directed_edges.append({"source": a, "target": b})
        else:
            directed_edges.append({"source": b, "target": a})

    best_metrics = _evaluate_orientation_metrics(n, undirected_edges, best_orient)

    return {
        "directed_edges": directed_edges,
        "fixed_edge_count": len(fixed_map),
        "optimized_edge_count": len([e for e in undirected_edges if e not in fixed_map]),
        "qubo_variable_count": m,
        "qubo_od_count": len(path_pool),
        "qubo_energy": float(best_e),
        "metrics": best_metrics,
    }


def _evaluate_orientation_metrics(num_vertices: int, undirected_edges: list, orientation: dict) -> dict:
    DG = nx.DiGraph()
    DG.add_nodes_from(range(num_vertices))
    for a, b in undirected_edges:
        sign = orientation[(a, b)]
        if sign == 1:
            DG.add_edge(a, b)
        else:
            DG.add_edge(b, a)

    pair_total = num_vertices * (num_vertices - 1)
    if pair_total == 0:
        return {
            "average_distance": 0.0,
            "unreachable_pairs": 0,
            "load_square_sum": 0.0,
            "max_load": 0.0,
        }

    # 辺負荷は無向キーで集計
    load = {(a, b): 0 for a, b in undirected_edges}
    distance_sum = 0.0
    unreachable = 0
    reachable = 0
    unreachable_by_source = [0 for _ in range(num_vertices)]

    for s in range(num_vertices):
        lengths, paths = nx.single_source_dijkstra(DG, s)
        for t in range(num_vertices):
            if s == t:
                continue
            if t not in lengths:
                unreachable += 1
                unreachable_by_source[s] += 1
                continue
            reachable += 1
            distance_sum += float(lengths[t])
            path = paths[t]
            for i in range(len(path) - 1):
                u = path[i]
                v = path[i + 1]
                key = (min(u, v), max(u, v))
                if key in load:
                    load[key] += 1

    avg_dist = distance_sum / reachable if reachable > 0 else float(num_vertices * 10)
    load_values = list(load.values())
    load_square_sum = float(sum(v * v for v in load_values))
    max_load = float(max(load_values)) if load_values else 0.0

    return {
        "average_distance": avg_dist,
        "unreachable_pairs": int(unreachable),
        "unreachable_vertices": [i for i, c in enumerate(unreachable_by_source) if c > 0],
        "unreachable_vertex_count": int(sum(1 for c in unreachable_by_source if c > 0)),
        "load_square_sum": load_square_sum,
        "max_load": max_load,
    }


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
