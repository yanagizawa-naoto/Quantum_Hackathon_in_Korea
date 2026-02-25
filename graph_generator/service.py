import math
import random
from typing import Optional

import networkx as nx
import numpy as np


def _optimize_layout(G, side, iterations=150):
    """planar_layout を正方形に引き伸ばし、4種の力で最適化する（交差0保証）。"""
    nodes = sorted(G.nodes())
    n = len(nodes)
    edges = list(G.edges())
    E = len(edges)
    edge_arr = np.array(edges, dtype=int)
    adj = [list(G.neighbors(v)) for v in range(n)]
    margin = side * 0.04

    # 初期配置: planar_layout → 正方形全体にスケーリング（アフィン変換は交差0を保持）
    planar_pos = nx.planar_layout(G)
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

    # ステップ2: 辺を削除して目標辺数に近づける
    edges_list = list(G.edges())
    random.shuffle(edges_list)

    for u, v in edges_list:
        if G.number_of_edges() <= num_edges:
            break
        if G.degree(u) <= 2 or G.degree(v) <= 2:
            continue
        G.remove_edge(u, v)
        if not nx.is_connected(G):
            G.add_edge(u, v)

    # ステップ3: レイアウト最適化
    side = math.sqrt(n)
    pos = _optimize_layout(G, side, iterations=150)

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
