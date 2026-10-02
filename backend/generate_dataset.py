from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RNG = np.random.default_rng(42)


def build_network():
    # A small realistic urban-zone topology: loops + branches + a source tank.
    # Coordinates are used only for visualization and distance-aware simulation.
    nodes = []
    idx = 1
    coords = [
        (50, 8), (180, 8), (310, 8), (440, 8), (570, 8), (700, 8),
        (95, 105), (220, 105), (350, 105), (480, 105), (610, 105),
        (50, 205), (170, 205), (300, 205), (430, 205), (560, 205), (700, 205),
        (95, 300), (220, 300), (350, 300), (480, 300), (610, 300),
        (50, 400), (170, 400), (300, 400), (430, 400), (560, 400), (700, 400),
        (110, 505), (250, 505), (390, 505), (530, 505), (670, 505),
        (390, 610),
    ]
    elevations = np.linspace(8, 26, len(coords)) + RNG.normal(0, 1.5, len(coords))
    demand_base = RNG.uniform(1.2, 5.0, len(coords))
    for i, ((x, y), elev, demand) in enumerate(zip(coords, elevations, demand_base), start=1):
        nodes.append({
            "node_id": f"J{i:02d}",
            "x": x,
            "y": y,
            "elevation_m": round(float(elev), 2),
            "base_demand_lps": round(float(demand), 2),
            "type": "source" if i == 1 else "junction",
        })
    nodes[0]["type"] = "source"
    nodes[-1]["type"] = "tank"

    edges = []
    def add(a, b):
        edges.append({"source": f"J{a:02d}", "target": f"J{b:02d}"})

    # Horizontal corridors.
    for a, b in [(1,2),(2,3),(3,4),(4,5),(5,6),(12,13),(13,14),(14,15),(15,16),(16,17),
                 (23,24),(24,25),(25,26),(26,27),(28,29),(29,30),(30,31),(31,32),(33,34)]:
        add(a,b)
    # Vertical / diagonal links forming loops.
    for a,b in [(1,7),(2,8),(3,9),(4,10),(5,11),(12,18),(13,19),(14,20),(15,21),(16,22),
                (23,29),(24,30),(25,31),(26,32),(28,34),(29,34),(30,34),
                (7,8),(8,9),(9,10),(10,11),(18,19),(19,20),(20,21),(21,22),
                (7,13),(11,16),(18,24),(22,27),(23,29),(27,32),(33,34)]:
        add(a,b)
    # A few strategic cross-links.
    for a,b in [(8,14),(10,16),(19,25),(21,27),(29,31)]:
        add(a,b)

    # Remove duplicate undirected edges while preserving ordering.
    seen = set(); clean = []
    for e in edges:
        k = tuple(sorted((e["source"], e["target"])))
        if k not in seen:
            seen.add(k); clean.append(e)
    return pd.DataFrame(nodes), pd.DataFrame(clean)


def graph_distances(nodes: pd.DataFrame, edges: pd.DataFrame) -> np.ndarray:
    n = len(nodes)
    A = np.zeros((n,n), dtype=np.float32)
    for _, e in edges.iterrows():
        a = int(e.source[1:]) - 1
        b = int(e.target[1:]) - 1
        A[a,b] = A[b,a] = 1
    dist = np.full((n,n), 99.0, dtype=np.float32)
    for i in range(n):
        dist[i,i] = 0
        q = [i]
        for u in q:
            for v in np.where(A[u] > 0)[0]:
                if dist[i,v] > dist[i,u] + 1:
                    dist[i,v] = dist[i,u] + 1
                    q.append(int(v))
    return dist


def generate():
    nodes, edges = build_network()
    n = len(nodes)
    dist = graph_distances(nodes, edges)
    candidates = np.arange(1, n-1)  # avoid source and tank
    rows = []
    snapshots = []

    # 1200 graph snapshots. Each snapshot has 34 nodes and exactly one leak node
    # in 75% of cases; 25% are healthy network states.
    for sid in range(1200):
        leak_present = RNG.random() < 0.75
        leak_node = int(RNG.choice(candidates)) if leak_present else -1
        demand_scale = float(RNG.uniform(0.75, 1.35))
        pump_factor = float(RNG.uniform(0.92, 1.08))
        ambient_noise = RNG.normal(0, 1, n)

        base_demand = nodes.base_demand_lps.to_numpy() * demand_scale
        demand = base_demand * (1 + RNG.normal(0, 0.08, n))
        demand = np.clip(demand, 0.25, None)

        # Leak intensity is deliberately moderate: the network must be used as context.
        leak_strength = float(RNG.uniform(2.0, 8.0)) if leak_present else 0.0
        spatial = np.zeros(n)
        if leak_present:
            spatial = np.exp(-0.95 * dist[leak_node]) * leak_strength

        elevation = nodes.elevation_m.to_numpy()
        pressure = 62 * pump_factor - 0.42 * elevation - 0.70 * demand + ambient_noise * 0.65 - 0.95 * spatial
        pressure = np.clip(pressure, 18, 70)

        # Flow imbalance acts like a simplified residual / flow-balance sensor.
        neighbor_pressure = np.zeros(n)
        degree = np.zeros(n)
        for _, e in edges.iterrows():
            a = int(e.source[1:]) - 1; b = int(e.target[1:]) - 1
            delta = abs(pressure[a] - pressure[b])
            neighbor_pressure[a] += delta; neighbor_pressure[b] += delta
            degree[a] += 1; degree[b] += 1
        pressure_gradient = neighbor_pressure / np.maximum(degree, 1)
        flow_imbalance = np.abs(RNG.normal(0.0, 0.35, n) + 0.60 * spatial + 0.05 * demand)
        acoustic = np.clip(RNG.normal(0.10, 0.035, n) + 0.19 * spatial + 0.015 * pressure_gradient, 0, None)
        vibration = np.clip(RNG.normal(0.16, 0.05, n) + 0.11 * spatial + 0.02 * flow_imbalance, 0, None)
        flow_in = demand + flow_imbalance * 0.55
        flow_out = np.maximum(flow_in - demand, 0)

        for i in range(n):
            rows.append({
                "snapshot_id": sid,
                "node_id": nodes.node_id.iloc[i],
                "pressure_m": round(float(pressure[i]), 4),
                "demand_lps": round(float(demand[i]), 4),
                "flow_in_lps": round(float(flow_in[i]), 4),
                "flow_out_lps": round(float(flow_out[i]), 4),
                "pressure_gradient": round(float(pressure_gradient[i]), 4),
                "acoustic_index": round(float(acoustic[i]), 4),
                "vibration_index": round(float(vibration[i]), 4),
                "leak_label": 1 if i == leak_node else 0,
                "leak_node_id": nodes.node_id.iloc[leak_node] if leak_present else "NONE",
            })
        snapshots.append({"snapshot_id": sid, "leak_present": int(leak_present), "leak_node_id": nodes.node_id.iloc[leak_node] if leak_present else "NONE"})

    snap_df = pd.DataFrame(rows)
    meta = {
        "title": "AquaGraph — Smart Water Network Leak Localization Dataset",
        "description": "A compact, reproducible physics-inspired dataset generated on a 34-node urban water graph. It is intended for a runnable academic GCN demo; public benchmark links are included in README.",
        "num_nodes": n,
        "num_edges": int(len(edges)),
        "num_snapshots": 1200,
        "positive_snapshots": int(sum(s["leak_present"] for s in snapshots)),
        "features": ["pressure_m", "demand_lps", "flow_in_lps", "flow_out_lps", "pressure_gradient", "acoustic_index", "vibration_index", "elevation_m", "degree"],
        "random_seed": 42,
        "public_benchmarks": [
            "LeakDB",
            "LeakG3PD",
            "Zenodo WDN network operation state snapshots (L-Town/C-Town/etc.)"
        ],
    }
    DATA.mkdir(exist_ok=True)
    nodes.to_csv(DATA / "nodes.csv", index=False)
    edges.to_csv(DATA / "edges.csv", index=False)
    snap_df.to_csv(DATA / "snapshots.csv", index=False)
    with open(DATA / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Generated {n} nodes, {len(edges)} edges, {len(snapshots)} snapshots -> {DATA}")


if __name__ == "__main__":
    generate()
