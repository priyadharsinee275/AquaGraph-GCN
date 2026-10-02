from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

try:
    from .model import VanillaGCN
except ImportError:
    from model import VanillaGCN

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ART = ROOT / "artifacts"
FEATURES = [
    "pressure_m", "demand_lps", "flow_in_lps", "flow_out_lps",
    "pressure_gradient", "acoustic_index", "vibration_index", "elevation_m", "degree"
]

nodes = pd.read_csv(DATA / "nodes.csv")
edges = pd.read_csv(DATA / "edges.csv")
snapshots = pd.read_csv(DATA / "snapshots.csv")

degree = pd.concat([edges.source, edges.target]).value_counts()
nodes["degree"] = nodes.node_id.map(degree).fillna(0).astype(float)
node_lookup = nodes.set_index("node_id")

with open(ART / "scaler.json", encoding="utf-8") as f:
    scaler = json.load(f)
with open(ART / "metrics.json", encoding="utf-8") as f:
    metrics = json.load(f)

checkpoint = torch.load(ART / "gcn_model.pt", map_location="cpu")
model = VanillaGCN(checkpoint["in_features"], checkpoint["hidden"], 2)
model.load_state_dict(checkpoint["state_dict"])
model.eval()

n = len(nodes)
idx = {nid: i for i, nid in enumerate(nodes.node_id)}
A = torch.zeros((n, n), dtype=torch.float32)
for _, e in edges.iterrows():
    a, b = idx[e.source], idx[e.target]
    A[a, b] = A[b, a] = 1
A += torch.eye(n)
d = A.sum(dim=1)
D = torch.diag(torch.pow(d, -0.5))
ADJ = D @ A @ D

neighbors = {nid: [] for nid in nodes.node_id}
for _, e in edges.iterrows():
    neighbors[e.source].append(e.target)
    neighbors[e.target].append(e.source)

app = FastAPI(title="AquaGraph GCN API", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def get_graph_snapshot(snapshot_id: int):
    valid = set(snapshots.snapshot_id.unique())
    if snapshot_id not in valid:
        raise HTTPException(404, "Snapshot not found")
    raw = snapshots[snapshots.snapshot_id == snapshot_id].sort_values("node_id").copy()
    work = raw.copy()
    for f in FEATURES:
        if f not in work:
            work[f] = node_lookup.loc[work.node_id, f].to_numpy()
        work[f] = (work[f] - scaler[f]["mean"]) / scaler[f]["std"]
    x = torch.tensor(work[FEATURES].to_numpy(np.float32))
    return raw, x


def predict_frame(snapshot_id: int):
    raw, x = get_graph_snapshot(snapshot_id)
    with torch.no_grad():
        logits, hidden = model.forward_with_activations(x, ADJ)
        probs = torch.softmax(logits, dim=1)[:, 1].numpy()

    out = []
    for i, r in enumerate(raw.itertuples()):
        out.append({
            "node_id": r.node_id,
            "risk": round(float(probs[i]), 4),
            "pressure_m": round(float(r.pressure_m), 2),
            "demand_lps": round(float(r.demand_lps), 2),
            "flow_in_lps": round(float(r.flow_in_lps), 2),
            "flow_out_lps": round(float(r.flow_out_lps), 2),
            "pressure_gradient": round(float(r.pressure_gradient), 2),
            "acoustic_index": round(float(r.acoustic_index), 3),
            "vibration_index": round(float(r.vibration_index), 3),
            "elevation_m": round(float(r.elevation_m), 2) if hasattr(r, "elevation_m") else round(float(node_lookup.loc[r.node_id, "elevation_m"]), 2),
            "ground_truth": bool(r.leak_label),
        })
    out.sort(key=lambda z: z["risk"], reverse=True)
    leak_node = str(raw.iloc[0].leak_node_id)
    return {
        "snapshot_id": snapshot_id,
        "leak_present": leak_node != "NONE",
        "ground_truth_node": leak_node,
        "predictions": out,
        "top_suspects": out[:5],
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "AquaGraph Vanilla GCN", "model_layers": 2}


@app.get("/api/network")
def network():
    return {
        "nodes": nodes.to_dict(orient="records"),
        "edges": edges.to_dict(orient="records"),
        "neighbors": neighbors,
    }


@app.get("/api/snapshots")
def snapshot_list():
    base = snapshots.groupby("snapshot_id").agg(leak_present=("leak_label", "sum")).reset_index()
    base["leak_present"] = (base.leak_present > 0).astype(int)
    return base.to_dict(orient="records")


@app.get("/api/metrics")
def get_metrics():
    return metrics


@app.get("/api/model-info")
def model_info():
    return {
        "name": "Vanilla GCN",
        "layers": 2,
        "layer_1": {"name": "GCN Layer 1", "input": 9, "output": 36, "activation": "ReLU"},
        "layer_2": {"name": "GCN Layer 2", "input": 36, "output": 2, "activation": "Softmax (output)"},
        "features": FEATURES,
        "formula": "H^(l+1) = sigma(D_hat^-1/2 A_hat D_hat^-1/2 H^l W^l)",
        "notes": "Pure PyTorch vanilla GCN with two explicit graph convolution layers; no PyTorch Geometric, GAT, GraphSAGE, or attention.",
    }


@app.get("/api/predict/{snapshot_id}")
def predict(snapshot_id: int):
    return predict_frame(snapshot_id)


@app.get("/api/node-detail/{snapshot_id}/{node_id}")
def node_detail(snapshot_id: int, node_id: str):
    raw, x = get_graph_snapshot(snapshot_id)
    if node_id not in idx:
        raise HTTPException(404, "Node not found")
    with torch.no_grad():
        logits, hidden = model.forward_with_activations(x, ADJ)
        probs = torch.softmax(logits, dim=1)[:, 1].numpy()
    i = idx[node_id]
    row = raw[raw.node_id == node_id].iloc[0]
    h = hidden[i].numpy()
    active = np.where(h > 0)[0]
    top_channels = np.argsort(h)[-6:][::-1]
    return {
        "node_id": node_id,
        "risk": round(float(probs[i]), 4),
        "ground_truth": bool(row.leak_label),
        "neighbors": neighbors[node_id],
        "feature_names": FEATURES,
        "feature_values": [round(float(x[i, j]), 3) for j in range(len(FEATURES))],
        "layer1": {
            "units": int(hidden.shape[1]),
            "active_units": int(len(active)),
            "top_channels": [{"channel": int(c + 1), "activation": round(float(h[c]), 3)} for c in top_channels],
        },
        "layer2_logits": [round(float(v), 3) for v in logits[i].numpy()],
    }


@app.get("/api/demo")
def demo():
    leaky = snapshots.groupby("snapshot_id").leak_label.sum()
    sid = int(leaky[leaky > 0].index[0])
    return predict_frame(sid)
