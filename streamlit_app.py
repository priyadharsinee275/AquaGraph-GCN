from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"
DATA = BACKEND / "data"
ART = BACKEND / "artifacts"
sys.path.insert(0, str(BACKEND))
from model import VanillaGCN  # noqa: E402

st.set_page_config(
    page_title="AquaGraph • Vanilla GCN",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEATURES = [
    "pressure_m", "demand_lps", "flow_in_lps", "flow_out_lps",
    "pressure_gradient", "acoustic_index", "vibration_index", "elevation_m", "degree"
]


def css():
    st.markdown(
        """
        <style>
        .stApp { background: linear-gradient(135deg,#07111f 0%,#081a2a 45%,#062b38 100%); color:#eaf9ff; }
        [data-testid="stHeader"] { background: rgba(0,0,0,0); }
        [data-testid="stSidebar"] { background: rgba(4,15,27,.94); border-right:1px solid rgba(91,220,255,.16); }
        .hero { padding: 24px 30px; border-radius:24px; margin-bottom:18px; background:linear-gradient(110deg,rgba(10,44,67,.95),rgba(0,73,86,.55)); border:1px solid rgba(94,220,255,.24); box-shadow:0 0 35px rgba(22,200,255,.10); }
        .hero h1 { margin:0; font-size:34px; letter-spacing:-.8px; }
        .hero p { margin:8px 0 0; color:#acd9e7; font-size:15px; }
        .pill { display:inline-block; padding:5px 10px; margin:0 5px 0 0; border-radius:999px; background:rgba(82,216,255,.11); border:1px solid rgba(82,216,255,.28); color:#9cecff; font-size:12px; }
        .metric-card { padding:15px 17px; border-radius:18px; background:rgba(9,31,49,.80); border:1px solid rgba(109,223,255,.13); }
        .metric-label { color:#7fb0c0; font-size:12px; }
        .metric-value { color:#e9fbff; font-size:25px; font-weight:800; }
        .section { margin:8px 0 12px; font-weight:800; font-size:18px; }
        .danger { color:#ff8f9f; font-weight:800; }
        .safe { color:#6fe8bc; font-weight:800; }
        div[data-testid="stMetric"] { background:rgba(10,33,50,.75); padding:12px 14px; border-radius:16px; border:1px solid rgba(100,220,255,.12); }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data

def load_data():
    nodes = pd.read_csv(DATA / "nodes.csv")
    edges = pd.read_csv(DATA / "edges.csv")
    snapshots = pd.read_csv(DATA / "snapshots.csv")
    degree = pd.concat([edges.source, edges.target]).value_counts()
    nodes["degree"] = nodes.node_id.map(degree).fillna(0).astype(float)
    return nodes, edges, snapshots


@st.cache_resource

def load_model():
    with open(ART / "scaler.json", encoding="utf-8") as f:
        scaler = json.load(f)
    checkpoint = torch.load(ART / "gcn_model.pt", map_location="cpu")
    model = VanillaGCN(checkpoint["in_features"], checkpoint["hidden"], 2)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, scaler


@st.cache_data

def make_adjacency(nodes: pd.DataFrame, edges: pd.DataFrame):
    idx = {nid: i for i, nid in enumerate(nodes.node_id)}
    n = len(nodes)
    A = torch.zeros((n, n), dtype=torch.float32)
    for e in edges.itertuples():
        a, b = idx[e.source], idx[e.target]
        A[a, b] = A[b, a] = 1
    A += torch.eye(n)
    d = A.sum(dim=1)
    D = torch.diag(torch.pow(d, -0.5))
    return D @ A @ D, idx


@st.cache_data

def predict_snapshot(snapshot_id: int):
    nodes, edges, snapshots = load_data()
    model, scaler = load_model()
    ADJ, idx = make_adjacency(nodes, edges)
    raw = snapshots[snapshots.snapshot_id == snapshot_id].sort_values("node_id").copy()
    work = raw.copy()
    node_lookup = nodes.set_index("node_id")
    for f in FEATURES:
        if f not in work:
            work[f] = node_lookup.loc[work.node_id, f].to_numpy()
        work[f] = (work[f] - scaler[f]["mean"]) / scaler[f]["std"]
    x = torch.tensor(work[FEATURES].to_numpy(np.float32))
    with torch.no_grad():
        logits, hidden = model.forward_with_activations(x, ADJ)
        probs = torch.softmax(logits, dim=1)[:, 1].numpy()
    out = raw.copy()
    out["risk"] = probs
    out["rank"] = out.risk.rank(method="first", ascending=False).astype(int)
    return out.sort_values("risk", ascending=False).reset_index(drop=True), hidden.numpy(), logits.numpy()


def network_figure(nodes, edges, pred, selected_node=None):
    pred_map = pred.set_index("node_id")["risk"]
    node_x = nodes["x"].astype(float)
    node_y = nodes["y"].astype(float)

    ex, ey = [], []
    for e in edges.itertuples():
        a = nodes[nodes.node_id == e.source].iloc[0]
        b = nodes[nodes.node_id == e.target].iloc[0]
        ex += [a.x, b.x, None]
        ey += [a.y, b.y, None]

    edge_trace = go.Scatter(
        x=ex, y=ey, mode="lines", line=dict(width=2.6), hoverinfo="none", name="Pipes"
    )
    risks = nodes.node_id.map(pred_map).fillna(0).to_numpy()
    sizes = 16 + 28 * risks
    if selected_node:
        sizes = np.where(nodes.node_id.to_numpy() == selected_node, 34, sizes)
    hover = [f"<b>{nid}</b><br>Leak risk: {pred_map.get(nid,0)*100:.1f}%<br>Elevation: {e:.1f} m" for nid, e in zip(nodes.node_id, nodes.elevation_m)]
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=nodes.node_id, textposition="top center", textfont=dict(size=10),
        hovertemplate="%{hovertext}<extra></extra>", hovertext=hover,
        marker=dict(size=sizes, color=risks, colorscale="Turbo", cmin=0, cmax=1, showscale=True,
                    colorbar=dict(title="Leak risk", tickformat=".0%"), line=dict(width=2)),
        name="Junctions",
    )
    fig = go.Figure([edge_trace, node_trace])
    fig.update_layout(
        height=600, margin=dict(l=10,r=10,t=10,b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
        showlegend=False, hovermode="closest",
    )
    return fig


css()
nodes, edges, snapshots = load_data()
model, scaler = load_model()
metrics_path = ART / "metrics.json"
metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}

st.markdown(
    """
    <div class="hero">
      <span class="pill">💧 WATER NETWORK</span><span class="pill">🧠 VANILLA GCN</span><span class="pill">⚡ 2 LAYERS</span>
      <h1>AquaGraph — Leak Localization Dashboard</h1>
      <p>Explore the graph, inspect sensor evidence, and see how a 2-layer GCN converts neighborhood information into a node-level leak prediction.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("🎛️ Controls")
    snap_ids = sorted(snapshots.snapshot_id.unique().tolist())
    snapshot_id = st.selectbox("Network snapshot", snap_ids, index=0)
    pred, hidden, logits = predict_snapshot(int(snapshot_id))
    node_ids = nodes.node_id.tolist()
    selected_node = st.selectbox("Inspect junction", node_ids, index=0)
    risk = float(pred.loc[pred.node_id == selected_node, "risk"].iloc[0])
    st.divider()
    st.caption("Model architecture")
    st.code("Input 9\n   ↓\nGCN Layer 1: 9 → 36 + ReLU\n   ↓\nGCN Layer 2: 36 → 2\n   ↓\nSoftmax: Normal / Leak")
    st.caption("Tip: use the graph controls to zoom and pan. Select any junction here to inspect it.")

leak_present = bool(pred.leak_label.sum() > 0)
gt = str(pred.loc[pred.leak_label == 1, "node_id"].iloc[0]) if leak_present else "NONE"
top_node = str(pred.iloc[0].node_id)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Junctions", len(nodes))
m2.metric("Pipes", len(edges))
m3.metric("Top suspect", top_node)
m4.metric("Top risk", f"{pred.iloc[0].risk*100:.1f}%")
m5.metric("Actual leak", gt if leak_present else "None")

st.markdown("<div class='section'>🌐 Interactive water network</div>", unsafe_allow_html=True)
st.plotly_chart(network_figure(nodes, edges, pred, selected_node), use_container_width=True, config={"scrollZoom": True, "displaylogo": False})

left, right = st.columns([1.1, .9])
with left:
    st.markdown("<div class='section'>🚨 Top leak suspects</div>", unsafe_allow_html=True)
    table = pred[["rank","node_id","risk","pressure_m","acoustic_index","vibration_index"]].head(10).copy()
    table["risk"] = (table["risk"] * 100).round(1).astype(str) + "%"
    table.columns = ["Rank", "Junction", "Leak Risk", "Pressure (m)", "Acoustic", "Vibration"]
    st.dataframe(table, use_container_width=True, hide_index=True)

with right:
    row = pred[pred.node_id == selected_node].iloc[0]
    i = node_ids.index(selected_node)
    st.markdown(f"<div class='section'>🔎 Inspector — {selected_node}</div>", unsafe_allow_html=True)
    st.metric("Leak probability", f"{row.risk*100:.1f}%")
    st.progress(float(row.risk))
    f1, f2 = st.columns(2)
    f1.metric("Pressure", f"{row.pressure_m:.2f} m")
    f2.metric("Pressure gradient", f"{row.pressure_gradient:.2f}")
    f1.metric("Acoustic index", f"{row.acoustic_index:.3f}")
    f2.metric("Vibration index", f"{row.vibration_index:.3f}")
    st.caption(f"Ground truth: {'LEAK' if bool(row.leak_label) else 'NORMAL'}")

st.markdown("<div class='section'>🧠 GCN Model Lab</div>", unsafe_allow_html=True)
t1, t2, t3 = st.tabs(["Architecture", "Layer 1 activation", "Layer 2 output"])
with t1:
    a,b,c = st.columns(3)
    a.metric("Input features", 9)
    b.metric("Hidden units", 36)
    c.metric("Output classes", 2)
    st.latex(r"H^{(l+1)}=\sigma\left(\hat D^{-1/2}\hat A\hat D^{-1/2}H^{(l)}W^{(l)}\right)")
    st.write("The first GCN layer aggregates each junction's sensor features from its graph neighborhood. The second GCN layer aggregates the learned 36-dimensional representation and produces Normal/Leak logits.")
with t2:
    vals = hidden[i]
    fig_h = go.Figure(go.Bar(x=np.arange(1, len(vals)+1), y=vals))
    fig_h.update_layout(height=320, margin=dict(l=10,r=10,t=10,b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis_title="Layer 1 channel", yaxis_title="ReLU activation")
    st.plotly_chart(fig_h, use_container_width=True)
    st.caption(f"Junction {selected_node}: {int((vals>0).sum())} / 36 hidden units are active after ReLU.")
with t3:
    lv = logits[i]
    fig_o = go.Figure(go.Bar(x=["Normal","Leak"], y=lv))
    fig_o.update_layout(height=320, margin=dict(l=10,r=10,t=10,b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis_title="Logit")
    st.plotly_chart(fig_o, use_container_width=True)
    st.write(f"Final prediction for **{selected_node}**: **{'LEAK' if risk >= 0.5 else 'NORMAL'}**.")

st.markdown("<div class='section'>📊 Model evaluation</div>", unsafe_allow_html=True)
c1,c2,c3,c4 = st.columns(4)
test_metrics = metrics.get("test", metrics)
c1.metric("Top-1 localization", f"{test_metrics.get('top1_localization', 0)*100:.1f}%")
c2.metric("Top-3 localization", f"{test_metrics.get('top3_localization', 0)*100:.1f}%")
c3.metric("ROC-AUC", f"{test_metrics.get('node_roc_auc', 0)*100:.1f}%")
c4.metric("Layers", "2")

st.caption("AquaGraph • Vanilla 2-Layer GCN • Synthetic physics-inspired benchmark network")
