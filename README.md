# AquaGraph — Vanilla GCN Water Leak Localization

A complete VS Code project for a real-world graph learning problem: localizing the most likely leaking junction in a water-distribution network from node sensor readings.

## What is actually implemented?

This is an explicit **2-layer vanilla Graph Convolutional Network**:

1. **GCN Layer 1:** 9 input features → 36 hidden units + ReLU
2. **GCN Layer 2:** 36 hidden units → 2 output classes (Normal / Leak)

The graph convolution is implemented directly with PyTorch using the normalized adjacency matrix. PyTorch Geometric, GAT, GraphSAGE and attention mechanisms are not used.

### Core equation

`H^(l+1) = sigma(D_hat^-1/2 A_hat D_hat^-1/2 H^l W^l)`

## Interactive UI

- Animated command-center dashboard
- Pan and zoom the graph
- Click nodes and top suspects
- One-hop neighbor highlighting
- Live snapshot playback
- Node sensor inspector
- Layer 1 activation visualization
- Layer 2 output visualization
- Animated inference status
- Responsive mobile/tablet/laptop layout

## Run in VS Code

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cd backend
python train.py
uvicorn main:app --reload --port 8000
```

Open another terminal:

```powershell
cd frontend
python -m http.server 5173
```

Open `http://127.0.0.1:5173`.

## Dataset

The included dataset is a deterministic, synthetic/physics-inspired water-network dataset designed for reproducibility. It contains 1,200 graph snapshots across 34 nodes and 54 pipe edges with 9 node features.

For a research version, the same pipeline can be adapted to public benchmark scenarios such as LeakDB / LeakG3PD / EPANET-derived networks.


## Streamlit version

The project can be run as a single Streamlit app; FastAPI and the old frontend are not required for the Streamlit UI.

```powershell
pip install -r requirements-streamlit.txt
streamlit run streamlit_app.py
```

Open the URL shown by Streamlit, usually `http://localhost:8501`.
