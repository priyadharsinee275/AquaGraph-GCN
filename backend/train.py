from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from model import VanillaGCN

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)

FEATURES = [
    "pressure_m", "demand_lps", "flow_in_lps", "flow_out_lps",
    "pressure_gradient", "acoustic_index", "vibration_index", "elevation_m", "degree"
]


def normalized_adjacency(edges: pd.DataFrame, nodes: pd.DataFrame):
    n = len(nodes)
    A = torch.zeros((n, n), dtype=torch.float32)
    idx = {nid: i for i, nid in enumerate(nodes.node_id)}
    for _, e in edges.iterrows():
        a, b = idx[e.source], idx[e.target]
        A[a,b] = A[b,a] = 1.0
    A += torch.eye(n)
    d = A.sum(dim=1)
    d_inv_sqrt = torch.pow(d, -0.5)
    D = torch.diag(d_inv_sqrt)
    return D @ A @ D


def prepare():
    nodes = pd.read_csv(DATA / "nodes.csv")
    edges = pd.read_csv(DATA / "edges.csv")
    snaps = pd.read_csv(DATA / "snapshots.csv")
    degree = pd.concat([edges.source, edges.target]).value_counts()
    nodes["degree"] = nodes.node_id.map(degree).fillna(0).astype(float)
    snaps = snaps.merge(nodes[["node_id", "elevation_m", "degree"]], on="node_id", how="left")
    scaler = {}
    for f in FEATURES:
        m = float(snaps[f].mean()); s = float(snaps[f].std() or 1.0)
        scaler[f] = {"mean": m, "std": s}
        snaps[f] = (snaps[f] - m) / s
    graphs=[]
    for sid in sorted(snaps.snapshot_id.unique()):
        g=snaps[snaps.snapshot_id==sid].sort_values("node_id")
        graphs.append((sid, g[FEATURES].to_numpy(np.float32), g.leak_label.to_numpy(np.int64), g.leak_node_id.iloc[0]))
    rng=np.random.default_rng(SEED); sids=np.array([g[0] for g in graphs]); rng.shuffle(sids)
    return nodes, edges, graphs, normalized_adjacency(edges,nodes), scaler, set(sids[:840]), set(sids[840:1020]), set(sids[1020:])


def batch_arrays(graphs):
    X=torch.tensor(np.stack([g[1] for g in graphs]), dtype=torch.float32)
    Y=torch.tensor(np.stack([g[2] for g in graphs]), dtype=torch.long)
    return X,Y


def evaluate(model, graphs, adj, threshold=0.5):
    model.eval()
    X,Y=batch_arrays(graphs)
    with torch.no_grad():
        logits=model(X,adj)  # [B,N,2]
        probs=torch.softmax(logits,dim=-1)[:,:,1].numpy()
    ys=Y.numpy(); pred=(probs>=threshold).astype(int)
    y_all=ys.reshape(-1).tolist(); p_all=pred.reshape(-1).tolist()
    ranks=[]; top1=top3=considered=0
    for i,g in enumerate(graphs):
        if g[3] != "NONE":
            considered += 1
            true_idx=int(np.where(ys[i]==1)[0][0]); order=np.argsort(-probs[i]); rank=int(np.where(order==true_idx)[0][0])+1
            ranks.append(rank); top1 += rank==1; top3 += rank<=3
    return {
        "node_precision": float(precision_score(y_all,p_all,zero_division=0)),
        "node_recall": float(recall_score(y_all,p_all,zero_division=0)),
        "node_f1": float(f1_score(y_all,p_all,zero_division=0)),
        "node_roc_auc": float(roc_auc_score(y_all,probs.reshape(-1))) if len(set(y_all))==2 else 0.0,
        "leak_snapshots": int(considered),
        "top1_localization": float(top1/max(1,considered)),
        "top3_localization": float(top3/max(1,considered)),
        "mean_rank": float(np.mean(ranks)) if ranks else 0.0,
    }


def train():
    nodes,edges,graphs,adj,scaler,train_ids,val_ids,test_ids=prepare()
    train_graphs=[g for g in graphs if g[0] in train_ids]; val_graphs=[g for g in graphs if g[0] in val_ids]; test_graphs=[g for g in graphs if g[0] in test_ids]
    model=VanillaGCN(len(FEATURES), 36, 2)
    opt=torch.optim.Adam(model.parameters(),lr=0.008,weight_decay=1e-4)
    weight=torch.tensor([1.0,8.0],dtype=torch.float32)
    X_train,Y_train=batch_arrays(train_graphs)
    best_state=None; best_val=-1; history=[]; patience=0
    for epoch in range(1,61):
        model.train(); opt.zero_grad(); logits=model(X_train,adj)
        loss=torch.nn.functional.cross_entropy(logits.reshape(-1,2),Y_train.reshape(-1),weight=weight)
        loss.backward(); opt.step()
        val=evaluate(model,val_graphs,adj); history.append({"epoch":epoch,"loss":float(loss.item()),"val_top1":val["top1_localization"],"val_top3":val["top3_localization"]})
        if val["top1_localization"] > best_val + 1e-6:
            best_val=val["top1_localization"]; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; patience=0
        else:
            patience += 1
        if epoch % 10 == 0: print(f"epoch={epoch:02d} loss={loss.item():.4f} val_top1={val['top1_localization']:.3f} val_top3={val['top3_localization']:.3f}")
        if patience>=12: break
    model.load_state_dict(best_state)
    # Calibrate the node-classification threshold on validation snapshots.
    model.eval(); VX,VY=batch_arrays(val_graphs)
    with torch.no_grad(): vprobs=torch.softmax(model(VX,adj),dim=-1)[:,:,1].numpy()
    vy=VY.numpy().reshape(-1); best_threshold=0.5; best_f1=-1
    for t in np.linspace(0.25,0.90,66):
        f=f1_score(vy,(vprobs.reshape(-1)>=t).astype(int),zero_division=0)
        if f>best_f1: best_f1=float(f); best_threshold=float(t)
    test=evaluate(model,test_graphs,adj,threshold=best_threshold); train_m=evaluate(model,train_graphs,adj,threshold=best_threshold)
    torch.save({"state_dict":model.state_dict(),"in_features":len(FEATURES),"hidden":36,"features":FEATURES},ART/"gcn_model.pt")
    with open(ART/"scaler.json","w") as f: json.dump(scaler,f,indent=2)
    with open(ART/"metrics.json","w") as f: json.dump({"train":train_m,"test":test,"classification_threshold":best_threshold,"history":history},f,indent=2)
    print(json.dumps(test,indent=2))

if __name__=="__main__": train()
