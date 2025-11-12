#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
uk_citation_density.py
Builds a directed citation graph from an Excel file (A: case name, B: node id, C–K: citer IDs),
computes density metrics, and saves a JSON summary to the outputs folder.

Usage (from repo root or notebooks):
  python ../src/uk_citation_density.py --data "../data/Aus-time-matched simple UK dataset.xlsx" --outputs "../outputs"
"""

from pathlib import Path
from datetime import datetime
from collections import Counter
import argparse
import json

import pandas as pd
import networkx as nx

def build_graph_from_excel(
    xlsx_path: Path,
    sheet_name=0,
    col_case_name=0,
    col_node_id=1,
    citer_cols_slice=(2, 11),  # C..K inclusive (end-exclusive 11)
):
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=0)

    # Nodes: B (id) with label from A (name)
    df_nodes = df.iloc[:, [col_case_name, col_node_id]].copy()
    df_nodes.columns = ["case_name", "node_id"]
    df_nodes["node_id"] = df_nodes["node_id"].astype(str).str.strip()

    # Edges: for each row, each numeric in C..K is a citer of this row's node_id
    edges = []
    for _, row in df.iterrows():
        cited_id = str(row.iloc[col_node_id]).strip()
        if cited_id in ("", "nan", "None"):
            continue
        citer_values = row.iloc[slice(*citer_cols_slice)].values
        citers = pd.to_numeric(pd.Series(citer_values), errors="coerce").dropna()
        for citer_id in citers.astype(int).astype(str):
            if citer_id != cited_id:
                edges.append((citer_id, cited_id))  # citer -> cited

    # Collapse duplicate edges to weights
    edge_counts = Counter(edges)

    G = nx.DiGraph()
    # Add nodes with labels
    for _, r in df_nodes.iterrows():
        G.add_node(r["node_id"], label=r["case_name"])
    # Ensure citer-only nodes are present and add edges
    for (u, v), w in edge_counts.items():
        if u not in G:
            G.add_node(u, label=u)
        if v not in G:
            G.add_node(v, label=v)
        G.add_edge(u, v, weight=float(w))

    return G

def compute_densities(G: nx.DiGraph):
    n = G.number_of_nodes()
    m = G.number_of_edges()
    dens_directed = nx.density(G)                         # m / (n*(n-1))
    dens_undirected = nx.density(G.to_undirected())       # undirected projection
    total_w = sum(d.get("weight", 1.0) for _, _, d in G.edges(data=True))
    max_possible_dir = n * (n - 1) if n > 1 else 0
    weighted_density = (total_w / max_possible_dir) if max_possible_dir else 0.0

    return {
        "nodes": n,
        "edges": m,
        "density_directed": float(dens_directed),
        "density_undirected": float(dens_undirected),
        "weighted_density_directed": float(weighted_density),
    }

def main():
    parser = argparse.ArgumentParser(description="Compute density for UK citation network.")
    parser.add_argument("--data", required=True, help="Path to Excel file")
    parser.add_argument("--outputs", default="../outputs", help="Folder to write results")
    parser.add_argument("--sheet", type=int, default=0, help="Sheet index (default 0)")
    # Advanced: override columns if needed
    parser.add_argument("--col-case", type=int, default=0, help="Column index for case name (A=0)")
    parser.add_argument("--col-id", type=int, default=1, help="Column index for node id (B=1)")
    parser.add_argument("--citer-start", type=int, default=2, help="Start index for citer cols (C=2)")
    parser.add_argument("--citer-end", type=int, default=11, help="End-exclusive index for citer cols (K->11)")
    args = parser.parse_args()

    data_path = Path(args.data)
    outputs = Path(args.outputs); outputs.mkdir(parents=True, exist_ok=True)

    G = build_graph_from_excel(
        data_path,
        sheet_name=args.sheet,
        col_case_name=args.col_case,
        col_node_id=args.col_id,
        citer_cols_slice=(args.citer_start, args.citer_end),
    )

    summary = compute_densities(G)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    summary.update({
        "created": stamp,
        "source": str(data_path),
        "citer_cols": [args.citer_start, args.citer_end],
    })

    out_json = outputs / f"network_density_summary_{stamp}.json"
    out_latest = outputs / "network_density_summary_latest.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    out_latest.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
