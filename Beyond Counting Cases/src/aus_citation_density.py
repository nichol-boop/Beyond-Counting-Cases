#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
aus_citation_density.py
Builds a directed citation graph from the Australian Excel file where:
  - Column A (0): Case ID
  - Column B (1): Case name
  - Column E (4): 'Cited by' (a free-text list of citer IDs)

Computes density metrics and saves a JSON summary to the outputs folder.

Usage (from repo root or notebooks):
  python ../src/aus_citation_density.py --data "../data/Completed Australian case set.xlsx" --outputs "../outputs"
"""

from pathlib import Path
from datetime import datetime
from collections import Counter
import argparse
import json
import re
import math

import pandas as pd
import networkx as nx


def clean_id(val):
    """Return canonical string ID: trims, converts 12.0->'12', drops empties/NaN."""
    if pd.isna(val):
        return None
    s = str(val).strip()
    if s == "" or s.lower() == "nan":
        return None
    try:
        f = float(s.replace(",", ""))
        if math.isfinite(f) and f.is_integer():
            return str(int(f))
        return str(f)
    except Exception:
        if s.endswith(".0"):
            s = s[:-2]
        return s


def parse_citer_ids(cell):
    """
    Parse the 'Cited by' cell into a list of canonical string IDs.
    Handles delimiters: commas, semicolons, spaces, newlines, brackets, etc.
    Example cell: "1, 2; 3\n[4]" -> ["1","2","3","4"]
    """
    if pd.isna(cell):
        return []
    return [str(int(m)) for m in re.findall(r"\d+", str(cell))]


def build_graph_from_excel(
    xlsx_path: Path,
    sheet_name=0,
    col_case_name=1,       # B
    col_node_id=0,         # A
    col_cited_by=4,        # E (single column with list of citer IDs)
):
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=0)

    # Nodes: ID with label from case name
    df_nodes = df.iloc[:, [col_case_name, col_node_id]].copy()
    df_nodes.columns = ["case_name", "node_id"]
    df_nodes["node_id"] = df_nodes["node_id"].apply(clean_id)
    df_nodes = df_nodes[df_nodes["node_id"].notna()].copy()

    # Edges: for each row (target), parse citer IDs from Column E
    edges = []
    for _, row in df.iterrows():
        cited_id = clean_id(row.iloc[col_node_id])
        if not cited_id:
            continue
        for citer_id in parse_citer_ids(row.iloc[col_cited_by]):
            if citer_id and citer_id != cited_id:
                edges.append((citer_id, cited_id))  # citer -> cited

    # Collapse duplicate edges to weights
    edge_counts = Counter(edges)

    G = nx.DiGraph()

    # Add nodes with labels
    for _, r in df_nodes.iterrows():
        G.add_node(r["node_id"], label=str(r["case_name"]).strip() if pd.notna(r["case_name"]) else r["node_id"])

    # Ensure citer-only nodes are present; add weighted edges
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
    parser = argparse.ArgumentParser(description="Compute density for Australian citation network.")
    parser.add_argument("--data", required=True, help="Path to Australian Excel file (e.g., '../data/Completed Australian case set.xlsx')")
    parser.add_argument("--outputs", default="../outputs", help="Folder to write results")
    parser.add_argument("--sheet", type=int, default=0, help="Sheet index (default 0)")
    # Advanced: override columns if needed (A=0, B=1, E=4 by default)
    parser.add_argument("--col-case", type=int, default=1, help="Column index for case name (B=1)")
    parser.add_argument("--col-id", type=int, default=0, help="Column index for node id (A=0)")
    parser.add_argument("--col-citedby", type=int, default=4, help="Column index for 'Cited by' (E=4)")
    args = parser.parse_args()

    data_path = Path(args.data)
    outputs = Path(args.outputs); outputs.mkdir(parents=True, exist_ok=True)

    G = build_graph_from_excel(
        data_path,
        sheet_name=args.sheet,
        col_case_name=args.col_case,
        col_node_id=args.col_id,
        col_cited_by=args.col_citedby,
    )

    summary = compute_densities(G)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    summary.update({
        "created": stamp,
        "source": str(data_path),
        "columns": {"case_name": args.col_case, "node_id": args.col_id, "cited_by": args.col_citedby},
    })

    out_json = outputs / f"network_density_summary_AUS_{stamp}.json"
    out_latest = outputs / "network_density_summary_AUS_latest.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    out_latest.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
