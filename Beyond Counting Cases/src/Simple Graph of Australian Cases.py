#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple Graph of Australian Cases
"""

# --- Build a citation network from "Completed Australian case set.xlsx"
# Node ID = Column A; "Cited by" = Column E (edge source list) ---

import re
import math
import pandas as pd
import networkx as nx
from pathlib import Path
from collections import Counter
from pyvis.network import Network

# ---- paths ----
DATA_PATH = Path("../data/Completed Australian case set.xlsx")
OUTPUTS = Path("../outputs"); OUTPUTS.mkdir(parents=True, exist_ok=True)
HTML_OUT = OUTPUTS / "Aus_citation_network_pyvis.html"
GEXF_OUT = OUTPUTS / "Aus_citation_network.gexf"

# ---- load first sheet ----
df = pd.read_excel(DATA_PATH, sheet_name=0, header=0)

# Column indices (0-based): A=0, B=1, C=2, D=3, E=4
COL_NODE_ID   = 0   # A: Node ID (1..138)
COL_CASE_NAME = 1   # B: (optional) Case name/title, if present
COL_CITED_BY  = 4   # E: "cited by" (IDs of cases that cite this row)

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
    Parse 'cited by' cell into a list of canonical string IDs.
    Handles delimiters: commas, semicolons, spaces, newlines, brackets, etc.
    """
    if pd.isna(cell):
        return []
    s = str(cell)
    # Extract all integers in the text (supports '1, 2; 3\n4 [5]')
    nums = re.findall(r"\d+", s)
    return [str(int(n)) for n in nums]

def safe_label(val, fallback):
    """String label safe for PyVis titles."""
    if pd.isna(val):
        return str(fallback)
    s = str(val).strip()
    return s if s else str(fallback)

# ---- prepare nodes (IDs + labels if available) ----
nodes_df = df.iloc[:, [COL_NODE_ID]].copy()
nodes_df.columns = ["node_id_raw"]
nodes_df["node_id"] = nodes_df["node_id_raw"].apply(clean_id)
nodes_df = nodes_df[nodes_df["node_id"].notna()].drop_duplicates(subset=["node_id"]).copy()

# Optional case names from column B if present
if df.shape[1] > COL_CASE_NAME:
    nodes_df["label"] = [
        safe_label(df.iloc[i, COL_CASE_NAME], nid)
        for i, nid in enumerate(nodes_df["node_id"])
    ]
else:
    nodes_df["label"] = nodes_df["node_id"]

# ---- build edges from 'cited by' column E ----
edges_raw = []
for _, row in df.iterrows():
    target = clean_id(row.iloc[COL_NODE_ID])  # the case being cited
    if not target:
        continue
    citers = parse_citer_ids(row.iloc[COL_CITED_BY])
    for citer in citers:
        if not citer or citer == target:
            continue  # drop self-loops and empties
        edges_raw.append((citer, target))  # directed: citer -> cited(target)

# Collapse duplicate edges to weights
edge_counts = Counter(edges_raw)

# ---- build DiGraph ----
G = nx.DiGraph()

# Add nodes (include any citer-only nodes as well)
all_ids = set(nodes_df["node_id"]).union({u for u, _ in edge_counts}).union({v for _, v in edge_counts})
id_to_label = dict(zip(nodes_df["node_id"], nodes_df["label"]))

for nid in all_ids:
    G.add_node(nid, label=id_to_label.get(nid, nid))

for (u, v), w in edge_counts.items():
    G.add_edge(u, v, weight=float(w))

print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

# ---- export GEXF (for Gephi) ----
try:
    nx.write_gexf(G, GEXF_OUT)
    print(f"GEXF written to: {GEXF_OUT}")
except Exception as e:
    print(f"Warning: could not write GEXF ({e})")

# ---- PyVis HTML ----
net = Network(height="800px", width="100%", directed=True, notebook=False, cdn_resources="in_line")
net.barnes_hut()

# Add nodes with safe string titles
for n, attrs in G.nodes(data=True):
    label = safe_label(attrs.get("label", n), n)
    net.add_node(n, label=label, title=label)

# Add edges with string titles showing weight
for u, v, a in G.edges(data=True):
    w = float(a.get("weight", 1.0))
    net.add_edge(u, v, value=w, title=f"weight: {w}")

# Safety: ensure titles are strings (avoid PyVis TypeError)
for nd in net.nodes:
    if "title" in nd and not isinstance(nd["title"], str):
        nd["title"] = str(nd["title"])
for ed in net.edges:
    if "title" in ed and not isinstance(ed["title"], str):
        ed["title"] = str(ed["title"])

html = net.generate_html(notebook=False)
HTML_OUT.write_text(html, encoding="utf-8")
print(f"PyVis HTML written to: {HTML_OUT}")
