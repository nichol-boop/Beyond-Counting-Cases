# --- Build a citation network from the "Aus-time-matched complete UK dataset.xlsx" ---

import pandas as pd
import networkx as nx
from pathlib import Path

# Adjust if your notebook isn't inside notebooks/
DATA_PATH = Path("../data/Aus-time-matched simple UK dataset.xlsx")
OUTPUTS = Path("../outputs")
OUTPUTS.mkdir(parents=True, exist_ok=True)

# Read the first sheet (change sheet_name if needed)
df = pd.read_excel(DATA_PATH, sheet_name=0, header=0)

# We’ll reference columns by position to be robust:
# A = col 0 (case name), B = col 1 (node id), C–K = cols 2..10 (citer IDs)
COL_CASE_NAME = 0
COL_NODE_ID   = 1
CITER_COLS_SLICE = slice(2, 11)  # <-- C..K inclusive

# Clean node ids (column B) to strings
df_nodes = df.iloc[:, [COL_CASE_NAME, COL_NODE_ID]].copy()
df_nodes.columns = ["case_name", "node_id"]
df_nodes["node_id"] = df_nodes["node_id"].astype(str).str.strip()

# Build edges: for each row, every value in C..K (if numeric) is a CITER of this row’s node_id
edges = []
for _, row in df.iterrows():
    cited_id = str(row.iloc[COL_NODE_ID]).strip()
    # Skip rows without a node id
    if cited_id in ("", "nan", "None"):
        continue

    citer_values = row.iloc[CITER_COLS_SLICE].values
    # Coerce to numeric; non-numeric -> NaN; then drop NaN
    citers = pd.to_numeric(pd.Series(citer_values), errors="coerce").dropna()

    for citer_id in citers.astype(int).astype(str):
        # Direction: citer -> cited
        edges.append((citer_id, cited_id))

# Build the directed graph
G = nx.DiGraph()

# Add nodes with labels from A and ids from B
for _, r in df_nodes.iterrows():
    G.add_node(r["node_id"], label=r["case_name"])

# Ensure citer-only nodes are included (if any citer id doesn’t appear in column B)
for u, v in edges:
    if u not in G:
        G.add_node(u, label=u)  # fallback label = id
    if v not in G:
        G.add_node(v, label=v)

# Add edges (collapse duplicates by summing weights)
from collections import Counter
edge_counts = Counter(edges)
for (u, v), w in edge_counts.items():
    G.add_edge(u, v, weight=float(w))

print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

# ---- Export: PyVis HTML + Gephi GEXF ----

from pyvis.network import Network
from IPython.display import IFrame

html_out = Path("../outputs/uk_citation_network_pyvis.html")

net = Network(height="800px", width="100%", directed=True, notebook=False, cdn_resources="in_line")
net.barnes_hut()

# nodes
for n, attrs in G.nodes(data=True):
    net.add_node(n, label=attrs.get("label", n), title=attrs.get("label", n))

# edges
for u, v, a in G.edges(data=True):
    net.add_edge(u, v, value=a.get("weight", 1.0), title=f"weight: {a.get('weight', 1.0)}")

# Write HTML explicitly as UTF-8 to avoid Windows cp1252 errors
html = net.generate_html(notebook=False)
html_out.write_text(html, encoding="utf-8")
print(f"PyVis HTML written to: {html_out}")

# show inline in Jupyter (optional)
IFrame(src=html_out.as_posix(), width="100%", height=800)
