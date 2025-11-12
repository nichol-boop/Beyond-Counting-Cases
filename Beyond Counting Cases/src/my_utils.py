# src/my_utils.py

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterable, Union, List, Dict

import pandas as pd
import networkx as nx

try:
    from pyvis.network import Network
    _HAS_PYVIS = True
except Exception:
    _HAS_PYVIS = False


# ----------------------------
# Data loading
# ----------------------------

@dataclass
class EdgeSchema:
    """
    Map your Excel columns to edge fields.
    src: column name for the citing/source node
    dst: column name for the cited/target node
    weight: optional numeric weight column
    year: optional numeric year column (e.g., citing year)
    direction: 'directed' or 'undirected'
    """
    src: str
    dst: str
    weight: Optional[str] = None
    year: Optional[str] = None
    direction: str = "directed"


def load_edges_from_excel(
    path: Union[str, Path],
    sheet: Union[str, int] = 0,
    schema: Optional[EdgeSchema] = None,
) -> pd.DataFrame:
    """
    Load an Excel sheet and return a standardized edges DataFrame with columns:
    ['source','target','weight','year','direction'].
    """
    df = pd.read_excel(Path(path), sheet_name=sheet)

    if schema is None:
        raise ValueError("Provide an EdgeSchema mapping your column names.")

    out = pd.DataFrame({
        "source": df[schema.src].astype(str),
        "target": df[schema.dst].astype(str),
    })

    if schema.weight and schema.weight in df:
        out["weight"] = pd.to_numeric(df[schema.weight], errors="coerce").fillna(1.0)
    else:
        out["weight"] = 1.0

    if schema.year and schema.year in df:
        out["year"] = pd.to_numeric(df[schema.year], errors="coerce")
    else:
        out["year"] = pd.NA

    out["direction"] = schema.direction
    out = out.dropna(subset=["source", "target"])
    return out


# ----------------------------
# Graph building & filtering
# ----------------------------

def filter_edges_by_year(edges: pd.DataFrame, max_year: Optional[int]) -> pd.DataFrame:
    """Return edges with year <= max_year (drops edges with missing year)."""
    if max_year is None or "year" not in edges:
        return edges.copy()
    e = edges.copy()
    e = e[pd.to_numeric(e["year"], errors="coerce").notna()]
    return e[e["year"] <= max_year]


def build_graph(edges: pd.DataFrame) -> nx.Graph:
    """
    Build a NetworkX graph from standardized edges.
    Uses a DiGraph if any row is 'directed'; aggregates duplicate edges by summing weights.
    """
    directed = (edges.get("direction", "directed") == "directed").any()
    G: Union[nx.DiGraph, nx.Graph] = nx.DiGraph() if directed else nx.Graph()

    grouped = (
        edges.groupby(["source", "target"], as_index=False)
             .agg(weight=("weight", "sum"),
                  year=("year", "min"))
    )
    for _, r in grouped.iterrows():
        G.add_edge(
            r["source"], r["target"],
            weight=float(r["weight"]),
            year=None if pd.isna(r["year"]) else int(r["year"])
        )
    return G


# ----------------------------
# Metrics
# ----------------------------

def compute_network_metrics(
    G: nx.Graph,
    centralities: Iterable[str] = ("degree", "betweenness", "pagerank", "clustering"),
    weighted: bool = True,
) -> pd.DataFrame:
    """Compute selected node metrics and return a DataFrame."""
    w = "weight" if weighted else None
    data: Dict[str, Dict[str, float]] = {}

    if "degree" in centralities:
        deg = dict(G.out_degree(weight=w)) if G.is_directed() else dict(G.degree(weight=w))
        data["degree"] = deg
    if "betweenness" in centralities:
        data["betweenness"] = nx.betweenness_centrality(G, weight=w, normalized=True)
    if "pagerank" in centralities:
        data["pagerank"] = nx.pagerank(G if G.is_directed() else G.to_directed(), weight=w)
    if "clustering" in centralities:
        data["clustering"] = nx.clustering(G if not G.is_directed() else G.to_undirected(), weight=w)

    rows = []
    for n in G.nodes():
        row = {"node": n}
        for k, d in data.items():
            row[k] = float(d.get(n, 0.0))
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------
# PyVis export
# ----------------------------

def export_pyvis_html(
    G: nx.Graph,
    html_path: Union[str, Path],
    height: str = "750px",
    width: str = "100%",
    physics: bool = True,
    size_metric: str = "pagerank",
    min_size: int = 5,
    max_size: int = 30,
    metrics_df: Optional[pd.DataFrame] = None,
) -> Path:
    """
    Export an interactive PyVis HTML network. Node sizes scale by `size_metric`.
    """
    if not _HAS_PYVIS:
        raise RuntimeError("pyvis is not installed. `pip/conda install pyvis`")

    if metrics_df is None:
        metrics_df = compute_network_metrics(G)

    m = metrics_df.set_index("node")
    s = m[size_metric] if size_metric in m.columns else pd.Series(1.0, index=m.index)
    if s.max() > 0:
        sizes = ( (s - s.min()) / (s.max() - s.min() + 1e-12) ) * (max_size - min_size) + min_size
    else:
        sizes = pd.Series(min_size, index=m.index)

    net = Network(height=height, width=width, directed=G.is_directed())
    if physics:
        net.barnes_hut()
    else:
        net.toggle_physics(False)

    # nodes
    for n, attrs in G.nodes(data=True):
        title = f"<b>{n}</b>"
        for col in m.columns:
            if n in m.index:
                title += f"<br>{col}: {round(float(m.loc[n, col]), 4)}"
        net.add_node(n, value=float(sizes.get(n, min_size)), title=title)

    # edges
    for u, v, a in G.edges(data=True):
        w = float(a.get("weight", 1.0))
        yr = a.get("year", "")
        net.add_edge(u, v, value=w, title=f"weight: {w}" + (f" | year: {yr}" if yr else ""))

    html_path = Path(html_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    net.show(str(html_path))
    return html_path


# ----------------------------
# Persistence helpers
# ----------------------------

def save_graph_gexf(G: nx.Graph, path: Union[str, Path]) -> Path:
    """Save the graph to GEXF (Gephi)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gexf(G, path)
    return path


