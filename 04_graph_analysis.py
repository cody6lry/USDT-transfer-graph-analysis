from __future__ import annotations

from collections import deque
import math
import pickle

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import PROCESSED_DIR, RESULTS_DIR, save_json

INPUT =r"D:\vscode_code\USDT research\usdt_computer_networks_project\data_processed\clean_edges_base.csv"
OUT = RESULTS_DIR / "graph_analysis"
OUT.mkdir(parents=True, exist_ok=True)

def build_graph(df: pd.DataFrame) -> nx.DiGraph:
    grouped = (
        df.groupby(["from", "to"], as_index=False)
          .agg(weight=("value_usdt", "sum"),
               count=("event_id", "count"))
    )

    G = nx.DiGraph()
    for row in grouped.itertuples(index=False):
        G.add_edge(row[0], row[1], weight=float(row.weight), count=int(row.count))
    return G

def bow_tie(G: nx.DiGraph) -> dict:
    sccs = list(nx.strongly_connected_components(G))
    if not sccs:
        return {}
    core = set(max(sccs, key=len))

    out_nodes = set()
    dq = deque(core)
    visited = set(core)
    while dq:
        u = dq.popleft()
        for v in G.successors(u):
            if v not in visited:
                visited.add(v)
                out_nodes.add(v)
                dq.append(v)

    in_nodes = set()
    R = G.reverse(copy=False)
    dq = deque(core)
    visited = set(core)
    while dq:
        u = dq.popleft()
        for v in R.successors(u):
            if v not in visited:
                visited.add(v)
                in_nodes.add(v)
                dq.append(v)

    classified = core | in_nodes | out_nodes
    others = set(G.nodes()) - classified
    largest_wcc = set(max(nx.weakly_connected_components(G), key=len))
    tendrils = others & largest_wcc
    disconnected = others - largest_wcc

    n = G.number_of_nodes()
    def pack(s: set) -> dict:
        return {"nodes": len(s), "share": len(s) / n if n else 0.0}

    return {
        "SCC": pack(core),
        "IN": pack(in_nodes),
        "OUT": pack(out_nodes),
        "Tendrils": pack(tendrils),
        "Disconnected": pack(disconnected),
    }

def normalized_rich_club(G: nx.DiGraph, thresholds=None, random_graphs=5, seed=42) -> pd.DataFrame:
    U = nx.Graph(G)
    degrees = dict(U.degree())
    max_degree = max(degrees.values()) if degrees else 0

    if thresholds is None:
        thresholds = [k for k in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000] if k < max_degree]

    def phi(graph: nx.Graph, k: int) -> float:
        rich = [n for n, d in graph.degree() if d > k]
        nr = len(rich)
        if nr < 2:
            return np.nan
        er = graph.subgraph(rich).number_of_edges()
        return 2 * er / (nr * (nr - 1))

    rng = np.random.default_rng(seed)
    rows = []

    random_phi = {k: [] for k in thresholds}
    for i in range(random_graphs):
        R = U.copy()
        nswap = min(max(R.number_of_edges(), 1), 200000)
        try:
            nx.double_edge_swap(R, nswap=nswap, max_tries=max(nswap * 20, 1000), seed=int(rng.integers(1, 2**31-1)))
        except Exception:
            pass
        for k in thresholds:
            random_phi[k].append(phi(R, k))

    for k in thresholds:
        observed = phi(U, k)
        baseline = float(np.nanmean(random_phi[k])) if random_phi[k] else np.nan
        rho = observed / baseline if np.isfinite(observed) and np.isfinite(baseline) and baseline > 0 else np.nan
        rows.append({
            "k": k,
            "phi_observed": observed,
            "phi_random_mean": baseline,
            "rho": rho,
            "nodes_above_k": sum(d > k for d in degrees.values())
        })

    return pd.DataFrame(rows)

def main() -> None:
    df = pd.read_csv(INPUT)
    G = build_graph(df)

    with (OUT / "usdt_graph.pkl").open("wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

    in_degrees = np.array([d for _, d in G.in_degree()])
    out_degrees = np.array([d for _, d in G.out_degree()])

    pagerank = nx.pagerank(G, alpha=0.85, weight="weight", max_iter=500, tol=1e-8)
    sorted_pr = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)

    largest_wcc = max(nx.weakly_connected_components(G), key=len)
    largest_scc = max(nx.strongly_connected_components(G), key=len)

    summary = {
        "nodes": G.number_of_nodes(),
        "unique_directed_edges": G.number_of_edges(),
        "density": nx.density(G),
        "average_in_degree": float(in_degrees.mean()),
        "average_out_degree": float(out_degrees.mean()),
        "max_in_degree": int(in_degrees.max()),
        "max_out_degree": int(out_degrees.max()),
        "largest_wcc_nodes": len(largest_wcc),
        "largest_wcc_share": len(largest_wcc) / G.number_of_nodes(),
        "largest_scc_nodes": len(largest_scc),
        "largest_scc_share": len(largest_scc) / G.number_of_nodes(),
        "degree_assortativity": float(nx.degree_assortativity_coefficient(G)),
        "pagerank_top1_share": float(sorted_pr[0][1]),
        "pagerank_top10_share": float(sum(v for _, v in sorted_pr[:10])),
        "pagerank_top50_share": float(sum(v for _, v in sorted_pr[:50])),
    }
    save_json(summary, OUT / "graph_summary.json")

    top_rows = []
    for rank, (node, score) in enumerate(sorted_pr[:100], 1):
        top_rows.append({
            "rank": rank,
            "address": node,
            "pagerank": score,
            "in_degree": G.in_degree(node),
            "out_degree": G.out_degree(node),
            "weighted_in_degree": G.in_degree(node, weight="weight"),
            "weighted_out_degree": G.out_degree(node, weight="weight"),
        })
    pd.DataFrame(top_rows).to_csv(OUT / "top100_pagerank.csv", index=False)

    bt = bow_tie(G)
    save_json(bt, OUT / "bow_tie.json")

    rc = normalized_rich_club(G)
    rc.to_csv(OUT / "rich_club.csv", index=False)

    plt.figure(figsize=(7, 5))
    positive_in = in_degrees[in_degrees > 0]
    positive_out = out_degrees[out_degrees > 0]
    bins = np.logspace(0, np.log10(max(positive_in.max(), positive_out.max())), 40)
    plt.hist(positive_in, bins=bins, alpha=0.6, label="In-degree")
    plt.hist(positive_out, bins=bins, alpha=0.6, label="Out-degree")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.title("Degree Distributions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "degree_distribution.pdf")
    plt.close()

    if not rc.empty:
        plt.figure(figsize=(7, 5))
        plt.plot(rc["k"], rc["rho"], marker="o")
        plt.axhline(1.0, linestyle="--")
        plt.xscale("log")
        plt.xlabel("Degree threshold k")
        plt.ylabel("Normalized rich-club coefficient")
        plt.title("Normalized Rich-Club Coefficient")
        plt.tight_layout()
        plt.savefig(OUT / "rich_club.pdf")
        plt.close()

    print(summary)

if __name__ == "__main__":
    main()
