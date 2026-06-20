from __future__ import annotations

"""
05_robustness_experiment_deterministic.py

Deterministic robustness experiment for the sampled Ethereum USDT transfer graph.

Main differences from the previous 05_robustness_experiment.py:
1. Deterministic tie-breaking for degree-targeted and PageRank-targeted removal:
   - primary key: larger centrality first
   - secondary key: hexadecimal address ascending
2. Deterministic random-removal runs:
   - node list is sorted first
   - each repeat uses seed = SEED + repeat
   - larger removal fractions reuse the prefix of the same shuffled order
3. Output names are different from the previous script:
   - results/robustness_deterministic/
   - robustness_raw_deterministic.csv
   - robustness_summary_deterministic.csv
   - robustness_wcc_deterministic.pdf/png
   - robustness_scc_deterministic.pdf/png
   - robustness_num_wcc_deterministic.pdf/png
   - robustness_metadata_deterministic.json

Important paper statement:
- The number of removed nodes is k = ceil(fraction * original_n).
- Ties in total degree or PageRank are resolved by hexadecimal address order.
"""

import json
import math
import pickle
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

try:
    from common import RESULTS_DIR
except Exception:
    RESULTS_DIR = Path(__file__).resolve().parent / "results"

PROJECT_DIR = Path(__file__).resolve().parent

# The script tries these graph paths in order.
# Keep your current final graph path first.
GRAPH_CANDIDATES = [
    Path(r"D:\vscode_code\USDT research\results\graph_analysis\usdt_graph_v2.pkl"),
    Path(r"D:\vscode_code\USDT research\results\graph_analysis\usdt_graph.pkl"),
    PROJECT_DIR / "results" / "graph_analysis" / "usdt_graph_v2.pkl",
    PROJECT_DIR / "results" / "graph_analysis" / "usdt_graph.pkl",
    PROJECT_DIR.parent / "results" / "graph_analysis" / "usdt_graph_v2.pkl",
    PROJECT_DIR.parent / "results" / "graph_analysis" / "usdt_graph.pkl",
]

OUT = RESULTS_DIR / "robustness_deterministic"
OUT.mkdir(parents=True, exist_ok=True)

REMOVAL_FRACTIONS = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]
RANDOM_REPEATS = 20
SEED = 42

PAGERANK_ALPHA = 0.85
PAGERANK_MAX_ITER = 500
PAGERANK_TOL = 1e-8

FRACTION_LABELS = {
    0.0: "0",
    0.001: "0.1",
    0.005: "0.5",
    0.01: "1",
    0.02: "2",
    0.05: "5",
}

STRATEGY_ORDER = [
    "random",
    "degree_targeted",
    "pagerank_targeted",
]

STRATEGY_LABELS = {
    "random": "Random node removal",
    "degree_targeted": "Initial-degree targeted",
    "pagerank_targeted": "Initial-PageRank targeted",
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 11,
        "legend.fontsize": 8.5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    }
)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def locate_graph() -> Path:
    for path in GRAPH_CANDIDATES:
        if path.exists():
            return path

    searched = "\n".join(str(path) for path in GRAPH_CANDIDATES)
    raise FileNotFoundError(
        "Could not find graph pickle file. Searched:\n" + searched
    )


def load_graph(path: Path) -> nx.DiGraph:
    with path.open("rb") as file:
        graph = pickle.load(file)

    if not isinstance(graph, nx.DiGraph):
        raise TypeError(f"Expected nx.DiGraph, got {type(graph)} from {path}")

    return graph


def removal_count(original_n: int, fraction: float) -> int:
    """Use ceil, consistent with the paper statement for this script."""
    if fraction <= 0:
        return 0
    return min(original_n, int(math.ceil(original_n * fraction)))


def largest_component_size(components) -> int:
    return max((len(component) for component in components), default=0)


def graph_metrics(graph: nx.DiGraph, original_n: int) -> dict[str, float | int]:
    n = graph.number_of_nodes()

    if n == 0:
        return {
            "remaining_nodes": 0,
            "remaining_node_share": 0.0,
            "largest_wcc_nodes": 0,
            "largest_scc_nodes": 0,
            "largest_wcc_share_original": 0.0,
            "largest_scc_share_original": 0.0,
            "num_wcc": 0,
            "num_scc": 0,
        }

    largest_wcc_nodes = largest_component_size(nx.weakly_connected_components(graph))
    largest_scc_nodes = largest_component_size(nx.strongly_connected_components(graph))

    return {
        "remaining_nodes": n,
        "remaining_node_share": n / original_n,
        "largest_wcc_nodes": largest_wcc_nodes,
        "largest_scc_nodes": largest_scc_nodes,
        "largest_wcc_share_original": largest_wcc_nodes / original_n,
        "largest_scc_share_original": largest_scc_nodes / original_n,
        "num_wcc": nx.number_weakly_connected_components(graph),
        "num_scc": nx.number_strongly_connected_components(graph),
    }


def remove_and_measure(
    graph: nx.DiGraph,
    nodes_to_remove: list[str],
    original_n: int,
) -> dict[str, float | int]:
    residual = graph.copy()
    residual.remove_nodes_from(nodes_to_remove)
    return graph_metrics(residual, original_n)


def deterministic_degree_order(graph: nx.DiGraph) -> list[str]:
    all_nodes = sorted(graph.nodes())
    return sorted(all_nodes, key=lambda node: (-graph.degree(node), node))


def deterministic_pagerank_order(graph: nx.DiGraph) -> tuple[list[str], dict[str, float]]:
    print("Computing weighted PageRank...")
    pagerank = nx.pagerank(
        graph,
        alpha=PAGERANK_ALPHA,
        weight="weight",
        max_iter=PAGERANK_MAX_ITER,
        tol=PAGERANK_TOL,
    )

    all_nodes = sorted(graph.nodes())
    order = sorted(all_nodes, key=lambda node: (-pagerank[node], node))
    return order, pagerank


def deterministic_random_orders(nodes: list[str]) -> list[list[str]]:
    """Create one shuffled node order for each repeat."""
    orders = []

    for repeat in range(RANDOM_REPEATS):
        order = nodes.copy()
        rng = random.Random(SEED + repeat)
        rng.shuffle(order)
        orders.append(order)

    return orders


def run_experiment(graph: nx.DiGraph) -> pd.DataFrame:
    original_n = graph.number_of_nodes()
    all_nodes = sorted(graph.nodes())

    print(f"Graph nodes: {original_n:,}")
    print(f"Graph edges: {graph.number_of_edges():,}")

    print("Preparing deterministic degree ranking...")
    degree_order = deterministic_degree_order(graph)

    pr_order, _ = deterministic_pagerank_order(graph)

    print("Preparing deterministic random orders...")
    random_orders = deterministic_random_orders(all_nodes)

    rows: list[dict[str, Any]] = []

    for fraction in REMOVAL_FRACTIONS:
        k = removal_count(original_n, fraction)
        print(f"Running fraction={fraction:.4f}, k={k:,}")

        # Random removal: 20 repeats, each uses a fixed shuffled order.
        # Larger fractions use a longer prefix of the same order.
        for repeat, order in enumerate(random_orders):
            selected = order[:k]
            metrics = remove_and_measure(graph, selected, original_n)
            rows.append(
                {
                    "strategy": "random",
                    "fraction_removed": fraction,
                    "removed_nodes": k,
                    "repeat": repeat,
                    **metrics,
                }
            )

        # Degree-targeted removal.
        metrics = remove_and_measure(graph, degree_order[:k], original_n)
        rows.append(
            {
                "strategy": "degree_targeted",
                "fraction_removed": fraction,
                "removed_nodes": k,
                "repeat": 0,
                **metrics,
            }
        )

        # PageRank-targeted removal.
        metrics = remove_and_measure(graph, pr_order[:k], original_n)
        rows.append(
            {
                "strategy": "pagerank_targeted",
                "fraction_removed": fraction,
                "removed_nodes": k,
                "repeat": 0,
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def aggregate_results(raw: pd.DataFrame) -> pd.DataFrame:
    summary = (
        raw.groupby(["strategy", "fraction_removed"], as_index=False)
        .agg(
            removed_nodes=("removed_nodes", "first"),
            largest_wcc_mean=("largest_wcc_share_original", "mean"),
            largest_wcc_std=("largest_wcc_share_original", "std"),
            largest_scc_mean=("largest_scc_share_original", "mean"),
            largest_scc_std=("largest_scc_share_original", "std"),
            num_wcc_mean=("num_wcc", "mean"),
            num_scc_mean=("num_scc", "mean"),
        )
    )

    # Targeted strategies are deterministic; standard deviations are not applicable.
    targeted = summary["strategy"].isin(["degree_targeted", "pagerank_targeted"])
    summary.loc[targeted, ["largest_wcc_std", "largest_scc_std"]] = np.nan

    strategy_rank = {name: index for index, name in enumerate(STRATEGY_ORDER)}
    summary["strategy_order"] = summary["strategy"].map(strategy_rank)
    summary = summary.sort_values(["strategy_order", "fraction_removed"]).drop(
        columns=["strategy_order"]
    )

    return summary


def prepare_series(
    summary: pd.DataFrame,
    strategy: str,
    metric_mean: str,
    metric_std: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    subset = (
        summary[summary["strategy"] == strategy]
        .set_index("fraction_removed")
        .reindex(REMOVAL_FRACTIONS)
    )

    x_positions = np.arange(len(REMOVAL_FRACTIONS))
    means = subset[metric_mean].to_numpy(dtype=float)
    stds = subset[metric_std].fillna(0.0).to_numpy(dtype=float)
    return x_positions, means, stds


def plot_metric(
    summary: pd.DataFrame,
    metric_mean: str,
    metric_std: str,
    ylabel: str,
    filename_stem: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.35))

    for strategy in STRATEGY_ORDER:
        x_positions, means, stds = prepare_series(
            summary,
            strategy,
            metric_mean,
            metric_std,
        )

        line = ax.plot(
            x_positions,
            means,
            marker="o",
            markersize=4,
            linewidth=1.6,
            label=STRATEGY_LABELS[strategy],
        )[0]

        if strategy == "random":
            lower = np.maximum(means - stds, 0.0)
            upper = means + stds
            ax.fill_between(
                x_positions,
                lower,
                upper,
                color=line.get_color(),
                alpha=0.18,
                linewidth=0,
                label="Random ± 1 SD",
            )

    ax.set_xlabel("Removed nodes (%)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(REMOVAL_FRACTIONS)))
    ax.set_xticklabels([FRACTION_LABELS[f] for f in REMOVAL_FRACTIONS])
    ax.set_xlim(-0.15, len(REMOVAL_FRACTIONS) - 0.85)
    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()

    pdf_path = OUT / f"{filename_stem}.pdf"
    png_path = OUT / f"{filename_stem}.png"

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {pdf_path}")
    print(f"Saved figure: {png_path}")


def make_plots(summary: pd.DataFrame) -> None:
    plot_metric(
        summary,
        metric_mean="largest_wcc_mean",
        metric_std="largest_wcc_std",
        ylabel="Largest WCC / original nodes",
        filename_stem="robustness_wcc_deterministic",
    )

    plot_metric(
        summary,
        metric_mean="largest_scc_mean",
        metric_std="largest_scc_std",
        ylabel="Largest SCC / original nodes",
        filename_stem="robustness_scc_deterministic",
    )

    plot_metric(
        summary,
        metric_mean="num_wcc_mean",
        metric_std="largest_wcc_std",
        ylabel="Number of weak components",
        filename_stem="robustness_num_wcc_deterministic",
    )


def main() -> None:
    graph_path = locate_graph()
    print(f"Loading graph from: {graph_path}")

    graph = load_graph(graph_path)

    raw = run_experiment(graph)
    summary = aggregate_results(raw)

    raw_path = OUT / "robustness_raw_deterministic.csv"
    summary_path = OUT / "robustness_summary_deterministic.csv"
    metadata_path = OUT / "robustness_metadata_deterministic.json"

    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)

    metadata = {
        "graph_path": str(graph_path),
        "output_directory": str(OUT),
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges(),
        "removal_fractions": REMOVAL_FRACTIONS,
        "random_repeats": RANDOM_REPEATS,
        "random_seed_rule": "repeat_seed = SEED + repeat",
        "seed": SEED,
        "removal_count_rule": "k = ceil(fraction * original_n)",
        "degree_targeted_order": "sorted by (-total_degree, hexadecimal_address)",
        "pagerank_targeted_order": "sorted by (-weighted_pagerank, hexadecimal_address)",
        "pagerank": {
            "alpha": PAGERANK_ALPHA,
            "weight": "weight",
            "max_iter": PAGERANK_MAX_ITER,
            "tol": PAGERANK_TOL,
        },
        "output_files": {
            "raw": str(raw_path),
            "summary": str(summary_path),
            "figures": [
                str(OUT / "robustness_wcc_deterministic.pdf"),
                str(OUT / "robustness_scc_deterministic.pdf"),
                str(OUT / "robustness_num_wcc_deterministic.pdf"),
            ],
        },
    }

    metadata_path.write_text(
        json.dumps(json_safe(metadata), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    make_plots(summary)

    print("\n" + "=" * 76)
    print("DETERMINISTIC ROBUSTNESS EXPERIMENT COMPLETED")
    print("=" * 76)
    print(summary.to_string(index=False))
    print(f"\nRaw results: {raw_path}")
    print(f"Summary: {summary_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
