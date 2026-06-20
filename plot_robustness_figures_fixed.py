from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent

CSV_PATH = Path(r"D:\vscode_code\USDT research\usdt_computer_networks_project\results\robustness\robustness_summary.csv")

OUTPUT_DIR = PROJECT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FRACTIONS = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]
X_POSITIONS = np.arange(len(FRACTIONS))
X_LABELS = ["0", "0.1", "0.5", "1", "2", "5"]

STRATEGIES = [
    ("random", "Random node removal"),
    ("degree_targeted", "Initial-degree targeted"),
    ("pagerank_targeted", "Initial-PageRank targeted"),
]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 11,
    "legend.fontsize": 8.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})


def locate_csv() -> Path:
    for path in CSV_PATH:
        if path.exists():
            return path

    raise FileNotFoundError(
        "robustness_summary.csv was not found.\nSearched:\n"
        + "\n".join(str(path) for path in CSV_PATH)
    )


def get_series(
    df: pd.DataFrame,
    strategy: str,
    mean_col: str,
    std_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    subset = (
        df[df["strategy"] == strategy]
        .set_index("fraction_removed")
        .reindex(FRACTIONS)
    )

    if subset[mean_col].isna().any():
        missing = subset[subset[mean_col].isna()].index.tolist()
        raise ValueError(
            f"Missing {mean_col} values for {strategy}: {missing}"
        )

    means = subset[mean_col].to_numpy(dtype=float)
    stds = subset[std_col].fillna(0.0).to_numpy(dtype=float)
    return means, stds


def draw(
    df: pd.DataFrame,
    mean_col: str,
    std_col: str,
    ylabel: str,
    filename: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.35))

    for strategy, label in STRATEGIES:
        means, stds = get_series(
            df,
            strategy,
            mean_col,
            std_col,
        )

        line = ax.plot(
            X_POSITIONS,
            means,
            marker="o",
            markersize=4,
            linewidth=1.6,
            label=label,
        )[0]

        if strategy == "random":
            lower = np.maximum(means - stds, 0.0)
            upper = means + stds
            ax.fill_between(
                X_POSITIONS,
                lower,
                upper,
                color=line.get_color(),
                alpha=0.18,
                linewidth=0,
                label="Random ± 1 SD",
            )

    ax.set_xlabel("Removed nodes (%)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(X_POSITIONS)
    ax.set_xticklabels(X_LABELS)
    ax.set_xlim(-0.15, len(X_POSITIONS) - 0.85)
    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / f"{filename}.pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        OUTPUT_DIR / f"{filename}.png",
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Saved {OUTPUT_DIR / f'{filename}.pdf'}")
    print(f"Saved {OUTPUT_DIR / f'{filename}.png'}")


def main() -> None:
    csv_path = CSV_PATH
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到文件: {csv_path}")
    print(f"Reading {csv_path}")

    df = pd.read_csv(csv_path)

    required = {
        "strategy",
        "fraction_removed",
        "largest_wcc_mean",
        "largest_wcc_std",
        "largest_scc_mean",
        "largest_scc_std",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    draw(
        df,
        "largest_wcc_mean",
        "largest_wcc_std",
        "Largest WCC / original nodes",
        "robustness_wcc",
    )

    draw(
        df,
        "largest_scc_mean",
        "largest_scc_std",
        "Largest SCC / original nodes",
        "robustness_scc",
    )


if __name__ == "__main__":
    main()

