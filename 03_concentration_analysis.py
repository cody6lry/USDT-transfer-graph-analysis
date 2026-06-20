from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import PROCESSED_DIR, RESULTS_DIR, save_json

INPUT = r"D:\vscode_code\USDT research\usdt_computer_networks_project\data_processed\clean_events_all.csv"
OUT = RESULTS_DIR / "concentration"
OUT.mkdir(parents=True, exist_ok=True)

def gini(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x) & (x >= 0)]
    if len(x) == 0 or x.sum() == 0:
        return float("nan")
    x = np.sort(x)
    n = len(x)
    return float((2 * np.sum(np.arange(1, n + 1) * x) - (n + 1) * x.sum()) / (n * x.sum()))

def top_share(values: np.ndarray, p: float) -> float:
    x = np.sort(values)[::-1]
    k = max(1, int(np.ceil(len(x) * p)))
    return float(x[:k].sum() / x.sum())

def main() -> None:
    df = pd.read_csv(INPUT)
    values = pd.to_numeric(df["value_usdt"], errors="coerce").dropna().to_numpy()
    values = values[values > 0]

    quantile_levels = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    q = {f"P{int(level*100):02d}": float(np.quantile(values, level)) for level in quantile_levels}

    top = {f"Top_{int(p*100)}pct_share": top_share(values, p) for p in [0.01, 0.05, 0.10, 0.25, 0.50]}

    summary = {
        "n_events": int(len(values)),
        "total_volume_usdt": float(values.sum()),
        "mean_usdt": float(values.mean()),
        "median_usdt": float(np.median(values)),
        "std_usdt": float(values.std(ddof=1)),
        "min_usdt": float(values.min()),
        "max_usdt": float(values.max()),
        "gini": gini(values),
        "quantiles": q,
        "top_shares": top,
    }
    save_json(summary, OUT / "concentration_summary.json")

    sorted_values = np.sort(values)
    cumulative_value = np.cumsum(sorted_values) / sorted_values.sum()
    cumulative_count = np.arange(1, len(values) + 1) / len(values)

    plt.figure(figsize=(6.5, 6.5))
    plt.plot(np.r_[0, cumulative_count], np.r_[0, cumulative_value], label="Lorenz curve")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect equality")
    plt.xlabel("Cumulative share of transfer events")
    plt.ylabel("Cumulative share of transferred value")
    plt.title(f"Lorenz Curve (Gini = {summary['gini']:.4f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "lorenz_curve.pdf")
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.hist(np.log10(values), bins=60)
    plt.xlabel("log10(Transaction value in USDT)")
    plt.ylabel("Frequency")
    plt.title("Distribution of USDT Transfer Values")
    plt.tight_layout()
    plt.savefig(OUT / "transaction_value_distribution.pdf")
    plt.close()

    rows = [{"metric": k, "value": v} for k, v in summary.items() if not isinstance(v, dict)]
    rows += [{"metric": k, "value": v} for k, v in q.items()]
    rows += [{"metric": k, "value": v} for k, v in top.items()]
    pd.DataFrame(rows).to_csv(OUT / "concentration_table.csv", index=False)

    print(summary)

if __name__ == "__main__":
    main()
