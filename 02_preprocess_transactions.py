from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from common import RAW_DIR, PROCESSED_DIR, RESULTS_DIR, load_json, save_json

INPUT = Path(
    r"D:\vscode_code\USDT research\data_raw\usdt_2025_10_bidirectional_stratified_sample.json"
)

def lower_clean(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.lower()

def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def build_event_id(df: pd.DataFrame) -> pd.Series:
    log_index = df.get("logIndex", pd.Series([""] * len(df), index=df.index)).astype("string")
    tx_index = df.get("transactionIndex", pd.Series([""] * len(df), index=df.index)).astype("string")

    has_log_index = log_index.notna() & (log_index != "") & (log_index != "<NA>")
    event_id = (
        df["hash"].astype("string") + "|" +
        df["from"].astype("string") + "|" +
        df["to"].astype("string") + "|" +
        df["value"].astype("string") + "|" +
        df["blockNumber"].astype("string")
    )
    event_id = event_id.where(~has_log_index, df["hash"].astype("string") + "|" + log_index)
    return event_id

def main() -> None:
    raw = load_json(INPUT)
    df = pd.DataFrame(raw)
    raw_count = len(df)

    required = ["hash", "from", "to", "value", "tokenDecimal", "timeStamp", "blockNumber"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["hash"] = lower_clean(df["hash"])
    df["from"] = lower_clean(df["from"])
    df["to"] = lower_clean(df["to"])
    df["tokenSymbol"] = lower_clean(df.get("tokenSymbol", pd.Series([""] * len(df))))
    df["tokenName"] = df.get("tokenName", pd.Series([""] * len(df))).astype("string").str.strip()

    df["raw_value"] = to_numeric(df["value"])
    df["token_decimal_num"] = to_numeric(df["tokenDecimal"])
    df["blockNumber_num"] = to_numeric(df["blockNumber"])
    df["timestamp_utc"] = pd.to_datetime(df["timeStamp"], utc=True, errors="coerce")

    for col in ["gas", "gasPrice", "gasUsed"]:
        if col in df.columns:
            df[col] = to_numeric(df[col])

    invalid_core = (
        df["hash"].isna() |
        df["from"].isna() |
        df["to"].isna() |
        df["timestamp_utc"].isna() |
        df["raw_value"].isna() |
        df["token_decimal_num"].isna()
    )

    df = df.loc[~invalid_core].copy()

    df["event_id"] = build_event_id(df)
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["event_id"], keep="first").copy()
    duplicate_count = before_dedup - len(df)

    df["value_usdt"] = df["raw_value"] / np.power(10.0, df["token_decimal_num"])
    df["is_zero_or_negative"] = df["value_usdt"] <= 0
    df["is_self_transfer"] = df["from"] == df["to"]

    # Some Etherscan responses contain isError / txreceipt_status, others do not.
    success_rule_available = False
    if "txreceipt_status" in df.columns:
        success_rule_available = True
        status = df["txreceipt_status"].astype("string").str.strip()
        df["is_success"] = status.eq("1")
    elif "isError" in df.columns:
        success_rule_available = True
        err = df["isError"].astype("string").str.strip()
        df["is_success"] = err.eq("0")
    else:
        df["is_success"] = True

    period_mask = (
        (df["timestamp_utc"] >= pd.Timestamp("2025-10-01T00:00:00Z")) &
        (df["timestamp_utc"] < pd.Timestamp("2025-11-01T00:00:00Z"))
    )
    df = df.loc[period_mask].copy()

    positive = df.loc[(df["value_usdt"] > 0) & df["is_success"]].copy()
    graph_base = positive.loc[~positive["is_self_transfer"]].copy()

    keep_cols = [
        "event_id", "blockNumber_num", "timestamp_utc", "hash", "from", "to",
        "value_usdt", "tokenName", "tokenSymbol", "token_decimal_num",
        "gas", "gasPrice", "gasUsed", "is_self_transfer", "is_success"
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]

    positive[keep_cols].to_csv(PROCESSED_DIR / "clean_events_all.csv", index=False)
    graph_base[keep_cols].to_csv(PROCESSED_DIR / "clean_edges_base.csv", index=False)

    summary = {
        "raw_input_rows": raw_count,
        "invalid_core_rows_removed": int(invalid_core.sum()),
        "duplicate_event_rows_removed": int(duplicate_count),
        "rows_in_utc_period_after_dedup": int(len(df)),
        "zero_or_negative_rows": int(df["is_zero_or_negative"].sum()),
        "self_transfer_rows": int(df["is_self_transfer"].sum()),
        "success_status_available": bool(success_rule_available),
        "positive_success_events": int(len(positive)),
        "graph_events_after_removing_self_transfers": int(len(graph_base)),
    }
    save_json(summary, RESULTS_DIR / "preprocessing_summary.json")
    print(summary)

if __name__ == "__main__":
    main()
