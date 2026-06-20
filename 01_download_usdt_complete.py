"""
download_usdt_stratified_sample.py

Temporally stratified USDT ERC-20 sampling for October 2025 (UTC).
Each UTC hour is divided into four 15-minute strata, with up to 100
transfer records retained from each stratum. Maximum size: 297,600 rows.
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

API_KEY = os.getenv("ETHERSCAN_API_KEY", "").strip()
if not API_KEY:
    raise RuntimeError(
        "ETHERSCAN_API_KEY is not set.\n"
        "PowerShell: $env:ETHERSCAN_API_KEY='YOUR_KEY'\n"
        "CMD: set ETHERSCAN_API_KEY=YOUR_KEY"
    )

CHAIN_ID = "1"
USDT_CONTRACT = "0xdac17f958d2ee523a2206206994597c13d831ec7"
BASE_URL = "https://api.etherscan.io/v2/api"

START_UTC = datetime(2025, 10, 1, 0, 0, 0, tzinfo=timezone.utc)
END_UTC_EXCLUSIVE = datetime(2025, 11, 1, 0, 0, 0, tzinfo=timezone.utc)

STRATUM_MINUTES = 15
RECORDS_PER_STRATUM = 100
REQUESTS_PER_SECOND = 3.5
MAX_RETRIES = 8
REQUEST_TIMEOUT = 40

OUTPUT_DIR = Path(__file__).resolve().parent / "data_raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON = OUTPUT_DIR / "usdt_2025_10_stratified_sample.json"
OUTPUT_CSV = OUTPUT_DIR / "usdt_2025_10_stratified_sample.csv"
SUMMARY_JSON = OUTPUT_DIR / "sampling_summary.json"
STRATA_CSV = OUTPUT_DIR / "sampling_strata_summary.csv"

SESSION = requests.Session()


@dataclass
class StratumSummary:
    start_utc: str
    end_utc: str
    start_block: int
    end_block: int
    returned_records: int
    retained_records_after_global_dedup: int = 0
    reached_cap: bool = False
    error: str = ""


def throttled_request(params: dict[str, Any]) -> dict[str, Any]:
    full_params = dict(params)
    full_params.update({"chainid": CHAIN_ID, "apikey": API_KEY})

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = SESSION.get(BASE_URL, params=full_params, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                data = response.json()
                result_text = str(data.get("result", ""))
                if "rate limit" not in result_text.lower():
                    time.sleep(1.0 / REQUESTS_PER_SECOND)
                    return data
        except (requests.RequestException, ValueError) as exc:
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"Request failed after retries: {exc}") from exc

        wait_seconds = min(2 ** attempt, 30)
        print(f"[WARN] API request failed/rate-limited. Attempt {attempt}/{MAX_RETRIES}; sleeping {wait_seconds}s.")
        time.sleep(wait_seconds)

    raise RuntimeError(f"Etherscan request failed: {full_params}")


def get_block_by_timestamp(timestamp: int, closest: str) -> int:
    data = throttled_request({
        "module": "block",
        "action": "getblocknobytime",
        "timestamp": str(timestamp),
        "closest": closest,
    })
    try:
        return int(data.get("result"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Unable to resolve block number: {data}") from exc


def fetch_transfer_stratum(start_block: int, end_block: int, limit: int) -> list[dict[str, Any]]:
    data = throttled_request({
        "module": "account",
        "action": "tokentx",
        "contractaddress": USDT_CONTRACT,
        "startblock": str(start_block),
        "endblock": str(end_block),
        "page": "1",
        "offset": str(limit),
        "sort": "asc",
    })

    result = data.get("result", [])
    if isinstance(result, list):
        return result
    if "No transactions found" in str(result):
        return []
    raise RuntimeError(f"Unexpected Etherscan response: {data}")


def normalize_timestamp(record: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    raw_ts = normalized.get("timeStamp")
    try:
        normalized["timeStamp"] = datetime.fromtimestamp(int(raw_ts), tz=timezone.utc).isoformat()
    except (TypeError, ValueError):
        normalized["timeStamp"] = str(raw_ts)
    return normalized


def event_key(record: dict[str, Any]) -> tuple[str, ...]:
    tx_hash = str(record.get("hash", "")).strip().lower()
    log_index = str(record.get("logIndex", "")).strip()
    if tx_hash and log_index:
        return ("log", tx_hash, log_index)
    return (
        "fallback",
        tx_hash,
        str(record.get("blockNumber", "")).strip(),
        str(record.get("from", "")).strip().lower(),
        str(record.get("to", "")).strip().lower(),
        str(record.get("value", "")).strip(),
    )


def save_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(obj, file, ensure_ascii=False, indent=2)


def save_records_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        path.write_text("", encoding="utf-8")
        return

    preferred_fields = [
        "blockNumber", "timeStamp", "hash", "nonce", "blockHash", "from",
        "contractAddress", "to", "value", "tokenName", "tokenSymbol",
        "tokenDecimal", "transactionIndex", "gas", "gasPrice", "gasUsed",
        "cumulativeGasUsed", "input", "confirmations", "logIndex",
        "isError", "txreceipt_status",
    ]
    all_fields = set()
    for record in records:
        all_fields.update(record.keys())
    fieldnames = [f for f in preferred_fields if f in all_fields]
    fieldnames += sorted(all_fields - set(fieldnames))

    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def save_strata_csv(path: Path, summaries: list[StratumSummary]) -> None:
    rows = [asdict(item) for item in summaries]
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    total_strata_expected = int(
        (END_UTC_EXCLUSIVE - START_UTC).total_seconds() // (STRATUM_MINUTES * 60)
    )

    print("=" * 72)
    print("USDT TEMPORALLY STRATIFIED SAMPLING")
    print("=" * 72)
    print(f"Study period: {START_UTC.isoformat()} to {END_UTC_EXCLUSIVE.isoformat()}")
    print(f"Stratum length: {STRATUM_MINUTES} minutes")
    print(f"Maximum records per stratum: {RECORDS_PER_STRATUM}")
    print(f"Theoretical maximum records: {total_strata_expected * RECORDS_PER_STRATUM:,}")

    retained: dict[tuple[str, ...], dict[str, Any]] = {}
    summaries: list[StratumSummary] = []
    current = START_UTC
    delta = timedelta(minutes=STRATUM_MINUTES)

    total_raw_returned = 0
    strata_with_cap = 0
    strata_with_errors = 0

    while current < END_UTC_EXCLUSIVE:
        stratum_end = min(current + delta, END_UTC_EXCLUSIVE)
        summary = StratumSummary(
            start_utc=current.isoformat(),
            end_utc=stratum_end.isoformat(),
            start_block=0,
            end_block=0,
            returned_records=0,
        )

        try:
            start_block = get_block_by_timestamp(int(current.timestamp()), "after")
            end_block = get_block_by_timestamp(int(stratum_end.timestamp()) - 1, "before")
            summary.start_block = start_block
            summary.end_block = end_block

            records = fetch_transfer_stratum(start_block, end_block, RECORDS_PER_STRATUM)
            summary.returned_records = len(records)
            summary.reached_cap = len(records) >= RECORDS_PER_STRATUM
            total_raw_returned += len(records)
            if summary.reached_cap:
                strata_with_cap += 1

            before = len(retained)
            for record in records:
                normalized = normalize_timestamp(record)
                retained[event_key(normalized)] = normalized
            summary.retained_records_after_global_dedup = len(retained) - before

        except Exception as exc:
            summary.error = str(exc)
            strata_with_errors += 1
            print(f"\n[ERROR] {current.isoformat()} - {stratum_end.isoformat()}: {exc}")

        summaries.append(summary)

        if current.minute == 45:
            print(f"[PROGRESS] Completed {current.strftime('%Y-%m-%d %H:00 UTC')} | unique retained: {len(retained):,}")

        current = stratum_end

    records = list(retained.values())
    records.sort(key=lambda item: (
        str(item.get("timeStamp", "")),
        str(item.get("blockNumber", "")),
        str(item.get("hash", "")),
        str(item.get("logIndex", "")),
    ))

    save_json(OUTPUT_JSON, records)
    save_records_csv(OUTPUT_CSV, records)
    save_strata_csv(STRATA_CSV, summaries)

    total_strata = len(summaries)
    summary_obj = {
        "study_period_utc": {
            "start": START_UTC.isoformat(),
            "end_exclusive": END_UTC_EXCLUSIVE.isoformat(),
        },
        "sampling_design": {
            "stratum_minutes": STRATUM_MINUTES,
            "records_per_stratum_cap": RECORDS_PER_STRATUM,
            "strata_per_hour": 60 // STRATUM_MINUTES,
            "maximum_records_per_hour": (60 // STRATUM_MINUTES) * RECORDS_PER_STRATUM,
            "total_strata": total_strata,
            "theoretical_maximum_records": total_strata * RECORDS_PER_STRATUM,
        },
        "results": {
            "raw_records_returned_across_strata": total_raw_returned,
            "unique_records_after_global_deduplication": len(records),
            "duplicate_records_removed": total_raw_returned - len(records),
            "strata_reaching_record_cap": strata_with_cap,
            "share_of_strata_reaching_cap": strata_with_cap / total_strata if total_strata else 0.0,
            "strata_with_errors": strata_with_errors,
        },
        "contract": {
            "chain": "Ethereum mainnet",
            "chain_id": CHAIN_ID,
            "token": "USDT",
            "contract_address": USDT_CONTRACT,
        },
        "methodological_note": (
            "This is a temporally stratified sample, not a complete census. "
            "Each UTC hour is divided into four 15-minute strata, and up to "
            "100 transfer records are retained from each stratum."
        ),
    }
    save_json(SUMMARY_JSON, summary_obj)

    print("\n" + "=" * 72)
    print("SAMPLING COMPLETED")
    print("=" * 72)
    print(f"Raw records returned: {total_raw_returned:,}")
    print(f"Unique records retained: {len(records):,}")
    print(f"Duplicates removed: {total_raw_returned - len(records):,}")
    print(f"Strata reaching cap: {strata_with_cap:,}/{total_strata:,}")
    print(f"Strata with errors: {strata_with_errors:,}")
    print(f"JSON output: {OUTPUT_JSON}")
    print(f"CSV output: {OUTPUT_CSV}")
    print(f"Summary: {SUMMARY_JSON}")
    print(f"Stratum details: {STRATA_CSV}")


if __name__ == "__main__":
    main()
