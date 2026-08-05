#!/usr/bin/env python3
"""Augment the audited GPT Work CAISO 2025 bundle with official OASIS data.

The original aligned table is treated as immutable lineage.  This script adds:

* PRC_AS v12 DAM ancillary-service clearing prices (31-day windows by
  ANC_TYPE; requesting both ancillary dimensions as ALL truncates to one day);
* AS_RESULTS v1 DAM total procured ancillary-service MW;
* ENE_SLRS v1 DAM supply/load schedules;
* ENE_EIM_TRANSFER_TIE v4 RTPD transfers for BAA_GRP_ID=CISO, retaining
  direction, neighboring BAA and tie name and using a strict four-quarter
  arithmetic hourly mean.

All output hours are UTC hour beginning in calendar year 2025.  No values are
interpolated, forward-filled, or converted from missing to zero.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import http.client
import io
import json
import re
import shutil
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


UTC_START = pd.Timestamp("2025-01-01T00:00:00Z")
UTC_END = pd.Timestamp("2026-01-01T00:00:00Z")
LOCAL_TZ = "America/Los_Angeles"
OASIS = "https://oasis.caiso.com/oasisapi/SingleZip"
OASIS_ENTRY = "https://oasis.caiso.com/"
OASIS_API_SPEC = "https://www.caiso.com/documents/oasisapispecification.pdf"


@dataclass(frozen=True)
class Report:
    key: str
    official_name: str
    queryname: str
    version: int
    market: str
    step_days: int
    extra: dict[str, str]


REPORTS = {
    "as_price": Report(
        "as_price",
        "Ancillary-service clearing prices",
        "PRC_AS",
        12,
        "DAM",
        31,
        {"anc_region": "ALL"},
    ),
    "as_result": Report(
        "as_result",
        "Ancillary-service results",
        "AS_RESULTS",
        1,
        "DAM",
        31,
        {"anc_type": "ALL", "anc_region": "ALL"},
    ),
    "slrs": Report(
        "slrs",
        "Supply and load resource schedules",
        "ENE_SLRS",
        1,
        "DAM",
        31,
        {},
    ),
    "tie": Report(
        "tie",
        "EIM transfer by tie",
        "ENE_EIM_TRANSFER_TIE",
        4,
        "RTPD",
        31,
        {"baa_grp_id": "CISO"},
    ),
}


def slug(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_boundary(local_date: date) -> pd.Timestamp:
    local = datetime(
        local_date.year,
        local_date.month,
        local_date.day,
        tzinfo=ZoneInfo(LOCAL_TZ),
    )
    return pd.Timestamp(local).tz_convert("UTC")


def oasis_datetime(value: pd.Timestamp) -> str:
    return value.strftime("%Y%m%dT%H:%M-0000")


def date_windows(step_days: int) -> list[tuple[date, date]]:
    first = date(2024, 12, 31)
    end_exclusive = date(2026, 1, 1)
    windows: list[tuple[date, date]] = []
    cursor = first
    while cursor < end_exclusive:
        nxt = min(cursor + timedelta(days=step_days), end_exclusive)
        windows.append((cursor, nxt))
        cursor = nxt
    return windows


class Downloader:
    def __init__(
        self,
        cache: Path,
        allow_unverified_tls: bool,
        pause: float,
    ) -> None:
        self.cache = cache
        self.cache.mkdir(parents=True, exist_ok=True)
        self.allow_unverified_tls = allow_unverified_tls
        self.pause = pause
        self.last_request = 0.0
        self.manifest: list[dict[str, object]] = []

    def _context(self) -> ssl.SSLContext:
        if self.allow_unverified_tls:
            return ssl._create_unverified_context()  # noqa: SLF001
        return ssl.create_default_context()

    def get(
        self,
        report: Report,
        local_start: date,
        local_end: date,
        extra_override: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        start = utc_boundary(local_start)
        end = utc_boundary(local_end)
        effective_extra = {**report.extra, **(extra_override or {})}
        params: dict[str, object] = {
            "resultformat": 6,
            "queryname": report.queryname,
            "version": report.version,
            "startdatetime": oasis_datetime(start),
            "enddatetime": oasis_datetime(end),
            "market_run_id": report.market,
            **effective_extra,
        }
        url = OASIS + "?" + urllib.parse.urlencode(params)
        extra_tag = "".join(
            f"_{slug(key)}-{slug(value)}"
            for key, value in sorted(effective_extra.items())
        )
        cache_path = self.cache / (
            f"{report.key}_{local_start:%Y%m%d}_{local_end:%Y%m%d}{extra_tag}.zip"
        )
        retrieved_at = datetime.now().astimezone().isoformat(timespec="seconds")
        status = 200
        if cache_path.exists():
            payload = cache_path.read_bytes()
            retrieval_mode = "cache"
        else:
            payload = b""
            retrieval_mode = (
                "download with certificate verification disabled after "
                "browser-byte identity control"
                if self.allow_unverified_tls
                else "download with verified TLS"
            )
            for attempt in range(7):
                wait = self.pause - (time.monotonic() - self.last_request)
                if wait > 0:
                    time.sleep(wait)
                request = urllib.request.Request(
                    url,
                    headers={"User-Agent": "academic-research-dataset-builder/2.0"},
                )
                try:
                    with urllib.request.urlopen(
                        request,
                        timeout=240,
                        context=self._context(),
                    ) as response:
                        status = int(response.status)
                        payload = response.read()
                    self.last_request = time.monotonic()
                    break
                except urllib.error.HTTPError as exc:
                    status = int(exc.code)
                    self.last_request = time.monotonic()
                    if exc.code not in (429, 500, 502, 503, 504) or attempt == 6:
                        raise
                    if exc.code == 429:
                        time.sleep(min(300, 90 * (attempt + 1)))
                    else:
                        time.sleep(min(120, 8 * 2**attempt))
                except (
                    TimeoutError,
                    urllib.error.URLError,
                    http.client.RemoteDisconnected,
                    ConnectionResetError,
                ):
                    self.last_request = time.monotonic()
                    if attempt == 6:
                        raise
                    time.sleep(min(90, 5 * 2**attempt))
            if not payload:
                raise RuntimeError(f"Empty OASIS response: {url}")
            cache_path.write_bytes(payload)

        try:
            archive = zipfile.ZipFile(io.BytesIO(payload))
        except zipfile.BadZipFile as exc:
            raise RuntimeError(f"Invalid OASIS ZIP: {url}") from exc
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not csv_names:
            xml = "\n".join(
                archive.read(name).decode("utf-8", errors="replace")
                for name in archive.namelist()
                if name.lower().endswith(".xml")
            )
            raise RuntimeError(f"OASIS returned no CSV: {url}\n{xml[:1000]}")
        csv_payloads = {name: archive.read(name) for name in csv_names}
        frames = [
            pd.read_csv(io.BytesIO(csv_payloads[name]), low_memory=False)
            for name in csv_names
        ]
        self.manifest.append(
            {
                "source_key": report.key,
                "official_report_name": report.official_name,
                "queryname": report.queryname,
                "version": report.version,
                "market_run_id": report.market,
                "period_start_utc": start.isoformat(),
                "period_end_exclusive_utc": end.isoformat(),
                "request_url": url,
                "http_status": status,
                "retrieval_mode": retrieval_mode,
                "retrieved_at": retrieved_at,
                "response_bytes": len(payload),
                "sha256": sha256_bytes(payload),
                "cached_filename": cache_path.name,
                "contained_csv": ";".join(csv_names),
                "contained_csv_sha256": ";".join(
                    f"{name}={sha256_bytes(csv_payloads[name])}"
                    for name in csv_names
                ),
            }
        )
        return pd.concat(frames, ignore_index=True)


def filter_year(df: pd.DataFrame, timestamp_field: str) -> pd.DataFrame:
    result = df.copy()
    result["timestamp_utc"] = pd.to_datetime(
        result[timestamp_field],
        utc=True,
        errors="raise",
    )
    return result[
        (result["timestamp_utc"] >= UTC_START)
        & (result["timestamp_utc"] < UTC_END)
    ].copy()


def collapse_identical(
    df: pd.DataFrame,
    keys: list[str],
    value: str,
    label: str,
) -> pd.DataFrame:
    duplicated = df.duplicated(keys, keep=False)
    if not duplicated.any():
        return df
    conflicts = (
        df.loc[duplicated]
        .groupby(keys, dropna=False)[value]
        .nunique(dropna=False)
    )
    if conflicts.gt(1).any():
        raise AssertionError(
            f"{label} has conflicting duplicate keys: "
            f"{conflicts[conflicts.gt(1)].head(5).to_dict()}"
        )
    return df.drop_duplicates(keys, keep="first")


def pivot_hourly(
    df: pd.DataFrame,
    dimensions: list[str],
    value: str,
    name_fn,
    label: str,
) -> pd.DataFrame:
    data = df.copy()
    data[value] = pd.to_numeric(data[value], errors="raise")
    keys = ["timestamp_utc", *dimensions]
    data = collapse_identical(data, keys, value, label)
    data["output_field"] = data.apply(name_fn, axis=1)
    if data.duplicated(["timestamp_utc", "output_field"]).any():
        raise AssertionError(f"{label} output names are not unique")
    return (
        data.pivot(index="timestamp_utc", columns="output_field", values=value)
        .sort_index(axis=1)
    )


def strict_tie_hourly(
    tie_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = filter_year(tie_rows, "INTERVAL_START_GMT")
    data = data[data["BAA_GRP_ID"].eq("CISO")].copy()
    data["VALUE"] = pd.to_numeric(data["VALUE"], errors="raise")
    data["hour_utc"] = data["timestamp_utc"].dt.floor("h")
    dimensions = ["TIE_NAME", "DIRECTION", "TO_BAA"]
    keys = ["timestamp_utc", *dimensions]
    data = collapse_identical(data, keys, "VALUE", "ENE_EIM_TRANSFER_TIE")
    expected_minutes = {0, 15, 30, 45}
    minute_sets = data.groupby(["hour_utc", *dimensions])["timestamp_utc"].agg(
        lambda x: set(x.dt.minute)
    )
    counts = data.groupby(["hour_utc", *dimensions])["VALUE"].count()
    complete = counts.eq(4) & minute_sets.eq(expected_minutes)
    means = data.groupby(["hour_utc", *dimensions])["VALUE"].mean()
    means.loc[~complete] = pd.NA
    long = means.rename("VALUE").reset_index()
    long["output_field"] = long.apply(
        lambda row: (
            "rtpd_eim_transfer__mw__"
            f"{'export' if row['DIRECTION'] == 'E' else 'import'}__"
            f"{slug(row['TO_BAA'])}__{slug(row['TIE_NAME'])}"
        ),
        axis=1,
    )
    if long.duplicated(["hour_utc", "output_field"]).any():
        raise AssertionError("EIM transfer output names are not unique")
    wide = (
        long.pivot(index="hour_utc", columns="output_field", values="VALUE")
        .sort_index(axis=1)
    )
    wide.index.name = "timestamp_utc"
    coverage = (
        long.groupby("output_field")["VALUE"]
        .agg(non_null_hours="count", missing_within_observed_hours=lambda s: int(s.isna().sum()))
        .reset_index()
    )
    coverage["source_intervals"] = coverage["output_field"].map(
        data.assign(
            output_field=data.apply(
                lambda row: (
                    "rtpd_eim_transfer__mw__"
                    f"{'export' if row['DIRECTION'] == 'E' else 'import'}__"
                    f"{slug(row['TO_BAA'])}__{slug(row['TIE_NAME'])}"
                ),
                axis=1,
            )
        ).groupby("output_field").size()
    )
    coverage["aggregation_rule"] = (
        "strict arithmetic mean of four distinct RTPD quarter-hours; "
        "incomplete groups are missing"
    )
    return wide, coverage


def add_dictionary_rows(
    dictionary: list[dict[str, object]],
    columns: list[str],
    *,
    report: Report,
    unit: str,
    raw_field: str,
    dimension_fields: str,
    cadence: str,
    rule: str,
    explanation_prefix: str,
) -> None:
    for column in columns:
        dictionary.append(
            {
                "output_field": column,
                "dtype": "float64",
                "unit": unit,
                "source_tier": "official CAISO OASIS",
                "source_key": report.key,
                "official_report_name": report.official_name,
                "queryname": report.queryname,
                "version": report.version,
                "market_run_id": report.market,
                "official_raw_field": raw_field,
                "official_dimension_fields": dimension_fields,
                "original_cadence": cadence,
                "hour_alignment_or_aggregation": rule,
                "official_entry_url": OASIS_API_SPEC,
                "explanation_zh": f"{explanation_prefix}；具体维度已编码在字段名中。",
            }
        )


def write_deterministic_gzip(frame: pd.DataFrame, path: Path) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            gz.write(
                frame.to_csv(
                    index=False,
                    lineterminator="\n",
                    float_format="%.10g",
                ).encode("utf-8")
            )


def validate_browser_identity(
    browser_sample: Path | None,
    shell_sample: Path | None,
) -> tuple[bool | None, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    for label, path in (
        ("Chrome normal-client download", browser_sample),
        ("command-line unverified-TLS control download", shell_sample),
    ):
        if path is not None and path.exists():
            rows.append(
                {
                    "artifact": label,
                    "path": str(path.resolve()),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    if len(rows) == 2:
        return rows[0]["sha256"] == rows[1]["sha256"], rows
    return None, rows


def build(args: argparse.Namespace) -> None:
    original_dir = args.original_dir.resolve()
    original_bundle = args.original_bundle.resolve()
    cache = args.raw_cache.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    identity_ok, identity_rows = validate_browser_identity(
        args.browser_sample.resolve() if args.browser_sample else None,
        args.shell_sample.resolve() if args.shell_sample else None,
    )
    if args.allow_unverified_caiso_tls and identity_ok is not True:
        raise RuntimeError(
            "Unverified CAISO TLS requires two byte-identical PRC_AS control files: "
            "one from normal Chrome and one from the command-line route."
        )

    downloader = Downloader(
        cache,
        allow_unverified_tls=args.allow_unverified_caiso_tls,
        pause=args.pause,
    )
    collected: dict[str, list[pd.DataFrame]] = {key: [] for key in REPORTS}
    for key, report in REPORTS.items():
        windows = date_windows(report.step_days)
        if key == "as_price":
            ancillary_types = ["NR", "RD", "RMD", "RMU", "RU", "SR"]
            total_requests = len(windows) * len(ancillary_types)
            number = 0
            for ancillary_type in ancillary_types:
                for local_start, local_end in windows:
                    number += 1
                    frame = downloader.get(
                        report,
                        local_start,
                        local_end,
                        {"anc_type": ancillary_type},
                    )
                    collected[key].append(frame)
                    if number == 1 or number % 10 == 0 or number == total_requests:
                        print(
                            f"{key}: {number}/{total_requests} type-window requests "
                            f"cached; latest rows={len(frame):,}",
                            flush=True,
                        )
        else:
            for number, (local_start, local_end) in enumerate(windows, start=1):
                frame = downloader.get(report, local_start, local_end)
                if key == "tie":
                    frame = frame[frame["BAA_GRP_ID"].eq("CISO")].copy()
                collected[key].append(frame)
                if number == 1 or number % 10 == 0 or number == len(windows):
                    print(
                        f"{key}: {number}/{len(windows)} windows cached; "
                        f"latest rows={len(frame):,}",
                        flush=True,
                    )

    as_price_raw = filter_year(
        pd.concat(collected["as_price"], ignore_index=True),
        "INTERVALSTARTTIME_GMT",
    )
    as_price = pivot_hourly(
        as_price_raw,
        ["ANC_TYPE", "ANC_REGION"],
        "MW",
        lambda row: (
            "dam_as_clearing_price__usd_per_mw__"
            f"{slug(row['ANC_TYPE'])}__{slug(row['ANC_REGION'])}"
        ),
        "PRC_AS",
    )

    as_result_raw = filter_year(
        pd.concat(collected["as_result"], ignore_index=True),
        "INTERVALSTARTTIME_GMT",
    )
    as_total_raw = as_result_raw[as_result_raw["RESULT_TYPE"].eq("AS_MW")].copy()
    as_total = pivot_hourly(
        as_total_raw,
        ["ANC_TYPE", "ANC_REGION"],
        "MW",
        lambda row: (
            "dam_as_total_procured__mw__"
            f"{slug(row['ANC_TYPE'])}__{slug(row['ANC_REGION'])}"
        ),
        "AS_RESULTS AS_MW",
    )

    slrs_raw = filter_year(
        pd.concat(collected["slrs"], ignore_index=True),
        "INTERVALSTARTTIME_GMT",
    )
    schedules = pivot_hourly(
        slrs_raw,
        ["TAC_ZONE_NAME", "SCHEDULE"],
        "MW",
        lambda row: (
            "dam_schedule__mw__"
            f"{slug(row['SCHEDULE'])}__{slug(row['TAC_ZONE_NAME'])}"
        ),
        "ENE_SLRS",
    )

    ties, tie_coverage = strict_tie_hourly(
        pd.concat(collected["tie"], ignore_index=True)
    )
    supplements = [as_price, as_total, schedules, ties]

    original_data_path = original_dir / "caiso_2025_hourly_aligned_full.csv.gz"
    original = pd.read_csv(original_data_path, low_memory=False)
    original["timestamp_utc"] = pd.to_datetime(
        original["timestamp_utc"],
        utc=True,
        errors="raise",
    )
    if len(original) != 8760 or original["timestamp_utc"].duplicated().any():
        raise AssertionError("Original CAISO bundle time axis is not 8,760 unique UTC hours")
    full = original.set_index("timestamp_utc")
    for supplement in supplements:
        overlap = set(full.columns).intersection(supplement.columns)
        if overlap:
            raise AssertionError(f"Supplement column collision: {sorted(overlap)[:5]}")
        full = full.join(supplement, how="left")
    if len(full) != 8760 or not full.index.is_monotonic_increasing:
        raise AssertionError("Augmented CAISO time axis failed")

    original_dictionary = pd.read_csv(original_dir / "column_dictionary.csv")
    dictionary = original_dictionary.to_dict("records")
    add_dictionary_rows(
        dictionary,
        list(as_price.columns),
        report=REPORTS["as_price"],
        unit="$/MW",
        raw_field="MW (price value in PRC_AS)",
        dimension_fields="ANC_TYPE + ANC_REGION",
        cadence="hourly DAM",
        rule="official hourly interval-start value; no aggregation",
        explanation_prefix="日前辅助服务清算价格",
    )
    add_dictionary_rows(
        dictionary,
        list(as_total.columns),
        report=REPORTS["as_result"],
        unit="MW",
        raw_field="MW filtered to RESULT_TYPE=AS_MW",
        dimension_fields="ANC_TYPE + ANC_REGION + RESULT_TYPE",
        cadence="hourly DAM",
        rule="official hourly total-procured value; no aggregation",
        explanation_prefix="日前辅助服务总采购量",
    )
    add_dictionary_rows(
        dictionary,
        list(schedules.columns),
        report=REPORTS["slrs"],
        unit="MW",
        raw_field="MW",
        dimension_fields="TAC_ZONE_NAME + SCHEDULE",
        cadence="hourly DAM",
        rule="official hourly interval-start value; no aggregation",
        explanation_prefix="日前供给、负荷、进口或出口计划",
    )
    add_dictionary_rows(
        dictionary,
        list(ties.columns),
        report=REPORTS["tie"],
        unit="MW",
        raw_field="VALUE filtered to BAA_GRP_ID=CISO",
        dimension_fields="DIRECTION + TO_BAA + TIE_NAME; BAA_GRP_ID=CISO",
        cadence="15-minute RTPD",
        rule=(
            "strict UTC-hour arithmetic mean of four quarter-hours; import and "
            "export retained separately; incomplete hours remain missing"
        ),
        explanation_prefix="CAISO 视角的实时预调度 EIM 联络线转移量",
    )
    dictionary_frame = pd.DataFrame(dictionary)

    result = full.reset_index()
    result["timestamp_utc"] = result["timestamp_utc"].map(
        lambda value: value.isoformat().replace("+00:00", "Z")
    )
    if set(result.columns) != set(dictionary_frame["output_field"]):
        missing = sorted(set(result.columns) - set(dictionary_frame["output_field"]))
        extra = sorted(set(dictionary_frame["output_field"]) - set(result.columns))
        raise AssertionError(f"Dictionary mismatch: missing={missing}, extra={extra}")
    if dictionary_frame["output_field"].duplicated().any():
        raise AssertionError("Dictionary has duplicate output fields")

    data_name = "caiso_2025_hourly_aligned_augmented.csv.gz"
    write_deterministic_gzip(result, output / data_name)
    dictionary_frame.to_csv(
        output / "column_dictionary.csv",
        index=False,
        lineterminator="\n",
    )

    original_manifest = pd.read_csv(original_dir / "source_manifest.csv")
    supplemental_manifest = pd.DataFrame(downloader.manifest)
    source_manifest = pd.concat(
        [original_manifest, supplemental_manifest],
        ignore_index=True,
    )
    source_manifest.to_csv(
        output / "source_manifest.csv",
        index=False,
        lineterminator="\n",
    )
    supplemental_manifest.to_csv(
        output / "supplemental_source_manifest.csv",
        index=False,
        lineterminator="\n",
    )
    tie_coverage.to_csv(
        output / "supplemental_interval_coverage.csv",
        index=False,
        lineterminator="\n",
    )

    lineage = [
        {
            "artifact": "GPT Work original CAISO bundle",
            "path": str(original_bundle),
            "bytes": original_bundle.stat().st_size,
            "sha256": sha256_file(original_bundle),
        },
        {
            "artifact": "GPT Work original aligned dataset",
            "path": str(original_data_path),
            "bytes": original_data_path.stat().st_size,
            "sha256": sha256_file(original_data_path),
        },
        *identity_rows,
    ]
    pd.DataFrame(lineage).to_csv(
        output / "source_lineage.csv",
        index=False,
        lineterminator="\n",
    )

    numeric_columns = [
        column for column in result.columns if column not in ("timestamp_utc", "timestamp_local")
    ]
    stats: list[dict[str, object]] = []
    for column in numeric_columns:
        values = pd.to_numeric(result[column], errors="coerce")
        stats.append(
            {
                "output_field": column,
                "non_null_count": int(values.notna().sum()),
                "missing_count": int(values.isna().sum()),
                "missing_pct": float(values.isna().mean() * 100),
                "min": float(values.min()) if values.notna().any() else None,
                "max": float(values.max()) if values.notna().any() else None,
                "mean": float(values.mean()) if values.notna().any() else None,
                "std": float(values.std()) if values.notna().any() else None,
                "constant_non_null": bool(values.nunique(dropna=True) <= 1),
            }
        )
    stats_frame = pd.DataFrame(stats)
    stats_frame.to_csv(
        output / "column_validation_stats.csv",
        index=False,
        lineterminator="\n",
    )

    added_columns = list(as_price.columns) + list(as_total.columns) + list(schedules.columns) + list(ties.columns)
    missing_by_report = {
        "PRC_AS": int(result[list(as_price.columns)].isna().sum().sum()),
        "AS_RESULTS_RESULT_TYPE_AS_MW": int(
            result[list(as_total.columns)].isna().sum().sum()
        ),
        "ENE_SLRS": int(result[list(schedules.columns)].isna().sum().sum()),
        "ENE_EIM_TRANSFER_TIE_BAA_GRP_ID_CISO": int(
            result[list(ties.columns)].isna().sum().sum()
        ),
    }
    summary = {
        "dataset": "CAISO 2025 UTC-hourly aligned augmented",
        "rows": len(result),
        "columns_total": len(result.columns),
        "numeric_fields": len(numeric_columns),
        "timestamp_fields": 2,
        "added_numeric_fields": len(added_columns),
        "added_by_report": {
            "PRC_AS": len(as_price.columns),
            "AS_RESULTS_RESULT_TYPE_AS_MW": len(as_total.columns),
            "ENE_SLRS": len(schedules.columns),
            "ENE_EIM_TRANSFER_TIE_BAA_GRP_ID_CISO": len(ties.columns),
        },
        "added_missing_cells_by_report": missing_by_report,
        "utc_start": result["timestamp_utc"].iloc[0],
        "utc_end": result["timestamp_utc"].iloc[-1],
        "duplicate_utc_timestamps": int(result["timestamp_utc"].duplicated().sum()),
        "total_numeric_missing_cells": int(result[numeric_columns].isna().sum().sum()),
        "columns_with_missing": int(stats_frame["missing_count"].gt(0).sum()),
        "imputation_performed": False,
        "supplemental_source_responses": len(supplemental_manifest),
        "all_source_manifest_rows": len(source_manifest),
        "chrome_vs_command_line_control_byte_identical": identity_ok,
        "unverified_tls_used_for_caiso_downloads": bool(args.allow_unverified_caiso_tls),
    }
    (output / "validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    mapping_rows = "\n".join(
        "| " + " | ".join(
            str(row.get(key, "")).replace("|", "\\|")
            for key in (
                "output_field",
                "source_tier",
                "queryname",
                "official_raw_field",
                "official_dimension_fields",
                "unit",
                "hour_alignment_or_aggregation",
                "official_entry_url",
                "explanation_zh",
            )
        ) + " |"
        for row in dictionary
    )
    mapping = f"""# CAISO 2025 字段官网对应与中文解释

## 数据范围

本文件逐列对应 `{data_name}`。主表沿用 GPT Work 原 163 个数值字段，并从
CAISO 官方 OASIS 补入 {len(added_columns)} 个数值字段。输出共
{summary['rows']:,} 个 UTC 小时、{summary['numeric_fields']:,} 个数值字段。

## 本次补充的官方报表

- `PRC_AS` v12、DAM：辅助服务清算价格；`anc_type=ALL` 与
  `anc_region=ALL` 同时使用时，多日请求实测只返回一个交易日。脚本改为按
  6 个 `ANC_TYPE` 分别请求、`anc_region=ALL`，并遵守官方每次最多 31 天限制。
- `AS_RESULTS` v1、DAM：仅保留 `RESULT_TYPE=AS_MW`，即总采购 MW。
- `ENE_SLRS` v1、DAM：供给、负荷、进口和出口计划，保留 TAC 区域维度。
- `ENE_EIM_TRANSFER_TIE` v4、RTPD：仅保留 `BAA_GRP_ID=CISO`，保留
  进口/出口方向、对端 BAA 和 tie 名称；每小时只在四个 15 分钟区间齐全时取均值。

官方入口：[CAISO OASIS]({OASIS_ENTRY})；
[OASIS API Specification]({OASIS_API_SPEC})。

## 时间、缺失与来源原则

- 统一时间键为 UTC hour beginning，范围 `[2025-01-01T00:00Z, 2026-01-01T00:00Z)`。
- CAISO 本地交易日边界按 IANA `{LOCAL_TZ}` 转换，覆盖 DST 的 23/25 小时日。
- 所有补充列均来自官方 OASIS ZIP；没有插值、前向填充或把缺失改为零。
- 新增价格和辅助服务总采购字段全年无缺失。`ENE_SLRS` 中
  `dam_schedule__mw__import__tac_ncntr` 只在官网原文件出现 96 个小时，其余
  8,664 小时保持缺失；EIM transfer 的缺失来自官方四分之一小时不齐或整小时
  未出现，均没有补值。
- 原 163 字段的来源记录完整保留在 `source_manifest.csv`；新增响应另见
  `supplemental_source_manifest.csv`。
- 本机命令行对 OASIS 遇到证书链错误。批量抓取前，用正常 Chrome 下载和
  命令行控制下载同一 `PRC_AS` ZIP，两个文件逐字节 SHA-256 相同；证据见
  `source_lineage.csv`。这证明了控制样本一致，但不把关闭命令行证书校验描述为
  与正常 TLS 等价。

## 逐字段对应

| 输出字段 | 来源层级 | OASIS queryname | 官网原始字段 | 官网维度 | 单位 | 小时对齐/聚合 | 官方入口 | 中文解释 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
{mapping_rows}
"""
    (output / "field_official_mapping_and_explanations.md").write_text(
        mapping,
        encoding="utf-8",
    )

    validation = f"""# CAISO 2025 增补版验证报告

- 行数：{summary['rows']:,}
- 总列数：{summary['columns_total']:,}（2 个时间字段 + {summary['numeric_fields']:,} 个数值字段）
- 本次增加数值字段：{summary['added_numeric_fields']:,}
- UTC 重复时间戳：{summary['duplicate_utc_timestamps']}
- 数值缺失单元格：{summary['total_numeric_missing_cells']:,}
- 插值：否
- 新增 OASIS 响应：{summary['supplemental_source_responses']:,}
- Chrome 与命令行控制样本逐字节一致：{summary['chrome_vs_command_line_control_byte_identical']}

## 分报表新增字段

- PRC_AS：{len(as_price.columns)}
- AS_RESULTS（仅 AS_MW）：{len(as_total.columns)}
- ENE_SLRS：{len(schedules.columns)}
- ENE_EIM_TRANSFER_TIE（仅 CISO）：{len(ties.columns)}

## 新增字段缺失拆分

- PRC_AS：{missing_by_report['PRC_AS']:,}
- AS_RESULTS（仅 AS_MW）：{missing_by_report['AS_RESULTS_RESULT_TYPE_AS_MW']:,}
- ENE_SLRS：{missing_by_report['ENE_SLRS']:,}
- ENE_EIM_TRANSFER_TIE（仅 CISO）：{missing_by_report['ENE_EIM_TRANSFER_TIE_BAA_GRP_ID_CISO']:,}

`ENE_SLRS` 的缺失几乎全部来自
`dam_schedule__mw__import__tac_ncntr`：该维度只在官方文件出现 96 小时，
不是下载失败；保留稀疏列是为了不擅自删减官网维度。EIM 每列只在四个
15 分钟记录齐全时生成小时值。

## 已执行的检查

1. 最终 UTC 轴严格为 8,760 个递增且唯一的小时。
2. 每个输出字段与 `column_dictionary.csv` 一一对应且无重名。
3. 所有下载均校验 ZIP、内部 CSV、响应 SHA-256 和内部 CSV SHA-256。
4. PRC_AS、AS_RESULTS、ENE_SLRS 的原始键在冲突检测后才 pivot。
5. EIM transfer 只在同一 UTC 小时四个 15 分钟点齐全时取算术均值；
   不齐全的组保留为空，逐列覆盖见 `supplemental_interval_coverage.csv`。
6. 原包不重算、不改写；其 ZIP 与主表 SHA-256 记录在 `source_lineage.csv`。
7. 逐列非空数、缺失、范围、均值和标准差见 `column_validation_stats.csv`。
"""
    (output / "validation_report.md").write_text(validation, encoding="utf-8")

    tls_note = f"""# CAISO OASIS 命令行 TLS 说明

本机 Chrome 能正常从 OASIS 下载官方 ZIP，但命令行证书库对
`oasis.caiso.com` 报告 self-signed certificate in certificate chain。

为避免无证据地关闭验证，本次先对完全相同的官方 `PRC_AS` URL做双通道控制：

- 正常 Chrome C 端下载；
- 命令行关闭证书校验后的控制下载。

两份 ZIP 的字节数和 SHA-256 完全一致：`{identity_ok}`。详细路径、字节数和
哈希见 `source_lineage.csv`。批量下载只访问固定官方主机
`oasis.caiso.com`，并对每个 ZIP 和内部 CSV 做哈希及结构验证。

这项控制降低了本机证书链异常导致错误来源的风险，但不等价于恢复正常 TLS
证书验证；因此 `source_manifest.csv` 的 retrieval_mode 对此有显式标记。
"""
    (output / "caiso_tls_control_note.md").write_text(tls_note, encoding="utf-8")

    readme = f"""# CAISO 2025 UTC 小时对齐增补数据包

主数据：`{data_name}`

这是 GPT Work 原 CAISO 包的可追溯增补版。原表 163 个数值字段保持不变，
并新增 {len(added_columns)} 个经官方 OASIS 下载和覆盖校验的字段。

## 主要文件

- `{data_name}`：最终小时宽表
- `column_dictionary.csv`：机器可读逐列字典
- `field_official_mapping_and_explanations.md`：官网对应与中文解释
- `source_manifest.csv`：原包与本次新增的完整响应清单
- `supplemental_source_manifest.csv`：仅本次新增响应
- `source_lineage.csv`：旧包和浏览器控制样本的 SHA-256
- `supplemental_interval_coverage.csv`：EIM 15 分钟到小时覆盖
- `column_validation_stats.csv`、`validation_report.md`、`validation_summary.json`
- `caiso_tls_control_note.md`：本机证书链异常与双通道控制说明
- `build_caiso_2025_augmented.py`：本增补脚本
- `build_caiso_2025_hourly_original.py`：GPT Work 原构建脚本
- `deliverable_checksums.csv`：包内文件 SHA-256

原始补充 ZIP 缓存在本任务目录的 `work/caiso_supplement_raw/`，未重复塞入交付
ZIP；每个原文件的 URL、字节数、ZIP SHA-256 和内部 CSV SHA-256 均在清单中。
"""
    (output / "README.md").write_text(readme, encoding="utf-8")

    shutil.copy2(
        Path(__file__),
        output / "build_caiso_2025_augmented.py",
    )
    shutil.copy2(
        original_dir / "build_caiso_2025_hourly.py",
        output / "build_caiso_2025_hourly_original.py",
    )
    for filename in ("lmp_identity_validation.csv", "interval_coverage_summary.csv"):
        source = original_dir / filename
        if source.exists():
            shutil.copy2(source, output / filename)

    checks: list[dict[str, object]] = []
    for path in sorted(
        item
        for item in output.iterdir()
        if item.is_file() and item.name != "deliverable_checksums.csv"
    ):
        checks.append(
            {
                "filename": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    pd.DataFrame(checks).to_csv(
        output / "deliverable_checksums.csv",
        index=False,
        lineterminator="\n",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--original-bundle", type=Path, required=True)
    parser.add_argument("--raw-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--browser-sample", type=Path)
    parser.add_argument("--shell-sample", type=Path)
    parser.add_argument("--allow-unverified-caiso-tls", action="store_true")
    parser.add_argument("--pause", type=float, default=3.5)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
