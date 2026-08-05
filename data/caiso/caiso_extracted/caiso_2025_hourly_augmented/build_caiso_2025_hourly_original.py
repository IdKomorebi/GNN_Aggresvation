#!/usr/bin/env python3
"""Build an auditable 2025 UTC-hourly CAISO research dataset from CAISO OASIS."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


UTC_START = pd.Timestamp("2025-01-01T00:00:00Z")
UTC_END = pd.Timestamp("2026-01-01T00:00:00Z")
LOCAL_TZ = "America/Los_Angeles"
OASIS = "https://oasis.caiso.com/oasisapi/SingleZip"
CAISO_OASIS_HOME = "https://oasis.caiso.com/"
CAISO_OASIS_PAGE = (
    "https://www.caiso.com/systems-applications/portals-applications/"
    "open-access-same-time-information-system-oasis"
)
CAISO_API_SPEC = "https://www.caiso.com/documents/oasisapispecification.pdf"
TRADING_HUBS = (
    "TH_NP15_GEN-APND",
    "TH_SP15_GEN-APND",
    "TH_ZP26_GEN-APND",
)
LMP_TYPE_INFO = {
    "LMP": ("lmp", "LMP_PRC", "总节点边际电价"),
    "MCE": ("energy", "LMP_ENE_PRC", "边际能源分量"),
    "MCC": ("congestion", "LMP_CONG_PRC", "边际拥塞分量"),
    "MCL": ("loss", "LMP_LOSS_PRC", "边际损耗分量"),
    "MGHG": ("ghg", "LMP_GHG_PRC", "温室气体分量；即使全年为零也保留"),
}
AS_TYPE = {
    "NR": "non_spinning_reserve",
    "RD": "regulation_down",
    "RU": "regulation_up",
    "SR": "spinning_reserve",
    "RMD": "regulation_mileage_down",
    "RMU": "regulation_mileage_up",
}


@dataclass(frozen=True)
class Report:
    key: str
    official_name: str
    queryname: str
    version: int
    market: str
    raw_value: str
    raw_dimensions: str
    original_cadence: str
    hourly_rule: str


REPORTS = {
    "actual_load": Report(
        "actual_load", "Demand forecast / actual load", "SLD_FCST", 1, "ACTUAL",
        "MW", "TAC_AREA_NAME", "hourly", "official hourly interval value",
    ),
    "dam_load": Report(
        "dam_load", "Demand forecast", "SLD_FCST", 1, "DAM",
        "MW", "TAC_AREA_NAME", "hourly", "official DAM hourly interval value",
    ),
    "actual_renew": Report(
        "actual_renew", "Renewable forecast / actual generation", "SLD_REN_FCST", 1, "ACTUAL",
        "MW", "TRADING_HUB + RENEWABLE_TYPE", "hourly", "official hourly interval value",
    ),
    "dam_renew": Report(
        "dam_renew", "Renewable forecast", "SLD_REN_FCST", 1, "DAM",
        "MW", "TRADING_HUB + RENEWABLE_TYPE", "hourly", "official DAM hourly interval value",
    ),
    "dam_lmp": Report(
        "dam_lmp", "Day-ahead LMP", "PRC_LMP", 12, "DAM",
        "MW", "NODE + LMP_TYPE", "hourly", "official DAM hourly interval value",
    ),
    "rt15_lmp": Report(
        "rt15_lmp", "Fifteen-minute market LMP", "PRC_RTPD_LMP", 3, "RTPD",
        "PRC", "NODE + LMP_TYPE", "15 minutes",
        "arithmetic UTC-hour mean only when all four quarter-hours exist",
    ),
    "as_requirement": Report(
        "as_requirement", "Ancillary-service requirements", "AS_REQ", 1, "DAM",
        "MW", "ANC_REGION + ANC_TYPE + XML_DATA_ITEM", "hourly",
        "official DAM hourly interval value",
    ),
    "as_price": Report(
        "as_price", "Ancillary-service clearing prices", "PRC_AS", 12, "DAM",
        "MW", "ANC_REGION + ANC_TYPE", "hourly", "official DAM hourly interval value",
    ),
    "as_result": Report(
        "as_result", "Ancillary-service results", "AS_RESULTS", 1, "DAM",
        "MW", "ANC_REGION + ANC_TYPE + RESULT_TYPE", "hourly",
        "official DAM AS_MW (total procured MW) only",
    ),
    "slrs": Report(
        "slrs", "Supply and load resource schedules", "ENE_SLRS", 1, "DAM",
        "MW", "TAC_ZONE_NAME + SCHEDULE", "hourly",
        "official DAM hourly interval value",
    ),
    "tie": Report(
        "tie", "EIM transfer by tie", "ENE_EIM_TRANSFER_TIE", 4, "RTPD",
        "VALUE", "TIE_NAME + DIRECTION + FROM_BAA + TO_BAA", "15 minutes",
        "imports negated, directions/neighbor BAAs summed by CAISO tie; strict four-quarter UTC-hour mean",
    ),
}


def slug(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "unknown"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def month_windows() -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return OASIS-safe windows.

    OASIS describes its limit as 31 days, but rejects an exact 31-day
    half-open request as exceeding that limit. Fixed 28-day windows avoid the
    boundary ambiguity and also keep the larger RTPD responses manageable.
    """
    windows = []
    start = UTC_START
    while start < UTC_END:
        end = min(start + pd.Timedelta(days=28), UTC_END)
        windows.append((start, end))
        start = end
    return windows


def oasis_datetime(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y%m%dT%H:%M-0000")


class Downloader:
    def __init__(self, cache: Path, pause: float = 2.0) -> None:
        self.cache = cache
        self.cache.mkdir(parents=True, exist_ok=True)
        self.pause = pause
        self.manifest: list[dict[str, object]] = []
        self._last_request = 0.0

    def get(
        self,
        report: Report,
        start: pd.Timestamp,
        end: pd.Timestamp,
        extra: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        params: dict[str, object] = {
            "resultformat": 6,
            "queryname": report.queryname,
            "version": report.version,
            "market_run_id": report.market,
            "startdatetime": oasis_datetime(start),
            "enddatetime": oasis_datetime(end),
        }
        if extra:
            params.update(extra)
        query = urllib.parse.urlencode(params)
        url = f"{OASIS}?{query}"
        tag = "_".join(
            [report.key, start.strftime("%Y%m%d"), end.strftime("%Y%m%d")]
            + [f"{slug(k)}-{slug(v)}" for k, v in sorted((extra or {}).items())]
        )
        cache_path = self.cache / f"{tag}.zip"
        status = 200
        retrieved = datetime.now().astimezone().isoformat(timespec="seconds")
        if cache_path.exists():
            payload = cache_path.read_bytes()
            retrieval = "cache"
        else:
            payload = b""
            retrieval = "download"
            for attempt in range(7):
                wait = self.pause - (time.monotonic() - self._last_request)
                if wait > 0:
                    time.sleep(wait)
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "academic-research-dataset-builder/1.0"},
                )
                try:
                    with urllib.request.urlopen(req, timeout=180) as response:
                        status = response.status
                        payload = response.read()
                    self._last_request = time.monotonic()
                    break
                except urllib.error.HTTPError as exc:
                    status = exc.code
                    self._last_request = time.monotonic()
                    if exc.code not in (429, 500, 502, 503, 504) or attempt == 6:
                        raise
                    time.sleep(min(90, 8 * (2 ** attempt)))
                except (TimeoutError, urllib.error.URLError):
                    self._last_request = time.monotonic()
                    if attempt == 6:
                        raise
                    time.sleep(min(90, 8 * (2 ** attempt)))
            if not payload:
                raise RuntimeError(f"Empty OASIS response: {url}")
            cache_path.write_bytes(payload)
        digest = sha256_bytes(payload)
        try:
            zf = zipfile.ZipFile(io.BytesIO(payload))
        except zipfile.BadZipFile as exc:
            raise RuntimeError(f"Invalid OASIS ZIP for {report.key}: {url}") from exc
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            xml_text = "\n".join(
                zf.read(n).decode("utf-8", errors="replace")
                for n in zf.namelist() if n.lower().endswith(".xml")
            )
            raise RuntimeError(f"No CSV in OASIS response for {report.key}: {xml_text[:600]}")
        csv_payloads = {name: zf.read(name) for name in csv_names}
        frames = [
            pd.read_csv(io.BytesIO(csv_payloads[name]), low_memory=False)
            for name in csv_names
        ]
        self.manifest.append({
            "source_key": report.key,
            "official_report_name": report.official_name,
            "queryname": report.queryname,
            "version": report.version,
            "market_run_id": report.market,
            "period_start_utc": start.isoformat(),
            "period_end_exclusive_utc": end.isoformat(),
            "request_url": url,
            "http_status": status,
            "retrieval_mode": retrieval,
            "retrieved_at": retrieved,
            "response_bytes": len(payload),
            "sha256": digest,
            "cached_filename": cache_path.name,
            "contained_csv": ";".join(csv_names),
            "contained_csv_sha256": ";".join(
                f"{name}={sha256_bytes(csv_payloads[name])}" for name in csv_names
            ),
        })
        return pd.concat(frames, ignore_index=True)


def fetch_family(
    downloader: Downloader,
    report_key: str,
    extra: dict[str, str] | None = None,
) -> pd.DataFrame:
    report = REPORTS[report_key]
    frames = []
    for start, end in month_windows():
        frames.append(downloader.get(report, start, end, extra))
    df = pd.concat(frames, ignore_index=True)
    return df


def time_filter(df: pd.DataFrame, field: str) -> pd.DataFrame:
    out = df.copy()
    out["timestamp_utc"] = pd.to_datetime(out[field], utc=True, errors="raise")
    return out[(out["timestamp_utc"] >= UTC_START) & (out["timestamp_utc"] < UTC_END)].copy()


def assert_unique(df: pd.DataFrame, keys: list[str], label: str) -> None:
    dup = df.duplicated(keys, keep=False)
    if dup.any():
        example = df.loc[dup, keys].head(10).to_dict("records")
        raise AssertionError(f"{label}: duplicate pivot keys; examples={example}")


def collapse_identical_duplicates(
    df: pd.DataFrame,
    keys: list[str],
    value: str,
    label: str,
) -> pd.DataFrame:
    """Collapse overlapping OASIS query-window rows only when values agree."""
    duplicate_rows = df.duplicated(keys, keep=False)
    if not duplicate_rows.any():
        return df
    check = (
        df.loc[duplicate_rows]
        .groupby(keys, dropna=False)[value]
        .nunique(dropna=False)
    )
    if check.gt(1).any():
        examples = check[check.gt(1)].head(10).to_dict()
        raise AssertionError(
            f"{label}: overlapping OASIS rows disagree; examples={examples}",
        )
    return df.drop_duplicates(keys, keep="first")


def pivot(
    df: pd.DataFrame,
    dimensions: list[str],
    value: str,
    name_fn,
    label: str,
) -> pd.DataFrame:
    keys = ["timestamp_utc", *dimensions]
    df = collapse_identical_duplicates(df, keys, value, label)
    temp = df[keys + [value]].copy()
    temp["column"] = temp.apply(name_fn, axis=1)
    assert_unique(temp, ["timestamp_utc", "column"], label + " output")
    return temp.pivot(index="timestamp_utc", columns="column", values=value).sort_index(axis=1)


def strict_quarter_hour_mean(
    df: pd.DataFrame,
    dimensions: list[str],
    value: str,
    name_fn,
    label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    work["hour_utc"] = work["timestamp_utc"].dt.floor("h")
    keys = ["timestamp_utc", *dimensions]
    work = collapse_identical_duplicates(work, keys, value, label)
    grouped = work.groupby(["hour_utc", *dimensions], dropna=False)[value].agg(["count", "mean"]).reset_index()
    grouped[value] = grouped["mean"].where(grouped["count"].eq(4))
    grouped["column"] = grouped.apply(name_fn, axis=1)
    assert_unique(grouped, ["hour_utc", "column"], label + " hourly")
    wide = grouped.pivot(index="hour_utc", columns="column", values=value).sort_index(axis=1)
    coverage = grouped.pivot(index="hour_utc", columns="column", values="count").sort_index(axis=1)
    wide.index.name = "timestamp_utc"
    coverage.index.name = "timestamp_utc"
    return wide, coverage


def base_frame() -> pd.DataFrame:
    idx = pd.date_range(UTC_START, UTC_END, freq="h", inclusive="left")
    local = idx.tz_convert(LOCAL_TZ)
    return pd.DataFrame({
        "timestamp_utc": idx,
        "timestamp_local": [x.isoformat() for x in local],
    }).set_index("timestamp_utc")


def add_dictionary(
    rows: list[dict[str, object]],
    columns: list[str],
    report_key: str,
    raw_field: str,
    raw_dims: str,
    unit: str,
    explanation: str,
    transform: str | None = None,
) -> None:
    report = REPORTS[report_key]
    for col in columns:
        rows.append({
            "output_field": col,
            "dtype": "float64",
            "unit": unit,
            "source_tier": "official CAISO OASIS",
            "source_key": report_key,
            "official_report_name": report.official_name,
            "queryname": report.queryname,
            "version": report.version,
            "market_run_id": report.market,
            "official_raw_field": raw_field,
            "official_dimension_fields": raw_dims,
            "original_cadence": report.original_cadence,
            "hour_alignment_or_aggregation": transform or report.hourly_rule,
            "official_entry_url": CAISO_OASIS_HOME,
            "explanation_zh": explanation,
        })


def build(args: argparse.Namespace) -> None:
    out = args.output.resolve()
    raw = args.raw_cache.resolve()
    out.mkdir(parents=True, exist_ok=True)
    downloader = Downloader(raw, pause=args.pause)
    full = base_frame()
    dictionary: list[dict[str, object]] = [
        {
            "output_field": "timestamp_utc", "dtype": "string (ISO 8601)", "unit": "UTC",
            "source_tier": "derived time axis", "source_key": "time_axis",
            "official_report_name": "", "queryname": "", "version": "", "market_run_id": "",
            "official_raw_field": "INTERVALSTARTTIME_GMT / INTERVAL_START_GMT",
            "official_dimension_fields": "", "original_cadence": "hourly",
            "hour_alignment_or_aggregation": "canonical calendar-year UTC hour beginning",
            "official_entry_url": CAISO_OASIS_HOME,
            "explanation_zh": "规范 UTC 小时起点；2025 年恰为 8,760 个小时。",
        },
        {
            "output_field": "timestamp_local", "dtype": "string (ISO 8601)", "unit": LOCAL_TZ,
            "source_tier": "derived time label", "source_key": "time_axis",
            "official_report_name": "", "queryname": "", "version": "", "market_run_id": "",
            "official_raw_field": "derived from timestamp_utc", "official_dimension_fields": "",
            "original_cadence": "hourly",
            "hour_alignment_or_aggregation": "IANA America/Los_Angeles conversion; offset retained",
            "official_entry_url": "https://www.iana.org/time-zones",
            "explanation_zh": "加州本地时间标签，保留 -08:00/-07:00 偏移以消除夏令时歧义。",
        },
    ]
    coverage_tables: list[tuple[str, pd.DataFrame]] = []

    for key, prefix, explanation in [
        ("actual_load", "actual_load__mw", "CAISO 发布的 TAC 区域实际小时综合负荷。"),
        ("dam_load", "dam_load_forecast__mw", "CAISO 日前市场负荷预测。"),
    ]:
        d = time_filter(fetch_family(downloader, key), "INTERVALSTARTTIME_GMT")
        d["MW"] = pd.to_numeric(d["MW"], errors="raise")
        w = pivot(
            d, ["TAC_AREA_NAME"], "MW",
            lambda r, p=prefix: f"{p}__{slug(r['TAC_AREA_NAME'])}", key,
        )
        full = full.join(w, how="left")
        add_dictionary(
            dictionary, list(w.columns), key, "MW", "TAC_AREA_NAME", "MW", explanation,
        )

    for key, prefix, explanation in [
        ("actual_renew", "actual_renewable_generation__mw", "交易枢纽×风/光类别实际发电。"),
        ("dam_renew", "dam_renewable_forecast__mw", "日前交易枢纽×风/光类别预测。"),
    ]:
        d = time_filter(fetch_family(downloader, key), "INTERVALSTARTTIME_GMT")
        d["MW"] = pd.to_numeric(d["MW"], errors="raise")
        w = pivot(
            d, ["TRADING_HUB", "RENEWABLE_TYPE"], "MW",
            lambda r, p=prefix: f"{p}__{slug(r['TRADING_HUB'])}__{slug(r['RENEWABLE_TYPE'])}", key,
        )
        full = full.join(w, how="left")
        add_dictionary(
            dictionary, list(w.columns), key, "MW", "TRADING_HUB + RENEWABLE_TYPE",
            "MW", explanation,
        )

    for key, prefix, value_field in [
        ("dam_lmp", "dam_lmp", "MW"),
        ("rt15_lmp", "rt15_lmp_hourly_mean", "PRC"),
    ]:
        parts = []
        for node in TRADING_HUBS:
            parts.append(fetch_family(downloader, key, {"node": node}))
        d = time_filter(pd.concat(parts, ignore_index=True), "INTERVALSTARTTIME_GMT")
        d[value_field] = pd.to_numeric(d[value_field], errors="raise")
        if key == "dam_lmp":
            w = pivot(
                d, ["NODE", "LMP_TYPE"], value_field,
                lambda r, p=prefix: (
                    f"{p}__{LMP_TYPE_INFO[str(r['LMP_TYPE'])][0]}_usd_per_mwh"
                    f"__{slug(r['NODE'])}"
                ),
                key,
            )
        else:
            w, cov = strict_quarter_hour_mean(
                d, ["NODE", "LMP_TYPE"], value_field,
                lambda r, p=prefix: (
                    f"{p}__{LMP_TYPE_INFO[str(r['LMP_TYPE'])][0]}_usd_per_mwh"
                    f"__{slug(r['NODE'])}"
                ),
                key,
            )
            coverage_tables.append((key, cov))
        full = full.join(w, how="left")
        add_dictionary(
            dictionary, list(w.columns), key, value_field, "NODE + LMP_TYPE",
            "$/MWh", "三个 CAISO 交易枢纽的 LMP 总价及官方分量。",
        )

    d = time_filter(fetch_family(
        downloader, "as_requirement", {"anc_type": "ALL", "anc_region": "ALL"},
    ), "INTERVALSTARTTIME_GMT")
    d["MW"] = pd.to_numeric(d["MW"], errors="raise")
    d["bound"] = np.where(
        d["XML_DATA_ITEM"].astype(str).str.contains("MAX"), "maximum", "minimum",
    )
    w = pivot(
        d, ["ANC_REGION", "ANC_TYPE", "bound"], "MW",
        lambda r: (
            f"dam_as_requirement__mw__{slug(r['ANC_REGION'])}"
            f"__{AS_TYPE.get(str(r['ANC_TYPE']), slug(r['ANC_TYPE']))}__{r['bound']}"
        ),
        "as_requirement",
    )
    full = full.join(w, how="left")
    add_dictionary(
        dictionary, list(w.columns), "as_requirement", "MW",
        "ANC_REGION + ANC_TYPE + XML_DATA_ITEM", "MW",
        "日前辅助服务区域需求上下限；仅保留官网实际发布的稀疏组合。",
    )

    if len(full) != 8760 or not full.index.is_unique or not full.index.is_monotonic_increasing:
        raise AssertionError("Canonical time axis failed")
    numeric = full.columns.drop("timestamp_local")
    if full[numeric].apply(lambda s: pd.api.types.is_numeric_dtype(s)).all() is False:
        raise AssertionError("Non-numeric research field detected")

    # LMP component identities; MGHG is reported separately but CAISO's LMP identity
    # is checked against the conventional energy+congestion+loss components.
    identity_rows = []
    for market in ("dam_lmp", "rt15_lmp_hourly_mean"):
        for hub in map(slug, TRADING_HUBS):
            cols = {
                typ: f"{market}__{name}_usd_per_mwh__{hub}"
                for typ, (name, _, _) in LMP_TYPE_INFO.items()
            }
            if all(cols[t] in full for t in ("LMP", "MCE", "MCC", "MCL")):
                residual = full[cols["LMP"]] - full[
                    [cols["MCE"], cols["MCC"], cols["MCL"]]
                ].sum(axis=1, min_count=3)
                identity_rows.append({
                    "market": market, "hub": hub,
                    "observations": int(residual.notna().sum()),
                    "max_abs_residual_usd_per_mwh": float(residual.abs().max()),
                    "mean_abs_residual_usd_per_mwh": float(residual.abs().mean()),
                })
    identity = pd.DataFrame(identity_rows)
    if identity["max_abs_residual_usd_per_mwh"].max() > 0.001:
        raise AssertionError("LMP component identity residual exceeds $0.001/MWh")

    result = full.reset_index()
    result["timestamp_utc"] = result["timestamp_utc"].map(lambda x: x.isoformat().replace("+00:00", "Z"))
    data_name = "caiso_2025_hourly_aligned_full.csv.gz"
    data_path = out / data_name
    with data_path.open("wb") as raw_gz:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_gz, mtime=0) as gz:
            gz.write(result.to_csv(index=False, lineterminator="\n", float_format="%.10g").encode("utf-8"))

    dict_df = pd.DataFrame(dictionary)
    if set(dict_df["output_field"]) != set(result.columns):
        missing = sorted(set(result.columns) - set(dict_df["output_field"]))
        extra = sorted(set(dict_df["output_field"]) - set(result.columns))
        raise AssertionError(f"Dictionary mismatch: missing={missing}, extra={extra}")
    dict_df.to_csv(out / "column_dictionary.csv", index=False, lineterminator="\n")
    pd.DataFrame(downloader.manifest).to_csv(
        out / "source_manifest.csv", index=False, lineterminator="\n",
    )
    identity.to_csv(out / "lmp_identity_validation.csv", index=False, lineterminator="\n")

    coverage_rows = []
    for key, cov in coverage_tables:
        for col in cov.columns:
            counts = cov[col].reindex(full.index)
            coverage_rows.append({
                "source_key": key,
                "output_field": col,
                "expected_intervals_per_hour": 4,
                "hours_with_4_intervals": int(counts.eq(4).sum()),
                "hours_with_1_to_3_intervals": int(counts.between(1, 3).sum()),
                "hours_with_0_intervals": int(counts.isna().sum() + counts.eq(0).sum()),
            })
    pd.DataFrame(coverage_rows).to_csv(
        out / "interval_coverage_summary.csv", index=False, lineterminator="\n",
    )

    stats = []
    for col in numeric:
        s = pd.to_numeric(full[col], errors="coerce")
        stats.append({
            "output_field": col,
            "non_null_count": int(s.notna().sum()),
            "missing_count": int(s.isna().sum()),
            "missing_pct": float(s.isna().mean() * 100),
            "min": float(s.min()) if s.notna().any() else None,
            "max": float(s.max()) if s.notna().any() else None,
            "mean": float(s.mean()) if s.notna().any() else None,
            "std": float(s.std()) if s.notna().any() else None,
            "constant_non_null": bool(s.nunique(dropna=True) <= 1),
        })
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(out / "column_validation_stats.csv", index=False, lineterminator="\n")
    total_missing = int(full[numeric].isna().sum().sum())
    summary = {
        "dataset": "CAISO 2025 UTC-hourly aligned full",
        "rows": len(result),
        "columns_total": len(result.columns),
        "numeric_fields": len(numeric),
        "timestamp_fields": 2,
        "utc_start": result["timestamp_utc"].iloc[0],
        "utc_end": result["timestamp_utc"].iloc[-1],
        "duplicate_utc_timestamps": int(result["timestamp_utc"].duplicated().sum()),
        "total_numeric_missing_cells": total_missing,
        "columns_with_missing": int(stats_df["missing_count"].gt(0).sum()),
        "imputation_performed": False,
        "lmp_identity_max_abs_residual": float(identity["max_abs_residual_usd_per_mwh"].max()),
        "source_responses": len(downloader.manifest),
    }
    (out / "validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )

    included_source_keys = set(dict_df["source_key"]) - {"time_axis"}
    report_table = "\n".join(
        f"| {r.key} | {r.official_name} | `{r.queryname}` v{r.version} | "
        f"{r.market} | `{r.raw_value}` | {r.raw_dimensions} | {r.hourly_rule} |"
        for r in REPORTS.values() if r.key in included_source_keys
    )
    mapping_rows = "\n".join(
        "| " + " | ".join(
            str(row[k]).replace("|", "\\|")
            for k in (
                "output_field", "queryname", "market_run_id", "official_raw_field",
                "official_dimension_fields", "unit",
                "hour_alignment_or_aggregation", "explanation_zh",
            )
        ) + " |"
        for row in dictionary
    )
    mapping_md = f"""# CAISO 2025 字段官网对应与解释

## 官方入口与可审计性

- 官方数据入口：{CAISO_OASIS_HOME}
- CAISO 官网 OASIS 说明页：{CAISO_OASIS_PAGE}
- CAISO OASIS API 规范：{CAISO_API_SPEC}
- API 端点：{OASIS}
- `source_manifest.csv` 保存每个实际请求的完整 URL、时间窗、HTTP 状态、字节数、响应 ZIP SHA-256，以及 ZIP 内每个 CSV 的独立 SHA-256。
- 数据字节全部来自 CAISO OASIS；没有第三方镜像，也没有插值。

## 报表级对应

| 内部键 | 官网报表 | OASIS query | 市场 | 原始值字段 | 官方维度 | 小时化规则 |
| --- | --- | --- | --- | --- | --- | --- |
{report_table}

## 时间、方向与筛选规则

- 主键为 UTC hour beginning，范围 `[2025-01-01 00:00Z, 2026-01-01 00:00Z)`，共 8,760 行。
- `timestamp_local` 由 UTC 按 `America/Los_Angeles` 转换并保留数值偏移；秋季重复小时因此不会冲突。
- RTPD 15 分钟 LMP 必须恰有四个 quarter-hour 才计算小时算术均值；不足四个则该小时为空。
- 官方 LMP 的 `MGHG` 分量保留，即使所选交易枢纽全年为零；代数校验使用 `LMP=MCE+MCC+MCL`，容差 $0.001/MWh。
- 所有官网空缺原样保留；不前向填充、不线性插值、不以零代替。

## 有意未纳入的报表

- `PRC_AS`（辅助服务价格）经实测：提交 28 天时间窗仍只返回一个交易日。为避免生成伪完整的全年字段，本版未纳入。
- `AS_RESULTS`、`ENE_SLRS` 和 `ENE_EIM_TRANSFER_TIE` 的长时间窗请求在自动化环境中未能稳定完成，因此同样未纳入。
- 上述取舍是完整性控制，不是字段同名去重；已纳入字段全部保留。

## 建模与重复信息提醒

- `SLD_FCST` 的 TAC 实体按官网完整保留，其中可能同时包含系统/聚合区域与组成区域；不要未经实体层级核查直接求和。
- 同一交易枢纽、同一市场的 LMP 总价与能源/拥塞/损耗分量具有机械恒等关系。若预测其中一个字段，应从特征中移除同组代数相关字段。
- AS requirement 的 minimum/maximum、AS clearing price 与 AS procured MW 是不同含义，不因字段名相近而删除；但具体目标建模时仍需做泄漏审查。

## 每个输出字段的严格对应

| 输出字段 | OASIS query | 市场 | 官网原始值字段 | 官网维度字段 | 单位 | 小时对齐/聚合 | 中文解释 |
| --- | --- | --- | --- | --- | --- | --- | --- |
{mapping_rows}
"""
    (out / "field_official_mapping_and_explanations.md").write_text(mapping_md, encoding="utf-8")

    validation_md = f"""# CAISO 2025 验证报告

## 结论

- 行数：{summary['rows']:,}
- 总列数：{summary['columns_total']:,}（2 个时间字段 + {summary['numeric_fields']:,} 个数值字段）
- UTC 重复时间戳：{summary['duplicate_utc_timestamps']}
- 数值缺失单元格：{summary['total_numeric_missing_cells']:,}；未做任何插值或补零
- LMP 分量恒等式最大绝对残差：{summary['lmp_identity_max_abs_residual']:.10g} $/MWh
- 官方 OASIS 响应数：{summary['source_responses']:,}，逐响应 SHA-256 见 `source_manifest.csv`

## 自动检查

1. UTC 小时主键严格递增、唯一且恰为 8,760 行。
2. 每个原始报表在透视前检查 `时间×官方维度` 唯一性。
3. 15 分钟数据仅在四个区间完整时小时化，覆盖明细见 `interval_coverage_summary.csv`。
4. 输出数据列与 `column_dictionary.csv` 一一对应。
5. 价格总分量恒等式逐交易枢纽、逐市场检查。
6. 缺失值统计、范围、均值、标准差和常数列见 `column_validation_stats.csv`。
"""
    (out / "validation_report.md").write_text(validation_md, encoding="utf-8")
    readme = f"""# CAISO 2025 UTC 小时对齐数据包

主数据：`{data_name}`

本包用于在独立于 PJM/NYISO 的 CAISO 系统上进行方法外部验证。主表恰为 2025 年 8,760 个 UTC 小时；所有数值字段来自 CAISO OASIS 公开报表，未插值。

## 文件

- `{data_name}`：小时宽表
- `column_dictionary.csv`：每列机器可读字段字典
- `field_official_mapping_and_explanations.md`：官网报表和逐字段严格对应
- `source_manifest.csv`：每个官方响应的 URL、ZIP 字节数及 ZIP 内 CSV 的 SHA-256
- `interval_coverage_summary.csv`：15 分钟到小时覆盖
- `lmp_identity_validation.csv`：LMP 恒等式检查
- `column_validation_stats.csv`：逐列统计和缺失
- `validation_report.md` / `validation_summary.json`：总体检查
- `build_caiso_2025_hourly.py`：完整可复现脚本
- `deliverable_checksums.csv`：包内交付文件 SHA-256

## 研究提醒

如果目标变量是某个 LMP，应剔除同地点同市场的代数分量/同义字段，避免机械泄漏；这属于建模阶段筛选，不应在原始对齐宽表中永久删除。

## 完整性边界

本版有意不包含经实测无法通过多日请求稳定取得全年结果的 `PRC_AS`、`AS_RESULTS`、`ENE_SLRS` 和 `ENE_EIM_TRANSFER_TIE`。这避免了将单日响应拼成看似完整的全年字段；原因与复现实测结论见字段说明文件。
"""
    (out / "README.md").write_text(readme, encoding="utf-8")

    script_target = out / "build_caiso_2025_hourly.py"
    script_target.write_bytes(Path(__file__).read_bytes())
    checks = []
    for path in sorted(p for p in out.iterdir() if p.is_file() and p.name != "deliverable_checksums.csv"):
        checks.append({
            "filename": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    pd.DataFrame(checks).to_csv(
        out / "deliverable_checksums.csv", index=False, lineterminator="\n",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("caiso_2025_hourly"))
    parser.add_argument("--raw-cache", type=Path, default=Path("work/caiso_raw"))
    parser.add_argument("--pause", type=float, default=2.0)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
