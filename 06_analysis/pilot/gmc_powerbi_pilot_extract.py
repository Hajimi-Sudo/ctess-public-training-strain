#!/usr/bin/env python3
"""Extract GMC NTS Power BI tables for the PMJ feasibility pilot.

Runs on the remote server. It uses the public GMC Education Data Tool embed
endpoint, then decodes Power BI's DSR row-compression format into CSV files.
"""

from __future__ import annotations

import argparse
import base64
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd
import requests


PROJECT = Path(__file__).resolve().parents[2]
WORKSPACE_ID = "7b218523-484c-49a8-8ffa-0c6fad62e3e6"
GMC_BASE = "https://edt.gmc-uk.org"


@dataclass(frozen=True)
class ReportJob:
    name: str
    report_id: str
    visual_config: Path
    out_csv: Path
    raw_json: Path
    patch_trainer_hidden_table: bool = False


JOBS = [
    ReportJob(
        name="trainee_postspec_trust_full_2026",
        report_id="5dd4e0bf-b20c-424a-912b-48ef9294baa0",
        visual_config=PROJECT
        / "05_data/raw_gmc_nts/discovery/visual_configs/trainee_postspec_trust_hit2_tableEx_f2b8ec9d1476a588e8eb.json",
        out_csv=PROJECT / "05_data/processed/gmc_trainee_postspec_trust_full_2026.csv",
        raw_json=PROJECT / "05_data/raw_gmc_nts/pilot_2026/gmc_trainee_postspec_trust_full_2026_querydata.json",
    ),
    ReportJob(
        name="trainer_spec_trust_full_2026",
        report_id="f89e6007-0d53-4476-b4a6-cf16b6d3c8ef",
        visual_config=PROJECT
        / "05_data/raw_gmc_nts/discovery/visual_configs/trainer_spec_trust_hit3_tableEx_f2b8ec9d1476a588e8eb.json",
        out_csv=PROJECT / "05_data/processed/gmc_trainer_spec_trust_full_2026.csv",
        raw_json=PROJECT / "05_data/raw_gmc_nts/pilot_2026/gmc_trainer_spec_trust_full_2026_querydata.json",
        patch_trainer_hidden_table=True,
    ),
    ReportJob(
        name="trainer_spec_trust_pivot_2026",
        report_id="f89e6007-0d53-4476-b4a6-cf16b6d3c8ef",
        visual_config=PROJECT
        / "05_data/raw_gmc_nts/discovery/visual_configs/trainer_spec_trust_hit0_pivotTable_0c225aae3fa847d67b3c.json",
        out_csv=PROJECT / "05_data/processed/gmc_trainer_spec_trust_pivot_2026.csv",
        raw_json=PROJECT / "05_data/raw_gmc_nts/pilot_2026/gmc_trainer_spec_trust_pivot_2026_querydata.json",
    ),
    ReportJob(
        name="trainee_postspec_trust_pivot_2026",
        report_id="5dd4e0bf-b20c-424a-912b-48ef9294baa0",
        visual_config=PROJECT
        / "05_data/raw_gmc_nts/discovery/visual_configs/trainee_postspec_trust_hit0_pivotTable_0c225aae3fa847d67b3c.json",
        out_csv=PROJECT / "05_data/processed/gmc_trainee_postspec_trust_pivot_2026.csv",
        raw_json=PROJECT / "05_data/raw_gmc_nts/pilot_2026/gmc_trainee_postspec_trust_pivot_2026_querydata.json",
    ),
]


DETAIL_CONFIGS = {
    "trainee": {
        "report_id": "5dd4e0bf-b20c-424a-912b-48ef9294baa0",
        "specialty_entity": "Post Specialty",
        "specialty_source": "p",
        "specialty_property": "Current Post Specialty",
        "specialty_name": "Post Specialty.Current Post Specialty",
        "out_pattern": "05_data/processed/gmc_trainee_postspec_trust_detail_{year}.csv",
        "manifest_pattern": "06_analysis/pilot/gmc_trainee_postspec_trust_detail_{year}_manifest.json",
    },
    "trainer": {
        "report_id": "f89e6007-0d53-4476-b4a6-cf16b6d3c8ef",
        "specialty_entity": "Specialty",
        "specialty_source": "s",
        "specialty_property": "Current Specialty",
        "specialty_name": "Specialty.Current Specialty",
        "out_pattern": "05_data/processed/gmc_trainer_spec_trust_detail_{year}.csv",
        "manifest_pattern": "06_analysis/pilot/gmc_trainer_spec_trust_detail_{year}_manifest.json",
    },
}


def year_path(pattern: str, year: int) -> Path:
    return PROJECT / pattern.format(year=year)


def load_json_response(response: requests.Response) -> Any:
    text = response.text.lstrip("\ufeff\n\r\t ")
    if text.startswith(")]}'"):
        text = text.split("\n", 1)[1]
    return json.loads(text)


def decode_embed_cluster(embed_url: str) -> str:
    config = parse_qs(urlparse(embed_url).query)["config"][0]
    padded = config + "=" * ((4 - len(config) % 4) % 4)
    decoded = json.loads(base64.urlsafe_b64decode(padded))
    return decoded["clusterUrl"].rstrip("/")


def get_powerbi_context(session: requests.Session, report_id: str) -> dict[str, Any]:
    response = session.get(
        f"{GMC_BASE}/api/informatics/embedreport/get",
        params={"reportid": report_id, "workspaceid": WORKSPACE_ID},
        timeout=45,
    )
    response.raise_for_status()
    embed = load_json_response(response)
    token = embed["EmbedToken"]["Token"]
    embed_url = embed["Report"]["EmbedUrl"]
    cluster = decode_embed_cluster(embed_url)
    headers = {
        "Authorization": f"EmbedToken {token}",
        "X-PowerBI-HostEnv": "Embed for Customers",
        "User-Agent": "Mozilla/5.0",
        "Content-Type": "application/json",
    }
    model_response = session.get(
        f"{cluster}/explore/reports/{report_id}/modelsAndExploration"
        "?preferReadOnlySession=true&skipQueryData=true",
        headers=headers,
        timeout=90,
    )
    model_response.raise_for_status()
    models = load_json_response(model_response)
    model = models["models"][0]
    return {
        "cluster": cluster,
        "headers": headers,
        "model_id": model["id"],
        "dataset_id": model["dbName"],
    }


def patch_trainer_hidden_query(query: dict[str, Any]) -> None:
    """Fix stale trainee entity names in the trainer hidden download table."""
    for item in query.get("From", []):
        if item.get("Name") == "p" and item.get("Entity") == "Post Specialty":
            item["Entity"] = "Specialty"

    benchmark_property_map = {
        "Mean": "Benchmark Mean",
        "Q1": "Benchmark Q1",
        "Q3": "Benchmark Q3",
        "Lower CI": "Benchmark Lower CI",
        "Upper CI": "Benchmark Upper CI",
        "Standard Deviation": "Benchmark Standard Deviation",
        "N": "Benchmark N",
    }

    def patch_expr(node: Any) -> None:
        if isinstance(node, dict):
            column = node.get("Column")
            if isinstance(column, dict):
                source_ref = column.get("Expression", {}).get("SourceRef", {})
                if source_ref.get("Source") == "p":
                    column["Property"] = "Current Specialty"
                if source_ref.get("Source") == "b":
                    prop = column.get("Property")
                    if prop in benchmark_property_map:
                        column["Property"] = benchmark_property_map[prop]
            for value in node.values():
                patch_expr(value)
        elif isinstance(node, list):
            for value in node:
                patch_expr(value)

    patch_expr(query)

    for select in query.get("Select", []):
        if select.get("Name") == "Post Specialty.Current Post Specialty":
            select["Name"] = "Specialty.Current Specialty"
            select["NativeReferenceName"] = "Current Specialty"


def add_year_filter(query: dict[str, Any], year: int) -> None:
    source = None
    for item in query.get("From", []):
        if item.get("Entity") == "Year":
            source = item["Name"]
            break
    if source is None:
        return
    query["Where"] = [
        {
            "Condition": {
                "In": {
                    "Expressions": [
                        {
                            "Column": {
                                "Expression": {"SourceRef": {"Source": source}},
                                "Property": "Survey Year",
                            }
                        }
                    ],
                    "Values": [[{"Literal": {"Value": f"{year}L"}}]],
                }
            }
        }
    ]


def powerbi_string_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def add_text_filter(query: dict[str, Any], source: str, property_name: str, value: str) -> None:
    query.setdefault("Where", []).append(
        {
            "Condition": {
                "In": {
                    "Expressions": [
                        {
                            "Column": {
                                "Expression": {"SourceRef": {"Source": source}},
                                "Property": property_name,
                            }
                        }
                    ],
                    "Values": [[{"Literal": {"Value": powerbi_string_literal(value)}}]],
                }
            }
        }
    )


def col(source: str, property_name: str, name: str) -> dict[str, Any]:
    return {
        "Column": {
            "Expression": {"SourceRef": {"Source": source}},
            "Property": property_name,
        },
        "Name": name,
    }


def agg(source: str, property_name: str, name: str, function: int = 0) -> dict[str, Any]:
    return {
        "Aggregation": {
            "Expression": {
                "Column": {
                    "Expression": {"SourceRef": {"Source": source}},
                    "Property": property_name,
                }
            },
            "Function": function,
        },
        "Name": name,
    }


def make_manual_detail_query(config: dict[str, Any], year: int, specialty: str | None = None) -> dict[str, Any]:
    spec_source = config["specialty_source"]
    query = {
        "Version": 2,
        "From": [
            {"Name": "t", "Entity": "Trust / Board", "Type": 0},
            {"Name": "i", "Entity": "Indicator", "Type": 0},
            {"Name": "y", "Entity": "Year", "Type": 0},
            {"Name": "o", "Entity": "Outlier Fact", "Type": 0},
            {
                "Name": spec_source,
                "Entity": config["specialty_entity"],
                "Type": 0,
            },
        ],
        "Select": [
            col(spec_source, config["specialty_property"], config["specialty_name"]),
            col("t", "Current Trust/Board", "Trust / Board.Current Trust/Board"),
            col("i", "Current Indicator", "Indicator.Current Indicator"),
            col("y", "Survey Year", "Year.Survey Year"),
            col("o", "Outcome", "Outlier Fact.Outcome"),
            agg("o", "Mean", "Sum(Outlier Fact.Mean)"),
            agg("o", "Lower CI", "Sum(Outlier Fact.Lower CI)"),
            agg("o", "Upper CI", "Sum(Outlier Fact.Upper CI)"),
            agg("o", "N All", "Sum(Outlier Fact.N)"),
            agg("o", "Standard Deviation", "Sum(Outlier Fact.Standard Deviation)"),
            agg("o", "Response Rate", "Sum(Outlier Fact.Response Rate)"),
        ],
        "OrderBy": [
            {
                "Direction": 1,
                "Expression": {
                    "Column": {
                        "Expression": {"SourceRef": {"Source": spec_source}},
                        "Property": config["specialty_property"],
                    }
                },
            },
            {
                "Direction": 1,
                "Expression": {
                    "Column": {
                        "Expression": {"SourceRef": {"Source": "t"}},
                        "Property": "Current Trust/Board",
                    }
                },
            },
            {
                "Direction": 1,
                "Expression": {
                    "Column": {
                        "Expression": {"SourceRef": {"Source": "i"}},
                        "Property": "Current Indicator",
                    }
                },
            },
        ],
    }
    add_year_filter(query, year)
    if specialty is not None:
        add_text_filter(query, spec_source, config["specialty_property"], specialty)
    return query


def make_specialty_list_query(config: dict[str, Any], year: int) -> dict[str, Any]:
    spec_source = config["specialty_source"]
    query = {
        "Version": 2,
        "From": [
            {"Name": "y", "Entity": "Year", "Type": 0},
            {"Name": "o", "Entity": "Outlier Fact", "Type": 0},
            {
                "Name": spec_source,
                "Entity": config["specialty_entity"],
                "Type": 0,
            },
        ],
        "Select": [
            col(spec_source, config["specialty_property"], config["specialty_name"]),
            agg("o", "N All", "Sum(Outlier Fact.N)"),
        ],
        "OrderBy": [
            {
                "Direction": 1,
                "Expression": {
                    "Column": {
                        "Expression": {"SourceRef": {"Source": spec_source}},
                        "Property": config["specialty_property"],
                    }
                },
            }
        ],
    }
    add_year_filter(query, year)
    return query


def build_query_body(
    query: dict[str, Any],
    visual: dict[str, Any] | None,
    report_id: str,
    visual_id: str,
    model_id: int,
    dataset_id: str,
    row_count: int,
) -> dict[str, Any]:
    select_names = [item["Name"] for item in query["Select"]]
    select_index = {name: idx for idx, name in enumerate(select_names)}
    projection_names = []
    if visual is not None:
        projections = visual["singleVisual"].get("projections", {})
        for role in ("Values", "Rows", "Columns"):
            for item in projections.get(role, []):
                query_ref = item.get("queryRef")
                if query_ref in select_index and query_ref not in projection_names:
                    projection_names.append(query_ref)
    if not projection_names:
        projection_names = select_names
    projection_indices = [select_index[name] for name in projection_names]
    command = {
        "SemanticQueryDataShapeCommand": {
            "Query": query,
            "Binding": {
                "Primary": {"Groupings": [{"Projections": projection_indices}]},
                "DataReduction": {
                    "DataVolume": 4,
                    "Primary": {"Window": {"Count": row_count}},
                },
            },
            "ExecutionMetricsKind": 1,
            "Shape": {
                "Name": "DSR",
                "Projections": [{"Field": {"QueryRef": name}} for name in projection_names],
                "PrimaryHierarchy": [{"Field": {"QueryRef": name}} for name in projection_names],
            },
        }
    }
    return {
        "version": "1.0.0",
        "queries": [
            {
                "Query": {"Commands": [command]},
                "ApplicationContext": {
                    "DatasetId": dataset_id,
                    "Sources": [{"ReportId": report_id, "VisualId": visual_id}],
                },
            }
        ],
        "cancelQueries": [],
        "modelId": model_id,
    }


def decode_value(value: Any, spec: dict[str, Any], value_dicts: dict[str, list[Any]]) -> Any:
    dict_name = spec.get("DN")
    if dict_name and isinstance(value, int):
        values = value_dicts.get(dict_name, [])
        if 0 <= value < len(values):
            return values[value]
    return value


def decode_row_block(rows: list[dict[str, Any]], value_dicts: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not rows or "S" not in rows[0]:
        return []
    schema = rows[0]["S"]
    columns = [item["N"] for item in schema]
    previous = [None] * len(columns)
    decoded: list[dict[str, Any]] = []

    for raw in rows:
        repeat_mask = int(raw.get("R", 0))
        null_mask = int(raw.get("\u00d8", raw.get("Ø", 0)))
        values = iter(raw.get("C", []))
        row = []
        for idx, spec in enumerate(schema):
            if repeat_mask & (1 << idx):
                value = previous[idx]
            elif null_mask & (1 << idx):
                value = None
            else:
                value = next(values, None)
                value = decode_value(value, spec, value_dicts)
            row.append(value)
        previous = row
        decoded.append(dict(zip(columns, row)))
    return decoded


def decode_dsr(payload: dict[str, Any]) -> pd.DataFrame:
    data = payload["results"][0]["result"]["data"]
    if "dsr" not in data:
        raise RuntimeError(json.dumps(data, indent=2)[:2000])
    descriptor = data.get("descriptor", {})
    name_map = {item["Value"]: item["Name"] for item in descriptor.get("Select", [])}
    all_rows: list[dict[str, Any]] = []
    for ds in data["dsr"].get("DS", []):
        value_dicts = ds.get("ValueDicts", {})
        for phase in ds.get("PH", []):
            for rows in phase.values():
                decoded = decode_row_block(rows, value_dicts)
                for row in decoded:
                    if not any(key.startswith(("G", "M")) for key in row):
                        continue
                    renamed = {name_map.get(key, key): value for key, value in row.items()}
                    all_rows.append(renamed)
    return pd.DataFrame(all_rows)


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    canonical = {}
    for column in df.columns:
        lowered = column.lower()
        if "current post specialty" in lowered or "current specialty" in lowered:
            canonical[column] = "specialty"
        elif "current trust/board" in lowered:
            canonical[column] = "trust_board"
        elif "current indicator" in lowered:
            canonical[column] = "indicator"
        elif "survey year" in lowered:
            canonical[column] = "year"
        elif "outlier fact.outcome" in lowered or "short outcome" in lowered:
            canonical[column] = "outcome"
        elif "outlier fact.mean" in lowered:
            canonical[column] = "mean"
        elif "lower ci" in lowered and "outlier fact" in lowered:
            canonical[column] = "lower_ci"
        elif "upper ci" in lowered and "outlier fact" in lowered:
            canonical[column] = "upper_ci"
        elif "standard deviation" in lowered and "outlier fact" in lowered:
            canonical[column] = "sd"
        elif "outlier fact.n" in lowered:
            canonical[column] = "n_all"
        elif "benchmark.mean" in lowered:
            canonical[column] = "benchmark_mean"
        elif "benchmark.q1" in lowered:
            canonical[column] = "benchmark_q1"
        elif "benchmark.q3" in lowered:
            canonical[column] = "benchmark_q3"
        elif "benchmark.lower ci" in lowered:
            canonical[column] = "benchmark_lower_ci"
        elif "benchmark.upper ci" in lowered:
            canonical[column] = "benchmark_upper_ci"
        elif "benchmark.n" in lowered:
            canonical[column] = "benchmark_n"
        elif "benchmark.standard deviation" in lowered:
            canonical[column] = "benchmark_sd"
    return df.rename(columns=canonical)


def run_job(session: requests.Session, job: ReportJob, year: int, row_count: int) -> dict[str, Any]:
    context = get_powerbi_context(session, job.report_id)
    visual = json.loads(job.visual_config.read_text(encoding="utf-8"))
    query = copy.deepcopy(visual["singleVisual"]["prototypeQuery"])
    if job.patch_trainer_hidden_table:
        patch_trainer_hidden_query(query)
    add_year_filter(query, year)
    body = build_query_body(
        query=query,
        visual=visual,
        report_id=job.report_id,
        visual_id=visual["name"],
        model_id=context["model_id"],
        dataset_id=context["dataset_id"],
        row_count=row_count,
    )
    job.raw_json.parent.mkdir(parents=True, exist_ok=True)
    job.out_csv.parent.mkdir(parents=True, exist_ok=True)

    response = session.post(
        f"{context['cluster']}/explore/querydata?synchronous=true",
        headers=context["headers"],
        json=body,
        timeout=180,
    )
    response.raise_for_status()
    job.raw_json.write_text(response.text, encoding="utf-8")
    payload = load_json_response(response)
    df = canonicalize_columns(decode_dsr(payload))
    df.to_csv(job.out_csv, index=False)

    summary = {
        "job": job.name,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "out_csv": str(job.out_csv),
        "raw_json": str(job.raw_json),
    }
    for col in ["year", "indicator", "specialty", "trust_board"]:
        if col in df.columns:
            summary[f"unique_{col}"] = int(df[col].nunique(dropna=True))
    return summary


def execute_query(
    session: requests.Session,
    context: dict[str, Any],
    report_id: str,
    query: dict[str, Any],
    row_count: int,
    visual_id: str = "manual",
) -> pd.DataFrame:
    body = build_query_body(
        query=query,
        visual=None,
        report_id=report_id,
        visual_id=visual_id,
        model_id=context["model_id"],
        dataset_id=context["dataset_id"],
        row_count=row_count,
    )
    response = session.post(
        f"{context['cluster']}/explore/querydata?synchronous=true",
        headers=context["headers"],
        json=body,
        timeout=180,
    )
    response.raise_for_status()
    return canonicalize_columns(decode_dsr(load_json_response(response)))


def run_chunked_detail(session: requests.Session, kind: str, year: int, row_count: int) -> dict[str, Any]:
    config = DETAIL_CONFIGS[kind]
    out_csv = year_path(config["out_pattern"], year)
    manifest = year_path(config["manifest_pattern"], year)
    context = get_powerbi_context(session, config["report_id"])
    specialties_df = execute_query(
        session=session,
        context=context,
        report_id=config["report_id"],
        query=make_specialty_list_query(config, year),
        row_count=5000,
        visual_id=f"manual_{kind}_specialty_list",
    )
    specialties = sorted(
        value
        for value in specialties_df.get("specialty", pd.Series(dtype=object)).dropna().unique().tolist()
        if str(value).strip()
    )
    frames = []
    chunks = []
    for idx, specialty in enumerate(specialties, start=1):
        print(f"CHUNK {kind} {idx}/{len(specialties)} {specialty}", flush=True)
        df = execute_query(
            session=session,
            context=context,
            report_id=config["report_id"],
            query=make_manual_detail_query(config, year, specialty=specialty),
            row_count=row_count,
            visual_id=f"manual_{kind}_detail",
        )
        df = df.dropna(subset=["specialty", "trust_board", "indicator"], how="any")
        frames.append(df)
        chunks.append(
            {
                "specialty": specialty,
                "rows": int(len(df)),
                "trusts": int(df["trust_board"].nunique(dropna=True)) if "trust_board" in df else 0,
                "indicators": int(df["indicator"].nunique(dropna=True)) if "indicator" in df else 0,
            }
        )
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined = combined.drop_duplicates()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_csv, index=False)
    summary = {
        "kind": kind,
        "year": year,
        "rows": int(len(combined)),
        "specialties": len(specialties),
        "unique_specialty": int(combined["specialty"].nunique(dropna=True)) if "specialty" in combined else 0,
        "unique_trust_board": int(combined["trust_board"].nunique(dropna=True)) if "trust_board" in combined else 0,
        "unique_indicator": int(combined["indicator"].nunique(dropna=True)) if "indicator" in combined else 0,
        "out_csv": str(out_csv),
        "chunks": chunks,
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--years", type=int, nargs="+")
    parser.add_argument("--row-count", type=int, default=200000)
    parser.add_argument("--chunked-detail", action="store_true")
    args = parser.parse_args()

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    summaries = []
    if args.chunked_detail:
        years = args.years if args.years else [args.year]
        for year in years:
            for kind in ("trainee", "trainer"):
                print(f"RUN chunked_detail {kind} {year}", flush=True)
                try:
                    summary = run_chunked_detail(session, kind, year, args.row_count)
                except Exception as exc:
                    summary = {"kind": kind, "year": year, "error": repr(exc)}
                print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
                summaries.append(summary)
        years_slug = "-".join(str(year) for year in years)
        out = PROJECT / f"06_analysis/pilot/gmc_powerbi_chunked_detail_summary_{years_slug}.json"
        out.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"WROTE {out}", flush=True)
        return

    for job in JOBS:
        print(f"RUN {job.name}", flush=True)
        try:
            summary = run_job(session, job, args.year, args.row_count)
        except Exception as exc:
            summary = {"job": job.name, "error": repr(exc)}
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
        summaries.append(summary)

    out = PROJECT / "06_analysis/pilot/gmc_powerbi_pilot_extract_summary.json"
    out.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"WROTE {out}", flush=True)


if __name__ == "__main__":
    main()
