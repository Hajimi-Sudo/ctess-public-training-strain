#!/usr/bin/env python3
"""Link NHS Staff Survey organisational stress signals to the GMC panel."""

from __future__ import annotations

import difflib
import json
import re
import zipfile
from pathlib import Path

import pandas as pd
import requests
import statsmodels.formula.api as smf


PROJECT = Path(__file__).resolve().parents[2]
OUTDIR = PROJECT / "06_analysis/pilot"
RAW = PROJECT / "05_data/raw_external"

NHS_STAFF_ZIP = (
    "https://www.nhsstaffsurveys.com/static/305912560d76c82103edb7111251805f/"
    "2025_Local_data_files_v1.zip"
)
NHS_SICKNESS_DEC2025 = (
    "https://files.digital.nhs.uk/11/102C5E/"
    "NHS%20Sickness%20Absence%20by%20reason%2C%20staff%20group%20and%20organisation%20CSV%2C%20December%202025.csv"
)

STRESS_VARIABLES = {
    "q12a": "emotionally_exhausting",
    "q12b": "burnt_out",
    "q12c": "work_frustrates",
    "q12d": "exhausted_next_shift",
    "q12e": "worn_out",
    "q12f": "every_hour_tiring",
    "q11c": "work_related_stress_unwell",
    "q3i": "not_enough_staff",
    "q5a": "unrealistic_time_pressures",
}


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=180)
    response.raise_for_status()
    path.write_bytes(response.content)


def normalise_name(value: str) -> str:
    text = str(value).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\b(nhs|foundation|trust|the|university)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_staff_survey() -> pd.DataFrame:
    zip_path = RAW / "nhs_staff_survey/2025_Local_data_files_v1.zip"
    download(NHS_STAFF_ZIP, zip_path)
    extract_dir = RAW / "nhs_staff_survey/2025_Local_data_files_v1"
    if not extract_dir.exists():
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(zip_path.parent)

    trend = pd.read_csv(extract_dir / "local_trend.csv")
    trend["year"] = pd.to_datetime(trend["year_date"], dayfirst=True).dt.year
    trend = trend[trend["year"].between(2021, 2025)].copy()
    trend = trend[trend["target_variable"].isin(STRESS_VARIABLES)].copy()
    trend["variable"] = trend["target_variable"].map(STRESS_VARIABLES)
    trend["value"] = pd.to_numeric(trend["value"], errors="coerce")
    trend["adverse_value"] = trend["value"]
    trend.loc[trend["target_variable"] == "q3i", "adverse_value"] = (
        1 - trend.loc[trend["target_variable"] == "q3i", "value"]
    )
    wide = trend.pivot_table(
        index=["year", "org_id"],
        columns="variable",
        values="adverse_value",
        aggfunc="mean",
    ).reset_index()
    stress_cols = [column for column in wide.columns if column not in ["year", "org_id"]]
    wide["nhs_staff_stress_index"] = wide[stress_cols].mean(axis=1)
    wide["nhs_staff_stress_n_vars"] = wide[stress_cols].notna().sum(axis=1)
    return wide


def load_org_code_names() -> pd.DataFrame:
    sickness_path = RAW / "nhs_sickness/nhs_sickness_absence_december_2025.csv"
    download(NHS_SICKNESS_DEC2025, sickness_path)
    usecols = ["ORG_CODE", "ORG_NAME"]
    sickness = pd.read_csv(sickness_path, usecols=usecols)
    names = (
        sickness.dropna()
        .drop_duplicates()
        .query("ORG_CODE != 'All organisations'")
        .groupby("ORG_CODE")["ORG_NAME"]
        .first()
        .reset_index()
        .rename(columns={"ORG_CODE": "org_id", "ORG_NAME": "org_name"})
    )
    names = names[~names["org_name"].str.contains(r"\bICB\b", case=False, na=False)].copy()
    names["org_name_norm"] = names["org_name"].map(normalise_name)
    return names


def build_name_crosswalk(panel: pd.DataFrame, org_names: pd.DataFrame) -> pd.DataFrame:
    gmc_names = (
        panel[["trust_norm"]]
        .drop_duplicates()
        .assign(gmc_match_norm=lambda x: x["trust_norm"].map(normalise_name))
    )
    candidates = gmc_names["gmc_match_norm"].tolist()
    lookup = dict(zip(gmc_names["gmc_match_norm"], gmc_names["trust_norm"]))
    rows = []
    for _, row in org_names.iterrows():
        matches = difflib.get_close_matches(row["org_name_norm"], candidates, n=1, cutoff=0.86)
        if not matches:
            continue
        score = difflib.SequenceMatcher(None, row["org_name_norm"], matches[0]).ratio()
        rows.append(
            {
                "org_id": row["org_id"],
                "org_name": row["org_name"],
                "org_name_norm": row["org_name_norm"],
                "trust_norm": lookup[matches[0]],
                "gmc_match_norm": matches[0],
                "match_score": score,
            }
        )
    crosswalk = pd.DataFrame(rows)
    if crosswalk.empty:
        return crosswalk
    return (
        crosswalk.sort_values(["trust_norm", "match_score"], ascending=[True, False])
        .drop_duplicates("trust_norm", keep="first")
        .sort_values(["org_id", "trust_norm"])
        .reset_index(drop=True)
    )


def fit_model(formula: str, data: pd.DataFrame) -> dict[str, float]:
    result = smf.ols(formula, data=data).fit(
        cov_type="cluster", cov_kwds={"groups": data["trust_norm"]}
    )
    return {
        "n": int(result.nobs),
        "coef_trainer": float(result.params.get("trainer_adverse_prop", float("nan"))),
        "p_trainer": float(result.pvalues.get("trainer_adverse_prop", float("nan"))),
        "coef_staff_stress": float(result.params.get("nhs_staff_stress_index", float("nan"))),
        "p_staff_stress": float(result.pvalues.get("nhs_staff_stress_index", float("nan"))),
        "r2": float(result.rsquared),
    }


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(OUTDIR / "gmc_panel_key_indicator_matrix_usable_2021_2026.csv")
    staff = load_staff_survey()
    org_names = load_org_code_names()
    staff = staff.merge(org_names, on="org_id", how="left")
    crosswalk = build_name_crosswalk(
        panel, staff[["org_id", "org_name", "org_name_norm"]].dropna().drop_duplicates()
    )
    staff_linked = staff.merge(crosswalk[["org_id", "trust_norm", "match_score"]], on="org_id", how="inner")
    linked = panel.merge(
        staff_linked.drop(columns=["org_name", "org_name_norm"], errors="ignore"),
        on=["year", "trust_norm"],
        how="inner",
    )

    linked_model = linked[
        (linked["year"].between(2021, 2025))
        & (linked["nhs_staff_stress_n_vars"] >= 5)
        & (linked["match_score"] >= 0.82)
    ].copy()

    models = {
        "gmc_plus_staff_year_specialty_fe": fit_model(
            "trainee_adverse_prop ~ trainer_adverse_prop + nhs_staff_stress_index + "
            "trainer_suppressed_prop + trainee_suppressed_prop + C(year) + C(spec_norm)",
            linked_model,
        )
    }

    summary = {
        "staff_rows": int(len(staff)),
        "staff_years": sorted(int(year) for year in staff["year"].dropna().unique()),
        "staff_orgs": int(staff["org_id"].nunique()),
        "org_name_mapped": int(staff["org_name"].notna().groupby(staff["org_id"]).max().sum()),
        "crosswalk_rows": int(len(crosswalk)),
        "gmc_usable_rows_2021_2025": int(panel[panel.year.between(2021, 2025)].shape[0]),
        "linked_rows": int(len(linked_model)),
        "linked_trusts": int(linked_model["trust_norm"].nunique()),
        "linked_pairs": int((linked_model["trust_norm"] + " || " + linked_model["spec_norm"]).nunique()),
        "models": models,
        "sources": {
            "nhs_staff_survey_zip": NHS_STAFF_ZIP,
            "nhs_sickness_org_name_map": NHS_SICKNESS_DEC2025,
        },
        "variables": STRESS_VARIABLES,
    }

    staff.to_csv(OUTDIR / "nhs_staff_survey_stress_org_year_2021_2025.csv", index=False)
    crosswalk.to_csv(OUTDIR / "nhs_staff_to_gmc_trust_crosswalk.csv", index=False)
    linked_model.to_csv(OUTDIR / "gmc_panel_plus_nhs_staff_2021_2025.csv", index=False)
    (OUTDIR / "nhs_staff_linkage_stats.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    report = f"""# NHS Staff Survey linkage report

## Data Status

- NHS Staff Survey source: 2025 local data zip containing 2021-2025 trend files.
- Organisational name map source: NHS Digital sickness absence December 2025 CSV, used only to map ODS-like `org_id` codes to organisation names.
- Staff Survey organisation-years: {summary['staff_rows']:,}
- Staff Survey organisations: {summary['staff_orgs']:,}
- Name-mapped organisations: {summary['org_name_mapped']:,}
- Fuzzy crosswalk rows to GMC trust names: {summary['crosswalk_rows']:,}

## Linkage To GMC Panel

- GMC usable units in 2021-2025: {summary['gmc_usable_rows_2021_2025']:,}
- Linked GMC + NHS Staff Survey units: {summary['linked_rows']:,}
- Linked trusts: {summary['linked_trusts']:,}
- Linked trust-specialty pairs: {summary['linked_pairs']:,}

## First External-Covariate Model

Year + specialty fixed-effect model clustered by trust:

- Trainer adverse burden beta: {models['gmc_plus_staff_year_specialty_fe']['coef_trainer']:.3f}, p={models['gmc_plus_staff_year_specialty_fe']['p_trainer']:.3g}
- NHS Staff Survey stress index beta: {models['gmc_plus_staff_year_specialty_fe']['coef_staff_stress']:.3f}, p={models['gmc_plus_staff_year_specialty_fe']['p_staff_stress']:.3g}
- R2: {models['gmc_plus_staff_year_specialty_fe']['r2']:.3f}

## Interpretation

The NHS Staff Survey external linkage is feasible but narrower than the GMC-only panel because the Staff Survey source covers England organisations and requires name-based linkage back to GMC trust/board labels. The external stress index should be used as a triangulation and adjustment variable, not as the main exposure.
"""
    (OUTDIR / "nhs_staff_linkage_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("WROTE", OUTDIR / "nhs_staff_linkage_report.md")


if __name__ == "__main__":
    main()
