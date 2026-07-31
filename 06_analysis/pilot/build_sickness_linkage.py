#!/usr/bin/env python3
"""Link NHS sickness absence rates to the GMC trainer-trainee panel."""

from __future__ import annotations

import difflib
import json
import re
from pathlib import Path

import pandas as pd
import requests
import statsmodels.formula.api as smf


PROJECT = Path(__file__).resolve().parents[2]
OUTDIR = PROJECT / "06_analysis/pilot"
RAW = PROJECT / "05_data/raw_external/nhs_sickness/benchmarking_december"

URLS = {
    2021: "https://files.digital.nhs.uk/4C/FDAA2F/NHS%20Sickness%20Absence%20benchmarking%20tool%2C%20December%202021.csv",
    2022: "https://files.digital.nhs.uk/E3/BB18E7/NHS%20Sickness%20Absence%20benchmarking%20tool%20CSV%2C%20December%202022.csv",
    2023: "https://files.digital.nhs.uk/11/9B2B85/NHS%20Sickness%20Absence%20benchmarking%20tool%20CSV%2C%20December%202023.csv",
    2024: "https://files.digital.nhs.uk/77/CA47EC/NHS%20Sickness%20Absence%20benchmarking%20tool%20CSV%2C%20December%202024.csv",
    2025: "https://files.digital.nhs.uk/D3/14D38C/NHS%20Sickness%20Absence%20benchmarking%20tool%20CSV%2C%20December%202025.csv",
}


def normalise_name(value: str) -> str:
    text = str(value).lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\b(nhs|foundation|trust|the|university)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=240)
    response.raise_for_status()
    path.write_bytes(response.content)


def read_year(year: int) -> pd.DataFrame:
    path = RAW / f"nhs_sickness_benchmarking_december_{year}.csv"
    download(URLS[year], path)
    df = pd.read_csv(path)
    if year == 2021:
        df = df.rename(
            columns={
                "Tm End Date": "DATE",
                "Org code": "ORG_CODE",
                "Org name": "ORG_NAME",
                "Staff group": "STAFF_GROUP",
                "FTE days lost": "FTE_DAYS_LOST",
                "FTE days available": "FTE_DAYS_AVAILABLE",
                "Sickness absence rate (%)": "SICKNESS_ABSENCE_RATE_PERCENT",
            }
        )
    date = pd.to_datetime(df["DATE"], dayfirst=True, errors="coerce")
    df = df[date.dt.year.eq(year) & date.dt.month.eq(12)].copy()
    df["year"] = year
    df = df[df["STAFF_GROUP"].str.lower().eq("all staff groups")].copy()
    df = df[~df["ORG_NAME"].str.contains(r"\bICB\b", case=False, na=False)].copy()
    rate = df["SICKNESS_ABSENCE_RATE_PERCENT"].astype(str).str.replace("%", "", regex=False)
    df["sickness_absence_rate_percent"] = pd.to_numeric(rate, errors="coerce")
    df["fte_days_lost"] = pd.to_numeric(df["FTE_DAYS_LOST"], errors="coerce")
    df["fte_days_available"] = pd.to_numeric(df["FTE_DAYS_AVAILABLE"], errors="coerce")
    return df[
        [
            "year",
            "ORG_CODE",
            "ORG_NAME",
            "sickness_absence_rate_percent",
            "fte_days_lost",
            "fte_days_available",
        ]
    ].rename(columns={"ORG_CODE": "org_id", "ORG_NAME": "org_name"})


def build_crosswalk(panel: pd.DataFrame, sickness: pd.DataFrame) -> pd.DataFrame:
    gmc_names = (
        panel[["trust_norm"]]
        .drop_duplicates()
        .assign(gmc_match_norm=lambda x: x["trust_norm"].map(normalise_name))
    )
    candidates = gmc_names["gmc_match_norm"].tolist()
    lookup = dict(zip(gmc_names["gmc_match_norm"], gmc_names["trust_norm"]))
    orgs = sickness[["org_id", "org_name"]].dropna().drop_duplicates().copy()
    orgs["org_name_norm"] = orgs["org_name"].map(normalise_name)

    rows = []
    for _, row in orgs.iterrows():
        matches = difflib.get_close_matches(row["org_name_norm"], candidates, n=1, cutoff=0.90)
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
        "coef_sickness": float(result.params.get("sickness_absence_rate_percent", float("nan"))),
        "p_sickness": float(result.pvalues.get("sickness_absence_rate_percent", float("nan"))),
        "r2": float(result.rsquared),
    }


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(OUTDIR / "gmc_panel_key_indicator_matrix_usable_2021_2026.csv")
    sickness = pd.concat([read_year(year) for year in range(2021, 2026)], ignore_index=True)
    crosswalk = build_crosswalk(panel, sickness)
    linked = panel.merge(
        sickness.merge(crosswalk[["org_id", "trust_norm", "match_score"]], on="org_id", how="inner"),
        on=["year", "trust_norm"],
        how="inner",
    )
    linked = linked[
        linked["year"].between(2021, 2025)
        & linked["sickness_absence_rate_percent"].notna()
        & (linked["match_score"] >= 0.90)
    ].copy()

    staff_linked_path = OUTDIR / "gmc_panel_plus_nhs_staff_2021_2025.csv"
    combined = None
    combined_model = None
    if staff_linked_path.exists():
        staff_linked = pd.read_csv(staff_linked_path)
        combined = staff_linked.merge(
            linked[
                [
                    "year",
                    "trust_norm",
                    "spec_norm",
                    "sickness_absence_rate_percent",
                    "fte_days_lost",
                    "fte_days_available",
                ]
            ],
            on=["year", "trust_norm", "spec_norm"],
            how="inner",
        )
        combined_model = fit_model(
            "trainee_adverse_prop ~ trainer_adverse_prop + nhs_staff_stress_index + "
            "sickness_absence_rate_percent + trainer_suppressed_prop + trainee_suppressed_prop + "
            "C(year) + C(spec_norm)",
            combined,
        )

    sickness_model = fit_model(
        "trainee_adverse_prop ~ trainer_adverse_prop + sickness_absence_rate_percent + "
        "trainer_suppressed_prop + trainee_suppressed_prop + C(year) + C(spec_norm)",
        linked,
    )

    summary = {
        "sickness_rows": int(len(sickness)),
        "sickness_years": sorted(int(year) for year in sickness["year"].dropna().unique()),
        "sickness_orgs": int(sickness["org_id"].nunique()),
        "crosswalk_rows": int(len(crosswalk)),
        "crosswalk_min_score": float(crosswalk["match_score"].min()) if len(crosswalk) else None,
        "linked_rows": int(len(linked)),
        "linked_trusts": int(linked["trust_norm"].nunique()),
        "linked_pairs": int((linked["trust_norm"] + " || " + linked["spec_norm"]).nunique()),
        "sickness_model": sickness_model,
        "combined_staff_sickness_model": combined_model,
        "sources": URLS,
    }

    sickness.to_csv(OUTDIR / "nhs_sickness_december_org_year_2021_2025.csv", index=False)
    crosswalk.to_csv(OUTDIR / "nhs_sickness_to_gmc_trust_crosswalk.csv", index=False)
    linked.to_csv(OUTDIR / "gmc_panel_plus_nhs_sickness_2021_2025.csv", index=False)
    if combined is not None:
        combined.to_csv(OUTDIR / "gmc_panel_plus_nhs_staff_sickness_2021_2025.csv", index=False)
    (OUTDIR / "nhs_sickness_linkage_stats.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    report = f"""# NHS Sickness Absence Linkage Report

## Data Status

- Source: NHS Digital December sickness absence benchmarking CSVs, 2021-2025.
- Sickness organisation-years: {summary['sickness_rows']:,}
- Sickness organisations: {summary['sickness_orgs']:,}
- Crosswalk rows to GMC trust names: {summary['crosswalk_rows']:,}
- Linked GMC units: {summary['linked_rows']:,}
- Linked trusts: {summary['linked_trusts']:,}
- Linked trust-specialty pairs: {summary['linked_pairs']:,}

## Model Results

Year + specialty fixed-effect model clustered by trust:

- Trainer adverse burden beta: {sickness_model['coef_trainer']:.3f}, p={sickness_model['p_trainer']:.3g}
- Sickness absence rate beta: {sickness_model['coef_sickness']:.3f}, p={sickness_model['p_sickness']:.3g}
- R2: {sickness_model['r2']:.3f}

Combined Staff Survey + sickness model:

- {json.dumps(combined_model, ensure_ascii=False)}

## Interpretation

Sickness absence is an objective organisational-stress triangulation variable. Its role should be reported as validation/adjustment rather than as a causal mediator.
"""
    (OUTDIR / "nhs_sickness_linkage_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("WROTE", OUTDIR / "nhs_sickness_linkage_report.md")


if __name__ == "__main__":
    main()
