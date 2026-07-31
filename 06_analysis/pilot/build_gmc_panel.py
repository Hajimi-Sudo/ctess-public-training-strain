#!/usr/bin/env python3
"""Build and analyze the 2021-2026 GMC trainer-trainee panel."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr


PROJECT = Path(__file__).resolve().parents[2]
OUTDIR = PROJECT / "06_analysis/pilot"
YEARS = list(range(2021, 2027))

TRAINEE_KEEP = [
    "Overall Satisfaction",
    "Workload",
    "Rota Design",
    "Supportive Environment",
    "Educational Supervision",
    "Clinical Supervision",
]
TRAINER_KEEP = [
    "Time to Train",
    "Resources to Train",
    "Time/Resources to Train",
    "Rota Issues",
    "Support for Training",
    "Supportive Environment",
    "Educational Governance",
]

TRAINER_INDICATOR_MAP = {
    "Time for training": "Time to Train",
    "Resources for trainers": "Resources to Train",
    "Rota Design": "Rota Issues",
    "Support for trainers": "Support for Training",
    "Trainer Development": "Professional Development",
    "Time & training resources": "Time/Resources to Train",
    "Handover & rota design": "Rota Issues",
    "Support & appraisal": "Support for Training",
}


def clean_text(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()


def load_side(side: str) -> pd.DataFrame:
    frames = []
    for year in YEARS:
        if side == "trainee":
            path = PROJECT / f"05_data/processed/gmc_trainee_postspec_trust_detail_{year}.csv"
        else:
            path = PROJECT / f"05_data/processed/gmc_trainer_spec_trust_detail_{year}.csv"
        df = pd.read_csv(path)
        df["year"] = year
        df["side"] = side
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["trust_norm"] = clean_text(out["trust_board"])
    out["spec_norm"] = clean_text(out["specialty"])
    out["indicator_norm"] = out["indicator"].astype(str).str.strip()
    if side == "trainer":
        out["indicator_norm"] = out["indicator_norm"].replace(TRAINER_INDICATOR_MAP)
    out["mean_num"] = pd.to_numeric(out["mean"], errors="coerce")
    out["n_num"] = pd.to_numeric(out["n_all"], errors="coerce")
    out["adverse"] = (
        out["outcome"].fillna("").str.contains("Below|Q1", case=False, regex=True).astype(int)
    )
    out["suppressed"] = (out["outcome"].fillna("") == "N less than 3").astype(int)
    return out


def aggregate_key_burden(df: pd.DataFrame, side: str) -> pd.DataFrame:
    keep = TRAINEE_KEEP if side == "trainee" else TRAINER_KEEP
    key = df[df.indicator_norm.isin(keep)].copy()
    grouped = (
        key.groupby(["year", "trust_norm", "spec_norm"])
        .agg(
            adverse_prop=("adverse", "mean"),
            suppressed_prop=("suppressed", "mean"),
            cells=("indicator_norm", "count"),
            total_n=("n_num", "sum"),
            mean_score=("mean_num", "mean"),
        )
        .reset_index()
    )
    grouped = grouped.rename(
        columns={
            "adverse_prop": f"{side}_adverse_prop",
            "suppressed_prop": f"{side}_suppressed_prop",
            "cells": f"{side}_cells",
            "total_n": f"{side}_n",
            "mean_score": f"{side}_mean_score",
        }
    )

    wide = key.pivot_table(
        index=["year", "trust_norm", "spec_norm"],
        columns="indicator_norm",
        values="mean_num",
        aggfunc="mean",
    )
    wide.columns = [f"{side}_mean_" + column.lower().replace(" ", "_") for column in wide.columns]
    return grouped.merge(wide.reset_index(), on=["year", "trust_norm", "spec_norm"], how="left")


def fit_model(formula: str, data: pd.DataFrame, groups: pd.Series | None = None) -> dict[str, float]:
    model = smf.ols(formula, data=data)
    if groups is None:
        result = model.fit(cov_type="HC3")
    else:
        result = model.fit(cov_type="cluster", cov_kwds={"groups": groups})
    key = "trainer_adverse_prop"
    if "trainer_adverse_lag1" in formula:
        key = "trainer_adverse_lag1"
    return {
        "n": int(result.nobs),
        "coef": float(result.params.get(key, float("nan"))),
        "p": float(result.pvalues.get(key, float("nan"))),
        "r2": float(result.rsquared),
    }


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    trainee = load_side("trainee")
    trainer = load_side("trainer")

    trainee.to_csv(PROJECT / "05_data/processed/gmc_trainee_postspec_trust_detail_2021_2026.csv", index=False)
    trainer.to_csv(PROJECT / "05_data/processed/gmc_trainer_spec_trust_detail_2021_2026.csv", index=False)

    trainee_burden = aggregate_key_burden(trainee, "trainee")
    trainer_burden = aggregate_key_burden(trainer, "trainer")
    panel = trainee_burden.merge(
        trainer_burden, on=["year", "trust_norm", "spec_norm"], how="inner"
    )
    panel["usable"] = (
        (panel.trainee_cells >= 4)
        & (panel.trainer_cells >= 4)
        & (panel.trainee_suppressed_prop < 0.75)
        & (panel.trainer_suppressed_prop < 0.75)
    )
    usable = panel[panel.usable].copy()
    usable = usable.sort_values(["trust_norm", "spec_norm", "year"])
    usable["pair_id"] = usable["trust_norm"] + " || " + usable["spec_norm"]
    usable["trainer_adverse_lag1"] = usable.groupby("pair_id")["trainer_adverse_prop"].shift(1)
    lagged = usable.dropna(subset=["trainer_adverse_lag1"]).copy()

    year_summary = (
        panel.groupby("year")
        .agg(
            merged_units=("usable", "size"),
            usable_units=("usable", "sum"),
            trainee_adverse_mean=("trainee_adverse_prop", "mean"),
            trainer_adverse_mean=("trainer_adverse_prop", "mean"),
        )
        .reset_index()
    )
    yearly_corr = []
    for year, frame in usable.groupby("year"):
        rho, pvalue = spearmanr(
            frame["trainer_adverse_prop"], frame["trainee_adverse_prop"], nan_policy="omit"
        )
        yearly_corr.append(
            {"year": int(year), "n": int(len(frame)), "spearman_rho": float(rho), "p": float(pvalue)}
        )

    pooled_rho, pooled_p = spearmanr(
        usable["trainer_adverse_prop"], usable["trainee_adverse_prop"], nan_policy="omit"
    )
    q75_trainer = usable.trainer_adverse_prop.quantile(0.75)
    q75_trainee = usable.trainee_adverse_prop.quantile(0.75)
    usable["coupled_high_strain"] = (
        (usable.trainer_adverse_prop >= q75_trainer)
        & (usable.trainee_adverse_prop >= q75_trainee)
    ).astype(int)

    models = {
        "pooled_hc3": fit_model(
            "trainee_adverse_prop ~ trainer_adverse_prop + trainer_suppressed_prop + trainee_suppressed_prop",
            usable,
        ),
        "year_fe_cluster_trust": fit_model(
            "trainee_adverse_prop ~ trainer_adverse_prop + trainer_suppressed_prop + trainee_suppressed_prop + C(year)",
            usable,
            groups=usable["trust_norm"],
        ),
        "year_specialty_fe_cluster_trust": fit_model(
            "trainee_adverse_prop ~ trainer_adverse_prop + trainer_suppressed_prop + trainee_suppressed_prop + C(year) + C(spec_norm)",
            usable,
            groups=usable["trust_norm"],
        ),
        "lag1_year_specialty_fe_cluster_trust": fit_model(
            "trainee_adverse_prop ~ trainer_adverse_lag1 + trainer_suppressed_prop + trainee_suppressed_prop + C(year) + C(spec_norm)",
            lagged,
            groups=lagged["trust_norm"],
        ),
    }

    availability = {
        "trainee": {
            str(year): sorted(trainee.loc[trainee.year == year, "indicator_norm"].dropna().unique().tolist())
            for year in YEARS
        },
        "trainer": {
            str(year): sorted(trainer.loc[trainer.year == year, "indicator_norm"].dropna().unique().tolist())
            for year in YEARS
        },
    }
    summary = {
        "raw_rows": {
            "trainee": int(len(trainee)),
            "trainer": int(len(trainer)),
            "total": int(len(trainee) + len(trainer)),
        },
        "coverage": {
            "years": YEARS,
            "trainee_trusts": int(trainee.trust_norm.nunique()),
            "trainer_trusts": int(trainer.trust_norm.nunique()),
            "trainee_specialties": int(trainee.spec_norm.nunique()),
            "trainer_specialties": int(trainer.spec_norm.nunique()),
            "matched_trusts": int(len(set(trainee.trust_norm) & set(trainer.trust_norm))),
            "matched_specialties": int(len(set(trainee.spec_norm) & set(trainer.spec_norm))),
        },
        "panel": {
            "merged_units": int(len(panel)),
            "usable_units": int(len(usable)),
            "unique_pairs_usable": int(usable.pair_id.nunique()),
            "lagged_units": int(len(lagged)),
            "coupled_high_strain_units": int(usable.coupled_high_strain.sum()),
        },
        "pooled_spearman": {"rho": float(pooled_rho), "p": float(pooled_p)},
        "yearly_spearman": yearly_corr,
        "models": models,
        "key_indicators": {"trainee": TRAINEE_KEEP, "trainer": TRAINER_KEEP},
        "indicator_availability": availability,
    }

    panel.to_csv(OUTDIR / "gmc_panel_key_indicator_matrix_2021_2026.csv", index=False)
    usable.to_csv(OUTDIR / "gmc_panel_key_indicator_matrix_usable_2021_2026.csv", index=False)
    year_summary.to_csv(OUTDIR / "gmc_panel_year_summary_2021_2026.csv", index=False)
    (OUTDIR / "gmc_panel_stats_2021_2026.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    report = f"""# GMC 2021-2026 trainer-trainee panel report

## Server Outputs

- Trainee stacked detail rows: {summary['raw_rows']['trainee']:,}
- Trainer stacked detail rows: {summary['raw_rows']['trainer']:,}
- Total GMC detail rows: {summary['raw_rows']['total']:,}
- Merged trust-specialty-year units: {summary['panel']['merged_units']:,}
- Usable primary analysis units: {summary['panel']['usable_units']:,}
- Usable unique trust-specialty pairs: {summary['panel']['unique_pairs_usable']:,}
- Lagged units for one-year-lag analysis: {summary['panel']['lagged_units']:,}

## Primary GMC-Only Signal

- Pooled Spearman association: rho={summary['pooled_spearman']['rho']:.3f}, p={summary['pooled_spearman']['p']:.3g}
- Pooled HC3 model: beta={models['pooled_hc3']['coef']:.3f}, p={models['pooled_hc3']['p']:.3g}, R2={models['pooled_hc3']['r2']:.3f}
- Year fixed-effect model clustered by trust: beta={models['year_fe_cluster_trust']['coef']:.3f}, p={models['year_fe_cluster_trust']['p']:.3g}, R2={models['year_fe_cluster_trust']['r2']:.3f}
- Year + specialty fixed-effect model clustered by trust: beta={models['year_specialty_fe_cluster_trust']['coef']:.3f}, p={models['year_specialty_fe_cluster_trust']['p']:.3g}, R2={models['year_specialty_fe_cluster_trust']['r2']:.3f}
- Lagged trainer strain model: beta={models['lag1_year_specialty_fe_cluster_trust']['coef']:.3f}, p={models['lag1_year_specialty_fe_cluster_trust']['p']:.3g}, R2={models['lag1_year_specialty_fe_cluster_trust']['r2']:.3f}

## Interpretation

The multi-year GMC panel is feasible and large enough for the PMJ study design. The GMC-only trainer-trainee coupling signal is statistically detectable but small. This means the manuscript should not be framed as a strong predictive model; the stronger PMJ angle is public-data surveillance and triangulation of coupled training strain, hardened by external trust-level organisational measures.

## Immediate Next Step

Add NHS Staff Survey and CQC/NHS organisational covariates at trust-year level, then rerun the panel models with external stress signals and a pre-specified Coupled Training Strain Index.
"""
    (OUTDIR / "gmc_panel_report_2021_2026.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("WROTE", OUTDIR / "gmc_panel_report_2021_2026.md")


if __name__ == "__main__":
    main()
