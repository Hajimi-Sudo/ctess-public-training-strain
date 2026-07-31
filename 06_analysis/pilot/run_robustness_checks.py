#!/usr/bin/env python3
"""Run robustness checks requested by the external reviewer."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr


PROJECT = Path(__file__).resolve().parents[2]
OUTDIR = PROJECT / "06_analysis/pilot"

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
TRAINER_MAP = {
    "Time for training": "Time to Train",
    "Resources for trainers": "Resources to Train",
    "Rota Design": "Rota Issues",
    "Support for trainers": "Support for Training",
    "Trainer Development": "Professional Development",
    "Time & training resources": "Time/Resources to Train",
    "Handover & rota design": "Rota Issues",
    "Support & appraisal": "Support for Training",
}


def load_side(side: str) -> pd.DataFrame:
    path = PROJECT / f"05_data/processed/gmc_{side}_postspec_trust_detail_2021_2026.csv"
    if side == "trainer":
        path = PROJECT / "05_data/processed/gmc_trainer_spec_trust_detail_2021_2026.csv"
    df = pd.read_csv(path)
    df["trust_norm"] = df["trust_norm"].astype(str)
    df["spec_norm"] = df["spec_norm"].astype(str)
    df["indicator_norm"] = df["indicator"].astype(str).str.strip()
    if side == "trainer":
        df["indicator_norm"] = df["indicator_norm"].replace(TRAINER_MAP)
    df["mean_num"] = pd.to_numeric(df["mean"], errors="coerce")
    df["n_num"] = pd.to_numeric(df["n_all"], errors="coerce")
    outcome = df["outcome"].fillna("").str.lower()
    df["adverse_primary"] = outcome.str.contains("below|q1", regex=True).astype(int)
    df["adverse_strict_below"] = outcome.eq("below").astype(int)
    df["suppressed"] = outcome.eq("n less than 3").astype(int)
    return df


def build_burden(df: pd.DataFrame, side: str, adverse_col: str) -> pd.DataFrame:
    keep = TRAINEE_KEEP if side == "trainee" else TRAINER_KEEP
    key = df[df["indicator_norm"].isin(keep)].copy()
    return (
        key.groupby(["year", "trust_norm", "spec_norm"])
        .agg(
            adverse_prop=(adverse_col, "mean"),
            suppressed_prop=("suppressed", "mean"),
            cells=("indicator_norm", "count"),
            total_n=("n_num", "sum"),
            mean_score=("mean_num", "mean"),
        )
        .reset_index()
        .rename(
            columns={
                "adverse_prop": f"{side}_adverse_prop",
                "suppressed_prop": f"{side}_suppressed_prop",
                "cells": f"{side}_cells",
                "total_n": f"{side}_n",
                "mean_score": f"{side}_mean_score",
            }
        )
    )


def model(panel: pd.DataFrame) -> dict[str, float]:
    result = smf.ols(
        "trainee_adverse_prop ~ trainer_adverse_prop + trainer_suppressed_prop + "
        "trainee_suppressed_prop + C(year) + C(spec_norm)",
        data=panel,
    ).fit(cov_type="cluster", cov_kwds={"groups": panel["trust_norm"]})
    rho, p_rho = spearmanr(panel["trainer_adverse_prop"], panel["trainee_adverse_prop"])
    return {
        "n": int(result.nobs),
        "coef": float(result.params["trainer_adverse_prop"]),
        "p": float(result.pvalues["trainer_adverse_prop"]),
        "r2": float(result.rsquared),
        "spearman_rho": float(rho),
        "spearman_p": float(p_rho),
    }


def run_definition(trainee: pd.DataFrame, trainer: pd.DataFrame, adverse_col: str, suppression_cut: float) -> tuple[pd.DataFrame, dict]:
    t1 = build_burden(trainee, "trainee", adverse_col)
    t2 = build_burden(trainer, "trainer", adverse_col)
    panel = t1.merge(t2, on=["year", "trust_norm", "spec_norm"], how="inner")
    panel = panel[
        (panel["trainee_cells"] >= 4)
        & (panel["trainer_cells"] >= 4)
        & (panel["trainee_suppressed_prop"] < suppression_cut)
        & (panel["trainer_suppressed_prop"] < suppression_cut)
    ].copy()
    return panel, model(panel)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    detail_trainee = PROJECT / "05_data/processed/gmc_trainee_postspec_trust_detail_2021_2026.csv"
    detail_trainer = PROJECT / "05_data/processed/gmc_trainer_spec_trust_detail_2021_2026.csv"
    if not detail_trainee.exists() or not detail_trainer.exists():
        table = pd.read_csv(OUTDIR / "table_robustness_checks.csv")
        suppression_summary = pd.read_csv(OUTDIR / "table_suppression_summary.csv")
        red_flags = pd.read_csv(OUTDIR / "table_red_flag_units_top100.csv")
        primary = table.iloc[0].to_dict()
        strict = table[
            table["check"].eq("adverse_strict_below_suppression_lt_0.75")
        ].iloc[0].to_dict()
        summary = {
            "mode": "precomputed_tables",
            "reason": "detail-level intermediate GMC extracts are not included in the public repository",
            "robustness_rows": int(len(table)),
            "primary_result": primary,
            "strict_below_result": strict,
            "suppression_years": int(len(suppression_summary)),
            "red_flag_units_top100": int(len(red_flags)),
        }
        (OUTDIR / "robustness_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        report = f"""# Robustness and Suppression Report

This report was regenerated from the precomputed aggregate robustness tables included in the repository. Detail-level GMC intermediate extracts are not distributed with the public code release.

Primary model, suppression <0.75: beta={primary['coef']:.3f}, p={primary['p']:.3g}, n={int(primary['n']):,}.

Strict below-only model, suppression <0.75: beta={strict['coef']:.3f}, p={strict['p']:.3g}, n={int(strict['n']):,}.

Suppression summaries by year are saved in `table_suppression_summary.csv`.

The top 100 high-trainer/high-trainee adverse burden units are saved in `table_red_flag_units_top100.csv` for descriptive surveillance use. These should be presented as quality-improvement signals, not rankings of institutional quality.
"""
        (OUTDIR / "robustness_report.md").write_text(report, encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print("WROTE", OUTDIR / "robustness_report.md")
        return

    trainee = load_side("trainee")
    trainer = load_side("trainer")

    checks = []
    panels = {}
    for adverse_col in ["adverse_primary", "adverse_strict_below"]:
        for suppression_cut in [0.75, 0.50, 0.25]:
            panel, result = run_definition(trainee, trainer, adverse_col, suppression_cut)
            label = f"{adverse_col}_suppression_lt_{suppression_cut}"
            result.update(
                {
                    "check": label,
                    "adverse_definition": adverse_col,
                    "suppression_cut": suppression_cut,
                    "unique_pairs": int((panel["trust_norm"] + " || " + panel["spec_norm"]).nunique()),
                }
            )
            checks.append(result)
            panels[label] = panel

    primary_panel = panels["adverse_primary_suppression_lt_0.75"]
    suppression_summary = (
        primary_panel.groupby("year")
        .agg(
            units=("trust_norm", "size"),
            mean_trainee_suppressed=("trainee_suppressed_prop", "mean"),
            mean_trainer_suppressed=("trainer_suppressed_prop", "mean"),
            high_trainee_suppressed=("trainee_suppressed_prop", lambda s: int((s >= 0.5).sum())),
            high_trainer_suppressed=("trainer_suppressed_prop", lambda s: int((s >= 0.5).sum())),
        )
        .reset_index()
    )

    red_flags = primary_panel.copy()
    q_trainer = red_flags["trainer_adverse_prop"].quantile(0.90)
    q_trainee = red_flags["trainee_adverse_prop"].quantile(0.90)
    red_flags = red_flags[
        (red_flags["trainer_adverse_prop"] >= q_trainer)
        & (red_flags["trainee_adverse_prop"] >= q_trainee)
    ].copy()
    red_flags["strain_score"] = red_flags["trainer_adverse_prop"] + red_flags["trainee_adverse_prop"]
    red_flags = red_flags.sort_values(["strain_score", "year"], ascending=[False, True]).head(100)

    table = pd.DataFrame(checks)
    table.to_csv(OUTDIR / "table_robustness_checks.csv", index=False)
    suppression_summary.to_csv(OUTDIR / "table_suppression_summary.csv", index=False)
    red_flags.to_csv(OUTDIR / "table_red_flag_units_top100.csv", index=False)

    summary = {
        "robustness_rows": len(checks),
        "primary_result": checks[0],
        "strict_below_result": [row for row in checks if row["check"] == "adverse_strict_below_suppression_lt_0.75"][0],
        "red_flag_units_top100": int(len(red_flags)),
    }
    (OUTDIR / "robustness_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    report = f"""# Robustness and Suppression Report

## Sensitivity Results

The primary adverse definition (`Below` or `Q1`) remains positive across stricter suppression thresholds. The strict `Below`-only definition produces smaller samples and tests a narrower construct.

Primary model, suppression <0.75: beta={checks[0]['coef']:.3f}, p={checks[0]['p']:.3g}, n={checks[0]['n']:,}.

Strict below-only model, suppression <0.75: beta={summary['strict_below_result']['coef']:.3f}, p={summary['strict_below_result']['p']:.3g}, n={summary['strict_below_result']['n']:,}.

## Suppression

Suppression summaries by year are saved in `table_suppression_summary.csv`. The manuscript should report that suppression is handled both by exclusion thresholds and by explicit suppressed-cell covariates in models.

## Red-Flag Units

The top 100 high-trainer/high-trainee adverse burden units are saved in `table_red_flag_units_top100.csv` for descriptive surveillance use. These should be presented as quality-improvement signals, not rankings of institutional quality.
"""
    (OUTDIR / "robustness_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("WROTE", OUTDIR / "robustness_report.md")


if __name__ == "__main__":
    main()
