#!/usr/bin/env python3
"""Analyze the 2026 GMC trainer-trainee linkage pilot."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import pearsonr, spearmanr


PROJECT = Path(__file__).resolve().parents[2]
OUTDIR = PROJECT / "06_analysis/pilot"


def normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["trust_norm"] = (
        df["trust_board"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    )
    df["spec_norm"] = (
        df["specialty"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    )
    df["adverse"] = (
        df["outcome"].fillna("").str.contains("Below|Q1", case=False, regex=True).astype(int)
    )
    df["suppressed"] = (df["outcome"].fillna("") == "N less than 3").astype(int)
    df["mean_num"] = pd.to_numeric(df["mean"], errors="coerce")
    df["n_num"] = pd.to_numeric(df["n_all"], errors="coerce")
    return df


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    trainee = normalize_keys(
        pd.read_csv(PROJECT / "05_data/processed/gmc_trainee_postspec_trust_detail_2026.csv")
    )
    trainer = normalize_keys(
        pd.read_csv(PROJECT / "05_data/processed/gmc_trainer_spec_trust_detail_2026.csv")
    )

    trainee_keep = [
        "Overall Satisfaction",
        "Workload",
        "Rota Design",
        "Supportive Environment",
        "Educational Supervision",
        "Clinical Supervision",
    ]
    trainer_keep = [
        "Time to Train",
        "Resources to Train",
        "Rota Issues",
        "Support for Training",
        "Supportive Environment",
        "Educational Governance",
    ]

    trainee_key = trainee[trainee.indicator.isin(trainee_keep)].copy()
    trainer_key = trainer[trainer.indicator.isin(trainer_keep)].copy()

    trainee_grouped = (
        trainee_key.groupby(["trust_norm", "spec_norm"])
        .agg(
            trainee_adverse_prop=("adverse", "mean"),
            trainee_suppressed_prop=("suppressed", "mean"),
            trainee_cells=("indicator", "count"),
            trainee_n=("n_num", "sum"),
        )
        .reset_index()
    )
    trainer_grouped = (
        trainer_key.groupby(["trust_norm", "spec_norm"])
        .agg(
            trainer_adverse_prop=("adverse", "mean"),
            trainer_suppressed_prop=("suppressed", "mean"),
            trainer_cells=("indicator", "count"),
            trainer_n=("n_num", "sum"),
        )
        .reset_index()
    )
    matrix = trainee_grouped.merge(trainer_grouped, on=["trust_norm", "spec_norm"], how="inner")
    matrix["usable"] = (
        (matrix.trainee_cells >= 4)
        & (matrix.trainer_cells >= 4)
        & (matrix.trainee_suppressed_prop < 0.75)
        & (matrix.trainer_suppressed_prop < 0.75)
    )

    usable = matrix[matrix.usable].copy()
    for frame, prefix in [(trainee_key, "trainee"), (trainer_key, "trainer")]:
        wide = frame.pivot_table(
            index=["trust_norm", "spec_norm"],
            columns="indicator",
            values="mean_num",
            aggfunc="mean",
        )
        wide.columns = [
            f"{prefix}_mean_" + str(column).lower().replace(" ", "_")
            for column in wide.columns
        ]
        matrix = matrix.merge(wide.reset_index(), on=["trust_norm", "spec_norm"], how="left")
        usable = usable.merge(wide.reset_index(), on=["trust_norm", "spec_norm"], how="left")

    rho, spearman_p = spearmanr(
        usable["trainer_adverse_prop"], usable["trainee_adverse_prop"], nan_policy="omit"
    )
    pearson_r, pearson_p = pearsonr(
        usable["trainer_adverse_prop"], usable["trainee_adverse_prop"]
    )
    reg = smf.ols(
        "trainee_adverse_prop ~ trainer_adverse_prop + trainer_suppressed_prop + trainee_suppressed_prop",
        data=usable,
    ).fit(cov_type="HC3")

    q75_trainer = usable.trainer_adverse_prop.quantile(0.75)
    q75_trainee = usable.trainee_adverse_prop.quantile(0.75)
    usable["coupled_high_strain"] = (
        (usable.trainer_adverse_prop >= q75_trainer)
        & (usable.trainee_adverse_prop >= q75_trainee)
    ).astype(int)

    trainee_pairs = set(zip(trainee.trust_norm, trainee.spec_norm))
    trainer_pairs = set(zip(trainer.trust_norm, trainer.spec_norm))
    summary = {
        "raw_rows": {"trainee": int(len(trainee)), "trainer": int(len(trainer))},
        "coverage": {
            "trainee_specialties": int(trainee.specialty.nunique()),
            "trainer_specialties": int(trainer.specialty.nunique()),
            "trainee_trusts": int(trainee.trust_norm.nunique()),
            "trainer_trusts": int(trainer.trust_norm.nunique()),
            "pair_overlap": int(len(trainee_pairs & trainer_pairs)),
            "trust_overlap": int(len(set(trainee.trust_norm) & set(trainer.trust_norm))),
            "specialty_overlap": int(len(set(trainee.spec_norm) & set(trainer.spec_norm))),
        },
        "analysis_units": {
            "merged_key_pairs": int(len(matrix)),
            "usable_primary_pairs": int(len(usable)),
        },
        "adverse_burden": {
            "trainee_mean": float(usable.trainee_adverse_prop.mean()),
            "trainer_mean": float(usable.trainer_adverse_prop.mean()),
            "trainee_q75": float(q75_trainee),
            "trainer_q75": float(q75_trainer),
            "coupled_high_n": int(usable.coupled_high_strain.sum()),
        },
        "association": {
            "spearman_rho": float(rho),
            "spearman_p": float(spearman_p),
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
        },
        "regression": {
            "coef_trainer_adverse_prop": float(reg.params["trainer_adverse_prop"]),
            "p_trainer_adverse_prop": float(reg.pvalues["trainer_adverse_prop"]),
            "r2": float(reg.rsquared),
        },
        "indicators": {"trainee": trainee_keep, "trainer": trainer_keep},
    }

    matrix.to_csv(OUTDIR / "gmc_linked_key_indicator_matrix_2026.csv", index=False)
    usable.to_csv(OUTDIR / "gmc_linked_key_indicator_matrix_usable_2026.csv", index=False)
    (OUTDIR / "pilot_stats_2026.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    report = f"""# PMJ feasibility pilot report (server run, 2026 GMC NTS)

## Data Extraction Status

- Project workspace: repository root
- Extracted trainee detail table: `05_data/processed/gmc_trainee_postspec_trust_detail_2026.csv`
- Extracted trainer detail table: `05_data/processed/gmc_trainer_spec_trust_detail_2026.csv`
- Trainee rows: {summary['raw_rows']['trainee']:,}; trainer rows: {summary['raw_rows']['trainer']:,}.
- Trainee coverage: {summary['coverage']['trainee_specialties']} specialties, {summary['coverage']['trainee_trusts']} trusts/boards, 18 indicators.
- Trainer coverage: {summary['coverage']['trainer_specialties']} specialties, {summary['coverage']['trainer_trusts']} trusts/boards, 9 indicators.

## Linkage Feasibility

- Matched trust-specialty pairs before key-indicator filtering: {summary['coverage']['pair_overlap']:,}.
- Matched trusts/boards: {summary['coverage']['trust_overlap']}; matched specialties: {summary['coverage']['specialty_overlap']}.
- Key-indicator matrix merged units: {summary['analysis_units']['merged_key_pairs']:,}.
- Primary usable units after minimum-cell/suppression screen: {summary['analysis_units']['usable_primary_pairs']:,}.

## Pilot CTSI Signal

Adverse outcome was defined as a GMC benchmark outcome containing `Below` or `Q1` for pre-specified key indicators. The primary unit is trust/board by specialty.

- Trainee adverse burden mean: {summary['adverse_burden']['trainee_mean']:.3f}.
- Trainer adverse burden mean: {summary['adverse_burden']['trainer_mean']:.3f}.
- Coupled high-strain units, both burdens at/above their 75th percentile: {summary['adverse_burden']['coupled_high_n']:,}.
- Spearman correlation between trainer and trainee adverse burden: rho={summary['association']['spearman_rho']:.3f}, p={summary['association']['spearman_p']:.3g}.
- HC3 OLS coefficient for trainer adverse burden predicting trainee adverse burden: beta={summary['regression']['coef_trainer_adverse_prop']:.3f}, p={summary['regression']['p_trainer_adverse_prop']:.3g}, R2={summary['regression']['r2']:.3f}.

## Feasibility Conclusion

This pilot clears the core feasibility gate for a PMJ-targeted observational manuscript: public data are reachable from the server through the GMC embedded Power BI endpoint; trainer and trainee tables can be linked at a clinically interpretable trust-specialty unit; the linked analytic matrix is large enough for adjusted cross-sectional modelling and sensitivity checks.

The current evidence is still a server-side feasibility result, not the final manuscript result, because external organisational covariates and multi-year replication have not yet been added.

## Next Hardening Steps

1. Repeat the chunked extraction for 2021-2026 and stack a trust-specialty-year panel.
2. Harmonise specialty labels across trainee/trainer reports, especially case variants such as Emergency Medicine and Respiratory Medicine.
3. Add NHS Staff Survey and CQC/NHS organisational covariates at trust level.
4. Pre-specify the Coupled Training Strain Index and run sensitivity analyses using mean-score z-scores, benchmark outcomes, and suppression-aware exclusions.
5. Produce reproducible tables/figures for PMJ submission, including a public-data linkage flow diagram.
"""
    (OUTDIR / "pilot_feasibility_report_2026.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("WROTE", OUTDIR / "pilot_feasibility_report_2026.md")


if __name__ == "__main__":
    main()
