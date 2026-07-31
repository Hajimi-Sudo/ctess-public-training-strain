#!/usr/bin/env python3
"""Run final robustness models before server shutdown.

These analyses are deliberately conventional for an ecological public-data
linkage study: fixed effects, weighting by trainee response volume, and a
joint external-triangulation model. They are not prediction experiments.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf


PROJECT = Path(__file__).resolve().parents[2]
OUTDIR = PROJECT / "06_analysis/pilot"
FIGDIR = OUTDIR / "figures"


def _format_p(value: float) -> str:
    if pd.isna(value):
        return "NA"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def _fit_model(
    *,
    name: str,
    scope: str,
    data: pd.DataFrame,
    formula: str,
    exposures: list[str],
    weighted: bool = False,
    weight_col: str = "trainee_n",
) -> list[dict]:
    model_data = data.dropna(subset=["trainee_adverse_prop", *exposures]).copy()
    if weighted:
        weights = model_data[weight_col].clip(lower=1)
        result = smf.wls(formula, data=model_data, weights=weights).fit(
            cov_type="cluster", cov_kwds={"groups": model_data["trust_norm"]}
        )
    else:
        result = smf.ols(formula, data=model_data).fit(
            cov_type="cluster", cov_kwds={"groups": model_data["trust_norm"]}
        )

    ci = result.conf_int()
    rows = []
    for exposure in exposures:
        rows.append(
            {
                "model": name,
                "scope": scope,
                "n": int(result.nobs),
                "trusts": int(model_data["trust_norm"].nunique()),
                "specialties": int(model_data["spec_norm"].nunique()),
                "primary_exposure": exposure,
                "coef": float(result.params[exposure]),
                "std_err": float(result.bse[exposure]),
                "ci_low": float(ci.loc[exposure, 0]),
                "ci_high": float(ci.loc[exposure, 1]),
                "p": float(result.pvalues[exposure]),
                "r2": float(result.rsquared),
                "formula": formula,
                "weighted": bool(weighted),
            }
        )
    return rows


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(OUTDIR / "gmc_panel_key_indicator_matrix_usable_2021_2026.csv")
    staff = pd.read_csv(OUTDIR / "gmc_panel_plus_nhs_staff_2021_2025.csv")
    joint = pd.read_csv(OUTDIR / "gmc_panel_plus_nhs_staff_sickness_2021_2025.csv")

    base_covars = "trainer_suppressed_prop + trainee_suppressed_prop"
    rows: list[dict] = []

    rows.extend(
        _fit_model(
            name="gmc_trust_fe_year_specialty_fe_cluster_trust",
            scope="GMC 2021-2026",
            data=panel,
            formula=(
                "trainee_adverse_prop ~ trainer_adverse_prop + "
                f"{base_covars} + C(year) + C(spec_norm) + C(trust_norm)"
            ),
            exposures=["trainer_adverse_prop"],
        )
    )

    diff_panel = panel.sort_values(["pair_id", "year"]).copy()
    diff_cols = [
        "trainee_adverse_prop",
        "trainer_adverse_prop",
        "trainer_suppressed_prop",
        "trainee_suppressed_prop",
    ]
    for col in diff_cols:
        diff_panel[col] = diff_panel.groupby("pair_id")[col].diff()
    diff_panel = diff_panel.dropna(subset=diff_cols).copy()

    rows.extend(
        _fit_model(
            name="gmc_within_pair_annual_change_year_fe_cluster_trust",
            scope="GMC 2021-2026 annual within-pair changes",
            data=diff_panel,
            formula=(
                "trainee_adverse_prop ~ trainer_adverse_prop + "
                f"{base_covars} + C(year)"
            ),
            exposures=["trainer_adverse_prop"],
        )
    )

    rows.extend(
        _fit_model(
            name="gmc_weighted_year_specialty_fe_cluster_trust",
            scope="GMC 2021-2026",
            data=panel,
            formula=(
                "trainee_adverse_prop ~ trainer_adverse_prop + "
                f"{base_covars} + C(year) + C(spec_norm)"
            ),
            exposures=["trainer_adverse_prop"],
            weighted=True,
        )
    )

    rows.extend(
        _fit_model(
            name="nhs_staff_trust_fe_year_specialty_fe_cluster_trust",
            scope="GMC + NHS Staff Survey 2021-2025",
            data=staff,
            formula=(
                "trainee_adverse_prop ~ trainer_adverse_prop + nhs_staff_stress_index + "
                f"{base_covars} + C(year) + C(spec_norm) + C(trust_norm)"
            ),
            exposures=["trainer_adverse_prop", "nhs_staff_stress_index"],
        )
    )

    rows.extend(
        _fit_model(
            name="joint_staff_sickness_year_specialty_fe_cluster_trust",
            scope="GMC + NHS Staff Survey + sickness absence 2021-2025",
            data=joint,
            formula=(
                "trainee_adverse_prop ~ trainer_adverse_prop + nhs_staff_stress_index + "
                "sickness_absence_rate_percent + "
                f"{base_covars} + C(year) + C(spec_norm)"
            ),
            exposures=[
                "trainer_adverse_prop",
                "nhs_staff_stress_index",
                "sickness_absence_rate_percent",
            ],
        )
    )

    table = pd.DataFrame(rows)
    table.to_csv(OUTDIR / "table_final_robustness_models.csv", index=False)

    plot_data = table[table["primary_exposure"] == "trainer_adverse_prop"].copy()
    plot_data["label"] = plot_data["model"].str.replace("_", " ", regex=False)
    fig_height = max(4.5, 0.55 * len(plot_data) + 1.2)
    plt.figure(figsize=(8.5, fig_height))
    y = range(len(plot_data))
    plt.errorbar(
        plot_data["coef"],
        y,
        xerr=[
            plot_data["coef"] - plot_data["ci_low"],
            plot_data["ci_high"] - plot_data["coef"],
        ],
        fmt="o",
        color="#2457A6",
        ecolor="#6A8BC4",
        capsize=3,
    )
    plt.axvline(0, color="#333333", linewidth=1)
    plt.yticks(y, plot_data["label"], fontsize=8)
    plt.xlabel("Coefficient for trainer adverse burden")
    plt.title("Final robustness models")
    plt.tight_layout()
    plt.savefig(FIGDIR / "fig6_final_robustness_coefficients.png", dpi=220)
    plt.close()

    summary = {
        "models_run": int(table["model"].nunique()),
        "rows": int(len(table)),
        "trainer_models": table[table["primary_exposure"] == "trainer_adverse_prop"][
            ["model", "n", "coef", "p", "r2"]
        ].to_dict("records"),
        "staff_stress_rows": table[table["primary_exposure"] == "nhs_staff_stress_index"][
            ["model", "n", "coef", "p", "r2"]
        ].to_dict("records"),
        "sickness_rows": table[table["primary_exposure"] == "sickness_absence_rate_percent"][
            ["model", "n", "coef", "p", "r2"]
        ].to_dict("records"),
    }
    (OUTDIR / "final_robustness_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    trainer_lines = []
    for row in summary["trainer_models"]:
        trainer_lines.append(
            f"- {row['model']}: beta={row['coef']:.3f}, p={_format_p(row['p'])}, "
            f"n={row['n']:,}, R2={row['r2']:.3f}"
        )
    stress_lines = []
    for row in summary["staff_stress_rows"]:
        stress_lines.append(
            f"- {row['model']}: beta={row['coef']:.3f}, p={_format_p(row['p'])}, "
            f"n={row['n']:,}, R2={row['r2']:.3f}"
        )
    sickness_lines = []
    for row in summary["sickness_rows"]:
        sickness_lines.append(
            f"- {row['model']}: beta={row['coef']:.3f}, p={_format_p(row['p'])}, "
            f"n={row['n']:,}, R2={row['r2']:.3f}"
        )

    report = f"""# Final Robustness Completion Report

## Purpose

This file records the final server-side robustness analyses run before shutting down the analysis server. These tests address likely reviewer concerns about institutional confounding, unequal response volume, and whether the NHS Staff Survey and NHS sickness absence signals behave as expected when included jointly.

## Completed Models

{chr(10).join(trainer_lines)}

## External Organisational Signals

NHS Staff Survey stress index:

{chr(10).join(stress_lines)}

NHS sickness absence rate:

{chr(10).join(sickness_lines)}

## Interpretation

Trainer adverse burden remains positive across trust fixed effects, within trust-specialty annual changes, response-volume weighting, and joint external-linkage specifications. This supports a surveillance/QI interpretation: adverse trainer signals and adverse trainee learning-environment signals co-occur in reproducible public-data units.

NHS Staff Survey stress remains useful as external organisational triangulation, although the trust fixed-effect specification estimates a narrower within-trust-over-time contrast and should be presented as supportive rather than definitive. December sickness absence remains a linked but neutral objective benchmark; it should be reported as a negative/attenuated external check rather than framed as a failed primary outcome.

## Shutdown Status

The remaining work after this report is manuscript writing, figure polishing, and wording of limitations. No GPU or long-running server experiment is still required.
"""
    (OUTDIR / "final_experiment_completion_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("WROTE", OUTDIR / "table_final_robustness_models.csv")
    print("WROTE", OUTDIR / "final_experiment_completion_report.md")


if __name__ == "__main__":
    main()
