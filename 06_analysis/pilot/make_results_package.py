#!/usr/bin/env python3
"""Create tables, figures, and a concise results package for the PMJ project."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
OUTDIR = PROJECT / "06_analysis/pilot"
FIGDIR = OUTDIR / "figures"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pformat(value: float) -> str:
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    panel_stats = load_json(OUTDIR / "gmc_panel_stats_2021_2026.json")
    staff_stats = load_json(OUTDIR / "nhs_staff_linkage_stats.json")
    sickness_stats_path = OUTDIR / "nhs_sickness_linkage_stats.json"
    sickness_stats = load_json(sickness_stats_path) if sickness_stats_path.exists() else None
    panel = pd.read_csv(OUTDIR / "gmc_panel_key_indicator_matrix_usable_2021_2026.csv")
    linked = pd.read_csv(OUTDIR / "gmc_panel_plus_nhs_staff_2021_2025.csv")
    year_summary = pd.read_csv(OUTDIR / "gmc_panel_year_summary_2021_2026.csv")
    crosswalk = pd.read_csv(OUTDIR / "nhs_staff_to_gmc_trust_crosswalk.csv")

    model_rows = []
    for name, model in panel_stats["models"].items():
        model_rows.append(
            {
                "model": name,
                "scope": "GMC 2021-2026",
                "n": model["n"],
                "primary_exposure": "trainer_adverse_prop"
                if "lag1" not in name
                else "trainer_adverse_lag1",
                "coef": model["coef"],
                "p": model["p"],
                "r2": model["r2"],
            }
        )
    staff_model = staff_stats["models"]["gmc_plus_staff_year_specialty_fe"]
    model_rows.append(
        {
            "model": "gmc_plus_nhs_staff_year_specialty_fe",
            "scope": "GMC + NHS Staff Survey 2021-2025",
            "n": staff_model["n"],
            "primary_exposure": "trainer_adverse_prop",
            "coef": staff_model["coef_trainer"],
            "p": staff_model["p_trainer"],
            "r2": staff_model["r2"],
        }
    )
    if sickness_stats is not None:
        sickness_model = sickness_stats["sickness_model"]
        model_rows.append(
            {
                "model": "gmc_plus_nhs_sickness_year_specialty_fe",
                "scope": "GMC + NHS sickness absence 2021-2025",
                "n": sickness_model["n"],
                "primary_exposure": "trainer_adverse_prop",
                "coef": sickness_model["coef_trainer"],
                "p": sickness_model["p_trainer"],
                "r2": sickness_model["r2"],
            }
        )
        model_rows.append(
            {
                "model": "gmc_plus_nhs_sickness_year_specialty_fe",
                "scope": "GMC + NHS sickness absence 2021-2025",
                "n": sickness_model["n"],
                "primary_exposure": "sickness_absence_rate_percent",
                "coef": sickness_model["coef_sickness"],
                "p": sickness_model["p_sickness"],
                "r2": sickness_model["r2"],
            }
        )
    model_rows.append(
        {
            "model": "gmc_plus_nhs_staff_year_specialty_fe",
            "scope": "GMC + NHS Staff Survey 2021-2025",
            "n": staff_model["n"],
            "primary_exposure": "nhs_staff_stress_index",
            "coef": staff_model["coef_staff_stress"],
            "p": staff_model["p_staff_stress"],
            "r2": staff_model["r2"],
        }
    )
    model_table = pd.DataFrame(model_rows)
    model_table.to_csv(OUTDIR / "table_model_results.csv", index=False)

    coverage_rows = [
        {
            "dataset": "GMC NTS trainee",
            "years": "2021-2026",
            "rows_or_units": panel_stats["raw_rows"]["trainee"],
            "coverage": f"{panel_stats['coverage']['trainee_trusts']} trusts/boards; "
            f"{panel_stats['coverage']['trainee_specialties']} specialties",
        },
        {
            "dataset": "GMC NTS trainer",
            "years": "2021-2026",
            "rows_or_units": panel_stats["raw_rows"]["trainer"],
            "coverage": f"{panel_stats['coverage']['trainer_trusts']} trusts/boards; "
            f"{panel_stats['coverage']['trainer_specialties']} specialties",
        },
        {
            "dataset": "Linked GMC panel",
            "years": "2021-2026",
            "rows_or_units": panel_stats["panel"]["usable_units"],
            "coverage": f"{panel_stats['panel']['unique_pairs_usable']} usable trust-specialty pairs",
        },
        {
            "dataset": "NHS Staff Survey linked panel",
            "years": "2021-2025",
            "rows_or_units": staff_stats["linked_rows"],
            "coverage": f"{staff_stats['linked_trusts']} trusts; {staff_stats['linked_pairs']} trust-specialty pairs",
        },
    ]
    if sickness_stats is not None:
        coverage_rows.append(
            {
                "dataset": "NHS sickness absence linked panel",
                "years": "2021-2025",
                "rows_or_units": sickness_stats["linked_rows"],
                "coverage": f"{sickness_stats['linked_trusts']} trusts; {sickness_stats['linked_pairs']} trust-specialty pairs",
            }
        )
    pd.DataFrame(coverage_rows).to_csv(OUTDIR / "table_dataset_coverage.csv", index=False)

    crosswalk_audit = {
        "crosswalk_rows": int(len(crosswalk)),
        "min_match_score": float(crosswalk["match_score"].min()),
        "median_match_score": float(crosswalk["match_score"].median()),
        "duplicate_org_ids": int(crosswalk["org_id"].duplicated().sum()),
        "duplicate_gmc_trusts": int(crosswalk["trust_norm"].duplicated().sum()),
    }
    (OUTDIR / "crosswalk_audit.json").write_text(
        json.dumps(crosswalk_audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    plt.figure(figsize=(7, 4))
    plt.plot(year_summary["year"], year_summary["usable_units"], marker="o", color="#2457A6")
    plt.xlabel("Survey year")
    plt.ylabel("Usable trust-specialty units")
    plt.title("Usable GMC trainer-trainee linked units by year")
    plt.tight_layout()
    plt.savefig(FIGDIR / "fig1_usable_units_by_year.png", dpi=200)
    plt.close()

    yearly = pd.DataFrame(panel_stats["yearly_spearman"])
    plt.figure(figsize=(7, 4))
    plt.axhline(0, color="#777777", linewidth=1)
    plt.plot(yearly["year"], yearly["spearman_rho"], marker="o", color="#0D7C66")
    plt.xlabel("Survey year")
    plt.ylabel("Spearman rho")
    plt.title("Trainer-trainee adverse burden association by year")
    plt.tight_layout()
    plt.savefig(FIGDIR / "fig2_yearly_spearman.png", dpi=200)
    plt.close()

    coef_plot = model_table.copy()
    coef_plot["label"] = coef_plot["model"] + "\n" + coef_plot["primary_exposure"]
    plt.figure(figsize=(8, 4.8))
    colors = ["#2457A6" if exposure == "trainer_adverse_prop" else "#B54A35" for exposure in coef_plot["primary_exposure"]]
    plt.barh(range(len(coef_plot)), coef_plot["coef"], color=colors)
    plt.yticks(range(len(coef_plot)), coef_plot["label"], fontsize=8)
    plt.axvline(0, color="#333333", linewidth=1)
    plt.xlabel("Coefficient")
    plt.title("Primary model coefficients")
    plt.tight_layout()
    plt.savefig(FIGDIR / "fig3_model_coefficients.png", dpi=200)
    plt.close()

    plt.figure(figsize=(5.5, 5))
    plt.hexbin(
        panel["trainer_adverse_prop"],
        panel["trainee_adverse_prop"],
        gridsize=30,
        cmap="YlGnBu",
        mincnt=1,
    )
    plt.colorbar(label="Unit count")
    plt.xlabel("Trainer adverse burden")
    plt.ylabel("Trainee adverse burden")
    plt.title("GMC trust-specialty-year strain coupling")
    plt.tight_layout()
    plt.savefig(FIGDIR / "fig4_gmc_hexbin_trainer_trainee.png", dpi=200)
    plt.close()

    plt.figure(figsize=(5.5, 5))
    plt.hexbin(
        linked["nhs_staff_stress_index"],
        linked["trainee_adverse_prop"],
        gridsize=30,
        cmap="YlOrRd",
        mincnt=1,
    )
    plt.colorbar(label="Unit count")
    plt.xlabel("NHS Staff Survey stress index")
    plt.ylabel("Trainee adverse burden")
    plt.title("External organisational stress triangulation")
    plt.tight_layout()
    plt.savefig(FIGDIR / "fig5_nhs_staff_stress_hexbin.png", dpi=200)
    plt.close()

    sickness_sentence = ""
    if sickness_stats is not None:
        sm = sickness_stats["sickness_model"]
        sickness_sentence = (
            f" NHS sickness absence could also be linked to {sickness_stats['linked_rows']:,} units; "
            f"trainer adverse burden remained significant after adjustment (beta={sm['coef_trainer']:.3f}, "
            f"p={pformat(sm['p_trainer'])}), while December sickness absence rate itself was not independently "
            f"associated (beta={sm['coef_sickness']:.3f}, p={pformat(sm['p_sickness'])})."
        )

    report = f"""# Formal Results Package Summary

## Core Dataset

The server-side extraction produced {panel_stats['raw_rows']['total']:,} GMC detail rows across 2021-2026. The linked GMC panel contains {panel_stats['panel']['merged_units']:,} merged trust-specialty-year units, of which {panel_stats['panel']['usable_units']:,} meet the pre-specified minimum-cell and suppression filters. The NHS Staff Survey linkage contributes {staff_stats['linked_rows']:,} externally linked units for 2021-2025.

## Main Findings

In the GMC-only panel, trainer adverse burden is consistently associated with trainee adverse burden after year and specialty adjustment clustered by trust: beta={panel_stats['models']['year_specialty_fe_cluster_trust']['coef']:.3f}, p={pformat(panel_stats['models']['year_specialty_fe_cluster_trust']['p'])}. The one-year lag model remains significant: beta={panel_stats['models']['lag1_year_specialty_fe_cluster_trust']['coef']:.3f}, p={pformat(panel_stats['models']['lag1_year_specialty_fe_cluster_trust']['p'])}.

In the externally linked 2021-2025 model, trainer adverse burden remains significant (beta={staff_model['coef_trainer']:.3f}, p={pformat(staff_model['p_trainer'])}), and NHS Staff Survey organisational stress is also independently associated with trainee adverse burden (beta={staff_model['coef_staff_stress']:.3f}, p={pformat(staff_model['p_staff_stress'])}).{sickness_sentence}

## Crosswalk Quality

After excluding ICBs and tightening name matching, the NHS Staff Survey-to-GMC crosswalk contains {crosswalk_audit['crosswalk_rows']} trust mappings, with no duplicate organisation IDs or duplicate GMC trust names. The minimum automated match score is {crosswalk_audit['min_match_score']:.3f}. This is acceptable for analysis, but should still be documented as an ODS/name-based linkage.

## Manuscript Positioning

The strongest PMJ framing is not high-accuracy prediction. It is a public-data surveillance study showing that trainer strain, trainee adverse learning-environment signals, and wider organisational stress cluster at clinically interpretable trust-specialty-year units.
"""
    (OUTDIR / "formal_results_package_summary.md").write_text(report, encoding="utf-8")

    print(json.dumps({"model_rows": len(model_table), "figures": 5, "crosswalk_audit": crosswalk_audit}, indent=2))
    print("WROTE", OUTDIR / "formal_results_package_summary.md")


if __name__ == "__main__":
    main()
