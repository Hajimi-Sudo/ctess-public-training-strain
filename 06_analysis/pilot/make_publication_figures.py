#!/usr/bin/env python3
"""Create publication-style figures for the PMJ manuscript."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "06_analysis/pilot"
FIG = OUT / "figures_publication"

COLORS = {
    "ink": "#1F2933",
    "muted": "#64748B",
    "blue": "#2563A9",
    "teal": "#0F8B8D",
    "green": "#2E7D32",
    "amber": "#C77700",
    "red": "#B03A2E",
    "purple": "#6B4FA3",
    "light_blue": "#D9E8F6",
    "light_teal": "#D7F0EE",
    "light_amber": "#F6E7C8",
    "grid": "#D6DEE8",
}


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": COLORS["muted"],
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "text.color": COLORS["ink"],
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.6,
            "grid.alpha": 0.8,
            "figure.dpi": 130,
            "savefig.dpi": 300,
        }
    )


def pformat(value: float) -> str:
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def p_label(value: float) -> str:
    return f"p{pformat(value)}" if value < 0.001 else f"p={pformat(value)}"


def add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.08,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
    )


def draw_box(ax, xy, text, face, edge, width=0.36, height=0.13, fontsize=9.0) -> None:
    from matplotlib.patches import FancyBboxPatch

    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.2,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(box)
    wrapped_text = "\n".join(fill(line, 24) for line in text.splitlines())
    ax.text(
        x + width / 2,
        y + height / 2,
        wrapped_text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        linespacing=1.15,
    )


def draw_arrow(ax, start, end, color=COLORS["muted"]) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4, shrinkA=5, shrinkB=5),
    )


def fig0_framework() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 3.9), constrained_layout=True)
    ax.set_axis_off()

    ax.text(
        0.5,
        0.95,
        "Coupled Training Environment Strain Surveillance framework",
        ha="center",
        va="top",
        fontsize=15,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.88,
        "A public-data framework for detecting coupled trainer-trainee strain signals at trust-specialty-year level",
        ha="center",
        va="top",
        fontsize=9.5,
        color=COLORS["muted"],
    )

    layers = [
        ("1. Data linkage layer", "GMC trainee + trainer records\nNHS Staff Survey\nNHS sickness absence", COLORS["light_blue"], COLORS["blue"]),
        ("2. Signal construction layer", "Trainer adverse burden\nTrainee adverse burden\nCoupled strain signal", COLORS["light_teal"], COLORS["teal"]),
        ("3. Evidence triangulation layer", "Fixed effects\nLagged models\nRobustness checks\nExternal linkage", COLORS["light_amber"], COLORS["amber"]),
        ("4. Quality-improvement layer", "High-high signal group\nLocal review of capacity,\nrota and supervision", "#E7E2F3", COLORS["purple"]),
    ]
    x_positions = [0.03, 0.275, 0.52, 0.765]
    for i, ((title, body, face, edge), x) in enumerate(zip(layers, x_positions)):
        draw_box(ax, (x, 0.37), f"{title}\n{body}", face, edge, width=0.205, height=0.30, fontsize=8.4)
        if i < len(layers) - 1:
            draw_arrow(ax, (x + 0.205, 0.52), (x_positions[i + 1], 0.52), edge)

    ax.text(
        0.5,
        0.18,
        "Framework output: interpretable aggregate signals for surveillance, triangulation, and local review.",
        ha="center",
        va="center",
        fontsize=9.2,
        color=COLORS["ink"],
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8FAFC", edgecolor=COLORS["grid"]),
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.savefig(FIG / "fig0_ctess_framework.png", bbox_inches="tight")
    plt.close(fig)


def fig1_data_assembly() -> None:
    panel_stats = json.loads((OUT / "gmc_panel_stats_2021_2026.json").read_text(encoding="utf-8"))
    staff_stats = json.loads((OUT / "nhs_staff_linkage_stats.json").read_text(encoding="utf-8"))
    sickness_stats = json.loads((OUT / "nhs_sickness_linkage_stats.json").read_text(encoding="utf-8"))
    yearly = pd.read_csv(OUT / "gmc_panel_year_summary_2021_2026.csv")

    fig = plt.figure(figsize=(11.2, 6.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.25], height_ratios=[1, 1])
    ax_flow = fig.add_subplot(gs[:, 0])
    ax_cov = fig.add_subplot(gs[0, 1])
    ax_year = fig.add_subplot(gs[1, 1])

    add_panel_label(ax_flow, "A")
    gmc_2021_2025 = staff_stats["gmc_usable_rows_2021_2025"]
    retention_labels = [
        "Merged GMC units",
        "Primary analytic units",
        "Lagged analytic units",
        "Staff Survey linked units",
        "Sickness linked units",
        "High-high signal units",
    ]
    retention_values = [
        panel_stats["panel"]["merged_units"],
        panel_stats["panel"]["usable_units"],
        panel_stats["panel"]["lagged_units"],
        staff_stats["linked_rows"],
        sickness_stats["linked_rows"],
        panel_stats["panel"]["coupled_high_strain_units"],
    ]
    retention_notes = [
        "100% merged panel",
        f"{panel_stats['panel']['usable_units'] / panel_stats['panel']['merged_units']:.0%} of merged",
        f"{panel_stats['panel']['lagged_units'] / panel_stats['panel']['usable_units']:.0%} of primary",
        f"{staff_stats['linked_rows'] / gmc_2021_2025:.0%} of 2021-2025 GMC",
        f"{sickness_stats['linked_rows'] / gmc_2021_2025:.0%} of 2021-2025 GMC",
        f"{panel_stats['panel']['coupled_high_strain_units'] / panel_stats['panel']['usable_units']:.0%} of primary",
    ]
    retention_colors = [COLORS["ink"], COLORS["purple"], COLORS["blue"], COLORS["amber"], COLORS["red"], COLORS["teal"]]
    y_ret = np.arange(len(retention_labels))
    ax_flow.barh(y_ret, retention_values, color=retention_colors, alpha=0.88)
    ax_flow.set_yticks(y_ret, retention_labels)
    ax_flow.invert_yaxis()
    ax_flow.set_xlabel("Trust-specialty-year units")
    ax_flow.set_title("Analytic retention and signal yield", loc="left", fontsize=12, fontweight="bold")
    ax_flow.grid(axis="y", visible=False)
    ax_flow.set_xlim(0, max(retention_values) * 1.33)
    for yi, value, note in zip(y_ret, retention_values, retention_notes):
        ax_flow.text(value + max(retention_values) * 0.025, yi, f"{value:,}\n{note}", va="center", fontsize=8.1)

    add_panel_label(ax_cov, "B")
    labels = [
        "GMC trainee detail rows",
        "GMC trainer detail rows",
        "Usable linked GMC units",
        "NHS Staff Survey linked units",
        "NHS sickness linked units",
    ]
    values = [
        panel_stats["raw_rows"]["trainee"],
        panel_stats["raw_rows"]["trainer"],
        panel_stats["panel"]["usable_units"],
        staff_stats["linked_rows"],
        sickness_stats["linked_rows"],
    ]
    colors = [COLORS["blue"], COLORS["teal"], COLORS["purple"], COLORS["amber"], COLORS["red"]]
    y = np.arange(len(labels))
    ax_cov.barh(y, values, color=colors, alpha=0.88)
    ax_cov.set_yticks(y, labels)
    ax_cov.invert_yaxis()
    ax_cov.set_xscale("log")
    ax_cov.set_xlabel("Records / analytic units (log scale)")
    ax_cov.set_title("Scale and linkage yield", loc="left", fontsize=12, fontweight="bold")
    for yi, value in zip(y, values):
        ax_cov.text(value * 1.08, yi, f"{value:,}", va="center", fontsize=8.5)

    add_panel_label(ax_year, "C")
    ax_year.bar(yearly["year"], yearly["usable_units"], color=COLORS["light_blue"], edgecolor=COLORS["blue"], linewidth=1.1)
    ax_year.plot(yearly["year"], yearly["usable_units"], color=COLORS["blue"], marker="o", lw=2)
    ax_year.set_xlabel("Survey year")
    ax_year.set_ylabel("Usable linked units")
    ax_year.set_title("Annual analytic coverage", loc="left", fontsize=12, fontweight="bold")
    ax_year.set_xticks(yearly["year"])
    ax_year.grid(axis="x", visible=False)
    for x, v in zip(yearly["year"], yearly["usable_units"]):
        ax_year.text(x, v + 40, f"{int(v):,}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Public-data assembly and linkage coverage", fontsize=15, fontweight="bold", x=0.52)
    fig.savefig(FIG / "fig1_public_data_assembly.png", bbox_inches="tight")
    plt.close(fig)


def fig2_coupled_landscape() -> None:
    panel = pd.read_csv(OUT / "gmc_panel_key_indicator_matrix_usable_2021_2026.csv")
    stats = json.loads((OUT / "gmc_panel_stats_2021_2026.json").read_text(encoding="utf-8"))
    yearly = pd.DataFrame(stats["yearly_spearman"])

    xq = panel["trainer_adverse_prop"].quantile(0.90)
    yq = panel["trainee_adverse_prop"].quantile(0.90)
    high_high = ((panel["trainer_adverse_prop"] >= xq) & (panel["trainee_adverse_prop"] >= yq)).sum()

    fig, (ax_hex, ax_year) = plt.subplots(1, 2, figsize=(11.2, 5.4), gridspec_kw={"width_ratios": [1.2, 1]}, constrained_layout=True)
    add_panel_label(ax_hex, "A")
    hb = ax_hex.hexbin(
        panel["trainer_adverse_prop"],
        panel["trainee_adverse_prop"],
        gridsize=34,
        mincnt=1,
        cmap="viridis",
        bins="log",
        linewidths=0,
    )
    cb = fig.colorbar(hb, ax=ax_hex, shrink=0.86)
    cb.set_label("Trust-specialty-year units (log scale)")
    ax_hex.axvline(xq, color=COLORS["red"], ls="--", lw=1.5)
    ax_hex.axhline(yq, color=COLORS["red"], ls="--", lw=1.5)
    ax_hex.text(
        xq + 0.015,
        yq + 0.035,
        f"High-high QI signal zone\nn={high_high:,}",
        fontsize=9,
        color=COLORS["red"],
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=COLORS["red"], alpha=0.92),
    )
    ax_hex.set_xlabel("Trainer adverse burden")
    ax_hex.set_ylabel("Trainee adverse burden")
    ax_hex.set_title("Coupled strain landscape", loc="left", fontsize=12, fontweight="bold")
    ax_hex.set_xlim(-0.02, min(1.02, panel["trainer_adverse_prop"].max() + 0.08))
    ax_hex.set_ylim(-0.02, min(1.02, panel["trainee_adverse_prop"].max() + 0.08))

    add_panel_label(ax_year, "B")
    colors = [COLORS["teal"] if p < 0.05 else COLORS["muted"] for p in yearly["p"]]
    ax_year.axhline(0, color=COLORS["ink"], lw=1)
    ax_year.vlines(yearly["year"], 0, yearly["spearman_rho"], color=colors, lw=3, alpha=0.75)
    ax_year.scatter(yearly["year"], yearly["spearman_rho"], s=70, color=colors, edgecolor="white", linewidth=1.1, zorder=3)
    for _, row in yearly.iterrows():
        ax_year.text(row["year"], row["spearman_rho"] + 0.008, f"{row['spearman_rho']:.3f}", ha="center", fontsize=8.5)
    ax_year.set_xlabel("Survey year")
    ax_year.set_ylabel("Spearman rho")
    ax_year.set_title("Directionally consistent yearly association", loc="left", fontsize=12, fontweight="bold")
    ax_year.set_xticks(yearly["year"])
    ax_year.set_ylim(-0.02, max(0.14, yearly["spearman_rho"].max() + 0.03))
    ax_year.grid(axis="x", visible=False)

    fig.suptitle("Trainer and trainee adverse burden co-occur in interpretable units", fontsize=15, fontweight="bold")
    fig.savefig(FIG / "fig2_coupled_strain_landscape.png", bbox_inches="tight")
    plt.close(fig)


def fig3_model_forest() -> None:
    final = pd.read_csv(OUT / "table_final_robustness_models.csv")

    trainer = final[final["primary_exposure"] == "trainer_adverse_prop"].copy()
    trainer_labels = {
        "gmc_trust_fe_year_specialty_fe_cluster_trust": "Trust FE + year + specialty FE",
        "gmc_within_pair_annual_change_year_fe_cluster_trust": "Within pair annual change",
        "gmc_weighted_year_specialty_fe_cluster_trust": "Response-volume weighted",
        "nhs_staff_trust_fe_year_specialty_fe_cluster_trust": "Staff Survey linked, trust FE",
        "joint_staff_sickness_year_specialty_fe_cluster_trust": "Joint Staff Survey + sickness",
    }
    trainer["label"] = trainer["model"].map(trainer_labels)
    trainer = trainer.dropna(subset=["label"]).iloc[::-1]

    external = final[final["primary_exposure"].isin(["nhs_staff_stress_index", "sickness_absence_rate_percent"])].copy()
    external_labels = {
        ("nhs_staff_trust_fe_year_specialty_fe_cluster_trust", "nhs_staff_stress_index"): "Staff stress, trust FE",
        ("joint_staff_sickness_year_specialty_fe_cluster_trust", "nhs_staff_stress_index"): "Staff stress, joint model",
        ("joint_staff_sickness_year_specialty_fe_cluster_trust", "sickness_absence_rate_percent"): "Sickness absence, joint model",
    }
    external["label"] = [external_labels.get((m, e)) for m, e in zip(external["model"], external["primary_exposure"])]
    external = external.dropna(subset=["label"]).iloc[::-1]

    fig, (ax_t, ax_e) = plt.subplots(1, 2, figsize=(11.4, 5.2), gridspec_kw={"width_ratios": [1.25, 1]}, constrained_layout=True)

    add_panel_label(ax_t, "A")
    y = np.arange(len(trainer))
    xerr = np.vstack([trainer["coef"] - trainer["ci_low"], trainer["ci_high"] - trainer["coef"]])
    ax_t.axvline(0, color=COLORS["ink"], lw=1)
    ax_t.errorbar(trainer["coef"], y, xerr=xerr, fmt="o", ms=7, color=COLORS["blue"], ecolor=COLORS["light_blue"], elinewidth=4, capsize=3)
    ax_t.set_yticks(y, trainer["label"])
    ax_t.set_xlabel("Coefficient for trainer adverse burden")
    ax_t.set_title("Trainer signal remains positive", loc="left", fontsize=12, fontweight="bold")
    ax_t.set_xlim(0, max(0.14, trainer["ci_high"].max() + 0.015))
    for yi, (_, row) in zip(y, trainer.iterrows()):
        ax_t.text(row["ci_high"] + 0.004, yi, p_label(row["p"]), va="center", fontsize=8.3, color=COLORS["muted"])

    add_panel_label(ax_e, "B")
    y2 = np.arange(len(external))
    colors = [COLORS["amber"] if "Staff stress" in label else COLORS["red"] for label in external["label"]]
    xerr2 = np.vstack([external["coef"] - external["ci_low"], external["ci_high"] - external["coef"]])
    ax_e.axvline(0, color=COLORS["ink"], lw=1)
    ax_e.errorbar(external["coef"], y2, xerr=xerr2, fmt="none", ecolor=COLORS["grid"], elinewidth=4, capsize=3, zorder=1)
    ax_e.scatter(external["coef"], y2, s=70, color=colors, edgecolor="white", linewidth=1.1, zorder=2)
    ax_e.set_yticks(y2, external["label"])
    ax_e.set_xlabel("Coefficient")
    ax_e.set_title("External organisational signals", loc="left", fontsize=12, fontweight="bold")
    xmin = min(-0.18, external["ci_low"].min() - 0.04)
    xmax = max(0.62, external["ci_high"].max() + 0.05)
    ax_e.set_xlim(xmin, xmax)
    for yi, (_, row) in zip(y2, external.iterrows()):
        ax_e.text(row["ci_high"] + 0.025, yi, p_label(row["p"]), va="center", fontsize=8.3, color=COLORS["muted"])

    fig.suptitle("Robustness and external triangulation model estimates", fontsize=15, fontweight="bold")
    fig.savefig(FIG / "fig3_model_robustness_forest.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    set_style()
    fig0_framework()
    fig1_data_assembly()
    fig2_coupled_landscape()
    fig3_model_forest()
    print("WROTE", FIG)


if __name__ == "__main__":
    main()
