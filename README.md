# CTESS public training strain analysis code

This repository contains the analysis code for the study:

**Coupled strain in postgraduate medical training: a public-data linkage study of trainer pressure, trainee adverse training signals, and NHS organisational stress signals, 2021--2026**

The project develops the **Coupled Training Environment Strain Surveillance (CTESS)** framework to identify aggregate postgraduate medical training environments where trainer adverse burden and trainee adverse training-environment burden co-occur.

This is a code-only repository. It does not include manuscript files, PDFs, generated figures, derived result tables, or local data snapshots.

## Repository structure

- `06_analysis/pilot/`: Python scripts for public-data extraction, panel construction, linkage, robustness modelling, result packaging, and figure generation.
- `requirements.txt`: Python package requirements for the analysis scripts.
- `CITATION.cff`: citation metadata for the code repository.

## Public data sources

The analysis uses public aggregate data from:

- General Medical Council National Training Survey Education Data Tool.
- NHS Staff Survey public local results.
- NHS Digital NHS sickness absence rates.

No individual-level trainee, trainer, staff, patient, or confidential records are required or included. Generated CSV, JSON, Markdown, and PNG outputs should be produced locally from public aggregate inputs and are intentionally ignored by Git.

## Environment

Python 3.10 or newer is recommended.

```bash
pip install -r requirements.txt
```

The analysis is CPU-only. A GPU server is not required for reproducing the statistical models or figures.

## Running the pipeline

From the repository root:

```bash
python 06_analysis/pilot/gmc_powerbi_pilot_extract.py
python 06_analysis/pilot/build_gmc_panel.py
python 06_analysis/pilot/build_nhs_staff_linkage.py
python 06_analysis/pilot/build_sickness_linkage.py
python 06_analysis/pilot/analyze_gmc_pilot.py
python 06_analysis/pilot/run_final_robustness_models.py
python 06_analysis/pilot/run_robustness_checks.py
python 06_analysis/pilot/make_results_package.py
python 06_analysis/pilot/make_publication_figures.py
```

The scripts use paths relative to the repository root. Some extraction steps depend on public web endpoints and historical Power BI report structure, so users may need to update source URLs or provide locally downloaded public aggregate files before running the full pipeline.

## Suggested citation

Please cite the associated manuscript and this code repository if using the analysis code. A `CITATION.cff` file is included.
