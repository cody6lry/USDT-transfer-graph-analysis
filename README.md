# Ethereum USDT Transfer Graph Multi-Signal Analysis

This repository provides the code and derived result files for the manuscript:

**Multi-Signal Analysis of Value Concentration and Connectivity Sensitivity in Ethereum USDT Transfer Graphs**

The project analyzes Ethereum-based USDT ERC-20 transfer records using a multi-signal analytical framework. The analysis integrates event-level value, address-level flow, graph centrality, directed mixing structure, component organization, node-removal response, and sampling-seed sensitivity.

## Data source

The transfer records are collected from the Etherscan V2 ERC-20 token-transfer endpoint for the Ethereum USDT contract:

`0xdac17f958d2ee523a2206206994597c13d831ec7`

Observation period:

`2025-10-01 00:00:00 UTC` to `2025-11-01 00:00:00 UTC`

No Etherscan API key is included in this repository.

## Sampling design

The observation month is divided into 2,976 non-overlapping 15-minute strata. For each stratum, up to 500 earliest and 500 latest transfer events are retrieved to form a candidate pool. Up to 100 unique events are selected from each candidate pool using deterministic pseudorandom sampling.

Primary sampling seed:

`20251031`

Sampling-seed sensitivity analysis uses:

`42`, `2025`, and `20251031`

## Main dataset summary

For the primary sample:

* Raw sampled records: 297,600
* Positive-value transfer events: 279,065
* Graph-construction events: 278,941
* Observed addresses: 201,833
* Unique directed edges: 190,653

## Main results

Key results reported in the manuscript include:

* Gini coefficient: 0.9834
* Top 1% sampled event-value share: 85.12%
* Largest weakly connected component: 67.15%
* Largest strongly connected component: 2.87%
* Random 5% node removal leaves 59.64% of original nodes in the largest WCC
* Degree-targeted 0.1% node removal reduces the largest WCC share to 7.23%

## Repository structure

```text
USDT-transfer-graph-analysis/
├── README.md
├── requirements.txt
├── src/
│   ├── 01_generate_sensitivity_samples.py
│   ├── 02_preprocess_transactions.py
│   ├── 03_concentration_analysis.py
│   ├── 04_graph_analysis.py
│   ├── 05_robustness_experiment_deterministic.py
│   ├── analyze_seed_sensitivity.py
│   └── plot_robustness_figures_fixed.py
├── results/
   ├── preprocessing_summary.json
   ├── concentration_summary.json
   ├── graph_summary.json
   ├── directed_assortativity.json
   ├── bow_tie.json
   ├── robustness_summary_deterministic.csv
   └── sampling_sensitivity_summary.csv

```

## Installation

```bash
pip install -r requirements.txt
```

## API key

Do not hard-code the Etherscan API key.

Windows PowerShell:

```powershell
$env:ETHERSCAN_API_KEY="YOUR_KEY"
```

Linux or macOS:

```bash
export ETHERSCAN_API_KEY="YOUR_KEY"
```

## Reproduction workflow

Run the scripts in the following order:

python 01_download_usdt_complete.py
python 02_preprocess_transactions.py
python 03_concentration_analysis.py
python 04_graph_analysis.py
python 05_robustness_experiment_deterministic.py
python analyze_seed_sensitivity.py
python plot_robustness_figures_fixed.py


## Notes

The repository does not claim that the sampled records form a complete census of all Ethereum USDT transfers during the observation period. The sampling design is two-sided and temporally stratified, but events occurring in the middle of highly active 15-minute intervals may be underrepresented.

The node-removal experiment measures static graph connectivity dependence in the reconstructed historical transfer graph. It does not simulate Ethereum consensus failure, address replacement, adaptive rerouting, or operational failure of the USDT payment system.

## Data availability

The code, sampling configuration, derived result files, and figure-generation scripts are provided for reproducibility. Raw ERC-20 transfer records can be re-collected from the Etherscan V2 API using the scripts and parameters provided in this repository, subject to Etherscan API access conditions.
