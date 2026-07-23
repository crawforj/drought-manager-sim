# drought-manager-sim

A water-utility demand-forecasting and causal-inference engine — price-elasticity
estimation, customer behavioral segmentation, a multi-model short/medium-range
demand forecast ensemble, and non-revenue-water (NRW) auditing — demonstrated on
a **fully synthetic dataset calibrated to reproduce the statistical structure of
a real utility's data**, without containing any real customer record or real
infrastructure location.

## Datasets

Quick links to the synthetic data itself:

- [`state/customer_history/`](state/customer_history/) — account-level daily consumption panel (9,000 fictional accounts, 2023–2026, monthly parquet files)
- [`state/entry_point_history.parquet`](state/entry_point_history.parquet) — master-meter daily aggregate input (system A)
- [`production_history.csv`](production_history.csv) — SCADA finished-water production (system B)
- [`state/meter_zone_map.parquet`](state/meter_zone_map.parquet) — account → pressure-zone / system mapping
- [`state/gis/pressure_zones.shp`](state/gis/pressure_zones.shp) — fictional pressure-zone shapefile (+ `.dbf`/`.shx`/`.prj`)
- [`config.yaml`](config.yaml) — fictional master-meter site table, pricing tiers, model config
- [`state/weather_cache.csv`](state/weather_cache.csv) — real public NOAA weather data used to drive the synthetic demand series

All `.parquet` files are stored via Git LFS — `git clone` resolves them automatically; the "Raw" button on GitHub shows only the LFS pointer, not the data.

## What this is

The analysis code here (causal inference, clustering, forecasting, water-audit
math) was originally built against a real water utility's real operational
data. That data — real customer accounts, real meter locations, a real service
area — can't be published: even with names and addresses stripped, a set of
real per-account consumption records and real infrastructure coordinates is
identifying on its own.

So this repo does something different from typical "anonymization": instead of
redacting a real dataset, it **generates a new one from scratch**, using a
fictional town ("Antler Creek, CO") and fictional customer accounts, but
calibrated so that running the actual analyses against it recovers
findings consistent with what the real data showed — same sign, same rough
order of magnitude, same qualitative structure. The methodology:

1. **Calibrate** (done once, privately, against real data — not part of this
   repo): fit the analysis pipelines and extract only aggregate statistical
   parameters — a treatment-effect size and its standard error, cluster
   centroids and covariances, seasonal/weather-response coefficients,
   autocorrelation structure, an input–consumption correlation. Never a raw
   per-account value.
2. **Fabricate a geography** (`synth/geography.py`): a fictional pressure-zone
   system and master-meter site table with the same *structural* properties
   as a real one (zone count, area distribution, hydraulic-system split,
   meter-role counts) — procedurally generated (Voronoi tessellation), not
   derived from or transformed out of any real GIS export.
3. **Generate** (`synth/generate_panel.py`): consume the calibrated
   parameters + fictional geography to synthesize a full account-level daily
   panel and system aggregate series (~9,000 fictional accounts,
   2023–2026) whose statistics match the calibration closely enough that the
   real analysis code, run *unmodified*, recovers consistent findings.
4. **Validate**: run the actual analysis pipelines against the synthetic
   panel and compare to the real findings. See the table below.

## Validation: does the synthetic data actually work?

Run against ~9,000 synthetic accounts vs. a real ~23,000-account population.
Percentages/correlations/ratios are the numbers that should match; absolute
volumes (MGD, account counts) scale down with the smaller synthetic
population and aren't expected to match 1:1.

| Metric | Real | Synthetic |
|---|---:|---:|
| Price-elasticity DiD: tier1 step (% of weather-expected demand) | -10.2% | -15.1% |
| Price-elasticity DiD: tier2/3 step (% of weather-expected demand) | -9.7% | -14.6% |
| DiD effect sign / significance | negative, p=0.033 | negative, p<0.0001 |
| Segmentation: top-consumers volume share | 1.0% of accounts → 25.5% of volume | 1.0% → 25.2% |
| Segmentation: vacant/intermittent share | 3.2% | 3.2% |
| Segmentation cluster count (k) | 3 | 2 |
| NRW correlation, system A (master-meter-summed input) | 0.988 | 0.988 |
| NRW correlation, system B (WTP production input) | 0.940 | 0.915 |
| Demand forecast series: mean (MGD) | 10.7 | 4.3 (≈ population ratio) |
| Demand forecast series: autocorrelation (lag 1 / 7 / 14) | 0.64 / 0.62 / 0.61 | 0.70 / 0.49 / 0.54 |

**Where it's close**: NRW correlations, segmentation's top-consumer and
vacant-account shares, and the price-elasticity effect's sign/significance
all land close to the real findings. Demand-series volatility scales with
population size as expected.

**Where it doesn't match exactly, and why**: the price-elasticity step
percentages run consistently hotter than their calibrated targets (by a
roughly proportional amount for both tiers, so the *difference between
them* — the actual causal estimate — stays much closer to the real value
than either tier's raw percentage does). During development, increasing
the synthetic population from 3,000 to 9,000 accounts closed a large
fraction of that gap, consistent with ordinary estimator variance shrinking
with sample size rather than a construction flaw — a full-scale run at
real population size would likely close it further. Segmentation recovers
2 clusters instead of 3; the synthetic account generator preserves each
behavioral feature's real marginal distribution but not the real
cross-feature correlations, which is the most likely cause.

## Repo layout

- `behavioral/` — weather normalization, difference-in-differences,
  interrupted time series, causal impact (Bayesian structural time series),
  double ML / R-learner heterogeneous treatment effects, k-means behavioral
  segmentation, response-profile analysis
- `ccdids/` — SARIMA + SVM + gradient-boosted multi-model demand forecast
  ensemble, drought-stage adjustment
- `nrw.py` — non-revenue-water daily water-balance auditing (AWWA M36
  ratio-of-sums convention)
- `model.py`, `model_selection.py` — OLS/climatology demand forecasting with
  auto-bias correction
- `pressure_zones.py`, `customer_store.py`, `data_quality.py`,
  `weather_history.py` — supporting infrastructure
- `synth/` — the calibrate-then-generate synthetic data pipeline described
  above
- `state/`, `config.yaml`, `production_history.csv` — the shipped synthetic
  dataset itself

## Running it

```bash
pip install -r requirements.txt
python -m pytest tests/
python -m behavioral.did          # self-test, synthetic fixture
python -m behavioral.segmentation # self-test, synthetic fixture
python nrw.py                     # self-test, synthetic fixture
python -m pressure_zones          # loads state/gis/pressure_zones.shp
```

To regenerate the synthetic dataset from scratch:

```bash
python -m synth.geography         # fictional service-area geometry
python -m synth.generate_panel    # fictional customer panel + system series
```

## What's deliberately not included

Real per-account meter-technology/share-count tier classification
(`tier_schedule.py` in the original pipeline) isn't included — it depends
on real per-account GIS data that was never part of this synthetic dataset.
`behavioral/did.py`'s `classify_tier_exposure(source="real")` path will
raise a clear `ImportError` if called without it; `source="proxy"` (usage-
volume-based) works standalone, and the shipped synthetic panel already
carries its own ground-truth tier assignment for the DiD demonstration
above.
