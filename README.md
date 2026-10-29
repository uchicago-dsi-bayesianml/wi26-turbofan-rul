<picture>
  <source media="(prefers-color-scheme: dark)" srcset=".github/assets/banner-dark.png">
  <img alt="The University of Chicago, MS in Applied Data Science. ADSP 32014 course project." src=".github/assets/banner-light.png">
</picture>

# Predictive Maintenance for Aircraft Engines: RUL Estimation

> **Example project from Winter 2026.** This is a copy of [asingh49-cmd/RUL-and-Engine-Degradation-Bayesian-ML-Project](https://github.com/asingh49-cmd/RUL-and-Engine-Degradation-Bayesian-ML-Project), shown with the authors' work and commit history intact. Their own write-up is [REPORT.md](REPORT.md). Course staff wrote this README and made the edits listed under [Changes by course staff](#changes-by-course-staff), including the section on [what to learn from this project](#what-to-learn-from-this-project) and a script, `staff/check_scores.py`, that recomputes the scores. The NASA C-MAPSS data is not in this repository; NASA's repository states no license and asks to be acknowledged, and [data/README.md](data/README.md) explains how to get it.

A turbofan engine wears out gradually, and its sensors drift as it does. Given an engine's sensor history up to today, how many more cycles will it run, and how sure can we be? On NASA's C-MAPSS FD001 benchmark, 100 simulated engines run to failure for training and 100 cut off before failure for testing, each of the five team members built a different model of the remaining useful life (RUL): a hierarchical Bayesian regression sampled with NUTS, a dynamic Bayesian network and an unscented Kalman filter, a linear Kalman filter beside a random-forest ensemble, Meta's Prophet, and a CNN, attention and LSTM network with Monte Carlo dropout. A Bayesian online changepoint detector, written from scratch, sets the cycle at which the network's training labels start to count down. The write-up ranks the network first, by the benchmark's asymmetric score.

## Summary

**Data.** C-MAPSS FD001, simulated at one operating condition with one fault mode: 20,631 rows for the 100 training engines and 13,096 for the 100 test engines, one row per cycle with three settings and 21 sensors, and `RUL_FD001.txt` with the true RUL of each test engine after its last cycle. The standard evaluation scores one prediction per test engine, at its last cycle, against that file, by RMSE and by the PHM08 score that the notebooks call the NASA score.

**Labels.** Several notebooks cap the training RUL at 120 or 125 cycles, since an engine far from failure looks healthy and its distance from failure cannot be read from the sensors. The changepoint detector instead finds, engine by engine, the cycle at which three sensors leave their healthy baseline; the average, 112 cycles, becomes the BRNN notebook's onset, before which the label is held flat.

**Models.** A Student-t regression of log RUL with a random intercept per engine, in PyMC. A two-slice dynamic Bayesian network over binned sensors, and an unscented Kalman filter on a one-dimensional health state. A linear Kalman filter on health and degradation rate, and five random forests with quantile gradient boosting for an 80 percent interval, which the authors call BSTS. Prophet with 18 regressors. A network of a 1-D convolution, self-attention and an LSTM over 30-cycle windows, trained with an asymmetric loss and sampled 100 times with dropout on, which the write-up calls the BRNN. A GRU sits beside them as a baseline.

**Results as reported.** The write-up's final table ranks the network first (score 1,138.28, RMSE 24.30), then Prophet (1,265.47, 22.69), the random forests (2,067, 19.51) and the unscented Kalman filter (3,311.77, 35.85).

The staff section below explains why that ranking does not hold: the notebooks for four of the models swap the score's two constants, the network's number was chosen on the test set and scored against labels that differ from the official ones for 34 of the 100 test engines, and the model called BSTS is a random forest. Rescored on the standard protocol, the forests come first.

## Team and contributions

Team "The Priors of the Caribbean", Winter 2026.

- **Adi Singh** ([@asingh49-cmd](https://github.com/asingh49-cmd)): the repository; the linear Kalman filter and the random-forest model, in both `bsts_kf_rul_comparison` notebooks, with their README and figures; the preprocessing in `LSTM_Model.ipynb`; the upload of the Prophet notebook into `prophet_model/`; and the BSTS and Kalman sections of the write-up.
- **Ryan Chen** ([@RyanChenJung](https://github.com/RyanChenJung)): `BOCPD.ipynb`, both BRNN notebooks and the BRNN README, the folder layout and ignore rules, the EDA figure, and the merges of the team's pull requests; Ryan also committed early copies of Eduardo's hierarchical model.
- **William Niu** ([@williamniu](https://github.com/williamniu), committing as CasperNiu): `Prophet_Meta.ipynb` and the Prophet README, `GRU_Final_Version.ipynb`, and the last round of edits to the write-up, including its results table.
- **Mukul Ramesh** ([@MukulRamesh](https://github.com/MukulRamesh)): `DBN1.ipynb` to `DBN4.ipynb`, which hold the dynamic Bayesian network and both versions of the unscented Kalman filter, the README for that folder, and the DBN and UKF sections of the write-up.
- **Eduardo (Lalo) Tovilla Rivera** ([@Eduardo-Tovilla](https://github.com/Eduardo-Tovilla)): the hierarchical Bayesian model notebook and its README, and the project overview in the write-up.

The repository ran from February 10 to March 13, 2026. Several notebooks were saved from Colab or uploaded through GitHub's web interface, so the account that committed a notebook is not necessarily its only author; the attributions follow the [commit history](https://github.com/uchicago-dsi-bayesianml/wi26-turbofan-rul/commits/main) and the contributor table in the Prophet README.

**Borrowed pieces.** `gen_sequence` and `gen_labels` in `brnn_v2.ipynb` cell 2 are adapted from the helpers in Microsoft's [Deep Learning Basics for Predictive Maintenance](https://github.com/Azure/lstms_for_predictive_maintenance) notebook (MIT), which many C-MAPSS projects reuse, among them [umbertogriffo/Predictive-Maintenance-using-LSTM](https://github.com/umbertogriffo/Predictive-Maintenance-using-LSTM). The changepoint detector follows R. P. Adams and D. J. C. MacKay, "Bayesian Online Changepoint Detection" (2007). `brnn_v1.ipynb` was produced with Gemini's Data Science Agent in Colab: its Task, Subtask and Reasoning cells and closing Q&A summary are that tool's format.

## Getting started

There is no requirements file. Beyond the scientific Python stack, the notebooks use PyTorch (BRNN, GRU), PyMC and ArviZ (hierarchical model), pgmpy (DBN), filterpy (Kalman filters), statsmodels (first BSTS notebook) and Prophet.

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install numpy pandas scipy scikit-learn matplotlib seaborn tqdm torch pymc arviz pgmpy networkx filterpy statsmodels prophet jupyter
```

Download C-MAPSS as described in [data/README.md](data/README.md) and put the three FD001 files where each notebook expects them; the data note lists the paths.

**The notebooks.** They are independent and can run in any order; only the BRNN's onset of 112 cycles comes from another notebook. Cell numbers in this README count every cell, markdown included, from 1.

1. `notebooks/BOCPD.ipynb`: degradation onsets from Bayesian online changepoint detection.
2. `notebooks/bayesian_recurrent_neural_network(brnn)/brnn_v2.ipynb`: the network, its MC dropout sampling and the post-hoc search. It was trained on a Colab T4 GPU.
3. `notebooks/hierarchical_bayesian_model(hbm)/Hirechical Bayesian Model.ipynb`: the PyMC model.
4. `notebooks/dynamic_bayesian_network_(dbn)/DBN1.ipynb` for the network and `DBN3.ipynb` for the unscented Kalman filter.
5. `notebooks/bayesian_time_series_model(bsts)/bsts_kf_rul_comparison_updated.ipynb`: the linear Kalman filter and the random forests.
6. `notebooks/prophet_model/Prophet_Meta.ipynb`: Prophet with a linear and a logistic trend.

**Staff check.** `staff/check_scores.py` reads the FD001 files and the committed notebooks without running them. It lists which notebooks use the swapped score, counts the BRNN test labels that differ from `RUL_FD001.txt`, refits and rescores the random forests, shows that the Kalman filter's Q/R search cannot change anything, and computes a persistence baseline for the DBN. It needs NumPy and scikit-learn and takes about 20 seconds:

```sh
pip install numpy scikit-learn
python staff/check_scores.py "path/to/CMAPSSData"
```

**What does not run as committed.** Every notebook reads the data from its own location, several from a Google Drive folder. `Prophet_Meta.ipynb` cell 19, which produced the logistic model's test numbers, cannot have run as saved: it groups by a `unit_id` column that its frame does not have, and it passes `untertainty_samples`, an argument `Prophet.predict` does not take. `brnn_v2.ipynb` prints its messages in Chinese while its source prints them in English, so the source was translated after the run. The warnings saved in `DBN1.ipynb` cell 5 are the ones the discretizer gives when it runs a second time on its own output (the staff script reproduces both cases), so the committed DBN results come from a session in which that cell ran twice. The hierarchical model ran with `FAST_DEBUG = True`, on one row in 25; its sampler took 3,681 seconds without a C compiler. Both BSTS notebooks were run out of order, and `LSTM_Model.ipynb` stops after preprocessing.

## Data

[data/README.md](data/README.md) has the download links, citation, terms and the path each notebook expects. In short:

- **Source:** the "Turbofan Engine Degradation Simulation Data Set" (C-MAPSS) from the NASA Prognostics Data Repository (Saxena and Goebel, 2008), described in Saxena, Goebel, Simon and Eklund, PHM08. The project uses FD001, one of four subsets, three text files of about 6 MB.
- **Terms:** NASA's pages state no license. The repository asks that publications acknowledge it and the data donors, and the data are used at the user's own risk.
- **The split:** the benchmark fixes the training and test engines. For validation, the hierarchical model, the BSTS notebook and the GRU hold out training engines rather than rows, which is the right way to do it.

## Project structure

| Path | Contents |
|---|---|
| `REPORT.md` | The authors' write-up: the business problem, EDA, each model, the comparison table and the conclusions. |
| `notebooks/BOCPD.ipynb` | Sensor trendability, engine lifetimes, a multivariate BOCPD written from scratch, per-engine onsets and two grid searches over its settings. |
| `notebooks/bayesian_recurrent_neural_network(brnn)/` | `brnn_v2.ipynb`, the final network with its MC dropout and post-hoc search; `brnn_v1.ipynb`, the first version, written with Gemini's Data Science Agent in Colab; and the BRNN README. |
| `notebooks/hierarchical_bayesian_model(hbm)/` | The PyMC model and its README. The misspelled file name is the authors'. |
| `notebooks/dynamic_bayesian_network_(dbn)/` | `DBN1.ipynb`, the dynamic Bayesian network; `DBN2.ipynb`, exploratory analysis; `DBN3.ipynb`, the unscented Kalman filter, Mark I and Mark II; `DBN4.ipynb`, Mark II with the extrapolation capped at 112 cycles and an evaluation on truncated training engines; and the folder README. |
| `notebooks/bayesian_time_series_model(bsts)/` | `bsts_kf_rul_comparison_updated.ipynb`, the linear Kalman filter and the random forests, with two figures; `bsts_kf_rul_comparison.ipynb`, the earlier version whose structural time-series model leaked the test labels; and the BSTS README. |
| `notebooks/prophet_model/` | The Prophet notebook and its README. A byte-identical copy in `time_series_models/` was removed. |
| `notebooks/time_series_models/` | `GRU_Final_Version.ipynb`, a two-layer GRU on 30-cycle windows, and `LSTM_Model.ipynb`, preprocessing for an LSTM that was not built. |
| `viz/eda_plot.png` | The EDA figure in the write-up. |
| `data/README.md`, `staff/check_scores.py` | Course staff's data note and check of the scores. |

## Method

**Changepoint detection.** `BOCPD.ipynb` follows Adams and MacKay's Bayesian online changepoint detection. Sensors s11, s4 and s12 of each training engine are standardized against its first 30 cycles, each gets a Normal-Gamma prior, the three are treated as independent, and the run-length posterior is updated every cycle with a constant hazard of 1/100. The onset is the first drop in the most probable run length after cycle 50. Cell 5 finds one in every engine, on average at cycle 112.38, from 76 to 135 among the first ten engines; cells 6 and 7 search 1,296 and 972 settings.

**The network.** `brnn_v2.ipynb` cuts the training engines into 17,731 windows of 30 cycles over two settings and 14 sensors. The label is RUL held flat until cycle 112, so an engine that lives *L* cycles starts at *L* − 112 rather than at a common cap. A 1-D convolution, four-head self-attention, a two-layer LSTM and a dense head with dropout 0.3 give an output clamped to [0, 130]. Training runs 100 epochs, with Huber loss for 40 and then a weighted Huber loss plus a penalty on late predictions; after every epoch the score is computed on the last windows of the 100 test engines and the best epoch is kept. MC dropout makes 100 passes, and grids over an offset, a multiple *k* of the dropout standard deviation and a ceiling post-process the mean.

**Hierarchical regression.** The target is log(1 + RUL), uncapped, on 46 features: cycle, the settings, the 21 sensors and their first differences. The likelihood is Student-t with ν ~ Exponential(1/10) and σ ~ HalfNormal(1), and the mean is α + *b*<sub>engine</sub> + *X*β with α and each β ~ N(0, 1) and non-centered engine intercepts *b* ~ N(0, τ₀²), τ₀ ~ HalfNormal(1). The split is 80 training and 20 validation engines; with `FAST_DEBUG` on, the model sees every 25th cycle, 702 rows, with 300 tuning and 300 NUTS draws on two chains. For a new engine the predictive variance adds τ₀² to σ². The write-up describes a different prior, β ~ N(0, τ²) with a shared τ, which the code does not have.

**Dynamic Bayesian network.** `DBN1.ipynb` bins 17 features (two settings and 15 sensors) into five quantile bins each and builds a two-slice network, 77 edges: every feature at *t* to itself at *t* + 1, and both settings to every sensor within a slice. Tables are fit by maximum likelihood on consecutive pairs of cycles, and variable elimination predicts the bin of sensor_11 at *t* + 1 from all features at *t*. It predicts a sensor, not the RUL.

**Unscented Kalman filter.** `DBN3.ipynb` tracks one health state *x* that starts at 1. Mark I observes two raw sensors with a linear decay; Mark II observes the first two principal components of 14 sensors, both as *x*, with decay rate *b* · exp(*c*(1 − *x*)). Q and R are learned by smoothing the first five training engines repeatedly and moving toward the residual covariances, and *b* and *c* come from a grid on training engines 81 to 90, cut 40 cycles before failure. The RUL is the number of steps the decay model needs, after the last cycle, to bring *x* to zero; the grid used a threshold of 0.2 instead.

**Linear Kalman filter.** Health falls by the degradation rate each cycle, the observation matrix comes from each sensor's change between the first and last tenth of a life, both states are clipped after every update, and the RUL is health divided by rate.

**The model called BSTS.** Five random forests of 100 trees and two gradient-boosting models for the 10th and 90th percentiles, on eight standardized sensors and the cycle number, with the RUL capped at 120. The markdown still describes a structural time-series model in statsmodels. That model was in the first notebook, fit to each test engine with labels built from the test truth (RMSE 1.01, `bsts_kf_rul_comparison.ipynb` cell 31), and the authors replaced it with the forests when they found the leak.

**Prophet.** All training rows are laid end to end on an hourly time axis, the test rows continue it, and Prophet fits a linear or logistic trend plus 18 regressors, without seasonality, by its default MAP optimization. Predictions are clipped to [0, 125].

### What is Bayesian here

Three parts are Bayesian in substance. The changepoint detector computes a posterior over the run length with conjugate updates; it is collapsed to one number, 112, that is then used for every engine. The hierarchical regression has priors on every parameter, partial pooling of the engine intercepts and a NUTS posterior, and its interval is checked: 90.2 percent of validation rows fall in its 90 percent interval (`Hirechical Bayesian Model.ipynb` cell 7). Its chains have not converged: R-hat is 1.20 for α and 1.12 for τ₀, with 11 and 14 effective draws (same cell). The two Kalman filters compute a Gaussian posterior over a hidden health state at every cycle, which is Bayesian filtering, but the RUL is then a deterministic extrapolation of the posterior mean, and the linear filter's interval is a fixed ±1.96 × 20 cycles rather than anything from its covariance.

The network's MC dropout approximates a Bayesian neural network. Its spread enters only the post-hoc search and is never checked against the truth: the markdown in `brnn_v2.ipynb` cell 9 lists coverage, interval width, NLL and calibration error, and the code computes none of them. `model.train()` also switches the batch normalization to training mode during sampling, so each pass normalizes with the test batch's statistics. The dynamic Bayesian network is a discrete graphical model with maximum-likelihood tables and exact inference, the subject of the course's first homework, with no priors. Prophet has priors on its changepoints and regressors and is fit by MAP, which makes it a penalized regression; its intervals are not evaluated. The random forests and the quantile boosting are not Bayesian, and neither are the GRU and the LSTM.

## Results

### As reported

The write-up's final comparison:

| Model | RMSE | NASA score | Write-up's notes |
|---|---:|---:|---|
| BRNN | 24.30 | 1,138.28 | Best performance; Bayesian uncertainty |
| Prophet | 22.69 | 1,265.47 | Excellent baseline |
| BSTS | 19.51 | 2,067.00 | Good uncertainty calibration |
| UKF (Mark II) | 35.85 | 3,311.77 | Sensitive to hyperparameters |

Each row matches a notebook output, but not the one a reader would assume:

- **BRNN:** `brnn_v2.ipynb` cell 20, after the post-hoc search. The network as trained scores 1,851.31 with RMSE 25.63 (cell 10).
- **Prophet:** the logistic model, cells 19 and 20, from the cell that cannot have run as saved. The linear model, whose cells can run, has RMSE 22.63 and score 1,289.23 (cells 12 and 14).
- **BSTS:** the random forests, `bsts_kf_rul_comparison_updated.ipynb` cells 29 and 31, scored with the swapped constants.
- **UKF:** `DBN3.ipynb` cell 11. `DBN4.ipynb` cell 12, which caps the extrapolation at 112 cycles instead of 500, prints the same two numbers.

Elsewhere the write-up and the notebooks disagree. The write-up gives the forests' 80 percent interval a coverage of 65 percent; cell 29 prints 66.0. It gives the linear Kalman filter 26 percent; cell 37 prints 33.0 percent, for a 95 percent interval. It gives the DBN a mean next-step accuracy of about 52.6 percent over the 100 test engines; `DBN1.ipynb` cell 11 prints 75.92 percent, and no committed output shows 52.6.

### On the standard protocol

One prediction at each test engine's last cycle, scored against `RUL_FD001.txt` with the PHM08 constants (early /13, late /10). Lower is better for both columns.

| Model | RMSE | Score | Source |
|---|---:|---:|---|
| Random forests, RUL cap 120 (chosen on the test set) | 19.51 | 790.7 | `staff/check_scores.py` |
| Random forests, RUL cap 125 (the notebook's own setting) | 19.81 | 913.2 | `staff/check_scores.py` |
| Prophet, linear trend | 22.63 | 1,289.2 | `Prophet_Meta.ipynb` cells 12 and 14 |
| Unscented Kalman filter, Mark II | 35.85 | 3,311.8 | `DBN3.ipynb` cell 11 |
| Linear Kalman filter | 77.80 | not recomputed | `bsts_kf_rul_comparison_updated.ipynb` cell 25 |
| BRNN | not a valid test result | | see below |

The staff refit reproduces the notebook's RMSE of 19.51, MAE of 13.94, coverage of 66.0 percent and swapped score of 2,067.3 exactly (its mean interval width is 41.19 cycles against the notebook's 41.42). With the correct constants the same predictions score 790.7, the best in the project. The cap of 120 was picked by trying 115, 120 and 125 on the test set (cells 29 and 40), so the notebook's configured cap of 125, at 19.81 and 913.2, is the fair number. The linear Kalman filter's predictions all lie between 0.0 and 32.6 cycles (cell 25).

The BRNN has no valid test number. The epoch kept had the best test score (cell 8, epoch 54), the post-processing was the best of 704 combinations scored on the same 100 engines (cell 20), the score has the swapped constants, and its truth differs from `RUL_FD001.txt` for 34 engines. Of the ten engines the notebook lists as its worst late predictions (cell 10), four (1, 15, 22 and 85) are scored against labels 36 to 81 cycles below the truth, and three of those were in fact predicted early: engine 1, predicted at 98.1 cycles, has a true RUL of 112 and a label of 31.

The hierarchical model was not scored at the test engines' last cycles. On all 13,096 test rows its RMSE is 55.57, and 40.16 on the rows with RUL of at most 120 (`Hirechical Bayesian Model.ipynb` cells 10 and 11). The GRU is evaluated on all 1,951 windows of 10 training engines it did not see, a different protocol: RMSE 12.81, with a swapped score (`GRU_Final_Version.ipynb` cell 9).

### Other numbers in the notebooks

- **The BRNN's post-hoc search** (`brnn_v2.ipynb`). Subtracting *k* standard deviations, the cautious direction, does not help: the best *k* is 0 (cell 14). A ceiling of 90 brings the score to 1,474.27 (cell 16), adding 12 cycles under that ceiling to 1,160.78 (cell 18), and the final rule adds 5 cycles plus 1.2 standard deviations and caps at 89 (cell 20). The winning shifts make predictions later, the direction the swapped constants favor.
- **Hierarchical model.** Validation RMSE 41.89 cycles over rows; ν is about 1.7, a very heavy-tailed likelihood, with no divergences (cells 7 and 9). The "last-cycle" evaluation in cell 8 uses validation engines that ran to failure, where the true RUL is 0: RMSE 10.79, and none of the 20 intervals covers 0.
- **Unscented Kalman filter.** Mark I scores RMSE 62.22 and 275,779.63 on the test set (`DBN3.ipynb` cell 5). Mark II's grid picks *b* = 0.003 and *c* = 2.0 (cell 9). On training engines cut at random points it scores RMSE 63.60 and 90,500,546.75 (`DBN4.ipynb` cell 15), much worse than on the test set.

## What to learn from this project

Course staff added this section. Five people built five working pipelines on one benchmark; the points below are about what their comparison can and cannot support.

- **Check a scoring function against its definition.** The PHM08 score charges exp(−d/13) − 1 for an early prediction and exp(d/10) − 1 for a late one. The BRNN, hierarchical, BSTS and GRU notebooks have the constants the other way round, so a prediction 20 cycles late costs 3.66 instead of 6.39 and one 20 cycles early costs 6.39 instead of 3.66. The UKF notebooks (`DBN3`, `DBN4`) and the Prophet notebook have them right. The write-up's table mixes the two, and its prose about penalizing late predictions describes the correct formula while its best number was tuned with the other.
- **Never let the test set choose anything.** The BRNN keeps the epoch with the best test score and picks its post-processing from 704 combinations on the same 100 engines; the forests' RUL cap was chosen by trying three caps on the test set. Each choice uses the answer key. Hold out training engines for these decisions, as the hierarchical model, the BSTS notebook and the GRU already do for validation.
- **Score against the truth the benchmark gives you.** The BRNN's labeling function, applied to the test engines, caps each one's answer at its total life minus 112 cycles. For the 34 test engines observed for fewer than 112 cycles, that lowers the answer by 40.8 cycles on average and by up to 81. A label scheme for training is a modeling choice; the test truth is not.
- **Call a model by what the code fits.** The "Bayesian time-series model" is five random forests and two quantile boosters, and Prophet's trend runs over a time axis made by laying 100 engines end to end. Both are fair baselines, and the forests are the best model in the project, but the names put them in the wrong place in a comparison of Bayesian methods.
- **Check that a hyperparameter search can change something.** The Kalman filter's search over 16 values of Q and R returns 94.08 for every one, because the function never passes them to the filter, whose Q and R are fixed in its constructor (staff script, section 4). A grid that prints the same number 16 times is a bug report.
- **Check the uncertainty you report.** The random forests' 80 percent interval covers 66 percent of test engines, which the notebook prints and the write-up calls good calibration. The linear filter's interval has a constant width, the network's dropout spread is never compared with its errors, and the hierarchical model's coverage is measured on rows rather than on the test engines' last cycles. Measure coverage where the decisions are made.
- **Compare with a trivial baseline.** For the next bin of a slowly drifting sensor, "the bin does not change" is the baseline: 45.07 percent per engine here, and in the training data staying put is the most likely move from every bin (staff script, section 5). The DBN reports 75.92 percent (`DBN1.ipynb` cell 11), a large gain for a model in which sensor_11's next bin depends on the past only through its current bin and two operational settings that barely vary in FD001. The committed outputs come from a run in which the binning ran twice, and the evaluation loop skips any query that raises an error. Report the baseline with the model so the reader can judge.
- **Make sure the committed outputs come from the committed code.** The logistic Prophet cell cannot run as saved, the BRNN source was translated after its outputs were produced, the DBN's binning ran twice, and three numbers in the write-up appear in no output. Restart the kernel and run top to bottom before the last commit.
- **Integrate the plan or say that you did not.** The methodology document proposed one hybrid framework joining the network, the DBN and the hierarchical model, on all four C-MAPSS subsets, with maintenance decisions based on predictive confidence. The repository has separate models on FD001, linked only by the changepoint onset, and the write-up presents them as a planned progression.

What the project does well: a Bayesian online changepoint detector written from scratch and used for a concrete purpose, setting the labels of another model; a hierarchical model in PyMC with non-centered intercepts, a split by engine, a coverage check, and convergence diagnostics printed rather than skipped; an unscented Kalman filter with noise matrices learned from smoothed states; a dynamic Bayesian network built and queried with exact inference; the standard last-cycle protocol in the BSTS, Prophet, UKF and BRNN notebooks; and, in the BSTS notebook, a first model that looked too good, a search for why, the leak found and the model replaced, all written up in the notebook's closing cells. That last habit is the one to copy.

## About this project

This project was built in Bayesian Machine Learning with Generative AI Applications (ADSP 32014), part of the Master of Science in Applied Data Science at the University of Chicago, in Winter 2026. It is shown here as an example for later cohorts.

## Changes by course staff

Course staff forked the authors' repository and made these edits. The authors' code is unchanged.

- Deleted the top-level `metadata.widgets` entry from `brnn_v2.ipynb` and `DBN1.ipynb`. The entry had no `state` key, which stops GitHub from rendering a notebook at all; the cells and outputs are byte-for-byte unchanged, and `brnn_v2.ipynb` shrinks from 5.1 MB to 0.6 MB.
- Removed `notebooks/time_series_models/Prophet_Meta.ipynb`, a byte-identical copy of `notebooks/prophet_model/Prophet_Meta.ipynb`.
- Edited `.gitignore` so that `data/README.md` and the banner are kept while the C-MAPSS files, the downloaded archives and model checkpoints stay out.
- Added `data/README.md`, on the source, terms, citation and download of C-MAPSS and the path each notebook expects, and `staff/check_scores.py`, the check described above.
- Renamed the authors' README to `REPORT.md`, unchanged, and added this README with the program banner.

## License

The authors chose the MIT License; see [LICENSE](LICENSE), which names Adi Singh as the copyright holder. The C-MAPSS data carries its own terms, described in [data/README.md](data/README.md).
