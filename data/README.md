# Data

No data is in this repository. Every notebook reads the FD001 subset of NASA's C-MAPSS turbofan simulation, three plain-text files (`train_FD001.txt`, `test_FD001.txt` and `RUL_FD001.txt`), and each notebook expects them in a different place (below). `.gitignore` keeps the files and the downloaded archives out of the repository.

## Where to get it

- **NASA Prognostics Data Repository**, entry 6, "Turbofan Engine Degradation Simulation": [pcoe-data-set-repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/). The download link is `https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip` (12.4 MB). The archive holds one folder, `6. Turbofan Engine Degradation Simulation Data Set`, with a second archive, `CMAPSSData.zip`, inside it.
- **Mirror:** the NASA Open Data Portal entry [CMAPSS Jet Engine Simulated Data](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data) links to `https://data.nasa.gov/docs/legacy/CMAPSSData.zip`, which is the inner archive on its own. On October 28, 2026 the two copies of `CMAPSSData.zip` were byte-identical (SHA-256 `74bef434a34db25c7bf72e668ea4cd52afe5f2cf8e44367c55a82bfd91a5a34f`).

`CMAPSSData.zip` holds the four subsets FD001 to FD004 (a `train_`, `test_` and `RUL_` file for each), a `readme.txt`, and the paper that describes the simulation. The project uses FD001 only, although the methodology plan named all four. To check a copy from anywhere else, the SHA-256 hashes of the three FD001 files are:

```
963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8  train_FD001.txt
3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851  test_FD001.txt
a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca  RUL_FD001.txt
```

## What FD001 is

From the archive's `readme.txt`: 100 training engines that run until failure and 100 test engines whose records stop some time before failure, at one operating condition (sea level) and with one fault mode (degradation of the high-pressure compressor). Each engine starts with an unknown amount of normal wear, and the sensor readings carry noise. Each row is one operating cycle with 26 space-separated columns: unit number, cycle, three operational settings and 21 sensor measurements. `RUL_FD001.txt` gives, for each test engine in order, the true number of cycles it ran after its last recorded cycle. The training file has 20,631 rows and the test file 13,096.

The data set comes with a standard evaluation. For each test engine, a model predicts the remaining useful life (RUL) at the last recorded cycle and is scored against `RUL_FD001.txt`, by RMSE and by the asymmetric score from the PHM08 challenge, which the notebooks call the NASA score. With d = predicted − true, an engine costs exp(−d/13) − 1 when the prediction is early and exp(d/10) − 1 when it is late, so a late prediction costs more than an early one by the same number of cycles. The main README explains which notebooks use this protocol and this formula.

## Terms and citation

Neither NASA page states a license for the data set; the Open Data Portal entry says "License not specified". The repository page asks that publications using its data acknowledge both the repository and the donors of the data, and says that users employ the data at their own risk and that neither NASA nor the donors assume any liability. The data are simulated engine runs, with no personal information.

The repository asks for this citation:

> A. Saxena and K. Goebel (2008). "Turbofan Engine Degradation Simulation Data Set", NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.

The archive's `readme.txt` cites the paper that describes the simulation and defines the score:

> A. Saxena, K. Goebel, D. Simon, and N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", in Proceedings of the 1st International Conference on Prognostics and Health Management (PHM08), Denver, CO, October 2008.

## Where each notebook looks for the files

The notebooks were run on different machines, mostly in Google Colab, and none of them reads from this folder. Either put the three files where a notebook expects them or edit its path cell.

| Notebook | Path it reads |
|---|---|
| `notebooks/BOCPD.ipynb` | `../data/train_FD001.txt`, that is, this folder when the notebook runs from `notebooks/` |
| `notebooks/bayesian_recurrent_neural_network(brnn)/brnn_v1.ipynb` | the working directory (`data_dir = ''`) |
| `notebooks/bayesian_recurrent_neural_network(brnn)/brnn_v2.ipynb` | the working directory |
| `notebooks/bayesian_time_series_model(bsts)/*.ipynb` | `6. Turbofan Engine Degradation Simulation Data Set/CMAPSSData/`, relative to the working directory: the outer archive unpacked, with `CMAPSSData.zip` unpacked into a `CMAPSSData` folder beside it |
| `notebooks/time_series_models/LSTM_Model.ipynb` | the same folder as the BSTS notebooks |
| `notebooks/time_series_models/GRU_Final_Version.ipynb` | `/content/Bayesian Machine Learning/` (Colab) |
| `notebooks/prophet_model/Prophet_Meta.ipynb` | `/content/BayesianFinalProjects/` (Colab) |
| `notebooks/dynamic_bayesian_network_(dbn)/DBN1.ipynb` to `DBN4.ipynb` | a Google Drive folder, `/content/drive/MyDrive/Academics/Winter 2026/Bayes/Group Proj Personal/CMAPSSData/`, after `drive.mount` |
| `notebooks/hierarchical_bayesian_model(hbm)/Hirechical Bayesian Model.ipynb` | the working directory; it also writes `train_rul_predictions.csv` and `test_rul_predictions.csv` there and reads them back |

`staff/check_scores.py` takes the folder as an argument instead:

```sh
python staff/check_scores.py "path/to/CMAPSSData"
```
