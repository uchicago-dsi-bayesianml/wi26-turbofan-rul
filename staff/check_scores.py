#!/usr/bin/env python3
"""Course-staff check of the scores this project reports on NASA C-MAPSS FD001.

The authors' notebooks are not run. The script reads the FD001 text files, refits one
of the authors' models from scratch, and reads the source and saved outputs of the
committed notebooks. It prints five things:

  1. The PHM08 scoring function (Saxena et al. 2008) next to the swapped version, and
     which notebooks use which. The early branch must divide by 13 and the late branch
     by 10, so that late predictions cost more.
  2. The test labels that the BRNN notebook (brnn_v2.ipynb) scores against, compared
     with RUL_FD001.txt, and its ten "worst offenders" re-read against the real truth.
  3. A refit of the random forest + quantile gradient boosting model that the authors
     call BSTS (bsts_kf_rul_comparison_updated.ipynb), with the RUL cap of 120 that was
     chosen on the test set and with the notebook's original cap of 125, scored both ways.
  4. The Kalman-filter grid search over Q and R in the same notebook, read from its
     committed source and output, and its fixed RUL interval.
  5. A persistence baseline (next bin = current bin) for the DBN's next-step prediction
     of the 5-bin sensor_11 state (DBN1.ipynb), and a check of the discretizer warnings
     saved in that notebook against binning the data once and twice.

Usage:  python staff/check_scores.py PATH/TO/CMAPSSData [--repo PATH]
        (the folder that holds train_FD001.txt, test_FD001.txt and RUL_FD001.txt;
        see data/README.md for the download)

Needs NumPy and scikit-learn. Takes under a minute on a laptop.
"""
import argparse
import json
import pathlib
import re

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

NB = {
    "brnn_v2": "notebooks/bayesian_recurrent_neural_network(brnn)/brnn_v2.ipynb",
    "bsts": "notebooks/bayesian_time_series_model(bsts)/bsts_kf_rul_comparison_updated.ipynb",
    "dbn1": "notebooks/dynamic_bayesian_network_(dbn)/DBN1.ipynb",
}


# ---------------------------------------------------------------- data and scores
def load_fd001(folder):
    """Columns: 0 unit, 1 cycle, 2-4 operational settings, 5-25 sensors s1..s21."""
    folder = pathlib.Path(folder)
    train = np.loadtxt(folder / "train_FD001.txt")
    test = np.loadtxt(folder / "test_FD001.txt")
    rul = np.loadtxt(folder / "RUL_FD001.txt")
    return train, test, rul


def sensor(k):
    """Column index of sensor s<k>."""
    return 4 + k


def nasa_score(pred, true):
    """PHM08 score, d = pred - true: exp(-d/13) - 1 if d < 0 (early), exp(d/10) - 1 if late."""
    d = np.asarray(pred, float) - np.asarray(true, float)
    return float(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1).sum())


def nasa_score_swapped(pred, true):
    """The version in brnn_v1/v2, the HBM, BSTS and GRU notebooks: early /10, late /13."""
    d = np.asarray(pred, float) - np.asarray(true, float)
    return float(np.where(d < 0, np.exp(-d / 10.0) - 1, np.exp(d / 13.0) - 1).sum())


def last_rows(data):
    """Index of the last row of every unit, in unit order (the files are sorted)."""
    units = data[:, 0]
    return np.flatnonzero(np.r_[units[1:] != units[:-1], True])


def code_and_outputs(path, index):
    nb = json.loads(path.read_text(encoding="utf-8"))
    cell = nb["cells"][index]
    text = "".join(
        "".join(o.get("text", [])) for o in cell.get("outputs", []) if o.get("output_type") == "stream"
    )
    return "".join(cell["source"]), text


# ---------------------------------------------------------------- 1. score formulas
def check_formulas(repo):
    print("1. NASA (PHM08) score: correct versus swapped constants")
    for d in (-20, 20):
        print(f"   one engine predicted {abs(d)} cycles {'early' if d < 0 else 'late'}: "
              f"correct {nasa_score([d], [0]):.2f}, swapped {nasa_score_swapped([d], [0]):.2f}")
    early = re.compile(r"exp\(\s*-\s*\(?\s*[A-Za-z_][\w.\[\]]*\s*\)?\s*/\s*(10|13)(?:\.0)?\b")
    print("   early-branch divisor in each notebook's code (13 is correct, 10 is swapped):")
    for p in sorted((repo / "notebooks").rglob("*.ipynb")):
        nb = json.loads(p.read_text(encoding="utf-8"))
        src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        found = sorted(set(early.findall(src)))
        if found:
            verdict = "correct" if found == ["13"] else ("SWAPPED" if found == ["10"] else "mixed")
            print(f"     {str(p.relative_to(repo)):<78} /{'/'.join(found):<3} {verdict}")
    print()


# ---------------------------------------------------------------- 2. BRNN labels
def check_brnn_labels(repo, test, rul):
    print("2. BRNN test labels (brnn_v2.ipynb, add_piecewise_rul with onset=112)")
    last = test[last_rows(test), 1]
    eol = last + rul
    label = np.minimum(rul, np.maximum(0, eol - 112))  # the notebook's label at each unit's last cycle
    shift = rul - label
    bad = shift > 0
    print(f"   units whose label differs from RUL_FD001.txt: {bad.sum()} of {len(rul)}")
    print(f"   label lower than the truth by {shift[bad].mean():.1f} cycles on average, "
          f"{shift.max():.0f} at most; these are the units observed for fewer than 112 cycles")
    _, out = code_and_outputs(repo / NB["brnn_v2"], 9)
    rows = re.findall(r"^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+(-?[\d.]+)\s+([\d.]+)\s*$", out, re.M)
    print("   the notebook's ten worst offenders (cell 10 output), against the real truth:")
    print("     unit  label used  real RUL  predicted  error vs label  error vs real")
    for u, t, p, _, _ in rows:
        u, t, p = int(u), float(t), float(p)
        assert abs(label[u - 1] - t) < 1e-9, "reconstructed label does not match the notebook"
        print(f"     {u:>4}  {t:>10.0f}  {rul[u - 1]:>8.0f}  {p:>9.1f}  {p - t:>+14.1f}  {p - rul[u - 1]:>+13.1f}")
    print("   (the labels in that table match the reconstruction above for all ten units)")
    print()


# ---------------------------------------------------------------- 3. RF + quantile GB
def check_rf(train, test, rul):
    print("3. Random forest + quantile gradient boosting ('BSTS'), refit as in cells 27 and 29")
    cols = [sensor(k) for k in (2, 3, 4, 7, 9, 12, 14, 17)]
    mu, sd = train[:, cols].mean(0), train[:, cols].std(0)
    X = np.c_[(train[:, cols] - mu) / sd, train[:, 1]]
    Xt = np.c_[(test[:, cols] - mu) / sd, test[:, 1]][last_rows(test)]
    units = train[:, 0].astype(int)
    max_cycle = np.zeros(units.max() + 1)
    np.maximum.at(max_cycle, units, train[:, 1])
    raw_rul = max_cycle[units] - train[:, 1]
    print("     cap  RMSE   MAE   score(swapped)  score(correct)  80% interval coverage  width")
    for cap, note in ((120, "chosen on the test set, as committed"), (125, "the notebook's configured cap")):
        y = np.minimum(raw_rul, cap)
        preds = [RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42 + i, n_jobs=-1)
                 .fit(X, y).predict(Xt) for i in range(5)]
        mean = np.clip(np.mean(preds, 0), 0, 2 * cap)
        lo = GradientBoostingRegressor(loss="quantile", alpha=0.10, n_estimators=100, max_depth=5,
                                       random_state=42).fit(X, y).predict(Xt)
        hi = GradientBoostingRegressor(loss="quantile", alpha=0.90, n_estimators=100, max_depth=5,
                                       random_state=42).fit(X, y).predict(Xt)
        lo, hi = np.clip(lo, 0, mean), np.clip(hi, mean, 2 * cap)
        e = mean - rul
        cover = np.mean((rul >= lo) & (rul <= hi)) * 100
        print(f"     {cap}  {np.sqrt(np.mean(e ** 2)):.2f}  {np.abs(e).mean():.2f}  "
              f"{nasa_score_swapped(mean, rul):>14.1f}  {nasa_score(mean, rul):>14.1f}  "
              f"{cover:>20.1f}%  {np.mean(hi - lo):.2f}   ({note})")
    print()


# ---------------------------------------------------------------- 4. Kalman filter grid
def check_kalman(repo):
    print("4. Kalman filter Q/R grid search (bsts_kf_rul_comparison_updated.ipynb)")
    nb = json.loads((repo / NB["bsts"]).read_text(encoding="utf-8"))
    for i, cell in enumerate(nb["cells"]):
        src = "".join(cell["source"])
        if "def evaluate_kalman_on_validation" in src:
            func = src.split("def evaluate_kalman_on_validation", 1)[1].split("\nprint(", 1)[0]
            body = func.split('"""', 2)[-1]
            call = re.search(r"EngineHealthKalmanFilter\((.*?)\n\s*\)", body, re.S).group(1)
            out = "".join("".join(o.get("text", [])) for o in cell.get("outputs", []))
            vals = re.findall(r"^\d+\.\d+\s+\d+\.\d+\s+(\d+\.\d+)", out, re.M)
            print(f"   cell {i + 1}: evaluate_kalman_on_validation{func.split(':', 1)[0].strip()}")
            print(f"   builds the filter as EngineHealthKalmanFilter({' '.join(call.split())})")
            print(f"   uses of process_noise / measurement_noise in the body: "
                  f"{body.count('process_noise')} / {body.count('measurement_noise')}")
            print(f"   validation RMSE over {len(vals)} grid points: {sorted(set(vals))}")
        if "class EngineHealthKalmanFilter" in src:
            for pat in (r"self\.kf\.Q = .*", r"self\.kf\.R = .*", r"rul_std = .*"):
                m = re.search(pat, src)
                print(f"   cell {i + 1}: {m.group(0).strip() if m else pat + ' not found'}")
    print()


# ---------------------------------------------------------------- 5. DBN baseline
def quantile_bins(tr, te, n_bins=5):
    """KBinsDiscretizer(strategy='quantile', encode='ordinal') as in scikit-learn 1.6, per column.

    Returns the binned arrays and the columns where bins were dropped (scikit-learn warns
    "Bins whose width are too small ... are removed") or the column is constant."""
    out_tr, out_te, narrow, constant = np.zeros(tr.shape, int), np.zeros(te.shape, int), [], []
    for j in range(tr.shape[1]):
        if tr[:, j].min() == tr[:, j].max():
            constant.append(j)
            continue
        edges = np.percentile(tr[:, j], np.linspace(0, 100, n_bins + 1))
        keep = np.ediff1d(edges, to_begin=np.inf) > 1e-8
        if not keep.all():
            narrow.append(j)
        edges = edges[keep]
        out_tr[:, j] = np.searchsorted(edges[1:-1], tr[:, j], side="right")
        out_te[:, j] = np.searchsorted(edges[1:-1], te[:, j], side="right")
    return out_tr, out_te, narrow, constant


def check_dbn_baseline(repo, train, test):
    print("5. Persistence baseline for the DBN's next-step sensor_11 bin (DBN1.ipynb)")
    # The 17 features of cell 4: op settings 1-2 and the sensors that are not dropped.
    cols = [2, 3] + [sensor(k) for k in (2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21)]
    j11 = cols.index(sensor(11))
    lo, hi = train[:, cols].min(0), train[:, cols].max(0)
    scale = 1.0 / (hi - lo)  # MinMaxScaler fitted on the training data, cell 4
    tr, te = train[:, cols] * scale - lo * scale, test[:, cols] * scale - lo * scale
    units = test[:, 0]

    def persistence(bins):
        return 100 * np.mean([np.mean(b[1:] == b[:-1]) for b in (bins[units == u, j11] for u in np.unique(units))])

    tr1, te1, narrow1, const1 = quantile_bins(tr, te)    # cell 5 run once
    tr2, te2, narrow2, const2 = quantile_bins(tr1, te1)  # cell 5 run a second time on its own output
    pairs = np.zeros((5, 5))
    for u in np.unique(train[:, 0]):
        b = tr1[train[:, 0] == u, j11]
        np.add.at(pairs, (b[:-1], b[1:]), 1)
    _, out = code_and_outputs(repo / NB["dbn1"], 10)
    m = re.search(r"Overall Mean Accuracy: ([\d.]+%)", out)
    _, warn = code_and_outputs(repo / NB["dbn1"], 4)
    nb_narrow = sorted({int(x) for x in re.findall(r"too small \(i\.e\., <= 1e-8\) in feature (\d+)", warn)})
    nb_const = sorted({int(x) for x in re.findall(r"Feature (\d+) is constant", warn)})
    print(f"   committed warnings of cell 5: bins dropped in features {nb_narrow}, constant {nb_const}")
    print(f"   binning once:  bins dropped in {narrow1}, constant {const1}")
    print(f"   binning twice: bins dropped in {narrow2}, constant {const2}")
    print(f"   most likely next bin from bins 0-4 in the training data: {pairs.argmax(1).tolist()} "
          f"(staying put, with probability {np.round(np.diag(pairs) / pairs.sum(1), 2).tolist()})")
    print(f"   DBN, mean per-unit accuracy over the test units (cell 11 output): {m.group(1) if m else 'not found'}")
    print(f"   persistence (next bin = current bin), same averaging: {persistence(te1):.2f}% "
          f"with bins made once, {persistence(te2):.2f}% with bins made twice")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmapss", help="folder with train_FD001.txt, test_FD001.txt, RUL_FD001.txt")
    ap.add_argument("--repo", default=pathlib.Path(__file__).resolve().parents[1], type=pathlib.Path)
    a = ap.parse_args()
    train, test, rul = load_fd001(a.cmapss)
    print(f"FD001: {len(np.unique(train[:, 0]))} training engines ({len(train)} rows), "
          f"{len(rul)} test engines ({len(test)} rows)\n")
    check_formulas(a.repo)
    check_brnn_labels(a.repo, test, rul)
    check_rf(train, test, rul)
    check_kalman(a.repo)
    check_dbn_baseline(a.repo, train, test)


if __name__ == "__main__":
    main()
