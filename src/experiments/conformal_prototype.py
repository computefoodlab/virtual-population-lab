"""
Standalone conformal recalibration of transfer intervals.

The conformal recalibration is shipped inside src/evaluation/loro.py (columns
coverage_*_conf). This script is the standalone prototype behind that code: for
each train-group it calibrates a per-feature conformal interval width on the NEAR
held-out reals (target 0.90) and transfers the SAME width off-support (FAR),
reporting raw vs conformal coverage for NEAR and FAR.

Run from repo root:  python -m src.experiments.conformal_prototype
"""
import sys, os; sys.path.insert(0, os.getcwd())
import numpy as np, torch, logging
logging.disable(logging.CRITICAL)
from sklearn.model_selection import train_test_split
from src.config_loader import load_config
from src.data_loading import load_data
from src.evaluation import loro as L


def ecdf(g, y):
    g = np.sort(g); return np.searchsorted(g, y, side="right") / len(g)


def central_cov(g, y, level=0.90):
    lo, hi = np.quantile(g, (1 - level) / 2), np.quantile(g, (1 + level) / 2)
    return float(np.mean((y >= lo) & (y <= hi)))


def conformal_shat(g, y_cal, level=0.90):
    s = np.abs(ecdf(g, y_cal) - 0.5)
    m = len(s); k = min(max(int(np.ceil((m + 1) * level)) - 1, 0), m - 1)
    return min(float(np.sort(s)[k]), 0.5)


def conf_cov(g, y_eval, shat):
    lo, hi = np.quantile(g, 0.5 - shat), np.quantile(g, 0.5 + shat)
    return float(np.mean((y_eval >= lo) & (y_eval <= hi)))


def run(cfg_name, seeds=(41, 42, 43)):
    cfg = load_config(cfg_name); G = cfg.LORO_GROUP
    df = load_data(cfg); feats = [f for f in cfg.FEATURES if f != G]
    groups = sorted(df[G].unique()); sizes = df[G].value_counts()
    trainable = [g for g in groups if sizes[g] >= L.MIN_TRAIN_GROUP]
    acc = {k: [] for k in ["raw_near", "conf_near", "raw_far", "conf_far"]}
    for s in seeds:
        L.SEED = s; np.random.seed(s); torch.manual_seed(s)
        for tg in trainable:
            reg = df[df[G] == tg]
            tr, te = train_test_split(reg, test_size=L.NEAR_TEST_FRAC, random_state=s)
            models, scaler = L._train_ensemble(tr, feats, G, groups)
            gen_near, _ = L._generate(models, scaler, groups.index(tg), groups, feats)
            rn, cn, shats = [], [], {}
            for f in feats:
                g = gen_near[f].values; yn = te[f].values
                rn.append(central_cov(g, yn)); shats[f] = conformal_shat(g, yn)
                cn.append(conf_cov(g, yn, shats[f]))
            acc["raw_near"].append(np.mean(rn)); acc["conf_near"].append(np.mean(cn))
            for og in groups:
                if og == tg:
                    continue
                gen_far, _ = L._generate(models, scaler, groups.index(og), groups, feats)
                yf = df[df[G] == og]
                rf = [central_cov(gen_far[f].values, yf[f].values) for f in feats]
                cf = [conf_cov(gen_far[f].values, yf[f].values, shats[f]) for f in feats]
                acc["raw_far"].append(np.mean(rf)); acc["conf_far"].append(np.mean(cf))
    print(f"\n=== {cfg_name}  (target 0.90) ===")
    for k in ["raw_near", "conf_near", "raw_far", "conf_far"]:
        v = np.array(acc[k]); print(f"  {k:<10} {v.mean():.3f} ± {v.std(ddof=1):.3f}")


def main():
    for ds in ["mango_composition", "biofood_safou_region"]:
        run(ds)


if __name__ == "__main__":
    main()
