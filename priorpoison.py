"""PriorPoison: data-poisoning, backdoor and defense experiments on TabPFN.

TabPFN does in-context learning: the training rows are part of the model input,
no weights are updated. So an attacker who can touch the training rows can steer
predictions. We measure how much, versus XGBoost / LogReg / kNN.

Usage:
  python priorpoison.py --dataset breast_cancer --backend local
  python priorpoison.py --dataset blood-transfusion-service-center --backend client
  python priorpoison.py --backend dummy      # smoke test, no TabPFN needed
"""
import argparse, json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml, load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier


# ---------------------------------------------------------------- data
def load_data(name, n_train, seed):
    if name == "breast_cancer":
        X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    else:
        X, y = fetch_openml(name, as_frame=True, return_X_y=True)
    X = X.copy()
    for c in X.columns:
        if X[c].dtype == object or str(X[c].dtype) == "category":
            X[c] = pd.factorize(X[c])[0]
    X = X.astype(float).to_numpy()
    y = LabelEncoder().fit_transform(np.asarray(y))
    assert len(np.unique(y)) == 2, "binary datasets only"
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    if len(Xtr) > n_train:
        Xtr, _, ytr, _ = train_test_split(Xtr, ytr, train_size=n_train, stratify=ytr, random_state=seed)
    return Xtr, ytr, Xte, yte


# -------------------------------------------------------------- models
def make_tabpfn(backend):
    if backend == "client":
        from tabpfn_client import TabPFNClassifier   # hosted API (needs login/token)
    else:
        from tabpfn import TabPFNClassifier          # local weights
    # Select the 3.5 checkpoint here if your version needs it (see Prior Labs docs).
    return TabPFNClassifier()


def model_zoo(backend, seed):
    zoo = {
        "XGBoost": lambda: XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=seed, verbosity=0),
        "LogReg": lambda: make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=2000)),
        "kNN": lambda: make_pipeline(SimpleImputer(), StandardScaler(), KNeighborsClassifier(5)),
    }
    if backend == "dummy":
        zoo["TabPFN"] = lambda: RandomForestClassifier(200, random_state=seed)  # stand-in for testing
    else:
        zoo["TabPFN"] = lambda: make_tabpfn(backend)
    return zoo


def fit_predict(factory, Xtr, ytr, Xte):
    m = factory()
    m.fit(Xtr, ytr)
    return m.predict(Xte), m.predict_proba(Xte)


# ------------------------------------------------------------- attacks
def random_flip(y, rate, rng):
    y = y.copy()
    idx = rng.choice(len(y), int(rate * len(y)), replace=False)
    y[idx] = 1 - y[idx]
    return y


def targeted_poison(X, y, Xt, yt, k, rng, noise=0.05):
    """Insert k noisy copies of each target point, labelled with the WRONG class."""
    if k == 0:
        return X, y
    sd = X.std(0) + 1e-9
    P = np.repeat(Xt, k, axis=0)
    P = P + rng.normal(0, noise, P.shape) * sd
    py = np.repeat(1 - yt, k)
    return np.vstack([X, P]), np.concatenate([y, py])


def backdoor_poison(X, y, rate, trig_feat, trig_val, target, rng):
    """Stamp a trigger on rows of the other class and relabel them as `target`."""
    if rate == 0:
        return X, y
    X, y = X.copy(), y.copy()
    cand = np.where(y != target)[0]
    idx = rng.choice(cand, max(1, int(rate * len(y))), replace=False)
    X[idx, trig_feat] = trig_val
    y[idx] = target
    return X, y


# ------------------------------------------------------------- defenses
def oof_filter(factory, X, y, thr=0.2, folds=5, seed=0):
    """Drop rows whose label gets low probability from a model trained on the other folds."""
    keep = np.ones(len(y), bool)
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        _, p = fit_predict(factory, X[tr], y[tr], X[te])
        keep[te] = p[np.arange(len(te)), y[te]] >= thr
    return X[keep], y[keep], keep


def dedup_filter(X, y, eps_scale=0.15, max_neighbors=1):
    """Drop rows sitting inside a tight cluster of near-duplicates (kills 'copy injection')."""
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    eps = eps_scale * np.sqrt(Z.shape[1])
    D = np.linalg.norm(Z[:, None] - Z[None], axis=-1)
    n_close = (D < eps).sum(1) - 1
    keep = n_close <= max_neighbors
    return X[keep], y[keep], keep


# ---------------------------------------------------------- experiments
def run(args):
    rng = np.random.default_rng(args.seed)
    Xtr, ytr, Xte, yte = load_data(args.dataset, args.n_train, args.seed)
    zoo = model_zoo(args.backend, args.seed)
    rows = []

    def log(**kw):
        rows.append(kw); print(kw, flush=True)

    # pick targets: test points, attacker wants the opposite label
    tidx = rng.choice(len(Xte), min(args.n_targets, len(Xte)), replace=False)
    Xt, yt = Xte[tidx], yte[tidx]

    # trigger feature for backdoor: highest variance-normalised range feature
    j = int(np.argmax(Xtr.std(0)))
    trig_val = Xtr[:, j].max() + Xtr[:, j].std()
    target_cls = 1

    for name, f in zoo.items():
        print(f"== {name}")
        # 1) random label flipping
        for r in args.flip_rates:
            yp = random_flip(ytr, r, rng)
            pred, _ = fit_predict(f, Xtr, yp, Xte)
            log(attack="label_flip", model=name, strength=r, metric="clean_acc", value=float((pred == yte).mean()))
        # 2) targeted copy-injection
        for k in args.copies:
            Xp, yp = targeted_poison(Xtr, ytr, Xt, yt, k, rng)
            pred, _ = fit_predict(f, Xp, yp, np.vstack([Xt, Xte]))
            pt, pe = pred[:len(Xt)], pred[len(Xt):]
            log(attack="targeted", model=name, strength=k, metric="attack_success", value=float((pt == 1 - yt).mean()))
            log(attack="targeted", model=name, strength=k, metric="clean_acc", value=float((pe == yte).mean()))
            if k == args.defend_k:
                for dname, fn in [("dedup", lambda X, y: dedup_filter(X, y)[:2]),
                                  ("oof", lambda X, y: oof_filter(f, X, y, seed=args.seed)[:2])]:
                    Xd, yd = fn(Xp, yp)
                    pd_, _ = fit_predict(f, Xd, yd, Xt)
                    log(attack="targeted", model=name, strength=k, metric=f"attack_success_after_{dname}",
                        value=float((pd_ == 1 - yt).mean()))
        # 3) backdoor
        Xn, ynn = Xte[yte != target_cls], yte[yte != target_cls]
        for r in args.bd_rates:
            Xp, yp = backdoor_poison(Xtr, ytr, r, j, trig_val, target_cls, rng)
            Xtrig = Xn.copy(); Xtrig[:, j] = trig_val
            pred_trig, _ = fit_predict(f, Xp, yp, np.vstack([Xtrig, Xn]))
            log(attack="backdoor", model=name, strength=r, metric="ASR_with_trigger",
                value=float((pred_trig[:len(Xtrig)] == target_cls).mean()))
            log(attack="backdoor", model=name, strength=r, metric="no_trigger_rate",
                value=float((pred_trig[len(Xtrig):] == target_cls).mean()))

    df = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    df.to_csv(f"{args.out}/results_{args.dataset}.csv", index=False)
    plot(df, args)
    return df


def plot(df, args):
    specs = [("label_flip", "clean_acc", "Label-flip rate", "Test accuracy"),
             ("targeted", "attack_success", "Poison copies per target", "Targeted attack success"),
             ("backdoor", "ASR_with_trigger", "Backdoor poison rate", "Backdoor ASR")]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, (atk, met, xl, yl) in zip(ax, specs):
        for m, g in df[(df.attack == atk) & (df.metric == met)].groupby("model"):
            g = g.sort_values("strength")
            a.plot(g.strength, g.value, marker="o", lw=3 if m == "TabPFN" else 1.5, label=m)
        a.set_xlabel(xl); a.set_ylabel(yl); a.grid(alpha=.3)
    ax[0].legend(); fig.suptitle(f"PriorPoison — {args.dataset}")
    fig.tight_layout(); fig.savefig(f"{args.out}/priorpoison_{args.dataset}.png", dpi=150)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="breast_cancer")
    p.add_argument("--backend", choices=["local", "client", "dummy"], default="local")
    p.add_argument("--n_train", type=int, default=500)
    p.add_argument("--n_targets", type=int, default=20)
    p.add_argument("--flip_rates", type=float, nargs="+", default=[0, .05, .1, .2, .3])
    p.add_argument("--copies", type=int, nargs="+", default=[0, 1, 2, 3, 5])
    p.add_argument("--bd_rates", type=float, nargs="+", default=[0, .01, .02, .05, .1])
    p.add_argument("--defend_k", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results")
    run(p.parse_args())
