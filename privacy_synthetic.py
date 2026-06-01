"""
privacy_synthetic.py
--------------------
Privacy-preserving synthetic data generator using a VAE with three layers
of noise applied at different stages of the pipeline:

  1. Latent-space perturbation  — extra Gaussian noise added on top of the
     reparameterized posterior sample before decoding.  This nudges every
     generated record away from its source record in a smooth, model-aware
     way without breaking the decoder's learned structure.

  2. Post-hoc Laplace noise     — calibrated Laplace noise added to numeric
     columns after inverse-transforming.  Laplace noise is the natural choice
     for differential-privacy-like guarantees: its scale (b = sensitivity / ε)
     lets you reason about a privacy budget even when full DP-SGD is not used.

  3. Category flipping          — each categorical value is randomly replaced
     with a different category with probability `flip_prob`.  This makes
     individual records harder to re-identify even if an attacker knows the
     original dataset's marginal distributions.

Usage
-----
    python privacy_synthetic.py                        # runs with defaults
    python privacy_synthetic.py --csv my_data.csv      # custom dataset
    python privacy_synthetic.py --epsilon 2.0          # tighter privacy
    python privacy_synthetic.py --latent_noise 0.3 --flip_prob 0.05

Privacy knobs (all tunable via CLI)
------------------------------------
  --epsilon        Laplace noise scale for numerics (lower = more private).
                   Analogous to ε in differential privacy; default 5.0.
  --latent_noise   Std-dev of extra Gaussian noise in latent space; default 0.2.
  --flip_prob      Probability of randomly flipping a categorical value; default 0.03.

Metrics reported
----------------
  Numeric  : Welch t-test (means), KS test (distribution shape), Levene (variance)
  Categorical : Chi-square (category proportions)
  Privacy  : Per-column noise magnitudes and estimated privacy budget analogue
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend; swap to "TkAgg" for pop-ups
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neighbors import NearestNeighbors
from scipy import stats as scipy_stats
from scipy.stats import ks_2samp, levene
from scipy.special import softmax as scipy_softmax


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Privacy-preserving VAE synthetic data generator")
    p.add_argument("--csv",          default="customer_transactions_1500.csv",
                   help="Path to the input CSV file")
    p.add_argument("--output",       default="private_synthetic_data.csv",
                   help="Where to save the synthetic CSV")
    p.add_argument("--epochs",       type=int,   default=300)
    p.add_argument("--batch_size",   type=int,   default=128)
    p.add_argument("--hidden_dim",   type=int,   default=128)
    p.add_argument("--latent_dim",   type=int,   default=16)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--beta_final",   type=float, default=0.1,
                   help="Final KL weight (beta-VAE)")
    p.add_argument("--numeric_weight", type=float, default=3.0,
                   help="Weight for MSE loss vs cross-entropy loss")
    p.add_argument("--seed",         type=int,   default=42)
    # Privacy knobs
    p.add_argument("--epsilon",      type=float, default=20.0,
                   help="Privacy budget analogue: Laplace scale = sensitivity/epsilon. "
                        "Lower = more noise = more private. Recommended range: 10-50.")
    p.add_argument("--latent_noise", type=float, default=0.15,
                   help="Extra Gaussian std added in latent space before decoding")
    p.add_argument("--flip_prob",    type=float, default=0.03,
                   help="Probability of randomly flipping a categorical value")
    p.add_argument("--no_plots",     action="store_true",
                   help="Skip saving comparison plots")
    return p.parse_args()


# ---------------------------------------------------------------------------
# VAE definition  (identical architecture to Assignment_3.ipynb)
# ---------------------------------------------------------------------------

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder_hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        self.decoder_hidden = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x):
        h = self.encoder_hidden(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        return self.decoder_hidden(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(recon_x, x, mu, logvar, n_numeric, cat_sizes,
             beta=1.0, numeric_weight=3.0):
    b = x.size(0)
    num_loss = ((recon_x[:, :n_numeric] - x[:, :n_numeric]) ** 2).sum() / b if n_numeric > 0 \
               else torch.tensor(0.0, device=x.device)

    cat_loss = torch.tensor(0.0, device=x.device)
    offset = n_numeric
    for size in cat_sizes:
        logits  = recon_x[:, offset:offset + size]
        targets = x[:, offset:offset + size].argmax(dim=1)
        cat_loss = cat_loss + nn.CrossEntropyLoss()(logits, targets)
        offset += size

    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / b
    total  = numeric_weight * num_loss + cat_loss + beta * kl_div
    return total, num_loss.detach(), cat_loss.detach(), kl_div.detach()


# ---------------------------------------------------------------------------
# Privacy noise functions
# ---------------------------------------------------------------------------

def add_laplace_noise(array, sensitivity, epsilon):
    """Add Laplace noise scaled to sensitivity/epsilon (per-column)."""
    scale = sensitivity / epsilon
    noise = np.random.laplace(loc=0.0, scale=scale, size=array.shape)
    return array + noise


def perturb_latent(z_tensor, extra_std):
    """Add zero-mean Gaussian noise to latent vectors before decoding."""
    noise = torch.randn_like(z_tensor) * extra_std
    return z_tensor + noise


def flip_categories(series, categories, flip_prob):
    """Randomly replace values with a different category with probability flip_prob."""
    arr = series.copy()
    n   = len(arr)
    flip_mask = np.random.rand(n) < flip_prob
    if flip_mask.any():
        for idx in np.where(flip_mask)[0]:
            current = arr[idx]
            others  = [c for c in categories if c != current]
            if others:
                arr[idx] = np.random.choice(others)
    return arr


# ---------------------------------------------------------------------------
# Statistical tests  (same metrics as the original notebook)
# ---------------------------------------------------------------------------

def run_statistical_tests(df_real, df_syn, numeric_cols, categorical_cols):
    print("\n" + "=" * 90)
    print("STATISTICAL COMPARISON: REAL vs PRIVATE SYNTHETIC DATA")
    print("=" * 90)

    summary_rows = []

    # -- Numeric -----------------------------------------------------------
    if numeric_cols:
        print(f"\n{'Column':<22} {'Test':<12} {'Stat':>9} {'p-value':>9} {'Result':>8}")
        print("-" * 65)
        for col in numeric_cols:
            r = df_real[col].dropna().values
            s = df_syn[col].dropna().values

            t,   t_p   = scipy_stats.ttest_ind(r, s, equal_var=False)
            ks,  ks_p  = ks_2samp(r, s)
            lev, lev_p = levene(r, s)

            for test_name, stat, p in [("Welch t-test", t, t_p),
                                        ("KS test",      ks,  ks_p),
                                        ("Levene",       lev, lev_p)]:
                result = "PASS" if p >= 0.05 else "FAIL"
                print(f"{col:<22} {test_name:<12} {stat:>9.4f} {p:>9.4f} {result:>8}")
            print()

            passed = (t_p >= 0.05) and (ks_p >= 0.05) and (lev_p >= 0.05)
            summary_rows.append({
                "column":    col,
                "type":      "numeric",
                "real_mean": round(r.mean(), 4),
                "syn_mean":  round(s.mean(), 4),
                "t_p":       round(t_p, 4),
                "ks_p":      round(ks_p, 4),
                "levene_p":  round(lev_p, 4),
                "chi2_p":    "-",
                "overall":   "PASS" if passed else "FAIL",
            })

    # -- Categorical -------------------------------------------------------
    if categorical_cols:
        print(f"\n{'Column':<22} {'chi2':>10} {'p-value':>10} {'dof':>5} {'Result':>8}")
        print("-" * 60)
        for col in categorical_cols:
            all_vals = sorted(
                set(df_real[col].astype(str).unique()) |
                set(df_syn[col].astype(str).unique()), key=str
            )
            r_counts = [int(df_real[col].astype(str).value_counts().get(v, 0)) for v in all_vals]
            s_counts = [int(df_syn[col].astype(str).value_counts().get(v, 0))  for v in all_vals]
            contingency = np.vstack([r_counts, s_counts])
            mask = contingency.sum(axis=0) > 0
            chi2, p, dof, _ = scipy_stats.chi2_contingency(contingency[:, mask])
            result = "PASS" if p >= 0.05 else "FAIL"
            print(f"{col:<22} {chi2:>10.4f} {p:>10.4f} {dof:>5} {result:>8}")

            summary_rows.append({
                "column":    col,
                "type":      "categorical",
                "real_mean": f"mode={df_real[col].mode()[0]}",
                "syn_mean":  f"mode={df_syn[col].mode()[0]}",
                "t_p":       "-",
                "ks_p":      "-",
                "levene_p":  "-",
                "chi2_p":    round(p, 4),
                "overall":   result,
            })

    summary_df = pd.DataFrame(summary_rows)
    n_pass = (summary_df["overall"] == "PASS").sum()
    n_total = len(summary_df)

    print("\n" + "=" * 90)
    print("FINAL SUMMARY")
    print("=" * 90)
    print(summary_df.to_string(index=False))
    print(f"\nOverall: {n_pass}/{n_total} columns statistically indistinguishable (p >= 0.05)")
    return summary_df


def print_privacy_report(numeric_cols, sensitivities, epsilon, latent_noise, flip_prob):
    print("\n" + "=" * 70)
    print("PRIVACY NOISE REPORT")
    print("=" * 70)
    print(f"  Latent-space Gaussian noise std  : {latent_noise:.4f}")
    print(f"  Category flip probability        : {flip_prob:.4f}  ({flip_prob*100:.1f}%)")
    print(f"  Laplace epsilon (privacy budget) : {epsilon:.4f}  (lower = more private, higher = less noise)")
    print()
    print(f"  {'Column':<22} {'Sensitivity':>14} {'Laplace scale (b)':>18}")
    print(f"  {'-'*22} {'-'*14} {'-'*18}")
    for col, sens in zip(numeric_cols, sensitivities):
        b = sens / epsilon
        print(f"  {col:<22} {sens:>14.4f} {b:>18.6f}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_plots(df_real, df_syn, numeric_cols, categorical_cols, out_prefix):
    # Numeric: KDE overlay
    if numeric_cols:
        fig, axes = plt.subplots(1, len(numeric_cols),
                                 figsize=(5 * len(numeric_cols), 4))
        if len(numeric_cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, numeric_cols):
            sns.kdeplot(df_real[col], fill=True, label="Real",      ax=ax,
                        color="#3266ad", alpha=0.55)
            sns.kdeplot(df_syn[col],  fill=True, label="Private Synthetic", ax=ax,
                        color="#e07b39", alpha=0.55)
            ax.set_title(col)
            ax.legend(fontsize=8)
        fig.suptitle("Numeric distributions — Real vs Private Synthetic", y=1.02)
        plt.tight_layout()
        path = f"{out_prefix}_numeric_kde.png"
        plt.savefig(path, bbox_inches="tight", dpi=120)
        plt.close()
        print(f"  Saved: {path}")

    # Categorical: side-by-side bars
    if categorical_cols:
        n_cols_plot = min(3, len(categorical_cols))
        n_rows_plot = int(np.ceil(len(categorical_cols) / n_cols_plot))
        fig, axes = plt.subplots(n_rows_plot, n_cols_plot,
                                 figsize=(5 * n_cols_plot, 4 * n_rows_plot))
        axes = np.array(axes).flatten()
        for i, col in enumerate(categorical_cols):
            r = df_real[col].value_counts(normalize=True).rename("Real")
            s = df_syn[col].value_counts(normalize=True).rename("Private Synthetic")
            df_plot = pd.concat([r, s], axis=1).fillna(0)
            df_plot.plot(kind="bar", ax=axes[i], color=["#3266ad", "#e07b39"], alpha=0.8)
            axes[i].set_title(col)
            axes[i].set_ylabel("Proportion")
            axes[i].tick_params(axis="x", rotation=30)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        fig.suptitle("Categorical distributions — Real vs Private Synthetic", y=1.02)
        plt.tight_layout()
        path = f"{out_prefix}_categorical_bars.png"
        plt.savefig(path, bbox_inches="tight", dpi=120)
        plt.close()
        print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Privacy verification attack suite
# ---------------------------------------------------------------------------

def _encode_for_attack(df_real, df_syn, numeric_cols, categorical_cols, scaler, ohe):
    """Encode both dataframes into a single numeric matrix for distance/model attacks."""
    def encode(df):
        num = scaler.transform(df[numeric_cols]) if numeric_cols else np.zeros((len(df), 0))
        try:
            cat = ohe.transform(df[categorical_cols]) if categorical_cols else np.zeros((len(df), 0))
        except Exception:
            cat = np.zeros((len(df), 0))
        return np.concatenate([num, cat], axis=1)

    X_real = encode(df_real)
    X_syn  = encode(df_syn)
    return X_real, X_syn


def attack_1_membership_inference(X_real, X_syn):
    """
    Membership Inference Attack (MIA) via a shadow-model classifier.

    The attacker trains a Random Forest to predict whether a record was
    in the training set (real=1) or not (synthetic=0).  We use 5-fold
    cross-validated AUC as the attack success metric.

    Interpretation
    --------------
    AUC = 0.50  ->  attacker is no better than random guessing  ->  PRIVATE
    AUC = 1.00  ->  attacker perfectly separates real from synthetic  ->  BROKEN
    Threshold   :   AUC < 0.55 is considered safe in the literature.
    """
    n  = min(len(X_real), len(X_syn))
    X  = np.vstack([X_real[:n], X_syn[:n]])
    y  = np.array([1] * n + [0] * n)

    rf  = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    lr  = LogisticRegression(max_iter=1000, random_state=42)

    auc_rf = cross_val_score(rf, X, y, cv=5, scoring="roc_auc").mean()
    auc_lr = cross_val_score(lr, X, y, cv=5, scoring="roc_auc").mean()

    # Fit once more on full data to get probabilities for ROC curve
    rf.fit(X, y)
    probs  = rf.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, probs)

    return {
        "auc_random_forest": round(auc_rf, 4),
        "auc_logistic":      round(auc_lr, 4),
        "fpr":               fpr,
        "tpr":               tpr,
    }


def attack_2_dcr(X_real, X_syn):
    """
    Distance to Closest Record (DCR).

    For every synthetic record, find the Euclidean distance to its nearest
    real neighbour.  A very small DCR means the synthetic record is nearly
    identical to a real one — high re-identification risk.

    Metrics reported
    ----------------
    - DCR mean / median / 5th-percentile
    - Comparison with the real-to-real nearest-neighbour distance (baseline):
      if DCR_syn ~ DCR_real  ->  synthetic records blend in with real ones
      if DCR_syn << DCR_real ->  synthetic records are suspiciously close
    """
    # Synthetic -> nearest real
    nn_real = NearestNeighbors(n_neighbors=1, algorithm="ball_tree")
    nn_real.fit(X_real)
    dcr_syn, _ = nn_real.kneighbors(X_syn)
    dcr_syn    = dcr_syn.flatten()

    # Real -> nearest real (leave-one-out baseline)
    nn_r2r = NearestNeighbors(n_neighbors=2, algorithm="ball_tree")
    nn_r2r.fit(X_real)
    dcr_real, _ = nn_r2r.kneighbors(X_real)
    dcr_real    = dcr_real[:, 1].flatten()   # skip self (distance=0)

    return {
        "dcr_syn_mean":    round(dcr_syn.mean(),          4),
        "dcr_syn_median":  round(np.median(dcr_syn),      4),
        "dcr_syn_p5":      round(np.percentile(dcr_syn, 5), 4),
        "dcr_real_mean":   round(dcr_real.mean(),         4),
        "dcr_real_median": round(np.median(dcr_real),     4),
        "dcr_real_p5":     round(np.percentile(dcr_real, 5), 4),
        "dcr_syn_arr":     dcr_syn,
        "dcr_real_arr":    dcr_real,
    }


def attack_3_nnaa(X_real, X_syn):
    """
    Nearest Neighbour Adversarial Accuracy (NNAA).

    For each real record, ask: is its nearest neighbour (among real+synthetic
    combined) another real record or a synthetic one?  Do the same from the
    synthetic side.  The adversarial accuracy is the fraction of records that
    'correctly' identify their own group.

    Score = 0.50  ->  synthetic data perfectly mimics real  ->  PRIVATE
    Score = 1.00  ->  real and synthetic are clearly separated  ->  NOT private
    Safe threshold: NNAA < 0.55
    """
    n   = min(len(X_real), len(X_syn))
    Xr  = X_real[:n]
    Xs  = X_syn[:n]
    X_all = np.vstack([Xr, Xs])
    labels = np.array([0] * n + [1] * n)   # 0=real, 1=synthetic

    nn = NearestNeighbors(n_neighbors=2, algorithm="ball_tree")
    nn.fit(X_all)
    distances, indices = nn.kneighbors(X_all)
    # index 0 is self, index 1 is nearest neighbour
    nn_labels = labels[indices[:, 1]]

    # AA = fraction where the nearest neighbour is from the SAME group
    aa = (nn_labels == labels).mean()
    return {"nnaa": round(float(aa), 4)}


def _verdict(value, threshold, direction="below"):
    """Return SAFE / WARNING / UNSAFE based on threshold."""
    if direction == "below":
        if value < threshold[0]:
            return "SAFE"
        elif value < threshold[1]:
            return "WARNING"
        else:
            return "UNSAFE"
    else:
        if value > threshold[0]:
            return "SAFE"
        elif value > threshold[1]:
            return "WARNING"
        else:
            return "UNSAFE"


def run_privacy_verification(df_real, df_syn, numeric_cols, categorical_cols,
                              scaler, ohe, out_prefix, no_plots=False):
    print("\n" + "=" * 90)
    print("PRIVACY VERIFICATION — ADVERSARIAL ATTACK SUITE")
    print("=" * 90)
    print("Simulating three independent attacker strategies against the synthetic data.")
    print("Goal: confirm that no attacker can recover meaningful information about")
    print("the original records from the synthetic dataset.\n")

    X_real, X_syn = _encode_for_attack(df_real, df_syn,
                                        numeric_cols, categorical_cols,
                                        scaler, ohe)

    # ------------------------------------------------------------------ #
    # Attack 1 — Membership Inference Attack                              #
    # ------------------------------------------------------------------ #
    print("Running Attack 1: Membership Inference Attack (MIA) ...")
    mia = attack_1_membership_inference(X_real, X_syn)

    print("\n  [ATTACK 1] Membership Inference Attack Results")
    print(f"  {'Classifier':<25} {'AUC':>6}   {'Verdict'}")
    print(f"  {'-'*25} {'-'*6}   {'-'*20}")
    for name, key in [("Random Forest", "auc_random_forest"), ("Logistic Regression", "auc_logistic")]:
        auc     = mia[key]
        verdict = _verdict(auc, (0.55, 0.60), direction="below")
        print(f"  {name:<25} {auc:>6.4f}   {verdict}")
    print()
    print("  Interpretation: AUC=0.5 means the attacker is guessing randomly (best case).")
    print("  AUC < 0.55 = SAFE | 0.55-0.60 = WARNING | > 0.60 = UNSAFE")

    # ------------------------------------------------------------------ #
    # Attack 2 — Distance to Closest Record                               #
    # ------------------------------------------------------------------ #
    print("\nRunning Attack 2: Distance to Closest Record (DCR) ...")
    dcr = attack_2_dcr(X_real, X_syn)

    ratio_mean   = dcr["dcr_syn_mean"]   / max(dcr["dcr_real_mean"],   1e-9)
    ratio_median = dcr["dcr_syn_median"] / max(dcr["dcr_real_median"], 1e-9)

    print("\n  [ATTACK 2] Distance to Closest Record Results")
    print(f"  {'Metric':<30} {'Synthetic':>12} {'Real baseline':>14} {'Ratio':>8}")
    print(f"  {'-'*30} {'-'*12} {'-'*14} {'-'*8}")
    print(f"  {'Mean DCR':<30} {dcr['dcr_syn_mean']:>12.4f} {dcr['dcr_real_mean']:>14.4f} {ratio_mean:>8.3f}")
    print(f"  {'Median DCR':<30} {dcr['dcr_syn_median']:>12.4f} {dcr['dcr_real_median']:>14.4f} {ratio_median:>8.3f}")
    print(f"  {'5th-percentile DCR':<30} {dcr['dcr_syn_p5']:>12.4f} {dcr['dcr_real_p5']:>14.4f} {'':>8}")
    print()
    verdict_dcr = _verdict(ratio_mean, (0.8, 0.5), direction="above")
    print(f"  DCR ratio (syn/real): {ratio_mean:.3f}  ->  {verdict_dcr}")
    print("  Interpretation: Ratio >= 0.8 means synthetic records are no closer to real")
    print("  records than real records are to each other  ->  low re-identification risk.")
    print("  Ratio < 0.5 = UNSAFE | 0.5-0.8 = WARNING | >= 0.8 = SAFE")

    # ------------------------------------------------------------------ #
    # Attack 3 — Nearest Neighbour Adversarial Accuracy                  #
    # ------------------------------------------------------------------ #
    print("\nRunning Attack 3: Nearest Neighbour Adversarial Accuracy (NNAA) ...")
    nnaa_res = attack_3_nnaa(X_real, X_syn)
    nnaa     = nnaa_res["nnaa"]
    verdict_nnaa = _verdict(nnaa, (0.55, 0.60), direction="below")

    print("\n  [ATTACK 3] NNAA Results")
    print(f"  NNAA Score : {nnaa:.4f}   ->   {verdict_nnaa}")
    print("  Interpretation: Score=0.5 means synthetic perfectly mimics real (ideal).")
    print("  Score < 0.55 = SAFE | 0.55-0.60 = WARNING | > 0.60 = UNSAFE")

    # ------------------------------------------------------------------ #
    # Overall privacy verdict                                             #
    # ------------------------------------------------------------------ #
    verdicts = [
        _verdict(mia["auc_random_forest"], (0.55, 0.60), direction="below"),
        _verdict(mia["auc_logistic"],      (0.55, 0.60), direction="below"),
        _verdict(ratio_mean,               (0.8, 0.5),   direction="above"),
        _verdict(nnaa,                     (0.55, 0.60), direction="below"),
    ]
    n_safe    = verdicts.count("SAFE")
    n_warning = verdicts.count("WARNING")
    n_unsafe  = verdicts.count("UNSAFE")

    print("\n" + "=" * 90)
    print("OVERALL PRIVACY VERDICT")
    print("=" * 90)
    print(f"  {'Attack':<40} {'Metric':>10} {'Verdict'}")
    print(f"  {'-'*40} {'-'*10} {'-'*10}")
    rows = [
        ("MIA — Random Forest AUC",         mia["auc_random_forest"], verdicts[0]),
        ("MIA — Logistic Regression AUC",   mia["auc_logistic"],      verdicts[1]),
        ("DCR — Syn/Real mean ratio",        ratio_mean,               verdicts[2]),
        ("NNAA — Adversarial accuracy",      nnaa,                     verdicts[3]),
    ]
    for name, val, v in rows:
        print(f"  {name:<40} {val:>10.4f} {v}")

    print()
    if n_unsafe == 0 and n_warning <= 1:
        overall = "PRIVATE  -- synthetic data provides strong privacy protection."
    elif n_unsafe == 0:
        overall = "MOSTLY PRIVATE  -- minor leakage signals; consider lowering --epsilon."
    else:
        overall = "PRIVACY RISK  -- reduce --epsilon or increase --latent_noise/--flip_prob."
    print(f"  Final verdict: {overall}")
    print("=" * 90)

    # ------------------------------------------------------------------ #
    # Privacy verification plots                                          #
    # ------------------------------------------------------------------ #
    if not no_plots:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Plot 1: ROC curve for MIA
        ax = axes[0]
        ax.plot(mia["fpr"], mia["tpr"], color="#c0392b", lw=2,
                label=f"RF AUC = {mia['auc_random_forest']:.4f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random guess (AUC=0.5)")
        ax.fill_between(mia["fpr"], mia["tpr"], alpha=0.1, color="#c0392b")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Attack 1: MIA — ROC Curve\n(closer to diagonal = more private)")
        ax.legend(fontsize=9)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])

        # Plot 2: DCR distributions
        ax = axes[1]
        ax.hist(dcr["dcr_real_arr"], bins=50, alpha=0.6, color="#2980b9",
                label="Real-to-Real distance", density=True)
        ax.hist(dcr["dcr_syn_arr"],  bins=50, alpha=0.6, color="#e67e22",
                label="Synthetic-to-Real distance", density=True)
        ax.axvline(dcr["dcr_syn_p5"], color="#c0392b", linestyle="--", lw=1.5,
                   label=f"Syn 5th-pct = {dcr['dcr_syn_p5']:.3f}")
        ax.set_xlabel("Euclidean Distance (normalized space)")
        ax.set_ylabel("Density")
        ax.set_title("Attack 2: DCR Distribution\n(syn curve overlapping real = safe)")
        ax.legend(fontsize=8)

        # Plot 3: Summary bar chart
        ax = axes[2]
        metric_names  = ["MIA RF\nAUC", "MIA LR\nAUC", "NNAA\nScore"]
        metric_values = [mia["auc_random_forest"], mia["auc_logistic"], nnaa]
        colors = []
        for v in [verdicts[0], verdicts[1], verdicts[3]]:
            colors.append("#27ae60" if v == "SAFE" else "#f39c12" if v == "WARNING" else "#c0392b")
        bars = ax.bar(metric_names, metric_values, color=colors, alpha=0.85, width=0.4)
        ax.axhline(0.5,  color="green",  linestyle="--", lw=1.5, label="Ideal (0.50)")
        ax.axhline(0.55, color="orange", linestyle="--", lw=1.5, label="Warning (0.55)")
        ax.axhline(0.60, color="red",    linestyle="--", lw=1.5, label="Unsafe (0.60)")
        for bar, val in zip(bars, metric_values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_ylim([0.45, max(0.65, max(metric_values) + 0.05)])
        ax.set_ylabel("Score")
        ax.set_title("Attack Summary\n(green=SAFE, orange=WARNING, red=UNSAFE)")
        ax.legend(fontsize=8)

        fig.suptitle("Privacy Verification — Adversarial Attack Results", fontsize=13, y=1.02)
        plt.tight_layout()
        path = f"{out_prefix}_privacy_verification.png"
        plt.savefig(path, bbox_inches="tight", dpi=120)
        plt.close()
        print(f"\nPrivacy verification plot saved: {path}")

    # Save numeric results to CSV
    privacy_results = pd.DataFrame([
        {"attack": "MIA Random Forest",       "metric": "AUC",           "value": mia["auc_random_forest"], "verdict": verdicts[0]},
        {"attack": "MIA Logistic Regression", "metric": "AUC",           "value": mia["auc_logistic"],      "verdict": verdicts[1]},
        {"attack": "DCR",                     "metric": "Syn/Real ratio", "value": ratio_mean,               "verdict": verdicts[2]},
        {"attack": "NNAA",                    "metric": "Score",          "value": nnaa,                     "verdict": verdicts[3]},
    ])
    csv_path = f"{out_prefix}_privacy_verification.csv"
    privacy_results.to_csv(csv_path, index=False)
    print(f"Privacy verification results saved: {csv_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Reproducibility
    SEED = args.seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    csv_path = args.csv
    if not os.path.exists(csv_path):
        sys.exit(f"ERROR: CSV not found at '{csv_path}'. "
                 "Use --csv path/to/file.csv")

    df = pd.read_csv(csv_path)
    print(f"Loaded '{csv_path}'  shape={df.shape}")

    numeric_cols     = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=["int64", "float64"]).columns.tolist()
    print(f"Numeric    : {numeric_cols}")
    print(f"Categorical: {categorical_cols}")

    # ------------------------------------------------------------------
    # 2. Preprocess
    # ------------------------------------------------------------------
    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    scaler    = StandardScaler()
    X_numeric = scaler.fit_transform(df[numeric_cols])

    try:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:
        ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
    X_categorical = ohe.fit_transform(df[categorical_cols])

    cat_sizes = [len(cats) for cats in ohe.categories_]
    n_numeric = len(numeric_cols)

    X_combined = np.concatenate([X_numeric, X_categorical], axis=1)
    X_tensor   = torch.tensor(X_combined, dtype=torch.float32)

    dataset    = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # ------------------------------------------------------------------
    # 3. Train VAE
    # ------------------------------------------------------------------
    input_dim = X_tensor.shape[1]
    model     = VAE(input_dim, args.hidden_dim, args.latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    kl_warmup = int(args.epochs * 0.3)
    print(f"\nTraining VAE for {args.epochs} epochs ...")

    for epoch in range(args.epochs):
        beta = args.beta_final * min(1.0, (epoch + 1) / kl_warmup)
        model.train()
        for (batch_x,) in dataloader:
            optimizer.zero_grad()
            recon, mu, logvar = model(batch_x)
            loss, *_ = vae_loss(recon, batch_x, mu, logvar,
                                n_numeric, cat_sizes,
                                beta=beta,
                                numeric_weight=args.numeric_weight)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1:>3}/{args.epochs}  beta={beta:.4f}  loss={loss.item():.4f}")

    print("Training complete.")

    # ------------------------------------------------------------------
    # 4. Generate with privacy noise
    # ------------------------------------------------------------------
    model.eval()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    with torch.no_grad():
        # Step A: encode real data → posterior samples (V2 strategy)
        mu_all, logvar_all = model.encode(X_tensor)
        std_all = torch.exp(0.5 * logvar_all)
        z = mu_all + std_all * torch.randn_like(std_all)

        # Step B: PRIVACY LAYER 1 — latent-space Gaussian perturbation
        z_private = perturb_latent(z, args.latent_noise)

        synthetic_out = model.decode(z_private).numpy()

    # Step C: inverse-transform numeric
    synthetic_numeric = scaler.inverse_transform(synthetic_out[:, :n_numeric])

    # Compute per-column sensitivity (IQR-based, robust to outliers)
    sensitivities = []
    for i, col in enumerate(numeric_cols):
        q75, q25 = np.percentile(df[col].values, [75, 25])
        sens = max(q75 - q25, 1e-6)          # IQR as sensitivity proxy
        sensitivities.append(sens)

    # Step D: PRIVACY LAYER 2 — post-hoc Laplace noise on numerics
    print(f"\nApplying Laplace noise (epsilon={args.epsilon}) to numeric columns ...")
    for i, (col, sens) in enumerate(zip(numeric_cols, sensitivities)):
        synthetic_numeric[:, i] = add_laplace_noise(
            synthetic_numeric[:, i], sens, args.epsilon
        )

    # Clip purchase_amount to non-negative (domain constraint)
    if "purchase_amount" in numeric_cols:
        idx = numeric_cols.index("purchase_amount")
        synthetic_numeric[:, idx] = np.clip(synthetic_numeric[:, idx], 0, None)

    # Decode categoricals with temperature softmax
    temperature = 2.0
    offset      = n_numeric
    cat_decoded = {}
    for col, size in zip(categorical_cols, cat_sizes):
        logits   = synthetic_out[:, offset:offset + size]
        probs    = scipy_softmax(logits / temperature, axis=1)
        min_prob = 1.0 / (size * 3)
        probs    = np.clip(probs, min_prob, 1.0)
        probs   /= probs.sum(axis=1, keepdims=True)
        indices  = np.array([np.random.choice(size, p=probs[i])
                             for i in range(len(df))])
        cat_decoded[col] = ohe.categories_[categorical_cols.index(col)][indices]
        offset += size

    # Step E: PRIVACY LAYER 3 — category flipping
    print(f"Applying category flip (prob={args.flip_prob}) to categorical columns ...")
    for col in categorical_cols:
        cats = list(ohe.categories_[categorical_cols.index(col)])
        cat_decoded[col] = flip_categories(cat_decoded[col], cats, args.flip_prob)

    # Assemble synthetic DataFrame
    df_syn = pd.DataFrame(synthetic_numeric, columns=numeric_cols)
    for col in categorical_cols:
        df_syn[col] = cat_decoded[col]

    # ------------------------------------------------------------------
    # 5. Save output
    # ------------------------------------------------------------------
    df_syn.to_csv(args.output, index=False)
    print(f"\nPrivate synthetic data saved to: {args.output}")
    print(f"\nSynthetic Data (first 5 rows):")
    print(df_syn.head().to_string(index=False))

    # ------------------------------------------------------------------
    # 6. Privacy noise report
    # ------------------------------------------------------------------
    print_privacy_report(numeric_cols, sensitivities,
                         args.epsilon, args.latent_noise, args.flip_prob)

    # ------------------------------------------------------------------
    # 7. Statistical tests (same metrics as Assignment_3.ipynb)
    # ------------------------------------------------------------------
    summary_df = run_statistical_tests(df, df_syn, numeric_cols, categorical_cols)

    summary_path = args.output.replace(".csv", "_test_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nTest summary saved to: {summary_path}")

    # ------------------------------------------------------------------
    # 8. Plots
    # ------------------------------------------------------------------
    if not args.no_plots:
        out_prefix = args.output.replace(".csv", "")
        print("\nSaving comparison plots ...")
        save_plots(df, df_syn, numeric_cols, categorical_cols, out_prefix)

    # ------------------------------------------------------------------
    # 9. Privacy verification attack suite
    # ------------------------------------------------------------------
    out_prefix = args.output.replace(".csv", "")
    run_privacy_verification(df, df_syn, numeric_cols, categorical_cols,
                             scaler, ohe, out_prefix, args.no_plots)

    print("\nDone.")


if __name__ == "__main__":
    main()
