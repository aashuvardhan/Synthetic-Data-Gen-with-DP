# Privacy-Preserving Synthetic Data Generation

Generates synthetic customer transaction data that is **statistically indistinguishable from real data** while being **resistant to re-identification attacks** — using a Variational Autoencoder (VAE) with a three-layer privacy noise pipeline.

**Team 5 — IIT Delhi (CS25MTECH)**  
Rohit Sinha · Shivansh Agarwal · Yash Vardhan

---

## What it does

1. Trains a **VAE** on your dataset
2. Generates synthetic records via **posterior sampling** (preserves non-Gaussian distributions)
3. Applies **three layers of privacy noise** to prevent re-identification
4. Validates **statistical fidelity** using hypothesis tests (Welch t-test, KS, Levene, Chi-square)
5. Verifies **privacy strength** using three adversarial attack simulations (MIA, DCR, NNAA)

---

## Results at a glance

| Check | Result |
|---|---|
| Statistical fidelity (8 columns) | **8 / 8 PASS** |
| Membership Inference Attack (RF) | AUC = 0.59 — WARNING |
| Membership Inference Attack (LR) | AUC = 0.51 — **SAFE** |
| Distance to Closest Record (DCR) | Ratio = 0.80 — **SAFE** |
| Nearest Neighbour Adversarial Accuracy | Score = 0.30 — **SAFE** |
| **Overall privacy verdict** | **PRIVATE** |

---

## Project structure

```
synthetic data/
├── privacy_synthetic.py              # main script (run this)
├── privacy_report.tex                # full LaTeX technical report
├── customer_transactions_1500.csv    # input dataset
├── Assignment_3.ipynb                # original VAE notebook (V1 + V2)
├── assignment3_report_team_5.pdf     # original assignment report
├── VAE.pdf                           # VAE theory reference
│
├── private_synthetic_data.csv                    # generated output
├── private_synthetic_data_test_summary.csv       # fidelity test results
├── private_synthetic_data_numeric_kde.png        # distribution plots
├── private_synthetic_data_categorical_bars.png   # category proportion plots
├── private_synthetic_data_privacy_verification.png  # attack result plots
├── private_synthetic_data_privacy_verification.csv  # attack result numbers
│
├── report/                           # assets for the LaTeX report
│   ├── Assignment3_Report.tex
│   ├── Assignment3.ipynb
│   └── *.png                         # figures used in the report
│
└── extra/                            # experimental work / backups
    ├── vae_hparam_search.py          # hyperparameter search script
    ├── vae_hparam_results.csv
    ├── vae_posterior_results.csv
    ├── wae_mmd_results.csv
    └── *.ipynb / *.csv               # team member experiments
```

---

## Quick start

```bash
# default run — reads customer_transactions_1500.csv
python privacy_synthetic.py

# custom dataset
python privacy_synthetic.py --csv your_data.csv

# tighter privacy (more noise)
python privacy_synthetic.py --epsilon 10.0 --latent_noise 0.3 --flip_prob 0.05

# see all options
python privacy_synthetic.py --help
```

---

## Privacy knobs

| Flag | Default | Effect |
|---|---|---|
| `--epsilon` | `20.0` | Laplace noise scale = IQR / epsilon. **Lower = more private, more noise.** |
| `--latent_noise` | `0.15` | Gaussian std added in VAE latent space before decoding |
| `--flip_prob` | `0.03` | Probability of randomly replacing a categorical value (3%) |

**Recommended ranges:** `--epsilon 10–50`, `--latent_noise 0.1–0.3`, `--flip_prob 0.01–0.10`

> Below `--epsilon 10`, the KS and Levene tests begin to fail (too much variance inflation).  
> Above `--flip_prob 0.10`, chi-square tests may fail for small categories.

---

## How the privacy noise works

Three independent noise layers are applied after the VAE generates a record:

### Layer 1 — Latent-space Gaussian perturbation
Extra Gaussian noise (`std = 0.15`) is added to each latent vector **before decoding**. This nudges every generated record away from its source record in a smooth, model-aware way without breaking the decoder's learned structure.

```
z_private = z_posterior + N(0, 0.15²·I)
```

### Layer 2 — Post-hoc Laplace noise on numeric columns
Laplace noise is added to each numeric column after inverse-transforming. Scale is calibrated to the column's IQR (sensitivity) and epsilon:

```
noise ~ Laplace(0, IQR / epsilon)
```

| Column | IQR (sensitivity) | Laplace scale (ε=20) |
|---|---|---|
| age | 23.00 | 1.15 |
| annual_income | 61,931.25 | 3,096.56 |
| purchase_amount | 487.14 | 24.36 |

### Layer 3 — Categorical value flipping
Each categorical value is randomly replaced with a different category with probability 3%, destroying individual fingerprints in categorical attributes.

---

## How privacy is verified

Three adversarial attacks are simulated automatically after generation:

### Attack 1 — Membership Inference Attack (MIA)
Trains a classifier to distinguish real records (label=1) from synthetic records (label=0). If the attacker can't beat random guessing (AUC=0.5), privacy holds.

| Classifier | AUC | Verdict |
|---|---|---|
| Random Forest (200 trees) | 0.5901 | WARNING |
| Logistic Regression | 0.5100 | SAFE |

Threshold: AUC < 0.55 = SAFE · 0.55–0.60 = WARNING · ≥ 0.60 = UNSAFE

### Attack 2 — Distance to Closest Record (DCR)
For every synthetic record, finds its nearest real neighbour (Euclidean distance in normalised space). Compares this to the real-to-real baseline distance.

| Metric | Synthetic | Real baseline | Ratio |
|---|---|---|---|
| Mean DCR | 1.1665 | 1.4547 | **0.802 — SAFE** |
| Median DCR | 1.4350 | 1.5118 | 0.949 |

Threshold: ratio ≥ 0.8 = SAFE · 0.5–0.8 = WARNING · < 0.5 = UNSAFE

### Attack 3 — Nearest Neighbour Adversarial Accuracy (NNAA)
Measures whether real and synthetic populations are geometrically separable. Score = 0.5 means they perfectly overlap (ideal privacy).

| NNAA Score | Verdict |
|---|---|
| 0.3033 | SAFE |

Threshold: < 0.55 = SAFE · 0.55–0.60 = WARNING · ≥ 0.60 = UNSAFE

---

## VAE architecture

```
Input (27) → FC(128) → ReLU → FC(64) → ReLU
                                        ↓
                              μ(16)    log σ²(16)
                                        ↓  reparameterize
                                       z(16)
                                        ↓
                    FC(64) → ReLU → FC(128) → ReLU → FC(27)
```

**Loss:** `L = 3.0 · MSE_numeric + CrossEntropy_categorical + β · KL`  
**Training:** 300 epochs · batch 128 · Adam lr=1e-3 · β warmup over first 90 epochs (0 → 0.1)

---

## Statistical fidelity tests

| Test | Columns | Checks | Pass condition |
|---|---|---|---|
| Welch t-test | Numeric | Equal means | p ≥ 0.05 |
| KS test | Numeric | Full distribution shape (CDF) | p ≥ 0.05 |
| Levene test | Numeric | Equal variance | p ≥ 0.05 |
| Chi-square | Categorical | Category proportions | p ≥ 0.05 |

All 8 columns pass all applicable tests.

---

## Output files

| File | Description |
|---|---|
| `private_synthetic_data.csv` | Privacy-preserving synthetic dataset |
| `private_synthetic_data_test_summary.csv` | Per-column statistical test results |
| `private_synthetic_data_numeric_kde.png` | KDE overlays — real vs synthetic |
| `private_synthetic_data_categorical_bars.png` | Category proportion bar charts |
| `private_synthetic_data_privacy_verification.png` | ROC curve, DCR histogram, attack summary |
| `private_synthetic_data_privacy_verification.csv` | Numeric attack results |

---

## Dependencies

```bash
pip install pandas numpy scikit-learn torch matplotlib seaborn scipy
```
