"""
NF1 tumour-status classification from co-recorded phenotype features.

A contextual bandit agent and a kNN-based local phenotype score are combined
with a fixed weighted fusion. Everything is evaluated with repeated stratified
cross-validation and the paper tables/figures are written to the working folder.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.special import expit
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss, log_loss,
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc,
)

DATA_PATH = "dataset_csv.csv"
TARGET = "Tumour Case"
DROP_COLS = [TARGET, "Case Type", "ID", "Healthy"]  # Healthy leaks overall status

N_SPLITS, N_REPEATS, SEED = 5, 10, 42

# bandit settings
EPSILON = 0.10
R_CORRECT, R_WRONG = 5.0, -10.0
LR = 0.01
MAX_ITER = 1000

# local score + fusion
K = 15
W_AGENT, W_LOCAL = 0.70, 0.30

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("colorblind")
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 10, "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def get_metrics(y_true, scores, thr=0.5):
    scores = np.clip(np.asarray(scores), 1e-7, 1 - 1e-7)
    pred = (scores >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "ROC_AUC": roc_auc_score(y_true, scores),
        "PR_AUC": average_precision_score(y_true, scores),
        "Brier": brier_score_loss(y_true, scores),
        "LogLoss": log_loss(y_true, scores),
        "Accuracy": accuracy_score(y_true, pred),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "F1": f1_score(y_true, pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, pred),
    }


class ContextualBandit:
    """Two linear value models (one per action), updated only for the chosen action."""

    def __init__(self, n_actions=2, epsilon=EPSILON, r_correct=R_CORRECT,
                 r_wrong=R_WRONG, lr=LR, max_iter=MAX_ITER, seed=SEED):
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.r_correct = r_correct
        self.r_wrong = r_wrong
        self.lr = lr
        self.max_iter = max_iter
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.models = []

    def _build(self):
        self.models = [
            SGDRegressor(loss="squared_error", penalty="l2", alpha=0.0001,
                         learning_rate="constant", eta0=self.lr,
                         max_iter=self.max_iter, tol=1e-3,
                         random_state=self.seed + a)
            for a in range(self.n_actions)
        ]

    def _q(self, X):
        return np.column_stack([m.predict(X) for m in self.models])

    def fit(self, X, y):
        self._build()
        X = np.asarray(X)
        y = np.asarray(y).astype(int)
        # partial_fit needs one initial call before predict works
        first = X[0].reshape(1, -1)
        for m in self.models:
            m.partial_fit(first, np.array([0.0]))
        for _ in range(3):
            for i in self.rng.permutation(len(X)):
                x = X[i].reshape(1, -1)
                q = np.array([m.predict(x)[0] for m in self.models])
                if self.rng.rand() < self.epsilon:
                    a = self.rng.randint(self.n_actions)
                else:
                    a = int(np.argmax(q))
                reward = self.r_correct if a == y[i] else self.r_wrong
                self.models[a].partial_fit(x, np.array([reward]))
        return self

    def score(self, X):
        q = self._q(np.asarray(X))
        return expit(q[:, 1] - q[:, 0])


def local_score(X_train, y_train, X_test, k=K):
    """Inverse-distance weighted tumour rate among the k nearest training patients."""
    X_train, X_test, y_train = map(np.asarray, (X_train, X_test, y_train))
    k = min(k, len(X_train))
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(X_train)
    dist, idx = nn.kneighbors(X_test)
    out = []
    for d, ix in zip(dist, idx):
        w = 1.0 / (d + 1e-6)
        out.append(np.sum(w * y_train[ix]) / np.sum(w))
    return np.asarray(out)


def fuse(agent, local, w_agent=W_AGENT, w_local=W_LOCAL):
    # plain weighted sum, no fuzzy inference involved
    return np.clip(w_agent * agent + w_local * local, 0.0, 1.0)


# ---- data ----
df = pd.read_csv(DATA_PATH)
print("shape:", df.shape)
print(df[TARGET].value_counts())

features = [c for c in df.columns if c not in DROP_COLS]
X = df[features].apply(pd.to_numeric, errors="coerce")
y = df[TARGET].astype(int)
print(f"{len(features)} predictors:", features)

# ---- repeated stratified CV ----
cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)

n = len(df)
oof = {"agent": np.zeros(n), "local": np.zeros(n), "fused": np.zeros(n)}
times_seen = np.zeros(n)
fold_metrics = {"agent": [], "local": [], "fused": []}

for fold, (tr, te) in enumerate(cv.split(X, y), start=1):
    X_tr, X_te = X.iloc[tr], X.iloc[te]
    y_tr, y_te = y.iloc[tr].values, y.iloc[te].values

    # imputer and scaler are fitted on the training part only
    imp = IterativeImputer(random_state=SEED)
    X_tr = imp.fit_transform(X_tr)
    X_te = imp.transform(X_te)
    sc = RobustScaler()
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)

    bandit = ContextualBandit(seed=SEED + fold).fit(X_tr, y_tr)
    s_agent = bandit.score(X_te)
    s_local = local_score(X_tr, y_tr, X_te)
    s_fused = fuse(s_agent, s_local)

    for name, s in (("agent", s_agent), ("local", s_local), ("fused", s_fused)):
        oof[name][te] += s
        fold_metrics[name].append(get_metrics(y_te, s))
    times_seen[te] += 1

    if fold % 10 == 0:
        print(f"fold {fold}/{N_SPLITS * N_REPEATS} done")

# average the scores each patient got across repeats
for name in oof:
    oof[name] /= times_seen

final = {name: get_metrics(y, oof[name]) for name in oof}
for name, m in final.items():
    print(f"\n{name}")
    for k, v in m.items():
        print(f"  {k:18s} {v:.4f}")

# ---- tables ----
labels = {"agent": "Contextual Bandit", "local": "Local Phenotype", "fused": "Fused Score"}
pd.DataFrame({labels[k]: v for k, v in final.items()}).T.round(4).to_csv(
    "Table1_Final_OOF_Metrics.csv")

fm = {k: pd.DataFrame(v) for k, v in fold_metrics.items()}
pd.DataFrame({
    "Agent_Mean": fm["agent"].mean(), "Agent_Std": fm["agent"].std(),
    "Local_Mean": fm["local"].mean(), "Local_Std": fm["local"].std(),
    "Fused_Mean": fm["fused"].mean(), "Fused_Std": fm["fused"].std(),
}).round(4).to_csv("Table2_50Fold_Mean_Std.csv")

# ---- figures ----
colors = {"agent": "#1f77b4", "local": "#ff7f0e", "fused": "#2ca02c"}

# Fig 1: metric comparison
fig, ax = plt.subplots(figsize=(8.5, 4.8))
metric_names = ["ROC_AUC", "F1", "MCC", "Balanced_Accuracy"]
pos = np.arange(len(metric_names))
width = 0.25
for shift, name in zip((-width, 0, width), ("agent", "local", "fused")):
    ax.bar(pos + shift, [final[name][m] for m in metric_names], width,
           label=labels[name], color=colors[name])
ax.set_ylabel("Score")
ax.set_xticks(pos)
ax.set_xticklabels(["ROC-AUC", "F1-Score", "MCC", "Balanced Accuracy"])
ax.set_ylim(0, 1.08)
ax.legend(loc="lower right")
ax.set_title("Patient-Level Aggregated Performance Comparison")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("Figure1_Performance_Comparison.png")
plt.close()

# Fig 2: confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(12.5, 4))
for ax, name in zip(axes, ("agent", "local", "fused")):
    cm = confusion_matrix(y, (oof[name] >= 0.5).astype(int), labels=[0, 1])
    ConfusionMatrixDisplay(cm, display_labels=["No Tumour", "Tumour"]).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(labels[name])
plt.suptitle("Confusion Matrices (Patient-Level Aggregated Scores)", y=1.03, fontsize=13)
plt.tight_layout()
plt.savefig("Figure2_Confusion_Matrices.png")
plt.close()

# Fig 3: ROC curves
fig, ax = plt.subplots(figsize=(6.8, 5.8))
for name in ("agent", "local", "fused"):
    fpr, tpr, _ = roc_curve(y, oof[name])
    ax.plot(fpr, tpr, color=colors[name], lw=2.2,
            label=f"{labels[name]} (AUC = {auc(fpr, tpr):.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Classifier")
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.05])
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves (Patient-Level Aggregated Scores)")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("Figure3_ROC_Curves.png")
plt.close()

# Fig 4: where agent and local model disagree
disagree = (oof["agent"] >= 0.5) != (oof["local"] >= 0.5)
dis_df = pd.DataFrame({
    "Group": np.where(disagree, "Disagreement", "Agreement"),
    "True_Tumour": y.values,
    "Fused_Score": oof["fused"],
})
order = ["Disagreement", "Agreement"]
bar_colors = ["#d62728", "#2ca02c"]

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
rates = dis_df.groupby("Group")["True_Tumour"].mean().reindex(order)
axes[0].bar(rates.index, rates.values, color=bar_colors, width=0.55)
axes[0].set_ylabel("Actual Tumour Rate")
axes[0].set_title("Tumour Rate by Model-Disagreement Status")
axes[0].set_ylim(0, 1.0)
for i, v in enumerate(rates.values):
    axes[0].text(i, v + 0.035, f"{v:.3f}", ha="center", fontweight="bold")
sns.boxplot(data=dis_df, x="Group", y="Fused_Score", ax=axes[1],
            order=order, palette=bar_colors, width=0.5)
axes[1].set_ylabel("Fused Score")
axes[1].set_title("Distribution of Fused Score")
plt.tight_layout()
plt.savefig("Figure4_Model_Disagreement_Analysis.png")
plt.close()

# Fig 5: fused score by true label
fig, ax = plt.subplots(figsize=(7.5, 4.8))
plot_df = pd.DataFrame({
    "Fused Score": oof["fused"],
    "True Label": y.map({0: "No Tumour", 1: "Tumour"}),
})
sns.histplot(data=plot_df, x="Fused Score", hue="True Label", bins=28, kde=True,
             palette={"No Tumour": "#2ca02c", "Tumour": "#d62728"}, ax=ax, alpha=0.65)
ax.set_xlabel("Fused Score")
ax.set_ylabel("Count")
ax.set_title("Distribution of Fused Score by True Tumour Status")
plt.tight_layout()
plt.savefig("Figure5_Score_Distribution.png")
plt.close()

print("\nTables and figures saved.")
