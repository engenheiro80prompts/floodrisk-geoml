import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data" / "raw" / "setores_censitarios.csv"
DOCS = ROOT / "docs"
PROC = ROOT / "data" / "processed"
DOCS.mkdir(exist_ok=True)
PROC.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
FEATURES = ["declividade_graus","dist_drenagem_m","impermeabilizacao_pct",
            "altitude_relativa_m","precipitacao_max_24h_mm","capacidade_drenagem_idx",
            "densidade_pop_hab_km2","tempo_concentracao_min","cobertura_vegetal_pct","manut_microdrenagem_idx"]

X = df[FEATURES]
y = df["risco_alto"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

rf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test, y_proba)
cv = cross_val_score(rf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="roc_auc")

print(f"AUC: {auc:.4f} | CV: {cv.mean():.4f} +/- {cv.std():.4f}")

df_full = df.copy()
df_full["prob_risco_alto"] = rf.predict_proba(X)[:,1]
df_full["pred_risco"] = rf.predict(X)
df_full.to_csv(PROC / "dataset_completo_com_score.csv", index=False)

LABELS = {"declividade_graus":"Declividade (graus)","dist_drenagem_m":"Dist. ao curso d'água (m)",
          "impermeabilizacao_pct":"Impermeabilização (%)","altitude_relativa_m":"Altitude relativa (m)",
          "precipitacao_max_24h_mm":"Precipitação máx. 24h (mm)","capacidade_drenagem_idx":"Capacidade de drenagem",
          "densidade_pop_hab_km2":"Densidade populacional","tempo_concentracao_min":"Tempo concentração (min)",
          "cobertura_vegetal_pct":"Cobertura vegetal (%)","manut_microdrenagem_idx":"Manutenção microdrenagem"}

feat_imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=True)
feat_imp.index = [LABELS[f] for f in feat_imp.index]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor("#0d1117")

ax1 = axes[0]
ax1.set_facecolor("#0d1117")
colors = ["#238636" if v > feat_imp.median() else "#1f6feb" for v in feat_imp.values]
bars = ax1.barh(feat_imp.index, feat_imp.values, color=colors, edgecolor="none", height=0.65)
ax1.set_xlabel("Importância (Gini)", color="#8b949e", fontsize=10)
ax1.set_title("Importância das Variáveis", color="#e6edf3", fontsize=12, fontweight="bold")
ax1.tick_params(colors="#8b949e", labelsize=9)
ax1.spines[:].set_visible(False)
for bar, val in zip(bars, feat_imp.values):
    ax1.text(val+0.002, bar.get_y()+bar.get_height()/2, f"{val:.3f}", va="center", color="#8b949e", fontsize=8)

ax2 = axes[1]
ax2.set_facecolor("#0d1117")
fpr, tpr, _ = roc_curve(y_test, y_proba)
ax2.plot(fpr, tpr, color="#238636", lw=2, label=f"Random Forest (AUC={auc:.3f})")
ax2.plot([0,1],[0,1], color="#30363d", lw=1.5, linestyle="--", label="Aleatório")
ax2.fill_between(fpr, tpr, alpha=0.08, color="#238636")
ax2.set_xlabel("Taxa de Falsos Positivos", color="#8b949e", fontsize=10)
ax2.set_ylabel("Taxa de Verdadeiros Positivos", color="#8b949e", fontsize=10)
ax2.set_title("Curva ROC", color="#e6edf3", fontsize=12, fontweight="bold")
ax2.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#8b949e", fontsize=9)
ax2.tick_params(colors="#8b949e")
ax2.spines[:].set_color("#30363d")
ax2.grid(alpha=0.1, color="#30363d")

plt.tight_layout(pad=2)
plt.savefig(DOCS/"feature_importance_roc.png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print("Graficos salvos!")
