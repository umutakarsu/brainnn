"""
Alzheimer genetiği + Gradient Boosting sentetik demo.

Nature Communications 2025 makalesini küçük ölçekte taklit eder:
- Sahte SNP verisi üretir (10.000 kişi, 500 SNP)
- Bazı SNP'lere gerçek risk sinyali gömer (APOE benzeri)
- Bazı SNP-SNP etkileşimi ekler (GWAS'ın kaçırdığı kısım)
- GBM vs klasik lojistik regresyon (PRS benzeri) karşılaştırır
- En önemli SNP'leri SHAP mantığıyla gösterir

Çalıştır:  python examples/alzheimer_gbm_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

rng = np.random.default_rng(42)

N_PEOPLE = 10_000
N_SNPS = 500
MAF = 0.25  # minor allele frequency

# SNP verisi: her kişi için her SNP 0, 1 veya 2 (kaç kopya riskli varyant)
X = rng.binomial(2, MAF, size=(N_PEOPLE, N_SNPS)).astype(np.float32)

# "Gerçek" hastalık mantığı — makaledeki gibi:
# 1) Bazı SNP'lerin lineer (additif) etkisi var (GWAS bunu bulur)
# 2) Bazı SNP çiftlerinin etkileşimi var (GWAS kaçırır, GBM yakalar)

linear_snps = [10, 42, 77, 128, 200, 301]  # APOE, BIN1 gibi bilinen lokuslar
linear_weights = [1.2, 0.8, 0.7, 0.6, 0.5, 0.4]

interaction_pairs = [(55, 180), (90, 333)]  # SNP-SNP etkileşimleri
interaction_weights = [1.0, 0.9]

logit = np.zeros(N_PEOPLE)
for snp, w in zip(linear_snps, linear_weights):
    logit += w * X[:, snp]
for (a, b), w in zip(interaction_pairs, interaction_weights):
    # Etkileşim: SADECE ikisi de varsa etki ortaya çıkar (GWAS göremez)
    logit += w * (X[:, a] > 0) * (X[:, b] > 0)

logit -= logit.mean() + 1.5  # baz hastalık oranını düşür
prob = 1 / (1 + np.exp(-logit))
y = rng.binomial(1, prob)

print(f"Hasta oranı: {y.mean():.1%}  (toplam {y.sum()} vaka / {N_PEOPLE})")

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y
)

# Model 1: Klasik yöntem — lojistik regresyon (PRS benzeri, lineer)
prs = LogisticRegression(max_iter=1000, C=1.0).fit(X_tr, y_tr)
prs_auc = roc_auc_score(y_te, prs.predict_proba(X_te)[:, 1])

# Model 2: Gradient Boosting
gbm = GradientBoostingClassifier(
    n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0
).fit(X_tr, y_tr)
gbm_auc = roc_auc_score(y_te, gbm.predict_proba(X_te)[:, 1])

print(f"\nPRS (lojistik regresyon) AUC: {prs_auc:.3f}")
print(f"GBM (gradient boosting)  AUC: {gbm_auc:.3f}")
print(f"Fark: {gbm_auc - prs_auc:+.3f}  (makale: GBM biraz üstün ama benzer)")

# GBM'e göre en önemli SNP'ler
importances = gbm.feature_importances_
top_idx = np.argsort(importances)[::-1][:15]

print("\nGBM'e göre en önemli 15 SNP:")
print(f"(gerçekten gömdüğümüz sinyaller: lineer={linear_snps}, "
      f"etkileşim={interaction_pairs})\n")
for rank, idx in enumerate(top_idx, 1):
    tag = ""
    if idx in linear_snps:
        tag = "  ← LINEER SINYAL"
    elif any(idx in pair for pair in interaction_pairs):
        tag = "  ← ETKILESIM SINYALI"
    print(f"  {rank:2d}. SNP {idx:3d}  önem={importances[idx]:.4f}{tag}")

# Görselleştirme
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# (a) ROC eğrileri
for name, model, auc in [("PRS", prs, prs_auc), ("GBM", gbm, gbm_auc)]:
    fpr, tpr, _ = roc_curve(y_te, model.predict_proba(X_te)[:, 1])
    axes[0].plot(fpr, tpr, label=f"{name}  AUC={auc:.3f}", linewidth=2)
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC: Klasik vs GBM")
axes[0].legend()

# (b) En önemli 20 SNP (Manhattan benzeri)
axes[1].bar(range(N_SNPS), importances, color="#cccccc", width=1.0)
for snp in linear_snps:
    axes[1].bar(snp, importances[snp], color="#d62728", width=3,
                label="Lineer sinyal" if snp == linear_snps[0] else "")
for pair in interaction_pairs:
    for snp in pair:
        axes[1].bar(snp, importances[snp], color="#2ca02c", width=3,
                    label="Etkileşim" if snp == interaction_pairs[0][0] else "")
axes[1].set_xlabel("SNP indeksi")
axes[1].set_ylabel("GBM önem skoru")
axes[1].set_title("Makaledeki Manhattan grafiğinin karşılığı")
axes[1].legend()

# (c) Tahmin dağılımı (makaledeki Fig 2c benzeri)
preds = gbm.predict_proba(X_te)[:, 1]
axes[2].hist(preds[y_te == 0], bins=40, alpha=0.6, label="Sağlıklı", color="#8888ff")
axes[2].hist(preds[y_te == 1], bins=40, alpha=0.6, label="Hasta",    color="#ff6666")
axes[2].set_xlabel("GBM tahmini (hastalık olasılığı)")
axes[2].set_ylabel("Kişi sayısı")
axes[2].set_title("Tahmin dağılımı (iki tepeli → model ayırt edebiliyor)")
axes[2].legend()

plt.tight_layout()
out = "examples/alzheimer_gbm_demo.png"
plt.savefig(out, dpi=120)
print(f"\nGrafik kaydedildi: {out}")
