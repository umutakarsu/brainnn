"""
Alzheimer genetiği + Gradient Boosting — CANLI eğitim görselleştirmesi.

GBM'in ağaç ağaç nasıl öğrendiğini gözünle görürsün:
- Her yeni ağaç eklendikçe AUC nasıl değişiyor
- Tahmin dağılımı nasıl iki tepeli hale geliyor
- İlk karar ağaçları nasıl görünüyor
"""

import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.tree import plot_tree

st.set_page_config(page_title="GBM canlı eğitim — Alzheimer", layout="wide")

st.title("Gradient Boosting — Ağaç Ağaç Öğrenmeyi İzle")
st.caption("Her ağaç eklendikçe model nasıl gelişiyor? Nature Comm 2025 sentetik taklidi.")

with st.sidebar:
    st.header("Sentetik evren")
    n_people = st.slider("Kişi sayısı", 1000, 10000, 4000, step=1000)
    n_snps = st.slider("SNP sayısı", 50, 500, 200, step=50)
    maf = st.slider("Riskli varyant sıklığı (MAF)", 0.05, 0.50, 0.25)

    st.header("Hastalık mantığı")
    n_linear = st.slider("Lineer sinyal sayısı", 2, 10, 5)
    n_interact = st.slider("Etkileşim çifti sayısı", 0, 4, 2)
    interact_weight = st.slider("Etkileşim gücü", 0.0, 2.0, 1.2)

    st.header("GBM hiperparametreleri")
    n_trees = st.slider("Toplam ağaç sayısı", 20, 300, 100, step=10)
    max_depth = st.slider("Ağaç derinliği", 1, 5, 3)
    lr = st.select_slider("Öğrenme hızı", [0.03, 0.05, 0.1, 0.2, 0.3], value=0.1)

    st.header("Animasyon")
    delay = st.slider("Her kare arası bekleme (sn)", 0.0, 0.3, 0.05)
    train_btn = st.button("▶️  Eğitimi başlat", type="primary", use_container_width=True)

# --- veri üretimi ---
rng = np.random.default_rng(42)
X = rng.binomial(2, maf, size=(n_people, n_snps)).astype(np.float32)

linear_snps = rng.choice(n_snps, size=n_linear, replace=False).tolist()
linear_weights = np.linspace(1.5, 0.3, n_linear)
interaction_pairs = [
    tuple(rng.choice(n_snps, size=2, replace=False).tolist())
    for _ in range(n_interact)
]

logit = np.zeros(n_people)
for snp, w in zip(linear_snps, linear_weights):
    logit += w * X[:, snp]
for (a, b) in interaction_pairs:
    logit += interact_weight * (X[:, a] > 0) * (X[:, b] > 0)
logit -= logit.mean() - (-1.5)
prob = 1 / (1 + np.exp(-logit))
y = rng.binomial(1, prob)

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y
)

# --- klasik baseline (anlık) ---
prs = LogisticRegression(max_iter=500).fit(X_tr, y_tr)
prs_auc = roc_auc_score(y_te, prs.predict_proba(X_te)[:, 1])

c1, c2, c3 = st.columns(3)
c1.metric("Hasta oranı", f"{y.mean():.1%}")
c2.metric("PRS (klasik) AUC", f"{prs_auc:.3f}")
c3.metric("Veri", f"{n_people} kişi × {n_snps} SNP")

st.divider()

# --- canlı eğitim ---
if not train_btn:
    st.info("⬅️  Soldan **Eğitimi başlat**'a bas. GBM'in her ağaç eklendikçe "
            "nasıl öğrendiğini gerçek zamanlı izleyeceksin.")
    st.stop()

gbm = GradientBoostingClassifier(
    n_estimators=n_trees, max_depth=max_depth,
    learning_rate=lr, random_state=0
)

# Tek seferde eğit, sonra staged_predict_proba ile aşama aşama oynatacağız
with st.spinner(f"{n_trees} ağaçlı model eğitiliyor..."):
    gbm.fit(X_tr, y_tr)

# staged_predict_proba → her ağaç eklendikten sonraki tahminler
stages_te = list(gbm.staged_predict_proba(X_te))
stages_tr = list(gbm.staged_predict_proba(X_tr))

aucs_te = [roc_auc_score(y_te, p[:, 1]) for p in stages_te]
aucs_tr = [roc_auc_score(y_tr, p[:, 1]) for p in stages_tr]

# --- canlı görselleştirme ---
top_row = st.columns([1, 1, 1])
auc_placeholder = top_row[0].empty()
dist_placeholder = top_row[1].empty()
roc_placeholder = top_row[2].empty()

tree_placeholder = st.empty()
importance_placeholder = st.empty()

progress = st.progress(0, text="Ağaç 0 / " + str(n_trees))
status = st.empty()

for i in range(n_trees):
    # 1) AUC grafiği — şimdiye kadar
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(range(1, i + 2), aucs_tr[:i + 1], label="Train", color="#4c78a8")
    ax.plot(range(1, i + 2), aucs_te[:i + 1], label="Test", color="#e45756", linewidth=2)
    ax.axhline(prs_auc, linestyle="--", color="gray", label=f"PRS {prs_auc:.3f}")
    ax.set_xlim(0, n_trees)
    ax.set_ylim(0.4, 1.0)
    ax.set_xlabel("Ağaç sayısı")
    ax.set_ylabel("AUC")
    ax.set_title(f"Ağaç #{i+1} → AUC={aucs_te[i]:.3f}")
    ax.legend(loc="lower right")
    auc_placeholder.pyplot(fig, clear_figure=True)

    # 2) Tahmin dağılımı
    fig, ax = plt.subplots(figsize=(5, 3.5))
    preds = stages_te[i][:, 1]
    ax.hist(preds[y_te == 0], bins=30, alpha=0.6, label="Sağlıklı", color="#4c78a8")
    ax.hist(preds[y_te == 1], bins=30, alpha=0.6, label="Hasta", color="#e45756")
    ax.set_xlabel("Tahmin olasılığı")
    ax.set_ylabel("Kişi")
    ax.set_title("Tahmin dağılımı (iki tepe ayrışıyor mu?)")
    ax.legend()
    dist_placeholder.pyplot(fig, clear_figure=True)

    # 3) ROC (canlı)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    fpr, tpr, _ = roc_curve(y_te, preds)
    ax.plot(fpr, tpr, linewidth=2, color="#e45756",
            label=f"GBM @ {i+1} ağaç = {aucs_te[i]:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC eğrisi")
    ax.legend()
    roc_placeholder.pyplot(fig, clear_figure=True)

    # Her 10 ağaçta bir en son eklenen ağacı göster
    if i < 5 or i % 10 == 0 or i == n_trees - 1:
        fig, ax = plt.subplots(figsize=(14, 3.5))
        tree = gbm.estimators_[i, 0]
        plot_tree(
            tree, ax=ax, max_depth=max_depth,
            feature_names=[f"SNP{j}" for j in range(n_snps)],
            filled=True, impurity=False, rounded=True, fontsize=8,
        )
        ax.set_title(f"Ağaç #{i+1} — önceki ağaçların hatalarını bu düzeltiyor")
        tree_placeholder.pyplot(fig, clear_figure=True)

    # Her 5 ağaçta bir önem grafiği
    if i % 5 == 0 or i == n_trees - 1:
        # O ana kadarki ağaçların ortalama önem skoru
        imp = np.zeros(n_snps)
        for t in range(i + 1):
            imp += gbm.estimators_[t, 0].feature_importances_
        imp /= (i + 1)

        fig, ax = plt.subplots(figsize=(14, 2.8))
        ax.bar(range(n_snps), imp, color="#cccccc", width=1.0)
        interaction_snps = {s for p in interaction_pairs for s in p}
        for k, snp in enumerate(linear_snps):
            ax.bar(snp, imp[snp], color="#d62728", width=3,
                   label="Lineer (gömülü)" if k == 0 else "")
        for k, snp in enumerate(interaction_snps):
            ax.bar(snp, imp[snp], color="#2ca02c", width=3,
                   label="Etkileşim (gömülü)" if k == 0 else "")
        ax.set_title(f"SNP önem skoru — {i+1} ağaç eklendikten sonra")
        ax.set_xlabel("SNP indeksi")
        ax.legend()
        importance_placeholder.pyplot(fig, clear_figure=True)

    progress.progress((i + 1) / n_trees, text=f"Ağaç {i+1} / {n_trees}")
    status.markdown(
        f"**Test AUC:** {aucs_te[i]:.3f}  |  "
        f"**Train AUC:** {aucs_tr[i]:.3f}  |  "
        f"**PRS (klasik):** {prs_auc:.3f}  |  "
        f"**Avantaj:** {aucs_te[i] - prs_auc:+.3f}"
    )

    if delay > 0:
        time.sleep(delay)

st.success(f"✅ Bitti — {n_trees} ağaç. Final test AUC = {aucs_te[-1]:.3f} "
           f"(PRS: {prs_auc:.3f}, fark {aucs_te[-1] - prs_auc:+.3f})")

# Final tablo
imp = gbm.feature_importances_
top_idx = np.argsort(imp)[::-1][:15]
interaction_snps = {s for p in interaction_pairs for s in p}
rows = []
for rank, idx in enumerate(top_idx, 1):
    if idx in linear_snps:
        tag = "✅ Lineer (gömülü)"
    elif idx in interaction_snps:
        tag = "✅ Etkileşim (gömülü)"
    else:
        tag = "❓ Gürültü"
    rows.append({
        "Sıra": rank, "SNP": int(idx),
        "Önem": round(float(imp[idx]), 4), "Tür": tag
    })
st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
