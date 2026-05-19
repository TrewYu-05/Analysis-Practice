"""
PCA 主成分分析 —— 对 data/winequality.csv 进行降维可视化。

输出（保存至 results/）:
  - pca_by_quality.svg / .pdf  —— 按 quality 着色
  - pca_by_type.svg    / .pdf  —— 按 type（红/白葡萄酒）着色
  - pca_scree.svg      / .pdf  —— 碎石图（各主成分解释方差）
  - pca_loadings.csv           —— 前两个主成分的特征载荷
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ─── 0. 路径与输出目录 ────────────────────────────────────────────────────────
DATA_PATH = "data/winequality.csv"
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

# ─── 1. 加载数据 ─────────────────────────────────────────────────────────────
print(f"[1/5] 加载数据: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"      数据维度: {df.shape}")
print(f"      列名: {list(df.columns)}")

# 分离特征与标签
feature_cols = [
    "fixed acidity", "volatile acidity", "citric acid",
    "residual sugar", "chlorides", "free sulfur dioxide",
    "total sulfur dioxide", "density", "pH", "sulphates", "alcohol",
]
X = df[feature_cols].values
y_quality = df["quality"].values
y_type = df["type"].values  # 0 = red, 1 = white

print(f"      特征数: {len(feature_cols)}, 样本数: {len(df)}")

# ─── 2. 标准化 ───────────────────────────────────────────────────────────────
print("[2/5] StandardScaler 标准化（均值 0，方差 1）...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ─── 3. PCA 拟合 ─────────────────────────────────────────────────────────────
print("[3/5] PCA 拟合（保留全部主成分）...")
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# 打印各主成分解释方差
evr = pca.explained_variance_ratio_
for i, v in enumerate(evr):
    print(f"      PC{i+1}: {v:.4f}  ({v*100:.1f}%)", end="")
    if i == 1:
        print(f"  ← 前2维累计解释方差: {evr[:2].sum()*100:.1f}%")
    else:
        print()

# ─── 4. 碎石图 (Scree Plot) ─────────────────────────────────────────────────
print("[4/5] 绘制碎石图...")
fig, ax = plt.subplots(figsize=(8, 5))

components = np.arange(1, len(evr) + 1)
bars = ax.bar(components, evr * 100, color="steelblue", edgecolor="black", alpha=0.85)

# 累计解释方差曲线
cumsum = np.cumsum(evr) * 100
ax.plot(components, cumsum, "o--", color="darkorange", linewidth=2, markersize=6,
        label="Cumulative explained variance")

# 标注前两个主成分的值
for i in range(len(evr)):
    ax.text(components[i], evr[i] * 100 + 0.6, f"{evr[i]*100:.1f}%",
            ha="center", va="bottom", fontsize=8)

ax.set_xlabel("Principal Component")
ax.set_ylabel("Explained Variance Ratio (%)")
ax.set_title("PCA Scree Plot — Wine Quality Dataset")
ax.set_xticks(components)
ax.legend(loc="upper right")
ax.set_ylim(0, max(cumsum) * 1.12)
plt.tight_layout()

scree_svg = os.path.join(OUT_DIR, "pca_scree.svg")
scree_pdf = os.path.join(OUT_DIR, "pca_scree.pdf")
fig.savefig(scree_svg)
fig.savefig(scree_pdf)
plt.close(fig)
print(f"      碎石图已保存: {scree_svg} / {scree_pdf}")

# ─── 5. 保存载荷矩阵 ────────────────────────────────────────────────────────
print("[5/5] 保存载荷矩阵 & 绘制两张 PCA 散点图...")
loadings = pca.components_[:2, :]  # PC1, PC2 × features
loadings_df = pd.DataFrame(
    loadings.T,
    columns=["PC1_loading", "PC2_loading"],
    index=feature_cols,
)
loadings_df.to_csv(os.path.join(OUT_DIR, "pca_loadings.csv"), encoding="utf-8-sig")
print(f"      载荷矩阵已保存: {OUT_DIR}/pca_loadings.csv")

# ─── 6. 图一：按 quality 着色 ────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(9, 7))

sc1 = ax1.scatter(
    X_pca[:, 0], X_pca[:, 1],
    c=y_quality, cmap="viridis",
    alpha=0.55, edgecolors="none", s=18,
)
cbar1 = plt.colorbar(sc1, ax=ax1)
cbar1.set_label("Quality", fontsize=11)

ax1.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
ax1.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
ax1.set_aspect("equal", adjustable="datalim")
plt.tight_layout()

fig1_svg = os.path.join(OUT_DIR, "pca_by_quality.svg")
fig1_pdf = os.path.join(OUT_DIR, "pca_by_quality.pdf")
fig1.savefig(fig1_svg)
fig1.savefig(fig1_pdf)
plt.close(fig1)
print(f"      PCA-quality 图已保存: {fig1_svg} / {fig1_pdf}")

# ─── 7. 图二：按 type 着色（红/白葡萄酒） ────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(9, 7))

# type=0 → red wine, type=1 → white wine
type_colors = {0: "#c0392b", 1: "#2980b9"}
type_labels = {0: "Red (type=0)", 1: "White (type=1)"}

for t in sorted(type_colors.keys()):
    mask = y_type == t
    ax2.scatter(
        X_pca[mask, 0], X_pca[mask, 1],
        c=type_colors[t], label=type_labels[t],
        alpha=0.5, edgecolors="none", s=18,
    )

ax2.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
ax2.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
ax2.legend(loc="upper right", fontsize=10)
ax2.set_aspect("equal", adjustable="datalim")
plt.tight_layout()

fig2_svg = os.path.join(OUT_DIR, "pca_by_type.svg")
fig2_pdf = os.path.join(OUT_DIR, "pca_by_type.pdf")
fig2.savefig(fig2_svg)
fig2.savefig(fig2_pdf)
plt.close(fig2)
print(f"      PCA-type 图已保存: {fig2_svg} / {fig2_pdf}")

# ─── 完成 ────────────────────────────────────────────────────────────────────
print(f"\n[完成] PCA 分析结束，结果保存在 {OUT_DIR}/ 中:")
print(f"  - pca_scree.svg / .pdf")
print(f"  - pca_by_quality.svg / .pdf")
print(f"  - pca_by_type.svg / .pdf")
print(f"  - pca_loadings.csv")
print(f"  前2维累计解释方差: {evr[:2].sum()*100:.1f}%")
