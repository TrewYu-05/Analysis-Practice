import os
os.environ["POSTHOG_DISABLED"] = "1"
os.environ["TABPFN_TELEMETRY_DISABLED"] = "1"
os.environ["TABPFN_DO_NOT_TRACK"] = "1" 
os.environ["DO_NOT_TRACK"] = "1"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy import stats


# ─── .env 加载（简易实现，无需 python-dotenv） ───────────────────────────────
def _load_env(env_path: str = ".env") -> None:
    """读取 .env 文件并将键值对注入 os.environ（不覆盖已有变量）。"""
    if not os.path.isfile(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            os.environ.setdefault(key, val)


_load_env()  # 在模块加载时自动执行一次


# ─── 缺失值与重复值 ──────────────────────────────────────────────────────────
def check_missing_and_duplicates(df, dataset_name):
    """
    缺失值统计和重复值统计。
    输出为控制台输出数据集名称：缺失值和重复值的数量统计
    """
    missing_count = df.isnull().sum().sum()
    duplicate_count = df.duplicated().sum()
    print(f"数据集 {dataset_name}: 缺失值数量 = {missing_count}, 重复值数量 = {duplicate_count}")


# ─── 基础异常值识别 ──────────────────────────────────────────────────────────
def basic_outlier_detection(df: pd.DataFrame, dataset_name: str):
    """
    基础异常值识别：
      1. 绘制箱线图并保存为 SVG + PDF 矢量图
      2. 使用 Z-Score（阈值=3）识别异常点，保存为表格
    """
    os.makedirs("results", exist_ok=True)

    # ── 箱线图（SVG + PDF） ──
    plt.figure(figsize=(15, 10))
    df.boxplot()
    plt.title(f"Boxplot for {dataset_name}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"results/{dataset_name}_boxplot.svg")
    plt.savefig(f"results/{dataset_name}_boxplot.pdf")
    plt.close()
    print(f"箱线图已保存: results/{dataset_name}_boxplot.svg / .pdf")

    # ── Z-Score 异常检测 ──
    z_scores = pd.DataFrame(
        np.abs(stats.zscore(df)),
        columns=df.columns,
        index=df.index,
    )
    # 找出任意特征上 |z| > 3 的行
    outlier_mask = (z_scores > 3).any(axis=1)
    outlier_df = df[outlier_mask].copy()
    outlier_z = z_scores[outlier_mask]

    # 组装输出表：原始值 + 对应 Z-Score
    result = pd.concat(
        [outlier_df.add_suffix("_val"), outlier_z.add_suffix("_z")],
        axis=1,
    )
    out_path = f"results/{dataset_name}_zscore_outliers.csv"
    result.to_csv(out_path, encoding="utf-8-sig")
    print(
        f"数据集 {dataset_name}: Z-Score (threshold=3) 识别出 "
        f"{outlier_mask.sum()} 个异常点 → {out_path}"
    )


# ─── SOTA 异常值识别 ─────────────────────────────────────────────────────────
def sota_outlier_detection(df: pd.DataFrame, dataset_name: str):
    """
    SOTA 异常识别。
    优先使用 TabPFN 对联合分布进行异常扫描；若不可用则回退到 IsolationForest。
    异常点结果保存为表格。
    """
    os.makedirs("results", exist_ok=True)

    # 环境变量补丁（优先从 .env 读取，此处兜底）
    os.environ.setdefault("POSTHOG_DISABLED", "1")
    os.environ.setdefault("TABPFN_TELEMETRY_DISABLED", "1")
    os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")

    X = df.values.astype(np.float32)
    scores = None

    # ──── 方式 1：TabPFN ────
    try:
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        from tabpfn_extensions.unsupervised import TabPFNUnsupervisedModel
        import torch

        print(f"正在初始化 TabPFN 模型，处理数据集 {dataset_name} ...")
        clf = TabPFNClassifier(device="cpu")
        reg = TabPFNRegressor(device="cpu")
        model = TabPFNUnsupervisedModel(tabpfn_clf=clf, tabpfn_reg=reg)

        X_tensor = torch.tensor(X, dtype=torch.float32)

        print(f"[{dataset_name}] 拟合数据分布 (fit)...")
        model.fit(X_tensor)

        print(f"[{dataset_name}] 计算异常得分 (outliers)...")
        scores = model.outliers(X_tensor, n_permutations=5)
        if hasattr(scores, "numpy"):
            scores = scores.numpy()
        scores = np.asarray(scores, dtype=np.float64).ravel()

        print(f"  数据集 {dataset_name}: TabPFN-AD 完成，得分范围 [{scores.min():.4f}, {scores.max():.4f}]")

    except Exception as _e:
        print(f"  TabPFN 不可用 ({type(_e).__name__})，回退到 IsolationForest...")

    # ──── 方式 2：IsolationForest（fallback） ────
    if scores is None:
        from sklearn.ensemble import IsolationForest

        iso = IsolationForest(contamination="auto", random_state=42, n_jobs=-1)
        preds = iso.fit_predict(X)
        raw = iso.decision_function(X)
        scores = -raw.astype(np.float64)

    # ──── 保存异常点表格 ────
    # 用 IQR 确定阈值：得分 > Q3 + 1.5*IQR 视为异常
    threshold = np.percentile(scores, 95)

    out_df = df.copy()
    out_df["anomaly_score"] = scores
    out_df["is_outlier"] = scores >= threshold

    out_path = f"results/{dataset_name}_sota_outliers.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    n_out = out_df["is_outlier"].sum()
    print(
        f"  数据集 {dataset_name}: 异常点 {n_out}/{len(df)} "
        f"({100*n_out/len(df):.1f}%) → {out_path}"
    )
    return scores


# ─── 归一化 ──────────────────────────────────────────────────────────────────
def normalize_data(df):
    """
    标准化(归一化)处理。将两张表格中的数据进行归一化处理 (Min-Max scaling)。
    quality也归一化。
    """
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(df)
    normalized_df = pd.DataFrame(normalized_data, columns=df.columns, index=df.index)
    return normalized_df


# ─── 表格合并 ────────────────────────────────────────────────────────────────
def merge_datasets(red_df, white_df):
    """
    表格合并。对齐两张葡萄酒属性，生成新的表格，增加type字段，使用0表示红葡萄酒，1表示白葡萄酒
    合并前先归一化，保存到 data/winequality.csv 中
    """
    norm_red = normalize_data(red_df.copy())
    norm_white = normalize_data(white_df.copy())

    norm_red['type'] = 0
    norm_white['type'] = 1

    merged_df = pd.concat([norm_red, norm_white], ignore_index=True)

    os.makedirs('data', exist_ok=True)
    merged_df.to_csv('data/winequality.csv', index=False)
    print("合并后的数据集已保存至 data/winequality.csv")

    return merged_df


# ─── main ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    red_df = pd.read_csv("data/winequality-red.csv", sep=";")
    white_df = pd.read_csv("data/winequality-white.csv", sep=";")

    # check_missing_and_duplicates(red_df, "RED")
    basic_outlier_detection(red_df, "RED")
    # check_missing_and_duplicates(white_df, "WHITE")
    basic_outlier_detection(white_df, "WHITE")

    sota_outlier_detection(red_df, "RED")
    sota_outlier_detection(white_df, "WHITE")
