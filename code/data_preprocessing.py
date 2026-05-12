import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from scipy import stats

def check_missing_and_duplicates(df, dataset_name):
    """
    缺失值统计和重复值统计。
    输出为控制台输出数据集名称：缺失值和重复值的数量统计
    """
    missing_count = df.isnull().sum().sum()
    duplicate_count = df.duplicated().sum()
    print(f"数据集 {dataset_name}: 缺失值数量 = {missing_count}, 重复值数量 = {duplicate_count}")

def basic_outlier_detection(df, dataset_name):
    """
    基础异常值识别。绘制数据集箱线图和使用Z-Scores识别异常
    报告异常对象，阈值为3，将箱线图保存到results/中
    """
    os.makedirs('results', exist_ok=True)

    plt.figure(figsize=(15, 10))
    df.boxplot()
    plt.title(f'Boxplot for {dataset_name}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'results/{dataset_name}_boxplot.png')
    plt.close()

    z_scores = np.abs(stats.zscore(df))
    outliers = (z_scores > 3).sum().sum()
    print(f"数据集 {dataset_name}: Z-Score (threshold=3) 识别出的异常值总数 = {outliers}")

def sota_outlier_detection(df, dataset_name):
    """
    SOTA异常识别。使用最新的基础模型，对理化指标的联合分布进行异常点扫描。
    """
    try:
        from tabpfn_extensions.unsupervised import TabPFNUnsupervisedModel
        import numpy as np

        X = df.values

        # User specified this code exactly in their prompt:
        model = TabPFNUnsupervisedModel()
        scores = model.predict_outlier_scores(X)

        print(f"数据集 {dataset_name}: TabPFN-AD 异常点扫描完成，部分异常得分: {scores[:5] if scores is not None else None}")
        return scores
    except Exception as e:
        print(f"SOTA异常识别出现错误: {e}")
        return None

def normalize_data(df):
    """
    标准化(归一化)处理。将两张表格中的数据进行归一化处理 (Min-Max scaling)。
    quality也归一化。
    """
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(df)
    normalized_df = pd.DataFrame(normalized_data, columns=df.columns, index=df.index)
    return normalized_df

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
