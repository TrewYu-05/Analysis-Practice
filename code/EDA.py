import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def load_and_prepare_data():
    """
    Loads raw red and white wine data, adds a 'type' column, and combines them.
    Uses the raw data to maintain physical meaning of the features.
    """
    # Create results directory if it doesn't exist
    os.makedirs("../results", exist_ok=True)

    # Load data
    red_df = pd.read_csv("../data/winequality-red.csv", sep=";")
    white_df = pd.read_csv("../data/winequality-white.csv", sep=";")

    red_df['type'] = 'red'
    white_df['type'] = 'white'

    # Combine the datasets
    df = pd.concat([red_df, white_df], ignore_index=True)

    return df

def overall_distribution_analysis(df):
    """
    Core Task 1: Overall Distribution Analysis
    - Calculates descriptive statistics (mean, variance, skewness, kurtosis)
    - Saves statistics to a CSV
    - Generates and saves distribution plots (histograms + KDE) and boxplots for all features
    """
    print("Starting Overall Distribution Analysis...")

    # Exclude 'type' column for statistical calculations
    numeric_df = df.select_dtypes(include=[np.number])

    # 1. Descriptive Statistics
    stats_df = numeric_df.describe().T
    stats_df['variance'] = numeric_df.var()
    stats_df['skewness'] = numeric_df.skew()
    stats_df['kurtosis'] = numeric_df.kurt()

    stats_df.to_csv("../results/overall_statistics.csv")
    print("Saved overall statistics to results/overall_statistics.csv")

    # 2. Distribution Plots (Histograms with KDE)
    features = [col for col in numeric_df.columns if col != 'quality']

    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 15))
    axes = axes.flatten()

    for i, col in enumerate(features):
        sns.histplot(df[col], kde=True, ax=axes[i], color='skyblue')
        axes[i].set_title(f"Distribution of {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")

    for j in range(len(features), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig("../results/overall_distributions.svg")
    plt.savefig("../results/overall_distributions.pdf")
    plt.close()
    print("Saved overall distribution plots.")

    # 3. Boxplots for Outlier Detection
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 15))
    axes = axes.flatten()

    for i, col in enumerate(features):
        sns.boxplot(y=df[col], ax=axes[i], color='lightgreen')
        axes[i].set_title(f"Boxplot of {col}")

    for j in range(len(features), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig("../results/overall_boxplots.svg")
    plt.savefig("../results/overall_boxplots.pdf")
    plt.close()
    print("Saved overall boxplots.")


def red_vs_white_analysis(df):
    """
    Core Task 2: Red vs. White Wine Differences
    - Compare feature distributions using grouped statistics and Mann-Whitney U test.
    - Generate comparative plots (violin plots).
    """
    print("Starting Red vs. White Wine Analysis...")

    # Exclude quality for feature differences
    features = [col for col in df.columns if col not in ['quality', 'type']]

    # 1. Grouped Statistics and Hypothesis Testing
    red_df = df[df['type'] == 'red']
    white_df = df[df['type'] == 'white']

    stats_list = []
    for col in features:
        red_data = red_df[col].dropna()
        white_data = white_df[col].dropna()

        # Mann-Whitney U test for non-parametric distribution comparison
        stat, p_val = stats.mannwhitneyu(red_data, white_data, alternative='two-sided')

        stats_list.append({
            'Feature': col,
            'Red_Mean': red_data.mean(),
            'White_Mean': white_data.mean(),
            'Red_Median': red_data.median(),
            'White_Median': white_data.median(),
            'U_Statistic': stat,
            'p_value': p_val,
            'Significant_Difference (alpha=0.05)': p_val < 0.05
        })

    diff_df = pd.DataFrame(stats_list)
    diff_df.to_csv("../results/red_vs_white_stats.csv", index=False)
    print("Saved red vs white comparative statistics.")

    # 2. Comparative Plots (Violin Plots)
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 15))
    axes = axes.flatten()

    for i, col in enumerate(features):
        sns.violinplot(x='type', y=col, data=df, ax=axes[i], palette="muted", inner="quartile", hue='type', legend=False)
        axes[i].set_title(f"{col} by Wine Type")

    for j in range(len(features), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig("../results/red_vs_white_violinplots.svg")
    plt.savefig("../results/red_vs_white_violinplots.pdf")
    plt.close()
    print("Saved red vs white violin plots.")


def quality_and_relationship_analysis(df):
    """
    Core Task 3: Quality Differences & Variable Relationships
    - Boxplots for features grouped by quality
    - Correlation matrix and heatmap
    """
    print("Starting Quality and Relationship Analysis...")

    # 1. Quality Differences (Boxplots grouped by quality)
    features = [col for col in df.columns if col not in ['quality', 'type']]

    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 15))
    axes = axes.flatten()

    for i, col in enumerate(features):
        sns.boxplot(x='quality', y=col, data=df, ax=axes[i], palette="viridis", hue='quality', legend=False)
        axes[i].set_title(f"{col} by Quality")

    for j in range(len(features), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig("../results/quality_boxplots.svg")
    plt.savefig("../results/quality_boxplots.pdf")
    plt.close()
    print("Saved quality grouped boxplots.")

    # 2. Variable Relationships (Correlation Matrix & Heatmap)
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()

    # Save correlation matrix
    corr_matrix.to_csv("../results/correlation_matrix.csv")
    print("Saved correlation matrix to CSV.")

    # Plot heatmap
    plt.figure(figsize=(12, 10))
    # We use a mask to only show the lower triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm',
                vmin=-1, vmax=1, center=0, square=True, linewidths=.5)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("../results/correlation_heatmap.svg")
    plt.savefig("../results/correlation_heatmap.pdf")
    plt.close()
    print("Saved correlation heatmap.")


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def advanced_analysis(df):
    """
    Advanced Analysis (Exceeding Requirements)
    - Dimensionality Reduction (PCA) to visualize red vs white and quality clusters.
    - Feature Importance using Random Forest to find which variables predict quality best.
    """
    print("Starting Advanced Analysis...")

    # Exclude non-numeric for PCA/RF
    X = df.drop(columns=['quality', 'type'])
    y_quality = df['quality']
    y_type = df['type']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. PCA Visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
    df_pca['type'] = y_type
    df_pca['quality'] = y_quality

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # PCA colored by Wine Type
    sns.scatterplot(x='PC1', y='PC2', hue='type', data=df_pca, ax=axes[0], palette='Set1', alpha=0.5)
    axes[0].set_title(f"PCA of Wine Data (colored by Type)\nExplained Variance: {sum(pca.explained_variance_ratio_):.2f}")

    # PCA colored by Quality
    # Treat quality as continuous for colormap purposes here
    scatter = axes[1].scatter(df_pca['PC1'], df_pca['PC2'], c=df_pca['quality'], cmap='viridis', alpha=0.5)
    axes[1].set_title("PCA of Wine Data (colored by Quality)")
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    plt.colorbar(scatter, ax=axes[1], label='Quality')

    plt.tight_layout()
    plt.savefig("../results/pca_analysis.svg")
    plt.savefig("../results/pca_analysis.pdf")
    plt.close()
    print("Saved PCA visualization.")

    # 2. Feature Importance (Random Forest)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y_quality)

    importances = rf.feature_importances_
    feature_names = X.columns

    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='magma', hue='Feature', legend=False)
    plt.title("Feature Importance for Predicting Quality (Random Forest)")
    plt.tight_layout()
    plt.savefig("../results/feature_importance.svg")
    plt.savefig("../results/feature_importance.pdf")
    plt.close()

    importance_df.to_csv("../results/feature_importance.csv", index=False)
    print("Saved feature importance plot and CSV.")

if __name__ == "__main__":
    df = load_and_prepare_data()
    print(f"Data loaded successfully. Total rows: {len(df)}")
    overall_distribution_analysis(df)
    red_vs_white_analysis(df)
    quality_and_relationship_analysis(df)
    advanced_analysis(df)
