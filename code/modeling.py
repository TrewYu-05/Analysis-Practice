import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# ML metrics
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    silhouette_score, davies_bouldin_score
)

# ML models
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# TabPFN
from tabpfn import TabPFNRegressor, TabPFNClassifier

# SCARF
import torch
from torch.utils.data import DataLoader, TensorDataset
from scarf.model import SCARF
from scarf.loss import NTXent
import torch.optim as optim


def load_datasets():
    """
    Loads the datasets and returns them as configurations.
    """
    # Load combined dataset
    df = pd.read_csv("data/winequality.csv")

    # Return different configurations
    configs = []

    # Configuration 1: Combined with 'type' feature
    configs.append({
        'name': 'Combined_with_type',
        'X': df.drop(columns=['quality']),
        'y': df['quality'],
        'type_col': True
    })

    # Configuration 2: Combined without 'type' feature
    configs.append({
        'name': 'Combined_without_type',
        'X': df.drop(columns=['quality', 'type']),
        'y': df['quality'],
        'type_col': False
    })

    # Configuration 3: Separate Red
    df_red = df[df['type'] == 0].drop(columns=['type'])
    configs.append({
        'name': 'Red_only',
        'X': df_red.drop(columns=['quality']),
        'y': df_red['quality'],
        'type_col': False
    })

    # Configuration 4: Separate White
    df_white = df[df['type'] == 1].drop(columns=['type'])
    configs.append({
        'name': 'White_only',
        'X': df_white.drop(columns=['quality']),
        'y': df_white['quality'],
        'type_col': False
    })

    return configs

def run_regression(configs):
    """
    Runs regression models on all dataset configurations.
    """
    print("--- Running Regression Tasks ---")
    results = []

    for conf in configs:
        name = conf['name']
        X = conf['X'].values
        y = conf['y'].values

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define models
        models = {
            'LinearRegression': LinearRegression(),
            'RandomForestRegressor': RandomForestRegressor(n_estimators=100, random_state=42),
            'TabPFNRegressor': TabPFNRegressor(device='cpu')
        }

        for model_name, model in models.items():
            print(f"[{name}] Training {model_name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            results.append({
                'Configuration': name,
                'Model': model_name,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            })

    # Save results
    df_res = pd.DataFrame(results)
    df_res.to_csv("results/regression_results.csv", index=False)
    print("Regression tasks complete. Results saved to results/regression_results.csv\n")
    return df_res

def run_classification(configs):
    """
    Runs classification models on all dataset configurations.
    """
    print("--- Running Classification Tasks ---")
    results = []

    for conf in configs:
        name = conf['name']
        X = conf['X'].values
        y_raw = conf['y'].values

        # Grouping quality into 3 categories: Low (<=5), Medium (6), High (>=7)
        y = np.where(y_raw <= 5, 0, np.where(y_raw == 6, 1, 2))

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define models
        models = {
            'LogisticRegression': LogisticRegression(max_iter=1000),
            'RandomForestClassifier': RandomForestClassifier(n_estimators=100, random_state=42),
            'TabPFNClassifier': TabPFNClassifier(device='cpu')
        }

        for model_name, model in models.items():
            print(f"[{name}] Training {model_name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted')
            rec = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            cm = confusion_matrix(y_test, y_pred)

            # Plot and save confusion matrix
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Low', 'Medium', 'High'],
                        yticklabels=['Low', 'Medium', 'High'])
            plt.title(f'Confusion Matrix - {name} - {model_name}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.tight_layout()
            cm_filename_prefix = f"results/CM_{name}_{model_name}"
            plt.savefig(f"{cm_filename_prefix}.svg")
            plt.savefig(f"{cm_filename_prefix}.pdf")
            plt.close()

            results.append({
                'Configuration': name,
                'Model': model_name,
                'Accuracy': acc,
                'Precision': prec,
                'Recall': rec,
                'F1_Score': f1
            })

    # Save results
    df_res = pd.DataFrame(results)
    df_res.to_csv("results/classification_results.csv", index=False)
    print("Classification tasks complete. Results saved to results/classification_results.csv\n")
    return df_res

def run_clustering(configs):
    """
    Runs clustering models on all dataset configurations.
    """
    print("--- Running Clustering Tasks ---")
    results = []

    for conf in configs:
        name = conf['name']
        X = conf['X'].values

        # We drop the actual y (quality) as clustering is unsupervised.
        # But we still evaluate the silhouette and DB scores based on X.

        # Define models
        # For K-Means and Agglomerative, we use k=3 to align with low/med/high quality roughly
        k = 3

        models = {
            'KMeans': KMeans(n_clusters=k, random_state=42),
            'Agglomerative': AgglomerativeClustering(n_clusters=k)
        }

        # Advanced: SCARF embeddings followed by KMeans
        print(f"[{name}] Training SCARF embeddings...")

        # Convert X to tensor
        X_tensor = torch.tensor(X, dtype=torch.float32)
        dataset = TensorDataset(X_tensor)
        data_loader = DataLoader(dataset, batch_size=128, shuffle=True)

        # SCARF setup
        input_dim = X.shape[1]

        # features_low and features_high must be equal in length to input_dim representing min and max of each column
        features_low = X.min(axis=0)
        features_high = X.max(axis=0)

        scarf = SCARF(
            input_dim=input_dim,
            features_low=features_low,
            features_high=features_high,
            dim_hidden_encoder=64,
            num_hidden_encoder=2,
            dim_hidden_head=64,
            num_hidden_head=2,
            corruption_rate=0.6
        )
        optimizer = optim.Adam(scarf.parameters(), lr=0.001)
        ntxent_loss = NTXent()

        scarf.train()
        epochs = 20
        for epoch in range(epochs):
            for batch in data_loader:
                x_batch = batch[0]
                optimizer.zero_grad()
                emb_anchor, emb_positive = scarf(x_batch)
                loss = ntxent_loss(emb_anchor, emb_positive)
                loss.backward()
                optimizer.step()

        # Extract embeddings
        scarf.eval()
        with torch.no_grad():
            embeddings = scarf.get_embeddings(X_tensor).numpy()

        # Run KMeans on SCARF embeddings
        models['SCARF_KMeans'] = KMeans(n_clusters=k, random_state=42)

        for model_name, model in models.items():
            print(f"[{name}] Training {model_name}...")

            if model_name == 'SCARF_KMeans':
                clusters = model.fit_predict(embeddings)
                eval_data = embeddings
            else:
                clusters = model.fit_predict(X)
                eval_data = X

            sil_score = silhouette_score(eval_data, clusters)
            db_score = davies_bouldin_score(eval_data, clusters)

            # PCA for visualization
            pca = PCA(n_components=2)
            pca_res = pca.fit_transform(eval_data)

            plt.figure(figsize=(6, 4))
            sns.scatterplot(x=pca_res[:, 0], y=pca_res[:, 1], hue=clusters, palette='viridis', alpha=0.6)
            plt.title(f'PCA Clustering - {name} - {model_name}')
            plt.xlabel('PCA 1')
            plt.ylabel('PCA 2')
            plt.tight_layout()
            cluster_filename_prefix = f"results/Cluster_PCA_{name}_{model_name}"
            plt.savefig(f"{cluster_filename_prefix}.svg")
            plt.savefig(f"{cluster_filename_prefix}.pdf")
            plt.close()

            results.append({
                'Configuration': name,
                'Model': model_name,
                'Silhouette_Score': sil_score,
                'Davies_Bouldin_Score': db_score
            })

    # Save results
    df_res = pd.DataFrame(results)
    df_res.to_csv("results/clustering_results.csv", index=False)
    print("Clustering tasks complete. Results saved to results/clustering_results.csv\n")
    return df_res

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    configs = load_datasets()

    run_regression(configs)
    run_classification(configs)
    run_clustering(configs)
