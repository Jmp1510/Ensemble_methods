# -*- coding: utf-8 -*-
"""customer_segmentation.ipynb

**Step :- 1 Data Preprocessing, Visualization, & Scaling**
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load dataset
df = pd.read_csv('datasets/customer_segmentation.csv')

# Handling missing values
columns_with_modes = ['Ever_Married', 'Work_Experience', 'Family_Size', 'Graduated', 'Profession']
for col in columns_with_modes:
    df[col] = df[col].fillna(df[col].mode()[0])

# Encoding binary categorical variables
binary_mappings = {
    'Gender': {'Male': 1, 'Female': 0},
    'Ever_Married': {'Yes': 1, 'No': 0},
    'Graduated': {'Yes': 1, 'No': 0}
}
for col, mapping in binary_mappings.items():
    df[col] = df[col].map(mapping)

# Encoding multi-class categorical variables
le = LabelEncoder()
categorical_cols = ['Spending_Score', 'Profession']
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Dropping columns
df.drop(columns=['ID', 'Var_1'], inplace=True)

# Plot the distribution of Segmentation
plt.figure(figsize=(8, 6))
sns.countplot(x='Segmentation', hue='Segmentation', data=df, palette='Set2', dodge=False, legend=False)
plt.title('Distribution of Segmentation')
plt.xlabel('Segmentation')
plt.ylabel('Count')
plt.show()

# Splitting the data
X = df.drop(columns='Segmentation').copy()
y = df['Segmentation']

# Encoding the target variable
y_encoded = le.fit_transform(y)

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(df.info())
print(df.head)

"""
for col in df.select_dtypes(include=['int64', 'float64']).columns:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True, bins=20, color='red')
    plt.title(f'Distribution of {col}')
    plt.show()
"""

# Identifying numerical features
numerical_features = X_train.select_dtypes(include=['number']).columns

"""
# Calculating and visualizing the correlation matrix
correlation_matrix = df[numerical_features].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, vmin=-1, vmax=1)
plt.title('Correlation Matrix of Numerical Features')
plt.show()
"""

from sklearn.cluster import KMeans

# Calculating Within-Cluster Sum of Squares (WCSS) for different cluster sizes
wcss_original = []
cluster_range = range(1, 11)

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_train_scaled)
    wcss_original.append(kmeans.inertia_)

# Visualizing the Elbow Plot
plt.figure(figsize=(8, 6))
plt.plot(cluster_range, wcss_original, marker='o', linestyle='-', color='blue')
plt.title('Elbow Method (Original Data)')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS(Within-Cluster Sum of Squares)')
plt.axvline(x=2, color='r', linestyle='--')
plt.grid(True)
plt.show()

"""**Step :- 2 Dimensionality Reduction with PCA**"""

from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
import numpy as np

# Applying PCA
pca = PCA()
pca.fit(X_train_scaled)

# Visualizing the first two principal components (2D PCA)
pca_2d = PCA(n_components=2)
pca_data_2d = pca_2d.fit_transform(X_train_scaled)

# Create a scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(pca_data_2d[:, 0], pca_data_2d[:, 1], c=y_train, cmap='viridis', s=15, alpha=0.7)
plt.colorbar(label='Target Labels')
plt.title('PCA Visualization (2D)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()

# Scree Plot (Explained Variance Ratio)
plt.figure(figsize=(12, 6))
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, color='blue')
plt.title('Scree Plot (Explained Variance Ratio)')
plt.xlabel('Number of Principal Components')
plt.ylabel('Variance Ratio')
plt.xticks(rotation=90)
plt.grid(axis='y')
plt.show()

# Scree Plot (Eigenvalues)
plt.figure(figsize=(12, 6))
plt.bar(range(1, len(pca.explained_variance_) + 1), pca.explained_variance_, color='blue')
plt.title('Scree Plot (Eigenvalues)')
plt.xlabel('Number of Principal Components')
plt.ylabel('Eigenvalues')
plt.xticks(rotation=90)
plt.grid(axis='y')
plt.show()

# Cumulative Variance Plot
variance_explained_threshold = 0.90
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
num_components = np.argmax(cumulative_variance >= variance_explained_threshold) + 1

plt.figure(figsize=(12, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-', color='red')
plt.axhline(y=variance_explained_threshold, color='blue', linestyle='--')
plt.axvline(x=num_components, color='black', linestyle='--')
plt.title('Cumulative Explained Variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance')
plt.grid(True)
plt.show()

print(f"Number of components explaining {variance_explained_threshold:.0%} variance: {num_components}")

# PCA with a fixed number of components
num_components_retained = 6
pca_model = PCA(n_components=num_components_retained)
pca_data_train = pca_model.fit_transform(X_train_scaled)
pca_data_test = pca_model.transform(X_test_scaled)

# Calculate Trustworthiness for the PCA Model
trust_score = trustworthiness(X_train_scaled, pca_data_train, n_neighbors=5)
print(f"Trustworthiness of PCA Model with {num_components_retained} components: {trust_score:.4f}")

# Elbow Plot for PCA-Reduced Data
wcss_pca = []

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pca_data_train)
    wcss_pca.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(cluster_range, wcss_pca, marker='o', linestyle='-', color='red')
plt.title('Elbow Method (PCA-Reduced Data)')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS(Within-cluster sum of squares)')
plt.axvline(x=2, color='r', linestyle='--')
plt.grid(True)
plt.show()

"""**Step :- 3 Dimensionality Reduction with UMAP**"""

import warnings
warnings.filterwarnings("ignore", message=".*Falling back to random initialisation.*")

import umap

# Initial UMAP model
ump = umap.UMAP(n_neighbors=5, min_dist=0.25)
ump.fit(X_train_scaled)

umap_data_train = ump.transform(X_train_scaled)
current_trustworthiness = trustworthiness(X_train_scaled, umap_data_train)

highest_umap_trustworthiness = 0
best_umap = None
best_umap_data = None
best_num_components = None

# Define parameter grid for manual tuning
umap_param_grid = {'n_neighbors': [5, 10, 15, 20], 'min_dist': [0.25, 0.5, 0.75]}

for n_neighbors in umap_param_grid['n_neighbors']:
    for min_dist in umap_param_grid['min_dist']:
        ump = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, init='random')
        ump.fit(X_train_scaled)
        umap_data_train = ump.transform(X_train_scaled)
        current_trustworthiness = trustworthiness(X_train_scaled, umap_data_train)

        if current_trustworthiness > highest_umap_trustworthiness:
            highest_umap_trustworthiness = current_trustworthiness
            best_umap = ump
            best_umap_data = umap_data_train
            best_num_components = umap_data_train.shape[1]

# Display the tuned parameters
best_params = {k: v for k, v in best_umap.get_params().items() if k in ['n_neighbors', 'min_dist']}
print("Best UMAP parameters:", best_params)
print("Best number of components:", best_num_components)
print("Best trustworthiness:", highest_umap_trustworthiness)

# Transform data using the best UMAP model
umap_data_train_best = best_umap.transform(X_train_scaled)
umap_data_test_best = best_umap.transform(X_test_scaled)

plt.figure(figsize=(10, 8))
plt.scatter(umap_data_train_best[:, 0], umap_data_train_best[:, 1], c=y_train, cmap='viridis', s=15, alpha=0.7)
plt.colorbar(label='Target Labels')
plt.title('UMAP Visualization (2D)')
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.grid(True)
plt.show()

wcss_umap = []

for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(umap_data_train_best)
    wcss_umap.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(cluster_range, wcss_umap, marker='o', linestyle='-', color='g')
plt.title('Elbow Plot (UMAP-Reduced Data)')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS(Within cluster sum of squares)')
plt.axvline(x=4, color='r', linestyle='--')
plt.grid(True)
plt.show()

"""**Step :- 4 Clustering and Evaluation & Visualizations**"""

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# Define parameter grids for KMeans and Agglomerative Clustering
kmeans_param_grid = {'n_clusters': [2, 3, 4, 5, 6]}
agglo_param_grid = {'n_clusters': [2, 3, 4, 5, 6], 'linkage': ['ward', 'single', 'complete', 'average']}

# Initialize data frames
silhouette_scores = pd.DataFrame(index=['Original', 'PCA Reduced', 'UMAP Reduced'],
                                 columns=['KMeans', 'Agglomerative', 'Best KMeans Clusters',
                                          'Best Agglo Clusters', 'Best Agglo Linkage'])

# Dictionaries to track best clusters
best_num_clusters_kmeans = {}
best_num_clusters_agglo = {}

# Loop through datasets
for method, method_name in zip([X_train_scaled, pca_data_train, umap_data_train_best],
                                ['Original', 'PCA Reduced', 'UMAP Reduced']):
    # KMeans Clustering
    best_kmeans_score = -1
    for n_clusters in kmeans_param_grid['n_clusters']:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(method)
        kmeans_score = silhouette_score(method, kmeans.labels_)
        if kmeans_score > best_kmeans_score:
            best_kmeans_score = kmeans_score
            best_num_clusters_kmeans[method_name] = n_clusters

    # Update silhouette scores DataFrame for KMeans
    silhouette_scores.loc[method_name, 'KMeans'] = best_kmeans_score
    silhouette_scores.loc[method_name, 'Best KMeans Clusters'] = best_num_clusters_kmeans[method_name]

    # Agglomerative Clustering
    best_agglo_score = -1
    for n_clusters in agglo_param_grid['n_clusters']:
        for linkage_type in agglo_param_grid['linkage']:
            agglo = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_type)
            agglo.fit(method)
            agglo_score = silhouette_score(method, agglo.labels_)
            if agglo_score > best_agglo_score:
                best_agglo_score = agglo_score
                best_num_clusters_agglo[method_name] = (n_clusters, linkage_type)

    # Update silhouette scores DataFrame for Agglomerative Clustering
    best_n_clusters, best_linkage = best_num_clusters_agglo[method_name]
    silhouette_scores.loc[method_name, 'Agglomerative'] = best_agglo_score
    silhouette_scores.loc[method_name, 'Best Agglo Clusters'] = best_n_clusters
    silhouette_scores.loc[method_name, 'Best Agglo Linkage'] = best_linkage

# Final output for silhouette scores
print("Silhouette Scores & Best clusters:")
print(silhouette_scores)

# Clustering Visualization: KMeans Plots
fig, axes = plt.subplots(2, figsize=(18, 12))
method = KMeans(n_clusters=2, random_state=42)

method.fit(X_train_scaled)
axes[0].scatter(X_train_scaled[:, 4], X_train_scaled[:, 5], c=method.labels_, cmap='viridis')
axes[0].set_title('KMeans Clustering (Original Data)')

method.fit(pca_data_train)
axes[1].scatter(pca_data_train[:, 4], pca_data_train[:, 5], c=method.labels_, cmap='viridis')
axes[1].set_title('KMeans Clustering (PCA Reduced Data)')

plt.tight_layout()
plt.show()

# Dendrogram Visualization
datasets = {
    'Original Data': X_train_scaled,
    'PCA Reduced Data': pca_data_train,
    'UMAP Reduced Data': umap_data_train_best
}

for dataset_name, data in datasets.items():
    print(f"\nDendrogram for {dataset_name}:")

    # Compute the linkage matrix using the 'ward' method
    linkage_matrix = linkage(data, method='ward')

    # Create a dendrogram
    plt.figure(figsize=(12, 8))
    dendrogram(linkage_matrix, truncate_mode='level', p=5, leaf_rotation=90, leaf_font_size=10)
    plt.title(f'Dendrogram for Hierarchical Clustering ({dataset_name})')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.axhline(y=90, color='r', linestyle='--')
    plt.grid(True)
    plt.show()

    # Use fcluster to extract clusters based on a distance threshold
    num_clusters_from_dendrogram = fcluster(linkage_matrix, t=90, criterion='distance')
    print(f"Clusters extracted from dendrogram ({dataset_name}):", len(set(num_clusters_from_dendrogram)))

# Visualization of KMeans on UMAP-reduced data
best_n_clusters_kmeans_umap = best_num_clusters_kmeans['UMAP Reduced']
kmeans_umap = KMeans(n_clusters=best_n_clusters_kmeans_umap, random_state=42)

# Fit the KMeans model to the UMAP-reduced data
kmeans_umap.fit(umap_data_train_best)

# Plot the clustering results for UMAP-reduced data
plt.figure(figsize=(10, 8))
plt.scatter(umap_data_train_best[:, 0], umap_data_train_best[:, 1], c=kmeans_umap.labels_, cmap='viridis', s=50)
plt.title(f'KMeans Clustering on UMAP Reduced Data ({best_n_clusters_kmeans_umap} Clusters)')
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')

# add cluster centers
centers = kmeans_umap.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.legend()

plt.show()

plt.figure(figsize=(10, 8))
plt.plot(cluster_range, wcss_original, marker='o', linestyle='-', color='b', label='Original Data')
plt.plot(cluster_range, wcss_pca, marker='o', linestyle='-', color='r', label='PCA-Reduced Data')
plt.plot(cluster_range, wcss_umap, marker='o', linestyle='-', color='g', label='UMAP-Reduced Data')
plt.title('Elbow Plot Comparison')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS(Within cluster sum of squares)')
plt.legend()
plt.grid(True)
plt.show()

"""**Step :- 5 Tuning and Evaluation of AdaBoost and Random Forest**


"""

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, validation_curve, learning_curve
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time


data_transformations = {
    'Original': (X_train_scaled, X_test_scaled),
    'PCA Reduced': (pca_data_train, pca_data_test),
    'UMAP Reduced': (umap_data_train_best, umap_data_test_best)
}

adaboost = AdaBoostClassifier(algorithm='SAMME', random_state=10)
adaboost_hyperparameters = {'n_estimators': [50, 100, 150, 200],'learning_rate':[0.5,1.0,1.5,2.0]}
rf = RandomForestClassifier(random_state=10,class_weight='balanced')
random_forest_hyperparameters = {'n_estimators': [50, 100, 150, 200], 'max_depth': [5, 10, 15, 20]}

optimal_models = {}
training_errors = {}
test_errors = {}
training_times = {}
test_times = {}

# Function to plot validation curve
def plot_validation_curve(estimator, X, y, param_name, param_range, model_name, method_name, cv=5, scoring='accuracy'):
    train_scores, val_scores = validation_curve(estimator, X, y, param_name=param_name, param_range=param_range, cv=cv, scoring=scoring)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(param_range, train_mean, label='Training Score', color='r')
    plt.plot(param_range, val_mean, label='Validation Score', color='b')
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color='r', alpha=0.2)
    plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, color='b', alpha=0.2)
    plt.title(f"Validation Curve for Customer ({method_name} - {model_name})\n{param_name}")
    plt.xlabel(f"Value of {param_name}")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.show()

# Function to plot learning curve
def plot_learning_curve(estimator, X, y, model_name, method_name, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10)):
    train_sizes, train_scores, val_scores = learning_curve(estimator, X, y, train_sizes=train_sizes, cv=cv, scoring=scoring)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, label='Training Score', color='r')
    plt.plot(train_sizes, val_mean, label='Validation Score', color='b')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='r', alpha=0.2)
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, color='b', alpha=0.2)
    plt.title(f"Learning Curve for Customer ({method_name} - {model_name})")
    plt.xlabel("Training Size")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.show()

# Loop through each data transformation and perform model training/evaluation
for method_name, (X_train_method, X_test_method) in data_transformations.items():
    # AdaBoost
    start_time = time.time()
    adaboost_grid_search = GridSearchCV(adaboost, adaboost_hyperparameters, cv=5, scoring='accuracy')
    adaboost_grid_search.fit(X_train_method, y_train)
    end_time = time.time()
    adaboost_training_time = end_time - start_time
    best_adaboost = adaboost_grid_search.best_estimator_
    optimal_models[method_name + ' AdaBoost'] = best_adaboost

    # Random Forest
    start_time = time.time()
    rf_grid = GridSearchCV(rf, random_forest_hyperparameters, cv=5, scoring='accuracy')
    rf_grid.fit(X_train_method, y_train)
    end_time = time.time()
    rf_training_time = end_time - start_time
    best_rf = rf_grid.best_estimator_
    optimal_models[method_name + ' Random Forest'] = best_rf

    # Predictions on test data
    adaboost_pred = best_adaboost.predict(X_test_method)
    rf_pred = best_rf.predict(X_test_method)

    # Training error
    adaboost_train_pred = best_adaboost.predict(X_train_method)
    rf_train_pred = best_rf.predict(X_train_method)
    adaboost_train_error = 1 - accuracy_score(y_train, adaboost_train_pred)
    rf_train_error = 1 - accuracy_score(y_train, rf_train_pred)
    training_errors[method_name + ' AdaBoost'] = adaboost_train_error
    training_errors[method_name + ' Random Forest'] = rf_train_error

    # Metrics for AdaBoost
    print(f"\nEvaluation Metrics for {method_name} AdaBoost:")
    print(classification_report(y_test, adaboost_pred, zero_division=0))
    print(f"Best Hyperparameters for {method_name} AdaBoost: {adaboost_grid_search.best_params_}")
    print(f"Training Time for {method_name} AdaBoost: {adaboost_training_time:.2f} seconds")
    try:
        roc_auc = roc_auc_score(y_test, best_adaboost.predict_proba(X_test_method), multi_class='ovr')
        print(f"ROC-AUC: {roc_auc}")
    except AttributeError:
        print("ROC-AUC not available for AdaBoost with SAMME algorithm.")
    print(f"Training Error for {method_name} AdaBoost: {adaboost_train_error:.4f}")

    # Metrics for Random Forest
    print(f"\nEvaluation Metrics for {method_name} Random Forest:")
    print(classification_report(y_test, rf_pred))
    print(f"Best Hyperparameters for {method_name} Random Forest: {rf_grid.best_params_}")
    print(f"Training Time for {method_name} Random Forest: {rf_training_time:.2f} seconds")
    try:
        roc_auc = roc_auc_score(y_test, best_rf.predict_proba(X_test_method), multi_class='ovr')
        print(f"ROC-AUC: {roc_auc}")
    except AttributeError:
        print("ROC-AUC not available for Random Forest.")
    print(f"Training Error for {method_name} Random Forest: {rf_train_error:.4f}")

    # Confusion Matrix for each model
    for model_name, pred in [("AdaBoost", adaboost_pred), ("Random Forest", rf_pred)]:
        cm = confusion_matrix(y_test, pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix: {method_name} {model_name}")
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

    # Validation Curve for AdaBoost
    print(f"\nValidation Curve for {method_name} AdaBoost:")
    plot_validation_curve(AdaBoostClassifier(algorithm='SAMME', random_state=10),
                          X_train_method, y_train,
                          param_name="n_estimators", param_range=adaboost_hyperparameters['n_estimators'], model_name="AdaBoost", method_name=method_name)

    # Validation Curve for Random Forest
    print(f"\nValidation Curve for {method_name} Random Forest:")
    plot_validation_curve(RandomForestClassifier(random_state=10),
                          X_train_method, y_train,
                          param_name="n_estimators", param_range=random_forest_hyperparameters['n_estimators'], model_name="Random Forest", method_name=method_name)
    print(f"\nValidation Curve for {method_name} Random Forest (max_depth):")
    plot_validation_curve(RandomForestClassifier(random_state=10),
                          X_train_method, y_train,
                          param_name="max_depth", param_range=random_forest_hyperparameters['max_depth'], model_name="Random Forest", method_name=method_name)

    # Learning Curve for AdaBoost
    print(f"\nLearning Curve for {method_name} AdaBoost:")
    plot_learning_curve(best_adaboost, X_train_method, y_train, model_name="AdaBoost", method_name=method_name)

    # Learning Curve for Random Forest
    print(f"\nLearning Curve for {method_name} Random Forest:")
    plot_learning_curve(best_rf, X_train_method, y_train, model_name="Random Forest", method_name=method_name)