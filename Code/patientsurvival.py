# -*- coding: utf-8 -*-
"""patientSurvival.ipynb

# Patient Survival Dataset

## EDA/Preprocessing
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('datasets/patient_survival_prediction.csv')

"""
# Set the option to display all columns & rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
"""

# 91,713 observations, 85 features
df.shape

# Drop the id columns
df = df.drop(["encounter_id","patient_id","hospital_id","icu_id"],axis=1)
# Drop the null column
df = df.drop("Unnamed: 83",axis=1)

# Replace values in the same column
df['gender'] = df['gender'].map({'M': 0, 'F': 1})


# Features with correlation <.01
numerical_features = df.select_dtypes(include=['number'])
# numerical_features.corrwith(df['hospital_death'])[abs(numerical_features.corrwith(df['hospital_death'])) < 0.01]

df.shape

# Check for null values
df.isnull().sum()

# Handle missing values

from sklearn.impute import SimpleImputer

# Imputation of the null values in the 'height' column
imputer_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
df['height'] = imputer_mean.fit_transform(df[['height']])

# Imputation of the null values in the 'age', 'bmi', and 'weight' columns
columns_to_impute = ['age', 'bmi', 'weight']
imputer_median = SimpleImputer(missing_values=np.nan, strategy='median')
df[columns_to_impute] = imputer_median.fit_transform(df[columns_to_impute])

# Removal of the null values in the other features
df.dropna(inplace=True)


df.isnull().sum()

df.shape

# There are two values in apache_2_bodysystem that represent the same thing, therefore they are to be consolidated into one
df['apache_2_bodysystem'] = df['apache_2_bodysystem'].replace({'Undefined Diagnoses': 'Undefined diagnoses'})

# Explicitly state the categorical features
categorical_features = df.select_dtypes(include=['object'])
print("\nCategorical Features:")
categorical_features.columns

# Convert the categorical features from 'object' to 'category' data type
for col in categorical_features.columns:
    df[col] = df[col].astype('category')

df.isnull().sum()

# One-Hot Enconding
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_array = ohe.fit_transform(df[categorical_features.columns])
encoded_df = pd.DataFrame(encoded_array, columns=ohe.get_feature_names_out(categorical_features.columns), index=df.index)
df = pd.concat([df.drop(columns=categorical_features.columns), encoded_df], axis=1)

df.shape

# Plot od the distribution of the target variable (significant class imbalance)
target_counts=df['hospital_death'].value_counts()
plt.pie(target_counts, labels= ['0 (Survived)', '1 (Died)'], autopct='%1.1f%%', startangle=75, colors=['#a2d2ff','#bde0fe'])
plt.title('Distribution of Hospital Deaths')
plt.show()

"""
# Count the outliers present in the data
def count_outliers(df):
    outlier_counts = {}

    for feature in df.select_dtypes(include=[np.number]).columns:
        # ignore binary features
        if df[feature].nunique() == 2:
            outlier_counts[feature] = 0
            continue

        # Calculate the upper and lower limits for outliers
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        # Count outliers
        outliers = df[(df[feature] < lower) | (df[feature] > upper)]
        outlier_counts[feature] = outliers.shape[0]

    return outlier_counts
"""

# Ultimately, outlier removal is not applied on this dataset because APACHE medical scores are present.

# Train/test splitting of dataset (80/20)
from sklearn.model_selection import train_test_split

X = df.drop('hospital_death', axis=1)
y = df['hospital_death']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Apply RandomUnderSampler on the training data to balance the classes
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)
X_train, y_train = rus.fit_resample(X_train, y_train)

# Feature scaling
from sklearn.preprocessing import RobustScaler

rs = RobustScaler()
X_train_scaled = rs.fit_transform(X_train)
X_test_scaled = rs.transform(X_test)

"""## Clustering Methods - Original Data

### K-means
"""

# Perform k-means
from sklearn.cluster import KMeans

wcss = []
cluster_range = range(1, 11)  # it is sufficient to test from 1 to 10 clusters
for k in cluster_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X_train_scaled)
    wcss.append(kmeans.inertia_)

"""#### Elbow Plot (Original)"""

from matplotlib.ticker import MaxNLocator

# Plot the elbow curve
plt.figure(figsize=(8, 6))
plt.plot(cluster_range, wcss, marker='o')
plt.title('Elbow Plot for Patient Survival (Original Data)')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS(Within-Cluster Sum of Squares)')
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):}'))
plt.axvline(x=2, color='r', linestyle='--') # Add a dashed vertical line at the elbow point (k=2)
plt.show()

from sklearn.metrics import silhouette_score

silhouette_scores = []

# Sample the data (10% of the dataset)
# Set the random seed for reproducibility
np.random.seed(42)
n_samples = int(X_train_scaled.shape[0] * 0.1)
sample_indices = np.random.choice(X_train_scaled.shape[0], size=n_samples, replace=False)
sample_data = X_train_scaled[sample_indices]

for n_clusters in range(2, 7):
    clusterer = KMeans(n_clusters=n_clusters, random_state=42, init='k-means++')
    cluster_labels = clusterer.fit_predict(sample_data)
    silhouette_avg = silhouette_score(sample_data, cluster_labels)
    silhouette_scores.append({'Number of Clusters': n_clusters, 'Silhouette Score': silhouette_avg})

# Convert the scores into a DataFrame
silhouette_df = pd.DataFrame(silhouette_scores)

"""#### Silhouette Score (Original)"""

# Plot the silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(silhouette_df['Number of Clusters'], silhouette_df['Silhouette Score'], marker='o')
plt.title('Silhouette Score Analysis for Patient Survival (Original Data)')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.axvline(x=2, color='r', linestyle='--') # Add a dashed vertical line at the maximum silhouette score (k=2)
plt.show()

# Tabulate the silhouette scores
print("Silhouette Scores Table for Patient Survival (Original Data):")
print(silhouette_df.to_string(index=False))

"""#### Optimal K = 2 (Original)"""

# Get the optimal number of clusters based on the maximum silhouette score
optimal_clusters = silhouette_df.loc[silhouette_df['Silhouette Score'].idxmax(), 'Number of Clusters']
print(f"Optimal number of clusters for Patient Survival (Original Data) using K-means: {optimal_clusters}")

"""#### Scatter Plot (Original)"""

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(X_train_scaled)

# Visualize the clusters
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=kmeans_labels, cmap='coolwarm', alpha=0.6)
plt.title('K-Means Clustering for Patient Survival (Original Data)')
unique_clusters = np.unique(kmeans_labels)
legend_labels = [f'Cluster {i}' for i in unique_clusters]
plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

"""### Hierarchical Clustering

#### Optimal K = 2 (Original)
"""

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# Perform hierarchical clustering using the 'ward' method
linked = linkage(X_test_scaled, method='ward')

# Test different numbers of clusters (k) and calculate silhouette scores
max_k = 10
silhouette_scores = []

print("Silhouette Scores for Hierarchical Clustering - Patient Survival (Original Data):")
for k in range(2, max_k+1):
    cluster_labels = fcluster(linked, t=k, criterion='maxclust')
    if len(set(cluster_labels)) > 1:
        silhouette_avg = silhouette_score(X_test_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print(f"Number of Clusters: {k}, Silhouette Score: {silhouette_avg}")

# Find the best k (number of clusters) with the highest silhouette score
best_k = np.argmax(silhouette_scores) + 2
print(f"Best Number of Clusters (k): {best_k}")

"""#### Dendrogram (Original)"""

# Create the dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linkage(sample_data, method='ward')) # sampled 10% of the data to decrease runtime
plt.axhline(y=130, color='r', linestyle='--')
plt.title('Dendrogram for Hierarchical Clustering (Original Data)')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

"""#### Scatter Plot (Original)"""

# Visualize the clusters
cluster_labels = fcluster(linked, t=best_k, criterion='maxclust')
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=cluster_labels, cmap='coolwarm', alpha=0.6)
plt.title('Hierarchical Clustering for Patient Survival (Original Data)')
unique_clusters = np.unique(cluster_labels)
legend_labels = [f'Cluster {i}' for i in unique_clusters]
plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

"""## Dimensionality Reduction Algorithms

### PCA

#### Scatter Plot
"""

from sklearn.decomposition import PCA

# Apply PCA
pca = PCA(random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Visualize the first two principal components (2D PCA)
pca_2d = PCA(n_components=2)
pca_data_2d = pca_2d.fit_transform(X_train_scaled)

# Create a scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(pca_data_2d[:, 0], pca_data_2d[:, 1], c=y_train, cmap='coolwarm', alpha=0.7)
plt.title('PCA Visualization for Patient Survival (2D)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='0 = Survived, 1 = Died', ticks=[0, 1])
plt.grid(True)
plt.show()

"""#### Principal Components: 115"""

# Print the number of principal components
print(f"Number of principal components calculated for Patient Survival using PCA: {len(pca.explained_variance_ratio_)}")

"""#### Scree Plot - Explained Variance Ratio"""

# Scree Plot (Explained Variance Ratio)
threshold = 0.005  # 0.5% threshold
explained_variance = pca.explained_variance_ratio_

# Filter out components with less than 0.5% variance
filtered_variance = explained_variance[explained_variance >= threshold]

# Plot the scree plot with only components that contribute more than 0.5% variance
plt.figure(figsize=(12, 6))
plt.bar(range(1, len(filtered_variance) + 1), filtered_variance, color='blue')
plt.title('Scree Plot for Patient Survival (Explained Variance Ratio)')
plt.xlabel('Number of Principal Components')
plt.ylabel('Variance Ratio')
plt.xticks(range(1, len(filtered_variance) + 1))
plt.grid(axis='y')
plt.show()

"""#### Scree Plot - Eigenvalues"""

# Scree Plot (Eigenvalues)
filtered_eigenvalues = pca.explained_variance_[pca.explained_variance_ratio_ > threshold]
plt.figure(figsize=(12, 6))
plt.bar(range(1, len(filtered_eigenvalues) + 1), filtered_eigenvalues, color='blue')
plt.title('Scree Plot for Patient Survival (Eigenvalues)')
plt.xlabel('Number of Principal Components')
plt.ylabel('Eigenvalues')
plt.xticks(range(1, len(filtered_eigenvalues) + 1))
plt.grid(axis='y')
plt.show()

"""#### Cummulative Variance Plot"""

# Cumulative Variance Plot
threshold = 0.90
cumulative_variance = np.cumsum(filtered_variance)
num_components = np.argmax(cumulative_variance >= threshold) + 1

plt.figure(figsize=(12, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-', color='red')
plt.axhline(y=threshold, color='blue', linestyle='--', label=f'{threshold:.0%} Variance Threshold')
plt.axvline(x=num_components, color='black', linestyle='--', label=f'{num_components} Components')
plt.title('Cumulative Explained Variance for Patient Survival')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance')
plt.legend()
plt.grid(True)
plt.show()

"""#### 21 PC explain 90% Variance"""

print(f"Number of components explaining {threshold:.0%} variance in Patient Survival: {num_components}")

"""### UMAP"""

import umap

# Applying UMAP
umap_model = umap.UMAP(random_state=42)
X_train_umap = umap_model.fit_transform(X_train_scaled)
X_test_umap = umap_model.transform(X_test_scaled)

"""#### Scatter Plot"""

# Visualize the first two principal components (2D PCA)
umap_2d = umap.UMAP(n_components=2)
umap_data_2d = umap_2d.fit_transform(X_train_scaled)

# Create a scatter plot of the UMAP transformed  data
plt.figure(figsize=(8, 6))
plt.scatter(umap_data_2d[:, 0], umap_data_2d[:, 1], c=y_train, cmap='coolwarm', alpha=0.7)
plt.colorbar(label='Hospital Death')
plt.title('UMAP Visualization for Patient Survival (2D)')
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.show()

"""## Clustering Methods - PCA Reduced Data

### K-means (PCA)

#### Elbow Plot (PCA)
"""

# Perform k-means
wcss = []
cluster_range = range(1, 11)  # it is sufficient to test from 1 to 10 clusters
for k in cluster_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X_train_pca)
    wcss.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(8, 6))
plt.plot(cluster_range, wcss, marker='o')
plt.title('Elbow Plot for Patient Survival (PCA Reduced Data)')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS(Within-Cluster Sum of Squares)')
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):}'))
plt.axvline(x=2, color='r', linestyle='--') # Add a dashed vertical line at the elbow point (k=2)
plt.show()

silhouette_scores = []

# Sample the data (10% of the dataset)
# Set the random seed for reproducibility
np.random.seed(42)
n_samples = int(X_train_pca.shape[0] * 0.1)
sample_indices = np.random.choice(X_train_pca.shape[0], size=n_samples, replace=False)
sample_data = X_train_pca[sample_indices]

for n_clusters in range(2, 7):
    clusterer = KMeans(n_clusters=n_clusters, random_state=42, init='k-means++')
    cluster_labels = clusterer.fit_predict(sample_data)
    silhouette_avg = silhouette_score(sample_data, cluster_labels)
    silhouette_scores.append({'Number of Clusters': n_clusters, 'Silhouette Score': silhouette_avg})

# Convert the scores into a DataFrame
silhouette_df = pd.DataFrame(silhouette_scores)

"""#### Silhouette Score (PCA)"""

# Plot the silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(silhouette_df['Number of Clusters'], silhouette_df['Silhouette Score'], marker='o')
plt.title('Silhouette Score Analysis for Patient Survival (PCA Reduced Data)')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.axvline(x=2, color='r', linestyle='--') # Add a dashed vertical line at the maximum silhouette score (k=2)
plt.show()

# Tabulate the silhouette scores
print("Silhouette Scores Table for Patient Survival (PCA Reduced Data):")
print(silhouette_df.to_string(index=False))

"""#### Optimal K = 2 (PCA)"""

# Get the optimal number of clusters based on the maximum silhouette score
optimal_clusters = silhouette_df.loc[silhouette_df['Silhouette Score'].idxmax(), 'Number of Clusters']
print(f"Optimal number of clusters for Patient Survival (PCA Reduced Data) using K-means: {optimal_clusters}")

"""#### Scatter Plot (PCA)"""

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(X_train_pca)

# Visualize the clusters
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=kmeans_labels, cmap='coolwarm', alpha=0.6)
plt.title('K-Means Clustering for Patient Survival (PCA Reduced Data)')
unique_clusters = np.unique(kmeans_labels)
legend_labels = [f'Cluster {i}' for i in unique_clusters]
plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

"""### Hierarchical Clustering (PCA)

#### Optimal K = 2 (PCA)
"""

# Perform hierarchical clustering using the 'ward' method
linked = linkage(X_train_pca, method='ward')

# Test different numbers of clusters (k) and calculate silhouette scores
max_k = 10
silhouette_scores = []

print("Silhouette Scores for Hierarchical Clustering - Patient Survival (PCA Reduced Data):")
for k in range(2, max_k+1):
    cluster_labels = fcluster(linked, t=k, criterion='maxclust')
    if len(set(cluster_labels)) > 1:
        silhouette_avg = silhouette_score(X_train_pca, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print(f"Number of Clusters: {k}, Silhouette Score: {silhouette_avg}")

# Find the best k (number of clusters) with the highest silhouette score
best_k = np.argmax(silhouette_scores) + 2
print(f"Best Number of Clusters (k): {best_k}")

"""#### Dendrogram (PCA)"""

# Create the dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linkage(sample_data, method='ward') ) # sampled 10% of the data to decrease runtime
plt.axhline(y=130, color='r', linestyle='--')
plt.title('Dendrogram for Hierarchical Clustering (PCA Reduced Data)')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

"""#### Scatter Plot (PCA)"""

# Visualize the clusters
cluster_labels = fcluster(linked, t=best_k, criterion='maxclust')
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=cluster_labels, cmap='coolwarm', alpha=0.6)
plt.title('Hierarchical Clustering for Patient Survival (PCA Reduced Data)')
unique_clusters = np.unique(cluster_labels)
legend_labels = [f'Cluster {i}' for i in unique_clusters]
plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

"""## Clustering Methods - UMAP Reduced Data

### K-means (UMAP)

#### Elbow Plot (UMAP)
"""

# Perform k-means
wcss = []
cluster_range = range(1, 11)  # it is sufficient to test from 1 to 10 clusters
for k in cluster_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X_train_umap)
    wcss.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(8, 6))
plt.plot(cluster_range, wcss, marker='o')
plt.title('Elbow Plot for Patient Survival (UMAP Reduced Data)')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS(Within-Cluster Sum of Squares)')
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):}'))
plt.axvline(x=2, color='r', linestyle='--') # Add a dashed vertical line at the elbow point (k=2)
plt.show()

silhouette_scores = []

# Sample the data (10% of the dataset)
# Set the random seed for reproducibility
np.random.seed(42)
n_samples = int(X_train_umap.shape[0] * 0.1)
sample_indices = np.random.choice(X_train_umap.shape[0], size=n_samples, replace=False)
sample_data = X_train_umap[sample_indices]

for n_clusters in range(2, 7):
    clusterer = KMeans(n_clusters=n_clusters, random_state=42, init='k-means++')
    cluster_labels = clusterer.fit_predict(sample_data)
    silhouette_avg = silhouette_score(sample_data, cluster_labels)
    silhouette_scores.append({'Number of Clusters': n_clusters, 'Silhouette Score': silhouette_avg})

# Convert the scores into a DataFrame
silhouette_df = pd.DataFrame(silhouette_scores)

"""#### Silhouette Score (UMAP)"""

# Plot the silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(silhouette_df['Number of Clusters'], silhouette_df['Silhouette Score'], marker='o')
plt.title('Silhouette Score Analysis for Patient Survival (UMAP Reduced Data)')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.axvline(x=2, color='r', linestyle='--') # Add a dashed vertical line at the maximum silhouette score (k=2)
plt.show()

# Tabulate the silhouette scores
print("Silhouette Scores Table for Patient Survival (UMAP Reduced Data):")
print(silhouette_df.to_string(index=False))

"""#### Optimal K = 2 (UMAP)"""

# Get the optimal number of clusters based on the maximum silhouette score
optimal_clusters = silhouette_df.loc[silhouette_df['Silhouette Score'].idxmax(), 'Number of Clusters']
print(f"Optimal number of clusters for Patient Survival (UMAP Reduced Data) using K-means: {optimal_clusters}")

"""#### Scatter Plot (UMAP)"""

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(X_train_umap)

# Visualize the clusters
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_train_umap[:, 0], X_train_umap[:, 1], c=kmeans_labels, cmap='coolwarm', alpha=0.6)
plt.title('K-Means Clustering for Patient Survival (UMAP Reduced Data)')
unique_clusters = np.unique(kmeans_labels)
legend_labels = [f'Cluster {i}' for i in unique_clusters]
plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

"""### Hierarchical Clustering (UMAP)

#### Optimal K = 2 (UMAP)
"""

# Perform hierarchical clustering using the 'ward' method
linked = linkage(X_train_umap, method='ward')

# Test different numbers of clusters (k) and calculate silhouette scores
max_k = 10
silhouette_scores = []

print("Silhouette Scores for Hierarchical Clustering - Patient Survival (PCA Reduced Data):")
for k in range(2, max_k+1):
    cluster_labels = fcluster(linked, t=k, criterion='maxclust')
    if len(set(cluster_labels)) > 1:
        silhouette_avg = silhouette_score(X_train_umap, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print(f"Number of Clusters: {k}, Silhouette Score: {silhouette_avg}")

# Find the best k (number of clusters) with the highest silhouette score
best_k = np.argmax(silhouette_scores) + 2
print(f"Best Number of Clusters (k): {best_k}")

"""#### Dendogram (UMAP)"""

# Create the dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linkage(sample_data, method='ward')) # sampled 10% of the data to decrease runtime
plt.axhline(y=100, color='r', linestyle='--')
plt.title('Dendrogram for Hierarchical Clustering (UMAP Reduced Data)')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

"""#### Scatter Plot (UMAP)"""

# Visualize the clusters
cluster_labels = fcluster(linked, t=best_k, criterion='maxclust')
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=cluster_labels, cmap='coolwarm', alpha=0.6)
plt.title('Hierarchical Clustering for Patient Survival (UMAP Reduced Data)')
unique_clusters = np.unique(cluster_labels)
legend_labels = [f'Cluster {i}' for i in unique_clusters]
plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

"""## Ensemble Methods"""

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, f1_score
from sklearn.model_selection import learning_curve, validation_curve

# Create models
ada_model = AdaBoostClassifier(random_state=42) # ada_model = AdaBoostClassifier(random_state=42, algorithm='SAMME')
rf_model = RandomForestClassifier(random_state=42)

# Define hyperparameter grids for tuning
ada_param_grid = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1]}
rf_param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, 30]}

import warnings

# Suppress the specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, message=".*SAMME.R algorithm.*")

# Function for hyperparameter tuning
import time

def tune_model(model, param_grid, X_train, y_train):
    start_time = time.time()
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, scoring='f1_macro', verbose=3)
    grid_search.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    print(f"Tuning completed in {elapsed_time:.2f} seconds")
    print(f"Best parameters for {type(model).__name__}: {grid_search.best_params_}")
    return grid_search.best_estimator_

# Perform tuning and get the best models (3 min)
print("Tuning AdaBoost on the original dataset...")
best_ada_original = tune_model(ada_model, ada_param_grid, X_train_scaled, y_train)

print("Tuning Random Forest on the original dataset...") # (2 min)
best_rf_original = tune_model(rf_model, rf_param_grid, X_train_scaled, y_train)

print("Tuning AdaBoost on the PCA reduced dataset...") # (12 min)
best_ada_pca = tune_model(ada_model, ada_param_grid, X_train_pca, y_train)

print("Tuning Random Forest on the PCA reduced dataset...") # (7 min)
best_rf_pca = tune_model(rf_model, rf_param_grid, X_train_pca, y_train)

print("Tuning AdaBoost on the UMAP reduced dataset...") # (<1 min)
best_ada_umap = tune_model(ada_model, ada_param_grid, X_train_umap, y_train)

print("Tuning Random Forest on the UMAP reduced dataset...") # (1 min)
best_rf_umap = tune_model(rf_model, rf_param_grid, X_train_umap, y_train)

# Evaluation Functions

# Classification Report
def print_classification_report(y_true, y_pred, dataset_name, model_name):
    print(f"\nClassification Report for {dataset_name} ({model_name}):\n", classification_report(y_true, y_pred))

# Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, dataset_name, model_name, cmap):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=False)
    plt.title(f"Confusion Matrix for {dataset_name} ({model_name})")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# ROC Curve
def plot_roc_curve(y_true, y_pred_proba, dataset_name, model_name, color):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color=color, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.title(f"ROC Curve for {dataset_name} ({model_name})")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

# Learning Curve
def plot_learning_curve(model, X_train, y_train, dataset_name, model_name, color):
    train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5))
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training F1 Score', color=color)
    plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-validation F1 Score', color='black')
    plt.title(f"Learning Curve for {dataset_name} ({model_name})")
    plt.xlabel('Training Size')
    plt.ylabel('F1 Score')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

# Validation Curve
def plot_validation_curve(model, X_train, y_train, dataset_name, model_name, param_name, param_range, color):
    train_scores, test_scores = validation_curve(
        model, X_train, y_train, param_name=param_name, param_range=param_range, cv=5, n_jobs=-1
    )
    plt.figure(figsize=(8, 6))
    plt.plot(param_range, np.mean(train_scores, axis=1), label='Training F1 Score', color=color)
    plt.plot(param_range, np.mean(test_scores, axis=1), label='Cross-validation F1 Score', color='black')
    plt.title(f"{param_name}: Validation Curve for {dataset_name} ({model_name})")
    plt.xlabel(f'{param_name}')
    plt.ylabel('F1 Score')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

# Evaluation Function for Model
def evaluate_model(best_model, X_train, X_test, y_train, y_test, dataset_name, model_name, color, cmap):
    # Predict and evaluate the model
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)

    # Print and plot metrics
    print_classification_report(y_test, y_pred, dataset_name, model_name)
    plot_confusion_matrix(y_test, y_pred, dataset_name, model_name, cmap)
    plot_roc_curve(y_test, y_pred_proba, dataset_name, model_name, color)
    plot_learning_curve(best_model, X_train, y_train, dataset_name, model_name, color)

    # Plot validation curve for RandomForest (max_depth, n_estimators) and AdaBoost (learning_rate, n_estimators)
    if isinstance(best_model, RandomForestClassifier):
        param_range_depth = np.arange(5, 31, 5)  # Range for max_depth
        param_range_estimators = np.arange(50, 201, 50)  # Range for n_estimators
        plot_validation_curve(best_model, X_train, y_train, dataset_name, model_name, 'max_depth', param_range_depth, color)
        plot_validation_curve(best_model, X_train, y_train, dataset_name, model_name, 'n_estimators', param_range_estimators, color)
    elif isinstance(best_model, AdaBoostClassifier):
        param_range_lr = [0.01, 0.1, 0.5, 1]  # Range for learning_rate
        param_range_estimators = [50, 100, 200]  # Range for n_estimators
        plot_validation_curve(best_model, X_train, y_train, dataset_name, model_name, 'learning_rate', param_range_lr, color)
        plot_validation_curve(best_model, X_train, y_train, dataset_name, model_name, 'n_estimators', param_range_estimators, color)

"""### Original Data"""

# Original Dataset - AdaBoost (4 min)
evaluate_model(best_ada_original, X_train_scaled, X_test_scaled, y_train, y_test, "Original Dataset", "AdaBoost", "blue", "Blues")

from sklearn.metrics import accuracy_score

# Training Error
y_train_pred = best_ada_original.predict(X_train_scaled)
train_accuracy = accuracy_score(y_train, y_train_pred)
training_error = 1 - train_accuracy

print(f"Training Error for Original Dataset (AdaBoost):: {training_error:.2f}")

# Original Dataset - Random Forest (2 min)
evaluate_model(best_rf_original, X_train_scaled, X_test_scaled, y_train, y_test, "Original Dataset", "RandomForest", "blue", "Blues")

# Training Error
y_train_pred = best_rf_original.predict(X_train_scaled)
train_accuracy = accuracy_score(y_train, y_train_pred)
training_error = 1 - train_accuracy

print(f"Training Error for Original Dataset (Random Forest): {training_error:.2f}")

"""### PCA Reduced Data"""

# PCA Reduced Dataset - AdaBoost (17 min)
evaluate_model(best_ada_pca, X_train_pca, X_test_pca, y_train, y_test, "PCA Reduced Dataset", "AdaBoost", "red", "Reds")

# Training Error
y_train_pred = best_ada_pca.predict(X_train_pca)
train_accuracy = accuracy_score(y_train, y_train_pred)
training_error = 1 - train_accuracy

print(f"Training Error for PCA Reduced Dataset (AdaBoost): {training_error:.2f}")

# PCA Reduced Dataset - Random Forest (13 min)
evaluate_model(best_rf_pca, X_train_pca, X_test_pca, y_train, y_test, "PCA Reduced Dataset", "RandomForest", "red", "Reds")

# Training Error
y_train_pred = best_rf_pca.predict(X_train_pca)
train_accuracy = accuracy_score(y_train, y_train_pred)
training_error = 1 - train_accuracy

print(f"Training Error for PCA Reduced Dataset (Random Forest): {training_error:.2f}")

"""### UMAP Reduced Data"""

# UMAP Reduced Dataset - AdaBoost
evaluate_model(best_ada_umap, X_train_umap, X_test_umap, y_train, y_test, "UMAP Reduced Dataset", "AdaBoost", "green", "Greens")

# Training Error
y_train_pred = best_ada_umap.predict(X_train_umap)
train_accuracy = accuracy_score(y_train, y_train_pred)
training_error = 1 - train_accuracy

print(f"Training Error for UMAP Reduced Dataset (AdaBoost): {training_error:.2f}")

# UMAP Reduced Dataset - Random Forest (1 min)
evaluate_model(best_rf_umap, X_train_umap, X_test_umap, y_train, y_test, "UMAP Reduced Dataset", "RandomForest", "green", "Greens")

# Training Error
y_train_pred = best_rf_umap.predict(X_train_umap)
train_accuracy = accuracy_score(y_train, y_train_pred)
training_error = 1 - train_accuracy

print(f"Training Error for UMAP Reduced Dataset (Random Forest): {training_error:.2f}")