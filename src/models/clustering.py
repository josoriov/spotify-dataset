# %%
# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# %%
# import os
# os.chdir('../../')

# %%
data = pd.read_csv('data/processed/data_cleaned.csv')

predictors = [
    'acousticness', 'danceability', 'energy', 'duration_ms',
    'instrumentalness', 'valence', 'tempo', 'liveness',
    'loudness', 'speechiness'
]

objectives = [
    'decade', 'year'
]

X = data[predictors].values
y = data[['year']].values
# %%
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

# %%
kmeans = KMeans(n_clusters=8)
kmeans.fit(X_scaled)
data['cluster'] = kmeans.labels_
# %%
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
# top left
axes[0, 0].set_xlabel('Acousticness')
sns.boxplot(data=data, x='cluster', y='acousticness', ax=axes[0, 0])
# top right
axes[0, 1].set_xlabel('Danceability')
sns.boxplot(data=data, x='cluster', y='danceability', ax=axes[0, 1])
# bottom left
axes[1, 0].set_xlabel('Loudness')
sns.boxplot(data=data, x='cluster', y='loudness', ax=axes[1, 0])
# bottom right
axes[1, 1].set_xlabel('Instrumentalness')
sns.boxplot(data=data, x='cluster', y='instrumentalness', ax=axes[1, 1])
plt.show()

# Clustering results does not seems good enough so it won't be used as a predictor
# the models will be fitted using only the other variables
