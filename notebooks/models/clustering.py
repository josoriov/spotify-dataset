
# %% Importing the dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dython.nominal import associations
from sklearn.utils.validation import check_random_state


sns.set()
# %% Reading the cleaned file
data = pd.read_csv('../../data/data_cleaned.csv')

# %%
predictors = [
    'acousticness','danceability','energy','duration_ms',
    'instrumentalness','valence','tempo','liveness',
    'loudness','speechiness'
]

target = ['decade']

# %%
corr = associations(data[predictors], plot=False)['corr']

# %%
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
fig, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.color_palette('Spectral', as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(
    corr, mask=mask, cmap=cmap, center=0,
    square=True, linewidths=.5, cbar_kws={"shrink": .5},
    vmin=0, vmax=1
)

# %%
# According to the corr plot, energy and loudness are highly correlated
# and thus one of them will be removed
predictors.remove('loudness')

# %%
# Since there is a lot of data to train, only 2000 songs per decade will be used
model_data = (
    data[target+predictors]
    .copy()
    .groupby('decade', group_keys=False)
    .apply(lambda x: x.sample(2000, random_state=10))
)

X = model_data[predictors].values
y = model_data[target].values

# %%
# Given that all predictors are numerical variables, they coould be grouped
# using the k-means algorithm. But first the variables need to be scaled.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    stratify=y,
    random_state=10
)

# %%

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

