# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

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

X = data[predictors].values
y = data[['decade']].values

# %% splitting the data and scaling it
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    stratify=y, test_size=0.5, shuffle=True, random_state=10
)
# val should be around 10% of the original dataset
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train,
    stratify=y_train, test_size=0.2, shuffle=True, random_state=10
)

# %% creating the model pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier())
])

# %%
pipe.fit(X_train, np.ravel(y_train))

y_pred_val = pipe.predict(X_val)

# %%
print(classification_report(y_val, y_pred_val))

# as of now the results are not satisfactory enough
# new features will need to be engineered and put to the test
# perhaps try more powerful models like ANNs
