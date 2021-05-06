# %%
import pandas as pd

# %%
# In case there are path related problems use this
import os
os.chdir('../../')

# %%
data = pd.read_csv('data/processed/data_cleaned.csv')

# %%
data.head()
# %%
# !TODO
# Leave this fill in case you want to process actual things
# as of now it seems that the features are enough
