# %%
import os
import numpy as np
import pandas as pd

# %%
os.chdir('../../')

data_raw = pd.read_csv('data/raw/data_o.csv')

# %%
def split_date(date_str: str) -> list[int]:
    """
    Expect a string containing a date formatted as %Y-%m-%d and parses it into a three element list

    Parameters:
        date_str: date to be formatted

    Returns:
        A list with three numbers that defaults to zero when said part cannot be parsed

    """
    separators = date_str.count('-')
    splitted = [int(x) for x in date_str.split('-')]
    if (separators == 0):
        return [int(date_str), 0, 0]
    elif (separators == 1):
        return [splitted[0], splitted[1], 0]
    elif (separators == 2):
        return [splitted[0], splitted[1], splitted[2]]
    else:
        return[0, 0, 0]

date_array = np.asarray(
    list(map(split_date, data_raw['release_date'].values))
)


#data_raw['year'] = date_array[:,0]
data_raw['month'] = date_array[:,1]
data_raw['day'] = date_array[:,2]

cond_year = (data_raw['year'] != 0)
cond_month = (data_raw['month'] != 0)
cond_day = (data_raw['day'] != 0)

data_raw = (
    data_raw[cond_year & cond_month & cond_day]
    .reset_index(drop=True)
)

# Round years to the decade
data_raw['decade'] = data_raw['year'].apply(lambda x: round(x, -1))
data_raw = data_raw[(data_raw['decade'] > 1920)]

# %%

predictors = [
    'acousticness','danceability','energy','duration_ms',
    'instrumentalness','valence','tempo','liveness',
    'loudness','speechiness'
]

data_raw[predictors].to_csv('data/processed/data_cleaned.csv', index=False)
