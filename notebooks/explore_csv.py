# %%

import pandas as pd

df = pd.read_csv("/Users/jonathanraphael/git/offshore-methane/data/sites.csv")

print(df.head())

# %%
month_start = 2  # inclusive
month_end = 12  # inclusive

# find all rows where the start and end datetimes are entirely within a single year between the months listed above
df["start_year"] = df["start"].str.split("-").str[0]
df["start_month"] = df["start"].str.split("-").str[1]
df["end_year"] = df["end"].str.split("-").str[0]
df["end_month"] = df["end"].str.split("-").str[1]
rejectables = df[df["start_year"] == df["end_year"]]
rejectables = rejectables[rejectables["start_month"].astype(int) >= month_start]
rejectables = rejectables[rejectables["end_month"].astype(int) <= month_end]

# remove rejectables from df
df = df[~df.index.isin(rejectables.index)]


# %%
df = df[["lat", "lon", "start", "end", "citation", "EEZ"]]

# %%
df.to_csv("/Users/jonathanraphael/git/offshore-methane/data/events.csv")

# %%
