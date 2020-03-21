#%%
import pandas as pd
import numpy as np
import holoviews as hv
import hvplot.pandas
from scipy.sparse.linalg import svds
from scipy.stats import chisquare, chi2_contingency
from sklearn.decomposition import TruncatedSVD
from umoja.ca import CA

hv.extension('bokeh')

#%%
X = context.io.load('xente_train')
Y = context.io.load('xente_sample_submission')

# %%
# lets view a sadom customer
X_mode = X.where(lambda df: df.acc == df.acc.sample(1).item()).dropna()
category = pd.DataFrame({'pid': pd.np.argmax(pd.get_dummies(X_mode.PID).to_numpy(), axis=1),
                         'time': pd.to_datetime(X_mode.date)})
category.hvplot.scatter(x='time', y='pid', color='pid')


# %%
data = X
# pivot on person-category
observed = pd.crosstab(data.acc, data.PID)
ca = CA(2, observed.columns, observed.index).fit(observed.to_numpy())

rows = pd.DataFrame(ca.row_coord_, index=observed.index, columns=["component_1", "component_2"])
columns = pd.DataFrame(ca.col_coord_, index=observed.columns, columns=["component_1", "component_2"])

plot = (
    hv.Labels(
        columns.reset_index(), ["component_1", "component_2"], "PID"
    ).opts(text_font_size="5pt", text_color="blue")
    * hv.Labels(
        rows.reset_index().assign(acc = lambda df: df.acc.astype(str)), ["component_1", "component_2"], "acc"
    ).opts(text_font_size="5pt", text_color="red")
).opts(width=800, height=500, title="Correspondance Analysis")

plot

#%%
# usertime = (X
# .assign(date = lambda df: pd.to_datetime(df.date))
# .groupby('acc')
# .apply(lambda df: (df.date
#                     .sort_values()
#                     .diff()
#                     .dt.total_seconds()
#                     .fillna(0)
#                     .cumsum() * pd.get_dummies(df.PID))
# ))

#%%
# usertransactions = (X.assign(date = lambda df: pd.to_datetime(df.date))
#                      .sort_values('date')
#                      .set_index('date')
#                      .pipe(lambda df: pd.concat([df.loc[:, ['acc']],
#                                                  pd.get_dummies(df.PID).astype(np.int)], axis=1))
#                      .groupby('acc')
#                      .apply(lambda df: df.transform(pd.Series.diff)
#                      .fillna(0)))

# %%
time_since_last = (X
                    .groupby(['acc', 'PID'])
                    .date
                    .apply(lambda df: pd.to_datetime(df).sort_values().diff().dt.total_seconds()
                    .fillna(0))
                    .reset_index()
                    .rename(columns={'date': 'time_since_last'})
                    .merge(X.loc[:,['date']], left_on=['level_2'], right_index=True)
                    .sort_values('date'))

dummies = pd.get_dummies(time_since_last.PID).astype(np.int)
dummies_nan_zero = dummies.replace(0, np.nan)
dummies_nan_neg = dummies.replace(0, -1)
dummies_nan_1 = np.abs(dummies - 1)

# %%
time_since_last_ffill = (dummies_nan_zero * time_since_last.time_since_last).ffill()
time_diff = time_since_last_ffill + (dummies_nan_1 * time_since_last.date.diff().dt.seconds())
time_diff_censor = time_diff_censor * dummies_nan_neg