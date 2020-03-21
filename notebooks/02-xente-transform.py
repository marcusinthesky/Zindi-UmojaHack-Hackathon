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


Y_wide = (Y
            .loc[:, 'Account X date X PID']
            .str.split(' X ', expand=True)
            .rename(columns={0:'acc', 1:'date',2:'PID'})
            .assign(test = True)
            )

context.io.save('xente_sample_submission_wide', Y_wide)

data = (pd.concat([Y_wide, X.assign(test = False)], axis=0)
        .reset_index()
        .rename(columns={'index':'old_index'})
        .assign(acc = lambda df: df.acc.astype('int64')))

context.io.save('xente_merged', data)


# %%
a = (data.drop(columns=['old_index'])
        .groupby(['acc', 'PID'])
        .date
        .apply(lambda df: pd.to_datetime(df).sort_values().diff().dt.total_seconds()
        .fillna(0))
        .reset_index()
        .rename(columns={'level_2': 'index'})
        .set_index('index')
        .rename(columns={'date': 'time_since_last'}))

b = data.loc[:,['date']].assign(date = lambda df: pd.to_datetime(df.date))
#%%
time_since_last = (a.join(b).sort_values('date'))

#%%
dummies = pd.get_dummies(time_since_last.PID).astype(np.int)
dummies_nan_zero = dummies.replace(0, np.nan)
dummies_nan_neg = dummies.replace(0, -1)
dummies_nan_one = dummies.replace({0:1, 1:0})
dummies_one_nan = dummies.replace({1: np.nan, 0: 1})
# %%
time_since_last_ffill = dummies_nan_zero.multiply(time_since_last.time_since_last, axis=0).ffill()
seconds_since_last_any = time_since_last.date.diff().dt.total_seconds()
time_diff = time_since_last_ffill.add(dummies_nan_one.multiply(seconds_since_last_any, axis=0))
time_diff_censor = time_diff.multiply(dummies_nan_neg)

target  = time_diff_censor.fillna(time_diff_censor.min())


# %%
sensor_pred = target.multiply(dummies_one_nan).abs()
features = sensor_pred.fillna(sensor_pred.mean(0))


# %%
context.io.save('xente_features', features)
context.io.save('xente_target', target)

# %%
