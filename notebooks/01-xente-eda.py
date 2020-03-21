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