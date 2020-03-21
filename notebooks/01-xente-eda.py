#%%
import pandas as pd
import numpy as np
import holoviews as hv
import hvplot.pandas
from scipy.sparse.linalg import svds
from scipy.stats import chisquare, chi2_contingency
from sklearn.decomposition import TruncatedSVD
from umoja.ca import CA
import logging
from pathlib import Path

import numpy as np
import hvplot.pandas
import holoviews as hv
import pandas as pd
import git
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.datasets import load_digits
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures
import xgboost as xgb
from skopt import BayesSearchCV
from sklearn.metrics import f1_score, get_scorer

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

rows = pd.DataFrame(ca.row_coord_, index=observed.index, columns=["PID_component_1", "PID_component_2"])
columns = pd.DataFrame(ca.col_coord_, index=observed.columns, columns=["acc_component_1", "acc_component_2"])

#%%
plot = (
    hv.Labels(
        columns.reset_index(), ["PID_component_1", "PID_component_2"], "PID"
    ).opts(text_font_size="5pt", text_color="blue")
    * hv.Labels(
        rows.reset_index().assign(acc = lambda df: df.acc.astype(str)), ["acc_component_1", "acc_component_2"], "acc"
    ).opts(text_font_size="5pt", text_color="red")
).opts(width=800, height=500, title="Correspondance Analysis")

plot

#%%
data = pd.concat([context.io.load('xente_sample_submission_wide').assign(test = True),
                  context.io.load('xente_train').assign(test = False)], axis=0)


#%%
sample = (data
            .assign(Prediction=1))

negsample = (data
            .assign(acc = lambda df: df.acc.sample(frac=1).to_numpy())
            .assign(PID = lambda df: df.PID.sample(frac=1).to_numpy())
            .assign(test=False)
            .assign(Prediction=0)
            .sample(X.shape[0]))

features =  (pd.concat([sample, negsample], axis=0)
            .merge(rows, left_on='acc', right_index=True, how='left')
            .merge(columns, left_on='PID', right_index=True, how='left')
            .assign(date = lambda df: pd.to_datetime(df.date))
            .assign(dow = lambda df: df.date.dt.dayofweek)
            .assign(hour = lambda df: df.date.dt.hour)
            .assign(month = lambda df: df.date.dt.hour))

features = features.fillna(features.mean(0))


train = features.where(lambda x: x.test==False).loc[:,['PID_component_1', 'PID_component_2', 'acc_component_1',
       'acc_component_2', 'dow', 'hour', 'month', 'Prediction']].dropna()

test = features.where(lambda x: x.test==True).loc[:,['PID_component_1', 'PID_component_2', 'acc_component_1',
       'acc_component_2', 'dow', 'hour', 'month', 'Prediction']].dropna()




#%%
# pipeline
preprocessing = FunctionTransformer(validate=False)
model = xgb.XGBClassifier()
pipeline = Pipeline([('transform', preprocessing),
                     ('model', model)])
link = FunctionTransformer(validate=False)

regressor = TransformedTargetRegressor(regressor=pipeline,
                                       transformer=link)

schema = KFold(3)
#%%
# search
params  = {
        'learning_rate': (0.01, 1.0, 'log-uniform'),
        'min_child_weight': (0, 10),
        'max_depth': (0, 50),
        'max_delta_step': (0, 20),
        'subsample': (0.01, 1.0, 'uniform'),
        'colsample_bytree': (0.01, 1.0, 'uniform'),
        'colsample_bylevel': (0.01, 1.0, 'uniform'),
        'reg_lambda': (1e-9, 1000, 'log-uniform'),
        'reg_alpha': (1e-9, 1.0, 'log-uniform'),
        'gamma': (1e-9, 0.5, 'log-uniform'),
        'min_child_weight': (0, 5),
        'n_estimators': (50, 100),
        'scale_pos_weight': (1e-6, 500, 'log-uniform')
    }


opt = BayesSearchCV(
    model,
    params,
    n_iter=5,
    cv=schema,
    refit=True,
    scoring='f1'
)

# %%
X_train, y_train = train.drop(columns=['Prediction']).astype(np.float32), train.Prediction.astype(np.int)
opt.fit(X_train, y_train)
context.io.save('xente_xgb', opt)

# %%
X_test, y_test = test.drop(columns=['Prediction']).astype(np.float32), test.Prediction.astype(np.int)
y_pred = opt.predict(X_test)

# %%
xente_sample_submission = context.io.load('xente_sample_submission').assign(Prediction = y_pred.astype(np.int))


context.io.save('xente_y_submission', xente_sample_submission)

# %%
