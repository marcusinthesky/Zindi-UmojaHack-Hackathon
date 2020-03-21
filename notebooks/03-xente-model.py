# %%
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
from sklearn.metrics import f1_score


hv.extension('bokeh')

#%%
#setup logging
file = Path(__file__).stem
repo = git.Repo(search_parent_directories=True)
sha = repo.git.rev_parse(repo.head.object.hexsha, short=4)
log = logging.getLogger('{}-{}'.format(file, sha))

# %%
# load data
X, y = context.io.load('xente_features').astype(np.float32), context.io.load('xente_target').astype(np.float32)


#%%
# pipeline
preprocessing = FunctionTransformer(validate=False)
model = MultiOutputRegressor(xgb.XGBRegressor(objective='survival:cox',
                             booster='gblinear',
                             base_score=1,
                             n_estimators=1000), n_jobs=-1)
pipeline = Pipeline([('transform', preprocessing),
                     ('model', model)])
link = FunctionTransformer(validate=False)

regressor = TransformedTargetRegressor(regressor=pipeline,
                                       transformer=link)

schema = KFold(3)

#%%
# search
params = {
    'regressor__model__estimator__gamma': (0., 1e+2, 'log-uniform'),
    'regressor__model__estimator__learning_rate': (1e-6, 1e+1, 'log-uniform'),
    'regressor__model__estimator__max_depth': (1, 8),  # integer valued parameter
    # 'regressor__model__estimator__min_child_weight': (1, 10, 'log-uniform'),  # categorical parameter
    'regressor__model__estimator__n_estimators': (1, 1000),  # integer valued parameter
    'regressor__model__estimator__reg_alpha': (0., 1., 'log-uniform'),  # integer valued parameter
    'regressor__model__estimator__reg_lambda': (1., 1.9, 'log-uniform'),  # integer valued parameter
    # 'regressor__model__estimator__subsample': (0.1, 1, 'log-uniform'),  # integer valued parameter
}


opt = BayesSearchCV(
    regressor,
    params,
    n_iter=5,
    cv=schema,
    refit=True
)


# %%
opt.fit(X, y)
context.io.save('xente_xgb', opt)
# pd.DataFrame(opt.cv_results_)

# %%
y_pred = pd.DataFrame(model.predict(X), index = y.index, columns=y.columns).fillna(0)
context.io.save('xente_y_prediction', y_pred)
pd.Series(y_pred.to_numpy().flatten()).hvplot.hist()

# %%
xente_sample_submission = context.io.load('xente_sample_submission')
xente_sample_submission_wide = context.io.load('xente_sample_submission_wide')
xente_merged = context.io.load('xente_merged')

# %%
y_pred_melt = y_pred.reset_index().melt(id_vars='index', var_name = 'PID')
test = (xente_merged
        .where(lambda df: df.test == True)
        .dropna()
        .reset_index()
        .merge(y_pred_melt.reset_index(), left_on=['index', 'PID'], right_on=['index', 'PID'], how='left', validate='one_to_one'))

#%%
train = (xente_merged
        .where(lambda df: df.test == False)
        .dropna()
        .reset_index()
        .merge(y_pred_melt.reset_index(), left_on=['index', 'PID'], right_on=['index', 'PID']))

# %%
xente_sample_submission_wide

# %%
submission = xente_sample_submission_wide.merge(test, on=['PID', 'acc', 'date'], how='left')
submission = submission.assign(Prediction = (submission.value > train.value.mean()).astype(np.int))

#%%
standardized_submission = xente_sample_submission.assign(Prediction = (submission.sort_values('old_index').Prediction))

context.io.save('xente_y_submission', standardized_submission)

# %%
