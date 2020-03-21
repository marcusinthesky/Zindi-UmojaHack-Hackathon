# %%
import logging
from pathlib import Path

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
preprocessing = FunctionTransformer()
model = MultiOutputRegressor(xgb.XGBRegressor(objective='survival:cox',
                             booster='gblinear',
                             base_score=1,
                             n_estimators=1000), n_jobs=-1)
pipeline = Pipeline([('transform', preprocessing),
                     ('model', model)])
link = FunctionTransformer()

regressor = TransformedTargetRegressor(regressor=pipeline,
                                       transformer=link)

schema = KFold(3)

#%%
# search
params = {
    'regressor__model__estimator__gamma': (1e-6, 1e+6, 'log-uniform'),
    'regressor__model__estimator__learning_rate': (1e-6, 1e+1, 'log-uniform'),
    'regressor__model__estimator__max_depth': (1, 8),  # integer valued parameter
    'regressor__model__estimator__min_child_weight': (10, 500, 'log-uniform'),  # categorical parameter
    'regressor__model__estimator__n_estimators': (1, 8),  # integer valued parameter
    'regressor__model__estimator__reg_alpha': (1, 8, 'log-uniform'),  # integer valued parameter
    'regressor__model__estimator__reg_lambda': (1, 8, 'log-uniform'),  # integer valued parameter
    'regressor__model__estimator__subsample': (1, 8, 'log-uniform'),  # integer valued parameter
}


opt = BayesSearchCV(
    regressor,
    params,
    n_iter=10,
    cv=schema,
    refit=True
)


# %%
opt.fit(X, y)
pd.DataFrame(opt.cv_results_)

# %%
y_pred = opt.predict(X)

# %%
