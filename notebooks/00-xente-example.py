#%%
import pandas as pd
import xgboost as xgb
from sksurv.datasets import load_aids
from sksurv.linear_model import CoxPHSurvivalAnalysis
from skopt import BayesSearchCV

# load and inspect the data
data_x, data_y = load_aids()
data_y[10:15]

#%%
# Since XGBoost only allow one column for y, the censoring information
# is coded as negative values:
data_y_xgb = [x[1] if x[0] else -x[1] for x in data_y]
data_y_xgb[10:15]


# %%
data_x = data_x[['age', 'cd4']]
data_x.head()


# %%
params = {
    'regressor__model__gamma': (1e-6, 1e+6, 'log-uniform'),
    'regressor__model__learning_rate': (1e-6, 1e+1, 'log-uniform'),
    'regressor__model__max_depth': (1, 8),  # integer valued parameter
    'regressor__model__min_child_weight': (10, 500, 'log-uniform'),  # categorical parameter
    'regressor__model__n_estimators': (1, 8),  # integer valued parameter
    'regressor__model__reg_alpha': (1, 8, 'log-uniform'),  # integer valued parameter
    'regressor__model__reg_lambda': (1, 8, 'log-uniform'),  # integer valued parameter
    'regressor__model__subsample': (1, 8, 'log-uniform'),  # integer valued parameter

}
#%%
# Since sksurv output log hazard ratios (here relative to 0 on predictors)
# we must use 'output_margin=True' for comparability.
estimator = CoxPHSurvivalAnalysis().fit(data_x, data_y)
gbm = xgb.XGBRegressor(objective='survival:cox',
                       booster='gblinear',
                       base_score=1,
                       n_estimators=1000)

search = BayesSearchCV(gbm, params, n_iter=3, cv=3)
search.fit(data_x, data_y_xgb)

#%%
prediction_sksurv = estimator.predict(data_x)
predictions_xgb = search.predict(data_x)
d = pd.DataFrame({'xgb': predictions_xgb,
                  'sksurv': prediction_sksurv})
d.head()

# %%
context.io.save('xente_xgb', gbm)

# %%
