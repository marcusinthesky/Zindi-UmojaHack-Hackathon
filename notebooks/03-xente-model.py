# %%
import logging
from pathlib import Path

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures
from sklearn.svm import SVC
from skopt import BayesSearchCV

#%%
#setup logging
file = Path(__file__).stem
repo = git.Repo(search_parent_directories=True)
sha = repo.git.rev_parse(repo.head.object.hexsha, short=4)
log = logging.getLogger('{}-{}'.format(file, sha))

# %%
# load data
X, y = load_digits(10, True)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=.25, random_state=0)

# pipeline
preprocessing = FunctionTransformer()
model = SVC()
pipeline = Pipeline([('transform', preprocessing),
                     ('model', SVC())])
link = FunctionTransformer()

regressor = TransformedTargetRegressor(regressor=pipeline,
                                       transformer=link)

schema = KFold(3)

#%%
# search
params = {
        'regressor__model__C': (1e-6, 1e+6, 'log-uniform'),
        'regressor__model__gamma': (1e-6, 1e+1, 'log-uniform'),
        'regressor__model__degree': (1, 8),  # integer valued parameter
        'regressor__model__kernel': ['linear', 'poly', 'rbf'],  # categorical parameter
    }


opt = BayesSearchCV(
    regressor,
    params,
    n_iter=3,
    cv=schema,
)


# %%
opt.fit(X_train, y_train)

# %%
