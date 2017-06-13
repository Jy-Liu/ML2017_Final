import logging 
import numpy as np
import xgboost as xgb


class XGBRegressor:

    def __init__(self, n_rounds, max_depth, valid=None):
        self.param = {'eta': 0.05,
                      'max_depth': max_depth,
                      'subsample': 1.0,
                      'colsample_bytree': 0.7,
                      'objective': 'reg:linear',
                      'eval_metric': 'mae',
                      'silent': 1}
        self.n_rounds = n_rounds
        self.valid = valid

    def fit(self, X, y):
        dtrain = xgb.DMatrix(data=X, label=y)
        if self.valid is not None:
            dvalid = xgb.DMatrix(data=self.valid['x'], label=self.valid['y'])
            evals = [(dvalid, 'val')]
        else:
            evals = ()

        self.estimator = xgb.train(self.param, dtrain,
                                   self.n_rounds, evals=evals,
                                   early_stopping_rounds=20,
                                   verbose_eval=False)
        best_iter = self.estimator.best_iteration + 1
        self.estimator = xgb.train(self.param, dtrain,
                                   best_iter, evals=evals,
                                   early_stopping_rounds=20)

    def predict(self, X):
        dtest = xgb.DMatrix(data=X)
        y_ = self.estimator.predict(dtest)
        return y_
