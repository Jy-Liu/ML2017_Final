import xgboost as xgb
from utils import root_mean_squared_log_error


def rmsle(preds, dtrain):
    truth = dtrain.get_label()
    return 'error', root_mean_squared_log_error(truth, preds)


class XGBRegressor:

    def __init__(self, n_rounds, max_depth, valid=None):
        self.param = {'eta': 0.05,
                      'max_depth': max_depth,
                      'subsample': 1.0,
                      'colsample_bytree': 0.7,
                      'objective': 'reg:linear',
                      'eval_metric': 'rmse'}
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
                                   early_stopping_rounds=20)

    def predict(self, X):
        dtest = xgb.DMatrix(data=X)
        y_ = self.estimator.predict(dtest)
        return y_
