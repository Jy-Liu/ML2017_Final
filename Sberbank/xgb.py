import xgboost as xgb


class XGBRegressor:

    def __init__(self, n_rounds):
        self.param = {}
        self.n_rounds = n_rounds

    def fit(self, X, y):
        dtrain = xgb.DMatrix(data=X, label=y)
        self.estimator = xgb.train(self.param, dtrain, self.n_rounds)

    def predict(self, X):
        dtest = xgb.DMatrix(data=X)
        y_ = self.estimator.predict(dtest)
        return y_
