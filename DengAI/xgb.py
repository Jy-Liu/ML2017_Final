import pickle
import numpy as np
import xgboost as xgb
from utils import DataWriter


class Regressor:
    def __init__(self):
        self.estimators = {}

    def train(self, X_raw, Y_raw, cv=0):

        for city_name in X_raw.keys():
            X = X_raw[city_name]
            Y = Y_raw[city_name]
            X.shape = (X.shape[0], -1)

            # shuffle training data
            perm = np.random.permutation(X.shape[0])
            X = X[perm]
            Y = Y[perm]

            num_boost_round=150
            param = {
                    'booster': 'gbtree',
                    'obective': 'reg:linear',
                    'eta': 0.05,
                    'eval_metric': 'mae',
                    'colsample_bytree': 0.5,
                    'max_depth': 12
                    }

            dtrain = xgb.DMatrix(data=X, label=Y, missing=np.nan)

            if cv > 0:
                res = self.cross_val(city_name, dtrain, param, cv)
                num_boost_round = res['test-mae-mean'].argmin()
                print('Best round on cv: {}'.format(num_boost_round))

            self.estimators[city_name] = xgb.train(param, dtrain=dtrain, num_boost_round=num_boost_round, maximize=False, verbose_eval=True)

    def cross_val(self, city_name, dtrain, param, cv):
        print('City Name {}'.format(city_name))
        res = xgb.cv(param, dtrain=dtrain, maximize=False, seed=42, num_boost_round=150, early_stopping_rounds=10, nfold=cv, verbose_eval=True)
        return res

    def predict(self, X_raw):
        cases = []
        for city_name in X_raw.keys():
            X = X_raw[city_name]
            X.shape = (X.shape[0], -1)
            
            dtest = xgb.DMatrix(data=X, missing=np.nan)
            
            predictions = self.estimators[city_name].predict(data=dtest)
            cases += np.round(predictions).astype('int64').tolist()
        return np.array(cases)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', help='features data path')
    parser.add_argument('--train', help='labels data path')
    parser.add_argument('--predict', help='outputs data path')
    parser.add_argument('--test_feature')
    parser.add_argument('--model', default='xgb-model.pkl')
    parser.add_argument('--cv', type=int, default=8)
    args = parser.parse_args()

    model_path = args.model

    if args.train:
        features_path = args.data_path
        labels_path = args.train
        features = np.load(features_path)
        labels = np.load(labels_path)

        regressor = Regressor()
        regressor.train(features, labels, cv=args.cv)

        with open(model_path, 'wb') as model_f:
            pickle.dump(regressor, model_f)

    if args.predict:
        features_path = args.data_path
        outputs_path = args.predict
        original_features_path = args.test_feature
        features = np.load(features_path)

        with open(model_path, 'rb') as model_f:
            regressor = pickle.load(model_f)

        cases = regressor.predict(features)
        print('cases.shape =', cases.shape)
        writer = DataWriter(original_features_path)
        writer.write_output(cases, outputs_path)

