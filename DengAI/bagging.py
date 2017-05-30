import pickle
import numpy as np
from utils import DataWriter
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import cross_val_score


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

            self.estimators[city_name] = BaggingRegressor(
                    n_estimators=100,
                    max_features=0.6,
                    max_samples=0.6)

            if cv > 0:
                self.cross_val(city_name, X, Y, cv)

            self.estimators[city_name].fit(X, Y)

    def cross_val(self, city_name, X, Y, cv):
        scores = cross_val_score(
                self.estimators[city_name],
                X, Y,
                scoring='neg_mean_absolute_error',
                cv=cv,
                n_jobs=-1)
        print('[Cross Validate]', city_name, scores, scores.mean(), scores.std())

    def predict(self, X_raw):
        cases = []
        for city_name in X_raw.keys():
            X = X_raw[city_name]
            X.shape = (X.shape[0], -1)
            predictions = self.estimators[city_name].predict(X)
            cases += np.round(predictions).astype('int64').tolist()
        return np.array(cases)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', help='features data path')
    parser.add_argument('--train', help='labels data path')
    parser.add_argument('--predict', help='outputs data path')
    parser.add_argument('--test_feature')
    parser.add_argument('--model', default='bg-model.pkl')
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
