import pickle
import numpy as np
from utils import DataWriter
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error


class Regressor:
    def __init__(self):
        self.boosters = {}

    def train(self, X_raw, Y_raw):
        for city_name in X_raw.keys():
            X = X_raw[city_name]
            Y = Y_raw[city_name]
            X.shape = (X.shape[0], -1)
            # X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.2)
            X_train = X
            Y_train = Y

            self.boosters[city_name] = GradientBoostingRegressor(learning_rate=0.05, n_estimators=1000, max_depth=7, loss='lad', criterion='mse')
            # self.boosters[city_name] = GradientBoostingRegressor(loss='ls', criterion='mse')
            # self.boosters[city_name].fit(X_train, Y_train)
            scores = cross_val_score(self.boosters[city_name], X_train, Y_train,
                    scoring='neg_mean_absolute_error', cv=8, n_jobs=-1)
            print(city_name, scores, scores.mean(), scores.std())

            # predictions = self.boosters[city_name].predict(X_valid)
            # mae_val = mean_absolute_error(Y_valid, np.round(predictions))
            # print('MAE on valid set({}) is {}'.format(city_name, mae_val))

    def predict(self, X_raw):
        cases = []
        for city_name in X_raw.keys():
            X = X_raw[city_name]
            X.shape = (X.shape[0], -1)
            predictions = self.boosters[city_name].predict(X)
            cases += np.round(predictions).astype('int64').tolist()
        return np.array(cases)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', help='features data path')
    parser.add_argument('--train', help='labels data path')
    parser.add_argument('--predict', help='outputs data path')
    parser.add_argument('--test_feature')
    parser.add_argument('--model', default='gb-model.pkl')
    args = parser.parse_args()

    model_path = args.model

    if args.train:
        features_path = args.data_path
        labels_path = args.train
        features = np.load(features_path)
        labels = np.load(labels_path)

        regressor = Regressor()
        regressor.train(features, labels)

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

