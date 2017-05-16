import pickle
import argparse
import numpy as np
from utils import DataReader
from pandas import DataFrame
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

class Regressor:
    def __init__(self):
        print('kerker')

    def train(self, X_raw, Y_raw):
        cities = np.unique(X_raw[:, 0])
        self.city2id = {city_name: idx for idx, city_name in enumerate(cities)}
        X_raw[:, 0] = [self.city2id[city_name] for city_name in X_raw[:, 0]]
        print('X_raw.shape =', X_raw.shape)
        X_raw = np.delete(X_raw, [1, 3], 1)
        print('X_raw.shape =', X_raw.shape)

        X_train, X_valid, Y_train, Y_valid = train_test_split(X_raw, Y_raw, test_size=0.01)

        self.booster = GradientBoostingRegressor()
        self.booster.fit(X_train, Y_train)

        predictions = self.booster.predict(X_valid)
        mae_val = mean_absolute_error(Y_valid, np.round(predictions))
        print('MAE on valid set is {}'.format(mae_val))

    def predict(self, X_raw, outputs_path):
        city_col = np.copy(X_raw[:, 0])
        year_col = np.copy(X_raw[:, 1])
        week_col = np.copy(X_raw[:, 2])
        X_raw[:, 0] = [self.city2id[city_name] for city_name in X_raw[:, 0]]
        X_raw = np.delete(X_raw, [1, 3], 1)
        predictions = self.booster.predict(X_raw)

        cols = {'city': city_col, 'year': year_col, 'weekofyear': week_col, 'total_cases': np.round(predictions).astype('int64')}
        data_frame = DataFrame(cols, columns=['city', 'year', 'weekofyear', 'total_cases'])
        data_frame.to_csv(outputs_path, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', help='features data path')
    parser.add_argument('--train', help='labels data path')
    parser.add_argument('--predict', help='outputs data path')
    parser.add_argument('--model', default='gb-model.pkl')
    args = parser.parse_args()

    model_path = args.model

    if args.train:
        features_path = args.data_path
        labels_path = args.train
        features = DataReader.read_features(features_path)
        labels = DataReader.read_labels(labels_path)

        regressor = Regressor()
        regressor.train(features, labels)

        with open(model_path, 'wb') as model_f:
            pickle.dump(regressor, model_f)

    if args.predict:
        features_path = args.data_path
        outputs_path = args.predict
        features = DataReader.read_features(features_path)

        with open(model_path, 'rb') as model_f:
            regressor = pickle.load(model_f)

        regressor.predict(features, outputs_path)
