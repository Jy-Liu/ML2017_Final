import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, Imputer


class DataProcessor():

    def __init__(self):
        pass

    def read_data(self, train_f, test_f):
        self.raw_train_features = pd.read_csv(train_f)
        self.raw_test_features = pd.read_csv(test_f)

    def check_NaN_exist(self, series):
        for i, v in enumerate(series.as_matrix()):
            try:
                int(v)
            except:
                print('ahhhh NaN')

    def to_numerical(self, series):
        imp = Imputer(missing_values='NaN', strategy='mode', axis=0)
        imp.fit_transform(series)

        le = LabelEncoder()
        target = le.fit_transform(series)
        print(len(target))

        enc = OneHotEncoder()
        ret = enc.fit_transform(target.reshape(-1, 1)).toarray()
        print(le.classes_)
        return le.classes_, ret

    def preprocess(self):
        train = self.raw_train_features
        test = self.raw_test_features
        train_label = train['price_doc']
        train_size = train.shape[0]
        del train['price_doc']
        features = train.columns.values
        all_corpus = pd.concat([train, test], ignore_index=True)

        for feature in features:
            if feature == 'timestamp' or feature == 'id':
                continue
            data_type = all_corpus[feature].dtype
            if data_type != 'float64' and data_type != 'int64':
                print(feature)
                new_dataframe = pd.get_dummies(all_corpus[feature])
                # print(new_dataframe)
                del all_corpus[feature]
                all_corpus = pd.concat([all_corpus, new_dataframe], axis=1)
            else:
                series = all_corpus[feature]
                series[series.isnull()] = -1
                all_corpus[feature] = series
        all_corpus_np = all_corpus.as_matrix()
        train = all_corpus_np[:train_size]
        test = all_corpus_np[train_size:]
        train = pd.concat([pd.DataFrame(train), train_label], axis=1)

        self.train = train.as_matrix()
        self.test = test

    def write_data(self, filename, pred):
        output = pd.DataFrame({'id': self.id, 'status_group': pred}, columns=['id', 'status_group'])
        output.to_csv(filename, index=False, columns=('id', 'status_group'))


def split_valid(data, valid_ratio):
    n_valid = int(data['x'].shape[0] * valid_ratio)

    train, valid = {}, {}
    for key in data:
        train[key] = data[key][:-n_valid]
        valid[key] = data[key][-n_valid:]

    return train, valid


def root_mean_squared_log_error(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    y_pred = np.where(y_pred > 0, y_pred, 0)
    rmsle = np.sqrt(np.mean(
        (np.log(y_true + 1) - np.log(y_pred + 1))**2
        ))

    return rmsle
