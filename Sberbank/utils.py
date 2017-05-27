import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, Imputer

class DataProcessor():

    def __init__(self):
        pass

    def read_data(self, train_f, test_f):
        self.raw_train_features = pd.read_csv(train_f)
        self.raw_test_features = pd.read_csv(test_f)

    def to_numerical(self, series):
        print(series)
        le = LabelEncoder()
        target = le.fit_transform(series)

        enc = OneHotEncoder()
        ret = enc.fit_transform(target.reshape(-1, 1)).toarray()
        return le.classes_, ret

    def preprocess(self):
        train = self.raw_train_features
        test = self.raw_test_features
        train_label = train['price_doc']
        del train['price_doc']
        features = train.columns.values
        all_corpus = pd.concat([train, test], ignore_index=True)
        # print(all_corpus)
        for feature in features:
            if feature == 'timestamp':
                continue
            data_type = all_corpus[feature].dtype
            if data_type != 'float64' and data_type != 'int64':
                print(feature)
                f, r = self.to_numerical(train[feature])
                # new_dataframe = pd.DataFrame(r, columns=f)
                # del all_corpus[feature]
                # all_corpus = pd.concat([all_corpus, new_dataframe], axis=1)



    def write_data(self, filename, pred):
        output = pd.DataFrame({'id': self.id, 'status_group': pred}, columns=['id', 'status_group'])
        output.to_csv(filename, index=False, columns=('id', 'status_group'))
