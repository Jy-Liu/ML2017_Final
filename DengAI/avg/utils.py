import numpy as np
import pandas as pd


class DataProcessor:

    @staticmethod
    def _read_csv(feature, label=None):
        df = pd.read_csv(feature)

        if label is not None:
            df_label = pd.read_csv(label)
            df_label = df_label['total_cases']
            df = pd.concat([df, df_label], axis=1)

        df = df.fillna(method='pad')
        dfs = {'sj': df.loc[df['city'] == 'sj'],
               'iq': df.loc[df['city'] == 'iq']}

        return dfs

    @staticmethod
    def _get_week_mean(data_week, data):
        """
        Get mean of data of each week.

        Parameter
        ---------
        data_week: np array of shape (n_rows,)
        Week of year of rows in the data.
        data: np array of shape (n_rows, n_features)
        Features to be normalized.

        Return
        ------
        mean_of_weeks: np array of shape (n_weeks, n_features)
        Mean of data of weeks.
        """
        mean_of_weeks = []
        for week in range(1, 54):
            # indices of data that is of the week
            indices = np.where(data_week == week)

            # calculate mean of the week
            week_mean = np.mean(data[indices], axis=0)
            mean_of_weeks.append(week_mean)

        return np.array(mean_of_weeks)

    @staticmethod
    def _weekwise_normalize(data_week, data, mean_of_weeks):
        """
        Minus data by mean of each week.

        Parameter
        ---------
        data_week: np array of shape (n_rows,)
            Week of year of rows in the data.
        data: np array of shape (n_rows, n_features)
            Features to be normalized.
        mean_of_weeks: np array of shape (n_weeks, n_features)
            Mean of data of weeks.

        Return
        ------
        normalized_data: np array of shape (n_rows, n_features)
            Normalized data.
        """
        normalized_data = np.zeros(data.shape)
        for week in range(1, 54):
            # indices of data that is of the week
            indices = np.where(data_week == week)

            # minus week mean
            normalized_data[indices] = data[indices] - mean_of_weeks[week - 1]

        return normalized_data

    @staticmethod
    def _split_valid(df, valid_ratio, shuffle=False):
        if shuffle:
            df = df.sample(frac=1)

        n_rows = df.shape[0]
        n_valid = int(n_rows * valid_ratio)
        df_train = df.head(n_rows - n_valid)
        df_valid = df.tail(n_valid)

        return df_train, df_valid

    def _get_label(self, df, city):
        """
        Do label preprocessing here.
        """
        week_of_year = df['weekofyear']
        total_cases = df['total_cases'].reshape(-1, 1)
        y = self._weekwise_normalize(week_of_year,
                                     total_cases,
                                     self.mean_cases_of_weeks[city])
        return y.reshape(-1,)

    def get_train(self, city):
        """
        Call preprocessing functions here.

        Parameter
        ---------
        city: string
            Either 'iq' or 'sj'.
        selected_features: list of strings.
            Names of culumns that should be used to train.

        Return
        ------
        train: dict
            train['x']: np array of shape (n_rows, len(selected_features))
                Preprocessed features.
            train['y']: np array of shape (n_rows,)
                Preprocessed labels.
        """
        train = {}

        df = self.train_dfs[city]
        train['x'] = df[self.selected_features].as_matrix()
        train['y'] = self._get_label(df, city)

        return train

    def get_valid(self, city):
        """
        Call preprocessing functions here.

        Parameter
        ---------
        city: string
            Either 'iq' or 'sj'.
        selected_features: list of strings.
            Names of culumns that should be used to train.

        Return
        ------
        valid: dict
            valid['x']: np array of shape (n_rows, len(selected_features))
                Preprocessed features.
            valid['y']: np array of shape (n_rows,)
                Preprocessed labels.
        """
        valid = {}

        df = self.valid_dfs[city]
        valid['x'] = df[self.selected_features].as_matrix()
        valid['y'] = self._get_label(df, city)

        return valid

    def get_test(self, city):
        """
        Do all the preprocessing.

        Parameter
        ---------
        city: string
            Either 'sj' or 'iq'.
        selected_features: list of strings.
            Names of culumns that should be used to train.

        Return
        ------
        test: dict
            train['x']: np array of shape (n_rows, len(selected_features))
                Preprocessed features.
            test['y']: np array of shape (n_rows,)
                Preprocessed labels.
        """
        test = {}
        test['x'] = self.test_dfs[city][self.selected_features].as_matrix()

        return test

    def __init__(self, train_feature, train_label,
                 test_feature, valid_ratio,
                 selected_features):
        """
        Data Preprocessor

        Parameter
        ---------
        train_feature: string
            Filename of the csv that contains training features.
        train_label: string
            Filename of the csv that contains training labels.
        test_feature: string
            Filename of the csv that contains testing features.
        selected_features: list of strings
            Names of culumns that should be used to train.
        """
        # read train
        whole_train_dfs = self._read_csv(train_feature, train_label)

        self.train_dfs = {}
        self.valid_dfs = {}
        self.mean_cases_of_weeks = {}

        for city in ['iq', 'sj']:
            # split valid
            self.train_dfs[city], self.valid_dfs[city] = \
                self._split_valid(whole_train_dfs['iq'], valid_ratio)

            # calculate week mean
            week_of_year = self.train_dfs[city]['weekofyear'].as_matrix()
            total_cases = self.train_dfs[city]['total_cases'].as_matrix()
            self.mean_cases_of_weeks[city] = \
                self._get_week_mean(week_of_year, total_cases)

        # read test
        self.test_dfs = self._read_csv(test_feature)
        self.selected_features = selected_features
