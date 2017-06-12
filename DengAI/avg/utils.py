import numpy as np
import pandas as pd


class DataProcessor:

    @staticmethod
    def _read_csv(self, feature, label=None):
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
            normalized_data[indices] = data[indices] - mean_of_weeks[week]

        return normalized_data

    def get_train(self, city, selected_features):
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
        train['x'] = df[selected_features].as_matrix()

        week_of_year = df['week_of_year']
        total_cases = df['total_cases']
        self.mean_cases_of_weeks['city'] = \
            self._get_week_mean(week_of_year, total_cases)

        train['y'] = self._weekwise_normalize(week_of_year,
                                              total_cases,
                                              self.mean_cases_of_weeks['city'])

        return train

    def get_test(self, city, selected_features):
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

        test['x'] = self.test_dfs[selected_features].as_matrix()

        return test

    def __init__(self, train_feature, train_label,
                 test_feature,
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
        self.train_dfs = self._read_csv(train_feature, train_label)
        self.test_dfs = self._read_csv(test_feature)
        self.mean_cases_of_weeks = {}
        self.selected_features = selected_features
