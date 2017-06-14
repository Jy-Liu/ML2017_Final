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
    def _get_week_median(data_week, data):
        """
        Get median of data of each week.

        Parameter
        ---------
        data_week: np array of shape (n_rows,)
        Week of year of rows in the data.
        data: np array of shape (n_rows, n_features)
        Features to be normalized.

        Return
        ------
        median_of_weeks: np array of shape (n_weeks, n_features)
            Median of data of weeks.
        """
        median_of_weeks = []
        for week in range(1, 54):
            # indices of data that is of the week
            indices = np.where(data_week == week)

            # calculate mean of the week
            week_mean = np.median(data[indices], axis=0)
            median_of_weeks.append(week_mean)

        return np.array(median_of_weeks)

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
        y = total_cases
        # y = self._weekwise_normalize(week_of_year,
        #                              total_cases,
        #                              self.mean_cases_of_weeks[city])
        return y.reshape(-1,)

    def _get_feature(self, df, city):
        week_of_year = df['weekofyear']
        features = df[self.selected_features].as_matrix()
        x = features
        # x = self._weekwise_normalize(week_of_year,
        #                              features,
        #                              self.feature_mean[city])
        return x

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
        # train['x'] = df[self.selected_features].as_matrix()
        train['x'] = self._get_feature(df, city)
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
        # valid['x'] = df[self.selected_features].as_matrix()
        valid['x'] = self._get_feature(df, city)
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
        # test['x'] = self.test_dfs[city][self.selected_features].as_matrix()
        test['x'] = self._get_feature(self.test_dfs[city], city)

        return test

    def write_predict(self, filename, sj_y, iq_y):
        """
        Parameter
        ---------
        """

        # normalized with minus mean to shift back
        sj_y = self._weekwise_normalize(self.test_dfs['sj']['weekofyear']
                                        .as_matrix(), sj_y,
                                        -self.mean_cases_of_weeks['sj'])
        iq_y = self._weekwise_normalize(self.test_dfs['iq']['weekofyear']
                                        .as_matrix(), iq_y,
                                        -self.mean_cases_of_weeks['iq'])

        # make df of sj and iq
        sj_df = self.test_dfs['sj'][['city', 'year', 'weekofyear']]
        sj_df['total_cases'] = np.array(np.round(sj_y), dtype=int)
        # sj_df['total_cases'] = sj_df['total_cases'].shift(6)
        # sj_df['total_cases'] = sj_df['total_cases'].fillna(method='bfill')
        iq_df = self.test_dfs['iq'][['city', 'year', 'weekofyear']]
        sj_df['total_cases'] = sj_df['total_cases'].astype(int)
        iq_df['total_cases'] = np.array(np.round(iq_y), dtype=int)

        # concat and dump out
        df = pd.concat([sj_df, iq_df])
        df.to_csv(filename, index=False)

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
        self.feature_mean = {}
        self.selected_features = selected_features

        for city in ['iq', 'sj']:
            # split valid
            self.train_dfs[city], self.valid_dfs[city] = \
                self._split_valid(whole_train_dfs[city], valid_ratio)

            # calculate week mean
            last_year = self.train_dfs[city]['year'].max()
            indices = self.train_dfs[city]['year'] > 0
            week_of_year = self.train_dfs[city].loc[
                indices]['weekofyear'].as_matrix()
            total_cases = self.train_dfs[city].loc[
                indices]['total_cases'].as_matrix()
            features = self.train_dfs[city].loc[
                indices][self.selected_features].as_matrix()
            self.mean_cases_of_weeks[city] = \
                self._get_week_mean(week_of_year, total_cases)
            self.feature_mean[city] = \
                self._get_week_mean(week_of_year, features)

        # read test
        self.test_dfs = self._read_csv(test_feature)
