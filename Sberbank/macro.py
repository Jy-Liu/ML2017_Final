import pandas as pd
import dateutil.parser
import numpy as np

chosen_cols = ['oil_urals', 'gdp_quart', 'cpi',
               'ppi', 'usdrub', 'eurrub', 'brent',
               'average_provision_of_build_contract_moscow', 'micex',
               'deposits_value', 'deposits_rate', 'mortgage_rate',
               'income_per_cap', 'salary', 'fixed_basket',
               'unemployment', 'employment', 'housing_fund_sqm',
               'rent_price_4+room_bus', 'rent_price_3room_bus',
               'rent_price_2room_bus', 'rent_price_1room_bus',
               'rent_price_3room_eco', 'rent_price_2room_eco',
               'rent_price_1room_eco']


class Macro:

    def __init__(self, filename):
        self.macro = pd.read_csv(filename, header=0)
        self.macro = self.macro.fillna(method='backfill')
        self.macro = self.macro.fillna(method='pad')

    def extract_features(self, data,
                         interval=(-20, 0),
                         cols=chosen_cols):
        """
        Extract features from macro for each transaction (row) in data.
        This function will average date in macro.csv for days before the
        transaction and return the result as features.

        Parameters
        ----------
        date: array-liked
            Note that its second column should be date (of ISO format).

        interval: tuple
            Interval of macro data of days to be considered for
            each transaction.

        cols: list
            List of columns name of macro.csv that should be considered.
        """
        date_list = list(self.macro['timestamp'])
        features = np.zeros((len(data), len(cols)))
        for i in range(len(data)):
            row_date = dateutil.parser.parse(data[i][1]).date()
            date_index = date_list.index(row_date.isoformat())
            related = self.macro[cols][date_index + interval[0]:
                                       date_index + interval[1]]
            features[i] = np.mean(related.as_matrix(), axis=0)

        return features
