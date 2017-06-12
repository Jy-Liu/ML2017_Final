import pdb
import sys
import traceback
import numpy as np
import argparse
from xgb import XGBRegressor
from utils import DataProcessor


def main():
    parser = argparse.ArgumentParser(description='DangAIIIIIII')
    parser.add_argument('feature_train', type=str, help='feature_train.csv')
    parser.add_argument('label_train', type=str, help='label_train.csv')
    parser.add_argument('feature_test', type=str, help='feature_test.csv')
    parser.add_argument('--n_iters', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--valid_ratio', type=float, default=0.05,
                        help='dimension of latent feature')
    args = parser.parse_args()

    selected_features = ['ndvi_nw', 'precipitation_amt_mm',
                         'reanalysis_max_air_temp_k',
                         'reanalysis_dew_point_temp_k',
                         'reanalysis_specific_humidity_g_per_kg']

    data_processor = DataProcessor(args.feature_train,
                                   args.label_train,
                                   args.feature_test,
                                   args.valid_ratio,
                                   selected_features)

    for city in ['iq', 'sj']:
        train = data_processor.get_train(city)
        valid = data_processor.get_valid(city)
        test = data_processor.get_test(city)

        # start training
        clf = XGBRegressor(n_rounds=args.n_iters, max_depth=6, valid=valid)
        clf.fit(train['x'], train['y'])

        train['y_'] = clf.predict(train['x'])
        valid['y_'] = clf.predict(valid['x'])
        test['y_'] = clf.predict(test['x'])


if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
