import pdb
import sys
import traceback
import numpy as np
import argparse
from xgb import XGBRegressor
from utils import DataProcessor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from plot import plot_pred


def main():
    parser = argparse.ArgumentParser(description='DangAIIIIIII')
    parser.add_argument('feature_train', type=str, help='feature_train.csv')
    parser.add_argument('label_train', type=str, help='label_train.csv')
    parser.add_argument('feature_test', type=str, help='feature_test.csv')
    parser.add_argument('--reg', type=str, default='xgb',
                        help='regressor to use')
    parser.add_argument('--n_iters', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--valid_ratio', type=float, default=0.05,
                        help='dimension of latent feature')
    args = parser.parse_args()

    all_features = ['ndvi_nw', 'ndvi_se', 'ndvi_sw',
                    'reanalysis_air_temp_k',
                    'reanalysis_avg_temp_k', 'reanalysis_dew_point_temp_k',
                    'reanalysis_max_air_temp_k', 'reanalysis_min_air_temp_k',
                    'reanalysis_precip_amt_kg_per_m2',
                    'reanalysis_relative_humidity_percent',
                    'reanalysis_sat_precip_amt_mm',
                    'reanalysis_specific_humidity_g_per_kg',
                    'reanalysis_tdtr_k', 'station_avg_temp_c',
                    'station_diur_temp_rng_c', 'station_max_temp_c',
                    'station_min_temp_c', 'weekofyear']
    selected_features = ['ndvi_nw', 'precipitation_amt_mm',
                         'reanalysis_max_air_temp_k',
                         'reanalysis_dew_point_temp_k',
                         'reanalysis_specific_humidity_g_per_kg',
                         'weekofyear']

    data_processor = DataProcessor(args.feature_train,
                                   args.label_train,
                                   args.feature_test,
                                   args.valid_ratio,
                                   all_features)

    for city in ['iq', 'sj']:
        train = data_processor.get_train(city)
        valid = data_processor.get_valid(city)
        test = data_processor.get_test(city)

        # start training
        regs = {'xgb': XGBRegressor(n_rounds=args.n_iters,
                                    max_depth=4, valid=valid),
                'rf': RandomForestRegressor(n_estimators=1000, n_jobs=-1),
                'et': ExtraTreesRegressor(n_estimators=500, n_jobs=-1)}
        reg = regs[args.reg]

        reg.fit(train['x'], train['y'])

        train['y_'] = reg.predict(train['x'])
        valid['y_'] = reg.predict(valid['x'])
        print('city = %s' % city)
        print('train mae = %f' % mean_absolute_error(train['y_'], train['y']))
        print('valid mae = %f' % mean_absolute_error(valid['y_'], valid['y']))
        test['y_'] = reg.predict(test['x'])

        plot_pred('%s-%s-train.png' % (args.reg, city),
                  train['y_'], train['y'])
        plot_pred('%s-%s-valid.png' % (args.reg, city),
                  valid['y_'], valid['y'])
        plot_pred('%s-%s-test.png' % (args.reg, city), test['y_'])
        print('====================================')


if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
