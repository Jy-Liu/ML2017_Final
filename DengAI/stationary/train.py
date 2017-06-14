import pdb
import sys
import traceback
import numpy as np
import argparse
from xgb import XGBRegressor
from utils import DataProcessor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from ensemble import Ensemble
from plot import plot_pred


def main():
    parser = argparse.ArgumentParser(description='DangAIIIIIII')
    parser.add_argument('feature_train', type=str, help='feature_train.csv')
    parser.add_argument('label_train', type=str, help='label_train.csv')
    parser.add_argument('feature_test', type=str, help='feature_test.csv')
    parser.add_argument('out', type=str, help='out.csv')
    parser.add_argument('--reg', type=str, default='xgb',
                        help='regressor to use')
    parser.add_argument('--n_iters', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--valid_ratio', type=float, default=0.05,
                        help='dimension of latent feature')
    parser.add_argument('--conv', type=bool, default=False)
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

    y_pred = {}
    iters = {'iq': 50, 'sj': 200}
    for city in ['iq', 'sj']:
        train = data_processor.get_train(city)
        valid = data_processor.get_valid(city)
        test = data_processor.get_test(city)

        # start training
        regs = {'xgb': XGBRegressor(n_rounds=iters[city],
                                    max_depth=7, valid=valid),
                'rf': RandomForestRegressor(n_estimators=1000, max_depth=5, n_jobs=-1),
                'et': ExtraTreesRegressor(n_estimators=5000, n_jobs=-1),
                'ensemble': Ensemble(valid=valid)}
        reg = regs[args.reg]

        # weights = np.exp(train['y']) / np.sum(np.exp(train['y']))
        # weights = (train['y'] - np.min(train['y'])) / (np.max(train['y']) - np.min(train['y']))
        # weights = weights ** 2
        # pdb.set_trace()
        # weights += 0.00001
        # weights = np.where(train['y'] > 100, train['y'], 1)
        weights = train['y'] + np.mean(train['y'])
        # weights[np.where(train['y'] < 100)] = 1
        reg.fit(train['x'], train['y'], sample_weight=weights)

        train['y_'] = reg.predict(train['x'])
        valid['y_'] = reg.predict(valid['x'])
        test['y_'] = reg.predict(test['x'])

        if args.conv:
            # v = np.array([5, 4, 3, 2, 1])
            v = np.ones(7)
            v = v / np.sum(v)
            train['y_'] = np.convolve(train['y_'], v, 'same')
            valid['y_'] = np.convolve(valid['y_'], v, 'same')
            test['y_'] = np.convolve(test['y_'], v, 'same')

        train['y_'] = np.roll(train['y_'], 6)
        valid['y_'] = np.roll(valid['y_'], 6)
        test['y_'] = np.roll(test['y_'], 6)

        print('city = %s' % city)
        print('train mae = %f' % mean_absolute_error(train['y_'], train['y']))
        print('valid mae = %f' % mean_absolute_error(valid['y_'], valid['y']))

        # plot here
        conv = '-conv' if args.conv else ''
        plot_pred('%s-%s-train%s.png' % (args.reg, city, conv),
                  train['y_'], train['y'])
        plot_pred('%s-%s-valid%s.png' % (args.reg, city, conv),
                  valid['y_'], valid['y'])
        plot_pred('%s-%s-test%s.png' % (args.reg, city, conv), test['y_'])
        print('====================================')

        y_pred[city] = test['y_']

    data_processor.write_predict(args.out, y_pred['sj'], y_pred['iq'])


if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
