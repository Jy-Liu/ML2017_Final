import argparse
from utils import DataProcessor, split_valid, root_mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, BayesianRidge, Lasso
from xgb import XGBRegressor
from macro import Macro
import numpy as np
import pdb
import sys
import traceback
from dnn import DNNRegressor
from sklearn.preprocessing import normalize


def parse_args():
    parser = argparse.ArgumentParser(description='Sberbank Housing Market')
    parser.add_argument('train', type=str)
    parser.add_argument('macro', type=str)
    parser.add_argument('test', type=str)
    parser.add_argument('--model', type=str, default='RandomForest')
    parser.add_argument('--valid_ratio', type=float, default=0.1)
    parser.add_argument('--output', type=str, default='output')
    return parser.parse_args()


def main(args):
    data = DataProcessor()
    data.read_data(args.train, args.test)
    data.preprocess()

    macro = Macro(args.macro)
    macro_features = macro.extract_features(data.train)
    train_data = {'x': data.train[:, 2:-1],
                  'y': np.array(data.train[:, -1], dtype=float)}
    train_data['x'] = np.concatenate([train_data['x'], macro_features],
                                     axis=1)

    # pdb.set_trace()
    train, valid = split_valid(train_data, args.valid_ratio)

    train['x'] = normalize(train['x'])
    valid['x'] = normalize(valid['x'])

    regressors = {
        'RandomForest': RandomForestRegressor(n_estimators=50, n_jobs=-1),
        'ExtraTrees': ExtraTreesRegressor(n_estimators=500, n_jobs=-1),
        'XGB': XGBRegressor(n_rounds=500, max_depth=13),
        'Ridge': Ridge(normalize=True, alpha=0),
        'Lasso': Lasso(normalize=True),
        'Bayesian': BayesianRidge(normalize=True),
        'DNN': DNNRegressor(valid=valid)
    }
    regressor = regressors[args.model]
    regressor.fit(train['x'], train['y'])
    train['y_'] = regressor.predict(train['x'])
    valid['y_'] = regressor.predict(valid['x'])
    train['rmsle'] = root_mean_squared_log_error(train['y'], train['y_'])
    valid['rmsle'] = root_mean_squared_log_error(valid['y'], valid['y_'])
    print('Train RMSLE = %f' % train['rmsle'])
    print('Valid RMSLE = %f' % valid['rmsle'])
    # pdb.set_trace()

    macro_features = macro.extract_features(data.test)
    test_data = data.test[:, 2:]
    test_data = np.concatenate([test_data, macro_features], axis=1)
    output = regressor.predict(test_data)

    result = []
    for i, v in enumerate(output):
        result.append('{},{}'.format(30474+i, v))
    with open(args.output, 'w+') as f:
        f.write('id,price_doc\n')
        f.write('\n'.join(result))


if __name__ == '__main__':
    try:
        args = parse_args()
        main(args)
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
