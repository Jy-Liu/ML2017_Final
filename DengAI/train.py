import pickle
import argparse
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import TimeSeriesSplit


def parse_args():
    parser = argparse.ArgumentParser('DengAI')
    parser.add_argument('--train', required=True)
    parser.add_argument('--test', required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--predict')
    parser.add_argument('--cv', action='store_true')
    return parser.parse_args()


def add_lags(df, features, lags=3):
    df_list = [df] + [df[features].add_suffix('_lag_{}'.format(i+1)).shift(i+1) for i in range(lags)]
    df_lags = pd.concat(df_list, axis=1)
    df_lags = df_lags.iloc[lags:]
   
    rolling = df_lags[features].rolling(window=lags+1, min_periods=1)
    df_rolling_mean, df_rolling_std = rolling.mean().add_prefix('rolling_mean_'), rolling.std().add_prefix('rolling_std_')
    
    df_all = pd.concat([df_lags, df_rolling_mean, df_rolling_std], axis=1)
    df_all.fillna(method='bfill', inplace=True)
    return df_all


def rolling(x, window=3):
    df = pd.DataFrame({'pred': x})
    return df['pred'].rolling(window, min_periods=1).mean().values.squeeze()


def time_series_cv_score(est, X, Y, n_splits=5):
    scores = []
    for train_index, test_index in TimeSeriesSplit(n_splits=n_splits).split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        Y_pred = est.fit(X_train, Y_train).predict(X_test)
        Y_pred = rolling(Y_pred)
        scores.append(mean_absolute_error(Y_test, Y_pred))
    return scores


def main(args):
    df_train, df_test, labels = pd.read_csv(args.train), pd.read_csv(args.test), pd.read_csv(args.label)

    df_train['test'] = 0
    df_test['test'] = 1

    data = pd.concat([df_train, df_test])

    data.drop(['ndvi_ne', 'precipitation_amt_mm','reanalysis_avg_temp_k',
		'reanalysis_specific_humidity_g_per_kg', 'station_diur_temp_rng_c'], axis=1, inplace=True)

    cities = ['sj', 'iq']
    indices = ['city', 'year', 'weekofyear']
    features = [col for col in data.columns if col not in indices+['week_start_date', 'test']]
    
    res = []
    lags = 5
    for city in cities:
        df = data[data.city == city]
        df.fillna(method='ffill', inplace=True)
        df = add_lags(df, features, lags=lags)

        if args.predict is None:
            X_train = df[df.test == 0].drop(indices + ['week_start_date', 'test'], axis=1).values
            Y_train = labels[labels.city == city].iloc[lags:].drop(indices, axis=1).values.squeeze()

            model = XGBRegressor(
                    n_estimators=200,
                    max_depth=12, 
                    learning_rate=0.05, 
                    colsample_bytree=0.5,
                    silent=False,
                    seed=5,
                    nthread=-1)

            '''
            model = RandomForestRegressor(
                    max_features=120, 
                    min_samples_split=70, 
                    n_estimators=840,
                    max_depth=12,
                    min_samples_leaf=3,
                    n_jobs=-1)

            model = ExtraTreesRegressor(
                    n_estimators=3000,
                    max_depth=3,
                    n_jobs=-1)
            '''

            if args.cv:
                scores = time_series_cv_score(model, X_train, Y_train, n_splits=4)
                print(scores, np.mean(scores), np.median(scores), np.std(scores))
            
            model.fit(X_train, Y_train)
            pickle.dump(model, open('model-{}.pkl'.format(city), 'wb'))

        else:
            X_test = df[df.test == 1].drop(indices + ['week_start_date', 'test'], axis=1).values

            fmodel = 'model-{}.pkl'.format(city)
            model = pickle.load(open(fmodel, 'rb'))
            pred = model.predict(X_test)
            pred = rolling(pred)
            res.append(pred)

    if args.predict is not None:
        df_out = df_test[indices].copy()
        df_out['total_cases'] = np.concatenate([pred for pred in res], axis=0)
        df_out['total_cases'] = df_out['total_cases'].round().astype(int)

        df_out.to_csv(args.predict, index=False)


if __name__ == '__main__':
    args = parse_args()
    main(args)
