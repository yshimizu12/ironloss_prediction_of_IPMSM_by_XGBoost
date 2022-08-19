import os
import numpy as np
import datetime

import pandas as pd
from sklearn import model_selection, metrics
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor as XGBR
import optuna

params = {
    'path_data': 'path_to_folder',
    'fold_number': 5,
    'fraction_of_validation_samples': 0.2,
    'seed': 0,
    'n_estimators': 1000,
    'n_trials': 100
}

def _sin(x): return np.sin(np.radians(x))
def _cos(x): return np.cos(np.radians(x))

if __name__ == '__main__':
    path = params['path_data']
    fold_number = params['fold_number']
    fraction_of_validation_samples = params['fraction_of_validation_samples']

    dt_now = datetime.datetime.now()
    date = dt_now.strftime('%Y%m%d')[2:]

    df_X = pd.read_csv(f'{path}xgeom_all_scaled_train.csv', index_col=0)
    df_current = pd.read_csv(f'{path}idiq_all_scaled_train.csv', index_col=0)
    df_speed = pd.read_csv(f'{path}speed_all_scaled_train.csv', index_col=0)
    df_hysteresis = pd.read_csv(f'{path}hysteresis_all_scaled_train.csv', index_col=0)
    df_joule = pd.read_csv(f'{path}joule_all_scaled_train.csv', index_col=0)
    cols = ['id', 'iq', 'N']
    cols.extend(list(df_X.columns))
    df_X = df_X.reindex(columns=cols)
    df_X['id'] = df_current['id']
    df_X['iq'] = df_current['iq']
    df_X['N'] = df_speed['N']

    df_X_test = pd.read_csv(f'{path}xgeom_all_scaled_test.csv', index_col=0)
    df_current_test = pd.read_csv(f'{path}idiq_all_scaled_test.csv', index_col=0)
    df_speed_test = pd.read_csv(f'{path}speed_all_scaled_test.csv', index_col=0)
    df_hysteresis_test = pd.read_csv(f'{path}hysteresis_all_scaled_test.csv', index_col=0)
    df_joule_test = pd.read_csv(f'{path}joule_all_scaled_test.csv', index_col=0)
    cols = ['id', 'iq', 'N']
    cols.extend(list(df_X_test.columns))
    df_X_test = df_X_test.reindex(columns=cols)
    df_X_test['id'] = df_current_test['id']
    df_X_test['iq'] = df_current_test['iq']
    df_X_test['N'] = df_speed_test['N']

    if not os.path.exists('models'): os.mkdir('models')

    seed = params['seed']

    characs = ['hysteresis', 'joule']
    df_characs = [df_hysteresis, df_joule]
    df_characs_test = [df_hysteresis_test, df_joule_test]

    for charac, df_charac, df_charac_test in zip(characs, df_characs, df_characs_test):
        print(charac)
        X_train = np.array(df_X)
        X_test = np.array(df_X_test)
        y_train = np.array(df_charac['total'])
        y_test = np.array(df_charac_test['total'])

        X_train_tmp, X_valid, y_train_tmp, y_valid = train_test_split(X_train,
                                                                    y_train,
                                                                    test_size=fraction_of_validation_samples,
                                                                    random_state=0)
        xgbr = XGBR(n_estimators=params['n_estimators'])
        xgbr.fit(X_train_tmp, 
                y_train_tmp,
                eval_set=[(X_valid, y_valid.reshape([len(y_valid), 1]))],
                eval_metric='rmse', 
                early_stopping_rounds=100)
        best_n_estimators_in_cv = xgbr.best_iteration

        def objective(trial):
            param = {
                'verbosity': 0,
                'objective': 'reg:squarederror',
                'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
                'lambda': trial.suggest_loguniform('lambda', 1e-8, 1.0),
                'alpha': trial.suggest_loguniform('alpha', 1e-8, 1.0)
            }

            if param['booster'] == 'gbtree' or param['booster'] == 'dart':
                param['max_depth'] = trial.suggest_int('max_depth', 1, 9)
                param['eta'] = trial.suggest_loguniform('eta', 1e-8, 1.0)
                param['gamma'] = trial.suggest_loguniform('gamma', 1e-8, 1.0)
                param['grow_policy'] = trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
            if param['booster'] == 'dart':
                param['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
                param['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
                param['rate_drop'] = trial.suggest_loguniform('rate_drop', 1e-8, 1.0)
                param['skip_drop'] = trial.suggest_loguniform('skip_drop', 1e-8, 1.0)

            xgbr = XGBR(**param, n_estimators=best_n_estimators_in_cv)
            estimated_y_in_cv = model_selection.cross_val_predict(xgbr, X_train, y_train, cv=fold_number)
            r2 = metrics.r2_score(y_train, estimated_y_in_cv)
            return 1.0 - r2

        study = optuna.create_study()
        study.optimize(objective, n_trials=params['n_trials'])

        xgbr = XGBR(**study.best_params, n_estimators=params['n_estimators'])
        xgbr.fit(X_train_tmp, 
                y_train_tmp,
                eval_set=[(X_valid, y_valid.reshape([len(y_valid), 1]))],
                eval_metric='rmse', 
                early_stopping_rounds=100)
        best_n_estimators = xgbr.best_iteration

        xgbr = XGBR(**study.best_params, n_estimators=best_n_estimators)
        xgbr.fit(X_train, y_train)

        print("  Training set score: {:.3f}".format(xgbr.score(X_train, y_train))) 
        print("  Test set score: {:.3f}".format(xgbr.score(X_test, y_test)))

        xgbr.save_model(f"models/{date}_Xgboost_{charac}.model")

