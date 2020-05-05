import numpy as np
import lightgbm as lgb
import catboost 
#import xgboost
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut, KFold
from scipy.stats import linregress
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy.stats import pearsonr as pr

class GBM:

    def __init__(self,
            package='lightgbm',
            X=None,
            y=None,
            model_scheme=None,
            feature_names=None,
            parameters=None,
            cv=5,
            test_size=0.1,
            grid_search=False,
            grid_search_scoring='neg_mean_squared_error',
            param_grid=None,
            eval_metric='rmse'):

        self.package = package
        self.X = X
        self.y = y
        self.model_scheme = model_scheme
        self.feature_names = feature_names
        self.parameters = parameters
        self.cv = cv
        self.test_size = test_size
        self.grid_search = grid_search
        self.grid_search_scoring = grid_search_scoring
        self.param_grid = param_grid

    def run_model(self):
        if self.grid_search:
            self.run_grid_search()
            self.parameters = self.best_params
        
        '''
        Xtr, Xts, ytr, yts, CT_RT_tr, CT_RT_ts = train_test_split(self.X, 
                self.y, self.CT_RT, test_size=self.test_size)
        '''

        if self.cv == 'loo':
            cv = LeaveOneOut()
        else:
            cv = KFold(n_splits=self.cv, shuffle=True)

        self.rmse_cv_train = []
        self.r2_cv_train = []
        self.rmse_cv_test = []
        self.r2_cv_test = []
        self.pr_cv_train = []
        self.pr_cv_test = []
        
        est = {'lightgbm': lgb.LGBMRegressor,
               'catboost': catboost.CatBoostRegressor}
               #'xgboost': xgboost.XGBRegressor}
        
        self.model = []
        model = est[self.package](**self.parameters)
        for n, (tr_id, ts_id) in enumerate(cv.split(self.y)):
            print('Running Validation {} of {}'.format(n, self.cv))
            if self.package == 'lightgbm':
                self.model.append(model.fit(self.X[tr_id], self.y[tr_id],
                    eval_set=[(self.X[ts_id], self.y[ts_id])],
                    eval_metric='rmse', early_stopping_rounds=20,
                    feature_name=self.feature_names))
            elif self.package == 'xgboost':
                self.model.append(model.fit(self.X[tr_id], self.y[tr_id],
                    eval_set=[(self.X[ts_id], self.y[ts_id])],
                    eval_metric='rmse', early_stopping_rounds=20))
            else:
                self.model.append(model.fit(self.X[tr_id], self.y[tr_id],
                    eval_set=[(self.X[ts_id], self.y[ts_id])],
                    early_stopping_rounds=20))
            if self.package == 'lightgbm':
                self.y_cv_tr_pred = self.model[-1].predict(self.X[tr_id],
                    num_iteration=self.model[-1].best_iteration_)
                self.y_cv_ts_pred = self.model[-1].predict(self.X[ts_id], 
                    num_iteration=self.model[-1].best_iteration_)
            else:
                self.y_cv_tr_pred = self.model[-1].predict(self.X[tr_id])
                self.y_cv_ts_pred = self.model[-1].predict(self.X[ts_id])
            self.y_cv_tr = self.y[tr_id]
            self.y_cv_ts = self.y[ts_id]
            self.rmse_cv_train.append(np.sqrt(mean_squared_error(
                self.y_cv_tr_pred, self.y[tr_id])))
            self.rmse_cv_test.append(np.sqrt(mean_squared_error(
                self.y_cv_ts_pred, self.y[ts_id])))
            self.r2_cv_train.append(linregress(self.y_cv_tr_pred, 
                self.y[tr_id])[2]**2)
            self.r2_cv_test.append(linregress(self.y_cv_ts_pred, 
                self.y[ts_id])[2]**2)
            self.pr_cv_train.append(pr(self.y_cv_tr_pred, self.y[tr_id]))
            self.pr_cv_test.append(pr(self.y_cv_ts_pred, self.y[ts_id]))
        
        self.N_dp = len(self.y)
        self.rmse_mean_train = np.mean(self.rmse_cv_train)
        self.rmse_std_train = np.std(self.rmse_cv_train)
        self.rmse_mean_test = np.mean(self.rmse_cv_test)
        self.rmse_std_test = np.std(self.rmse_cv_test)
        self.r2_mean_train = np.mean(self.r2_cv_train)
        self.r2_std_train = np.std(self.r2_cv_train)
        self.r2_mean_test = np.mean(self.r2_cv_test)
        self.r2_std_test = np.std(self.r2_cv_test)
        self.pr_mean_train = np.mean([i[0] for i in self.pr_cv_train])
        self.pr_std_train = np.std([i[0] for i in self.pr_cv_train])
        self.pr_mean_test = np.mean([i[0] for i in self.pr_cv_test])
        self.pr_std_test = np.std([i[0] for i in self.pr_cv_test])
    
