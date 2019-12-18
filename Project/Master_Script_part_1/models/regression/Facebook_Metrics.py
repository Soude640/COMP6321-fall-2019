import os
import pandas as pd
import numpy as np
import scipy
import scipy.stats
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV

from models import settings

from models.regression.ada_boost_regressor import Ada_boost_regressor
from models.regression.decision_tree_regressor import Decision_tree_regressor
from models.regression.gaussian_process_regressor import Gaussian_process_regressor
from models.regression.linear_least_squares import Linear_least_squares
from models.regression.neural_network_regressor import Neural_network_regressor
from models.regression.random_forest_regressor import Random_forest_regressor
from models.regression.support_vector_regressor import Support_vector_regressor

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class Facebook_metrics:
    data = []
    targets = []
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    def __init__(self):
        filepath = 'datasets/regression_datasets/5_Facebook_metrics'
        filename = 'dataset_Facebook.csv'

        # read data from the source file
        f = lambda s: (0 if s == 'Photo' else (1 if s == 'Status' else (2 if s == 'Link' else 3)))
        r = pd.read_csv(os.path.join(settings.ROOT_DIR, filepath,
                                     filename), sep=';', converters={1: f})
        r = self.missing_rows_with_most_frequent_value(r).astype(np.int)
        self.targets = r[:, 7:]
        self.data = r[:, :7]

        # separate into train and test sets
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(self.data, self.targets, test_size=0.33,
                             random_state=0)

        # datasets normalization
        # train_matrix = np.column_stack((self.x_train, self.y_train))
        # test_matrix = np.column_stack((self.x_test, self.y_test))
        scaler = sklearn.preprocessing.StandardScaler().fit(self.x_train)
        self.x_train = scaler.transform(self.x_train)
        self.x_test = scaler.transform(self.x_test)
        # self.y_train = scaler.transform(transformed_train_m[:, 7:])
        # self.x_train = scaler.transform(transformed_train_m[:, :7])
        # self.y_test = scaler.transform(transformed_test_m)

    def printself(self):
        print(self.data)
        print(self.targets)

    ##################### Data pre-processing (by Yixuan Li)#####################
    def missing_rows_with_missing_values_ignore(self, data):
        # drop the rows with NANs
        return data.dropna()

    def missing_rows_with_the_mean(self, data):
        # train an imputer
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp.fit(data)
        # impute the missing values with the mean
        return imp.transform(data)

    def missing_rows_with_most_frequent_value(self, data):
        # train an imputer
        imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        imp.fit(data)
        # impute the missing values with the most frequent value
        return imp.transform(data)

    ###################### model training ######################
    def support_vector_regression(self):
        # coef0 = scipy.stats.uniform(0, 5)
        # epsilon = scipy.stats.reciprocal(0.01, 1)
        np.random.seed(0)
        res = []
        train_mse = []
        train_r2 = []
        test_mse = []
        test_r2 = []
        for i in range(12):
            svr = Support_vector_regressor(
                x_train=self.x_train,
                y_train=self.y_train[:, i],
                cv=3,
                n_iter=10,
                n_jobs=-1,
                C=scipy.stats.reciprocal(1, 100),
                kernel=['sigmoid', 'rbf', 'linear'],
                gamma=scipy.stats.reciprocal(0.01, 20),
                random_search=True)
            # training
            train_r2.append(svr.r2_score(
                x_test=self.x_train,
                y_test=self.y_train[:, i]))
            train_mse.append(svr.mean_squared_error(
                x_test=self.x_train,
                y_test=self.y_train[:, i]))
            # test
            test_r2.append(svr.r2_score(
                x_test=self.x_train,
                y_test=self.y_train[:, i]))
            test_mse.append(svr.mean_squared_error(
                x_test=self.x_train,
                y_test=self.y_train[:, i]))
        res.append((train_mse, train_r2))
        res.append((test_mse, test_r2))
        return res

    def decision_tree_regression(self):
        np.random.seed(0)
        dtr = Decision_tree_regressor(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            n_iter=10,
            max_depth=range(1, 20),
            min_samples_leaf=range(1, 20),
            n_jobs=-1,
            random_search=True)

#         dtr.print_parameter_candidates()
#         dtr.print_best_estimator()

        return (dtr.evaluate(data=self.x_train, targets=self.y_train),
                dtr.evaluate(data=self.x_test, targets=self.y_test))


    def random_forest_regression(self):
        np.random.seed(0)
        rfr = Random_forest_regressor(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            n_iter=10,
            n_jobs=-1,
            n_estimators=range(1, 100),
            max_depth=range(1, 20),
            random_search=True)

#         rfr.print_parameter_candidates()
#         rfr.print_best_estimator()

        return (rfr.evaluate(data=self.x_train, targets=self.y_train),
                rfr.evaluate(data=self.x_test, targets=self.y_test))


    def ada_boost_regression(self):
        np.random.seed(0)
        res = []
        train_mse = []
        train_r2 = []
        test_mse = []
        test_r2 = []
        for i in range(12):
            abr = Ada_boost_regressor(
                x_train=self.x_train,
                y_train=self.y_train[:,i],
                cv=3,
                n_iter=15,
                n_estimators=range(1, 100),
                n_jobs=-1,
                random_search=True)
            # training
            train_r2.append(abr.r2_score(
                x_test=self.x_train,
                y_test=self.y_train[:, i]))
            train_mse.append(abr.mean_squared_error(
                x_test=self.x_train,
                y_test=self.y_train[:, i]))
            # test
            test_r2.append(abr.r2_score(
                x_test=self.x_train,
                y_test=self.y_train[:, i]))
            test_mse.append(abr.mean_squared_error(
                x_test=self.x_train,
                y_test=self.y_train[:, i]))
        res.append((train_mse, train_r2))
        res.append((test_mse, test_r2))
        return res
        

#         abr.print_parameter_candidates()
#         abr.print_best_estimator()

#         return abr.mean_squared_error(
#             x_test=self.x_test,
#             y_test=y_test)

    def gaussian_process_regression(self):
        np.random.seed(0)
        res = []
        train_mse = []
        train_r2 = []
        test_mse = []
        test_r2 = []
        for i in range(12):
            gpr = Gaussian_process_regressor(
                x_train=self.x_train,
                y_train=self.y_train[:,i],
                cv=3,
                n_iter=15,
                # kernel=kernel,
                alpha=scipy.stats.reciprocal(1e-11, 1e-8),
                n_jobs=-1,
                random_search=True)
            # training
            train_r2.append(gpr.r2_score(
                x_test=self.x_train,
                y_test=self.y_train[:, i]))
            train_mse.append(gpr.mean_squared_error(
                x_test=self.x_train,
                y_test=self.y_train[:, i]))
            # test
            test_r2.append(gpr.r2_score(
                x_test=self.x_train,
                y_test=self.y_train[:, i]))
            test_mse.append(gpr.mean_squared_error(
                x_test=self.x_train,
                y_test=self.y_train[:, i]))
        res.append((train_mse, train_r2))
        res.append((test_mse, test_r2))
        return res
            

        # print all possible parameter values and the best parameters
#         gpr.print_parameter_candidates()
#         gpr.print_best_estimator()

        # return the mean squared error
#         return gpr.mean_squared_error(
#             x_test=self.x_test,
#             y_test=y_test)

    def linear_least_squares(self):
        alpha = np.logspace(start=3, stop=9, base=2, num=7, dtype=np.float32)
        max_iter = np.logspace(start=2, stop=4, base=10, num=3, dtype=np.int)
        res = []
        train_mse = []
        train_r2 = []
        test_mse = []
        test_r2 = []
        np.random.seed(0)
        for i in range(12):
            lls = Linear_least_squares(
                x_train=self.x_train,
                y_train=self.y_train[:,i],
                cv=3,
                n_iter=15,
                alpha=alpha,
                max_iter=max_iter,
                n_jobs = -1,
                random_search=True
            )
            # training
            train_r2.append(lls.r2_score(
                x_test=self.x_train,
                y_test=self.y_train[:, i]))
            train_mse.append(lls.mean_squared_error(
                x_test=self.x_train,
                y_test=self.y_train[:, i]))
            # test
            test_r2.append(lls.r2_score(
                x_test=self.x_train,
                y_test=self.y_train[:, i]))
            test_mse.append(lls.mean_squared_error(
                x_test=self.x_train,
                y_test=self.y_train[:, i]))
        res.append((train_mse, train_r2))
        res.append((test_mse, test_r2))
        return res

    def neural_network_regression(self):
        np.random.seed(0)
        nnr = Neural_network_regressor(
            x_train=self.x_train,
            y_train=self.y_train,
            cv=3,
            n_iter=10,
            hidden_layer_sizes=range(100, 1000),
            activation=['logistic', 'tanh', 'relu'],
            max_iter=range(1000, 10000),
            n_jobs=-1,
            random_search=True)

        # print all possible parameter values and the best parameters
#         nnr.print_parameter_candidates()
#         nnr.print_best_estimator()

        # # return the mean squared error
        return (nnr.evaluate(data=self.x_train, targets=self.y_train),
                nnr.evaluate(data=self.x_test, targets=self.y_test))


if __name__ == '__main__':
    fm = Facebook_metrics()
    
