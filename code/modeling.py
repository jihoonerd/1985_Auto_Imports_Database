# -*- coding: utf-8 -*-
"""
Created on 8/6/17
Author: Jihoon Kim
"""

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor
from keras.layers import Dense, BatchNormalization
from keras.models import Sequential
from keras import losses
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from config import random_state, LOG_DIR, FIGURE_DIR
import matplotlib.pyplot as plt
import os


# -------------------- [Ridge] --------------------
def ridge_grid(X_train, y_train, X_test=None, y_test=None):
    """Returns Optimized grid model of ridge regression"""
    param_grid = [{'alpha': np.logspace(-2, 2, 50)}]
    ridge_reg = Ridge()
    grid_search = GridSearchCV(ridge_reg, param_grid, cv=10, n_jobs=-1, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    if X_test is not None and y_test is not None:
        print("\n\n========== [Ridge Regression Grid Search Report] ==========")
        print("Best Param: ", grid_search.best_params_)
        print("========== [Ridge Regression Performance Report] ==========")
        y_train_pred = grid_search.predict(X_train)
        print("Training Set Error (MSE): ", mean_squared_error(y_train, y_train_pred))
        y_pred = grid_search.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print("Test Set Error     (MSE): ", mse)
        print("Test Set Error    (RMSE): ", np.sqrt(mse))

        save_report = pd.DataFrame(data=[[random_state, 'Ridge', mse, np.sqrt(mse)]],
                                   columns=['Random State', 'Model', 'MSE', 'RMSE'])

        if not os.path.isfile(LOG_DIR + 'ridge.csv'):
            save_report.to_csv(LOG_DIR + 'ridge.csv', header=save_report.columns)
        else:
            save_report.to_csv(LOG_DIR + 'ridge.csv', mode='a', header=False)

    return grid_search


# ----------------- [Elastic Net] -----------------
def elastic_net_grid(X_train, y_train, X_test=None, y_test=None):
    """Returns Optimized grid model of elastic net regression"""
    param_grid = [{'alpha': np.logspace(-2, 2, 50), 'l1_ratio': np.linspace(0.1, 0.98, 40)}]
    elastic_net_reg = ElasticNet(tol=1e-2)
    grid_search = GridSearchCV(elastic_net_reg, param_grid, cv=10, n_jobs=-1,
                               verbose=0, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    if X_test is not None and y_test is not None:
        print("\n\n========== [Elastic Net Regression Grid Search Report] ==========")
        print("Best Param: ", grid_search.best_params_)
        print("========== [Elastic Net Regression Performance Report] ==========")
        y_train_pred = grid_search.predict(X_train)
        print("Training Set Error (MSE): ", mean_squared_error(y_train, y_train_pred))
        y_pred = grid_search.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print("Test Set Error     (MSE): ", mse)
        print("Test Set Error    (RMSE): ", np.sqrt(mse))

        save_report = pd.DataFrame(data=[[random_state, 'Elastic Net', mse, np.sqrt(mse)]],
                                   columns=['Random State', 'Model', 'MSE', 'RMSE'])

        if not os.path.isfile(LOG_DIR + 'elastic_net.csv'):
            save_report.to_csv(LOG_DIR + 'elastic_net.csv', header=save_report.columns)
        else:
            save_report.to_csv(LOG_DIR + 'elastic_net.csv', mode='a', header=False)

    return grid_search


# ---------------- [Random Forest] ----------------
def rf_grid(X_train, y_train, X_test=None, y_test=None):
    """Returns Optimized grid model of random forest regression"""
    param_grid = [{'n_estimators': np.arange(5, 15), 'max_depth': np.arange(1, 3),
                   'max_features': np.arange(20, 50)}]
    rf_reg = RandomForestRegressor(criterion='mse')
    grid_search = GridSearchCV(rf_reg, param_grid, cv=10, n_jobs=-1, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    if X_test is not None and y_test is not None:
        print("\n\n========== [Random Forest Grid Search Report] ==========")
        print("Best Param: ", grid_search.best_params_)
        print("========== [Random Forest Performance Report] ==========")
        y_train_pred = grid_search.predict(X_train)
        print("Training Set Error (MSE): ", mean_squared_error(y_train, y_train_pred))
        y_pred = grid_search.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print("Test Set Error     (MSE): ", mse)
        print("Test Set Error    (RMSE): ", np.sqrt(mse))

        save_report = pd.DataFrame(data=[[random_state, 'Random Forest', mse, np.sqrt(mse)]],
                                   columns=['Random State', 'Model', 'MSE', 'RMSE'])

        if not os.path.isfile(LOG_DIR + 'random_forest.csv'):
            save_report.to_csv(LOG_DIR + 'random_forest.csv', header=save_report.columns)
        else:
            save_report.to_csv(LOG_DIR + 'random_forest.csv', mode='a', header=False)

    return grid_search


# ----------------- [Extra Tree] ------------------
def et_grid(X_train, y_train, X_test=None, y_test=None):
    """Returns Optimized grid model of extra tree regression"""
    param_grid = [{'n_estimators': np.arange(5, 15), 'max_depth': np.arange(1, 3),
                   'max_features': np.arange(20, 50)}]

    et_reg = ExtraTreesRegressor(criterion='mse')
    grid_search = GridSearchCV(et_reg, param_grid, cv=10, n_jobs=-1, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    if X_test is not None and y_test is not None:
        print("\n\n========== [Extra Tree Grid Search Report] ==========")
        print("Best Param: ", grid_search.best_params_)
        print("========== [Extra Tree Performance Report] ==========")
        y_train_pred = grid_search.predict(X_train)
        print("Training Set Error (MSE): ", mean_squared_error(y_train, y_train_pred))
        y_pred = grid_search.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print("Test Set Error     (MSE): ", mse)
        print("Test Set Error    (RMSE): ", np.sqrt(mse))

        save_report = pd.DataFrame(data=[[random_state, 'ExtraTree', mse, np.sqrt(mse)]],
                                   columns=['Random State', 'Model', 'MSE', 'RMSE'])

        if not os.path.isfile(LOG_DIR + 'extra_tree.csv'):
            save_report.to_csv(LOG_DIR + 'extra_tree.csv', header=save_report.columns)
        else:
            save_report.to_csv(LOG_DIR + 'extra_tree.csv', mode='a', header=False)

    return grid_search


# ------------------- [XGBoost] -------------------
def xgb_grid(X_train, y_train, X_test=None, y_test=None):
    """Returns Optimized grid model of gradient boosting regression"""
    param_grid = [{'n_estimators': np.arange(5, 25), 'max_depth': np.arange(1, 3)}]
    xgb_reg = XGBRegressor()
    grid_search = GridSearchCV(xgb_reg, param_grid, cv=10, n_jobs=-1, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    if X_test is not None and y_test is not None:
        print("\n\n========== [Gradient Boosting Grid Search Report] ==========")
        print("Best Param: ", grid_search.best_params_)
        print("========== [Gradient Boosting Performance Report] ==========")
        y_train_pred = grid_search.predict(X_train)
        print("Training Set Error (MSE): ", mean_squared_error(y_train, y_train_pred))
        y_pred = grid_search.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print("Test Set Error     (MSE): ", mse)
        print("Test Set Error    (RMSE): ", np.sqrt(mse))

        save_report = pd.DataFrame(data=[[random_state, 'Gradient Boosting', mse, np.sqrt(mse)]],
                                   columns=['Random State', 'Model', 'MSE', 'RMSE'])

        if not os.path.isfile(LOG_DIR + 'xgb.csv'):
            save_report.to_csv(LOG_DIR + 'xgb.csv', header=save_report.columns)
        else:
            save_report.to_csv(LOG_DIR + 'xgb.csv', mode='a', header=False)

    return grid_search


# ------------------[Neural Net] ------------------
def neural_net(X_train, y_train, X_test=None, y_test=None):
    """Returns Optimized grid model of neural network"""
    model = Sequential()
    model.add(Dense(15, input_shape=(60, )))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='relu'))
    model.compile(optimizer='adam', loss=losses.mean_squared_error)

    model.fit(np.array(X_train), np.array(y_train), epochs=2000, verbose=0)

    if X_test is not None and y_test is not None:
        print("\n\n========== [Neural Network Performance Report] ==========")
        y_train_pred = model.predict(np.array(X_train))
        print("Training Set Error (MSE): ", mean_squared_error(y_train, y_train_pred))
        y_pred = model.predict(np.array(X_test))
        mse = mean_squared_error(y_test, y_pred)
        print("Test Set Error     (MSE): ", mse)
        print("Test Set Error    (RMSE): ", np.sqrt(mse))

        save_report = pd.DataFrame(data=[[random_state, 'Neural Network', mse, np.sqrt(mse)]],
                                   columns=['Random State', 'Model', 'MSE', 'RMSE'])

        if not os.path.isfile(LOG_DIR + 'neural_net.csv'):
            save_report.to_csv(LOG_DIR + 'neural_net.csv', header=save_report.columns)
        else:
            save_report.to_csv(LOG_DIR + 'neural_net.csv', mode='a', header=False)

    return model


# --------------- [Stacking] ---------------
def stacking_setup(X_train, y_train, X_heldout, y_heldout, X_test, y_test):
    """returns blended train input, blended train output, test_input, test_output"""
    print("\n\n=========== [Stacking Setup] ==========")
    ridge_stack_layer1 = ridge_grid(X_train=X_train, y_train=y_train)
    elastic_net_stack_layer1 = elastic_net_grid(X_train=X_train, y_train=y_train)
    rf_stack_layer1 = rf_grid(X_train=X_train, y_train=y_train)
    et_stack_layer1 = et_grid(X_train=X_train, y_train=y_train)
    xgb_stack_layer1 = xgb_grid(X_train=X_train, y_train=y_train)
    nn_stack_layer1 = neural_net(X_train=X_train, y_train=y_train)

    train_ridge_h_out = ridge_stack_layer1.predict(X_heldout)
    train_elastic_h_out = elastic_net_stack_layer1.predict(X_heldout)
    train_rf_h_out = rf_stack_layer1.predict(X_heldout)
    train_et_h_out = et_stack_layer1.predict(X_heldout)
    train_xgb_h_out = xgb_stack_layer1.predict(X_heldout)
    train_nn_h_out = nn_stack_layer1.predict(np.array(X_heldout))

    test_ridge_h_out = ridge_stack_layer1.predict(X_test)
    test_elastic_h_out = elastic_net_stack_layer1.predict(X_test)
    test_rf_h_out = rf_stack_layer1.predict(X_test)
    test_et_h_out = et_stack_layer1.predict(X_test)
    test_xgb_h_out = xgb_stack_layer1.predict(X_test)
    test_nn_h_out = nn_stack_layer1.predict(np.array(X_test))

    train_blender_input = pd.DataFrame(
        {'ridge': train_ridge_h_out, 'el': train_elastic_h_out, 'rf': train_rf_h_out, 'et': train_et_h_out,
         'xgb': train_xgb_h_out, 'nn': train_nn_h_out.flatten()})
    train_blender_output = y_heldout

    test_blender_input = pd.DataFrame(
        {'ridge': test_ridge_h_out, 'el': test_elastic_h_out, 'rf': test_rf_h_out, 'et': test_et_h_out,
         'xgb': test_xgb_h_out, 'nn': test_nn_h_out.flatten()})
    test_blender_output = y_test

    return train_blender_input, train_blender_output, test_blender_input, test_blender_output


def stacking_average_blender(X_train, y_train, X_test, y_test, report=True):
    """Returns y_true, y_test"""
    y_train_pred = X_train.apply(lambda x: x.mean(), axis=1)
    y_pred = X_test.apply(lambda x: x.mean(), axis=1)

    if report:
        print("\n\n========== [Stacking Report: AVG Blender] ==========")
        print("AVG Blender Stacking Training Error (MSE): ", mean_squared_error(y_train, y_train_pred))
        mse = mean_squared_error(y_test, y_pred)
        print("AVG Blender Stacking Test Error (MSE)    : ", mse)
        print("AVG Blender Stacking Test Error(RMSE)    : ", np.sqrt(mse))

        save_report = pd.DataFrame(data=[[random_state, '[Stacking] Average', mse, np.sqrt(mse)]],
                                   columns=['Random State', 'Model', 'MSE', 'RMSE'])

        if not os.path.isfile(LOG_DIR + 'STACKING_avg.csv'):
            save_report.to_csv(LOG_DIR + 'STACKING_avg.csv', header=save_report.columns)
        else:
            save_report.to_csv(LOG_DIR + 'STACKING_avg.csv', mode='a', header=False)

    return y_test, y_pred


def stacking_linear_regression_blender(X_train, y_train, X_test, y_test, report=True):
    """Returns linear regression blender model"""
    blender = LinearRegression()
    blender.fit(X_train, y_train)
    y_train_pred = blender.predict(X_train)
    y_pred = blender.predict(X_test)
    if report:
        print("\n\n========== [Stacking Report: OLS Blender] ==========")
        print("OLS Blender Stacking Training Error (MSE): ", mean_squared_error(y_train, y_train_pred))
        mse = mean_squared_error(y_test, y_pred)
        print("OLS Blender Stacking Test Error (MSE)    : ", mse)
        print("OLS Blender Stacking Test Error(RMSE)    : ", np.sqrt(mse))

        save_report = pd.DataFrame(data=[[random_state, '[Stacking] Linear Regression', mse, np.sqrt(mse)]],
                                   columns=['Random State', 'Model', 'MSE', 'RMSE'])

        if not os.path.isfile(LOG_DIR + 'STACKING_linear_regression.csv'):
            save_report.to_csv(LOG_DIR + 'STACKING_linear_regression.csv', header=save_report.columns)
        else:
            save_report.to_csv(LOG_DIR + 'STACKING_linear_regression.csv', mode='a', header=False)

    return y_test, y_pred


def stacking_neural_net_blender(X_train, y_train, X_test, y_test, report=True):
    """returns neural network blender model"""
    blender = Sequential()
    blender.add(Dense(4, input_shape=(6, )))
    blender.add(BatchNormalization())
    blender.add(Dense(1, activation='relu'))
    blender.compile(optimizer='adam', loss=losses.mean_squared_error)
    blender.fit(np.array(X_train), np.array(y_train), epochs=3000, verbose=0)

    y_train_pred = blender.predict(np.array(X_train))
    y_pred = blender.predict(np.array(X_test))

    if report:
        print("\n\n========== [Stacking Report: Neural Net Blender] ==========")
        print("Neural Net Blender Stacking Training Error (MSE): ", mean_squared_error(y_train, y_train_pred))
        mse = mean_squared_error(y_test, y_pred)
        print("Neural Net Blender Stacking Test Error (MSE)    : ", mse)
        print("Neural Net Blender Stacking Test Error(RMSE)    : ", np.sqrt(mse))

        save_report = pd.DataFrame(data=[[random_state, '[Stacking] Neural Network', mse, np.sqrt(mse)]],
                                   columns=['Random State', 'Model', 'MSE', 'RMSE'])

        if not os.path.isfile(LOG_DIR + 'STACKING_neural_network.csv'):
            save_report.to_csv(LOG_DIR + 'STACKING_neural_network.csv', header=save_report.columns)
        else:
            save_report.to_csv(LOG_DIR + 'STACKING_neural_network.csv', mode='a', header=False)

    return y_test, y_pred


# -------------------------- [Plot Performance Figure] --------------------------
def plot_performance(test_blender_input, test_blender_output, stk_avg_y_pred, stk_lr_y_pred, stk_nn_y_pred):

    f, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    ax[0].plot(test_blender_input.index, test_blender_output, label="y_true")
    ax[0].plot(test_blender_input.index, stk_avg_y_pred, label="y_pred")
    ax[0].legend()
    ax[0].set_ylabel('Normalized Loss')
    ax[0].set_title('Average Blender')

    ax[1].plot(test_blender_input.index, test_blender_output, label="y_true")
    ax[1].plot(test_blender_input.index, stk_lr_y_pred, label="y_pred")
    ax[1].legend()
    ax[1].set_ylabel('Normalized Loss')
    ax[1].set_title("Linear Regression Blender")

    ax[2].plot(test_blender_input.index, test_blender_output, label="y_true")
    ax[2].plot(test_blender_input.index, stk_nn_y_pred, label="y_pred")
    ax[2].legend()
    ax[2].set_xlabel('Test data Index')
    ax[2].set_ylabel('Normalized Loss')
    ax[2].set_title("Neural Network Blender")

    plt.savefig(FIGURE_DIR + str(random_state) + '_Performance_Figure.png')
