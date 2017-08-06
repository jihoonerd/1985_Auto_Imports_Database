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


# -------------------- [Ridge] --------------------
def ridge_grid(X_train, y_train, X_test=None, y_test=None):

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
        print("Test Set Error     (MSE): ", mean_squared_error(y_test, y_pred))
    return grid_search


# ----------------- [Elastic Net] -----------------
def elastic_net_grid(X_train, y_train, X_test=None, y_test=None):

    param_grid = [{'alpha': np.logspace(-2, 2, 50), 'l1_ratio': np.linspace(0.1, 0.98, 40)}]
    elastic_net_reg = ElasticNet(tol=1e-2)
    grid_search = GridSearchCV(elastic_net_reg, param_grid, cv=10, n_jobs=-1, verbose=0, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    if X_test is not None and y_test is not None:
        print("\n\n========== [Elastic Net Regression Grid Search Report] ==========")
        print("Best Param: ", grid_search.best_params_)
        print("========== [Elastic Net Regression Performance Report] ==========")
        y_train_pred = grid_search.predict(X_train)
        print("Training Set Error (MSE): ", mean_squared_error(y_train, y_train_pred))
        y_pred = grid_search.predict(X_test)
        print("Test Set Error     (MSE): ", mean_squared_error(y_test, y_pred))
    return grid_search


# ---------------- [Random Forest] ----------------
def rf_grid(X_train, y_train, X_test=None, y_test=None):

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
        print("Test Set Error     (MSE): ", mean_squared_error(y_test, y_pred))
    return grid_search


# ----------------- [Extra Tree] ------------------
def et_grid(X_train, y_train, X_test=None, y_test=None):

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
        print("Test Set Error     (MSE): ", mean_squared_error(y_test, y_pred))
    return grid_search


# ------------------- [XGBoost] -------------------
def xgb_grid(X_train, y_train, X_test=None, y_test=None):

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
        print("Test Set Error     (MSE): ", mean_squared_error(y_test, y_pred))
    return grid_search


# ------------------[Neural Net] ------------------
def neural_net(X_train, y_train, X_test=None, y_test=None):

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
        print("Test Set Error     (MSE): ", mean_squared_error(y_test, y_pred))
    return model


# --------------- [Stacking] ---------------
def stacking_setup(X_train, y_train, X_heldout, y_heldout, X_test, y_test):

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


def stacking_average_blender_predictor(X_train, y_train, X_test, y_test, report=True):
    y_train_pred = X_train.apply(lambda x: x.mean(), axis=1)
    y_pred = X_test.apply(lambda x: x.mean(), axis=1)

    if report:
        print("\n\n========== [Stacking Report: AVG Blender] ==========")
        print("AVG Blender Stacking Training Error (MSE): ", mean_squared_error(y_train, y_train_pred))
        print("AVG Blender Stacking Test Error (MSE)    : ", mean_squared_error(y_test, y_pred))

    return y_pred, y_test


def stacking_linear_regression_blender(X_train, y_train, X_test, y_test, report=True):
    blender = LinearRegression()
    blender.fit(X_train, y_train)
    y_train_pred = blender.predict(X_train)
    y_pred = blender.predict(X_test)
    if report:
        print("\n\n========== [Stacking Report: OLS Blender] ==========")
        print("OLS Blender Stacking Training Error (MSE): ", mean_squared_error(y_train, y_train_pred))
        print("OLS Blender Stacking Test Error (MSE)    : ", mean_squared_error(y_test, y_pred))

    return y_pred, y_test


def stacking_neural_net_blender(X_train, y_train, X_test, y_test, report=True):

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
        print("Neural Net Blender Stacking Test Error (MSE)    : ", mean_squared_error(y_test, y_pred))

    return y_pred, y_test