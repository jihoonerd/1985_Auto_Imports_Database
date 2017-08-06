# -*- coding: utf-8 -*-
"""
Created on 8/6/17
Author: Jihoon Kim
"""

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


# -------------------- [Ridge] --------------------
def ridge_grid(X_train, X_test, y_train, y_test, report=True):

    param_grid = [{'alpha': np.logspace(-2, 2, 50)}]
    ridge_reg = Ridge()
    grid_search = GridSearchCV(ridge_reg, param_grid, cv=10, n_jobs=-1, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    if report:
        print("\n\n========== [Ridge Regression Grid Search Report] ==========")
        print("Best Param: ", grid_search.best_params_)
        print("========== [Ridge Regression Performance Report] ==========")
        y_train_pred = grid_search.predict(X_train)
        print("Training Set Error (MSE): ", mean_squared_error(y_train, y_train_pred))
        y_pred = grid_search.predict(X_test)
        print("Test Set Error     (MSE): ", mean_squared_error(y_test, y_pred))
    return grid_search


# ----------------- [Elastic Net] -----------------
def elastic_net_grid(X_train, X_test, y_train, y_test, report=True):

    param_grid = [{'alpha': np.logspace(-2, 2, 50), 'l1_ratio': np.linspace(0.1, 0.98, 40)}]
    elastic_net_reg = ElasticNet(tol=1e-2)
    grid_search = GridSearchCV(elastic_net_reg, param_grid, cv=10, n_jobs=-1, verbose=0, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    if report:
        print("\n\n========== [Elastic Net Regression Grid Search Report] ==========")
        print("Best Param: ", grid_search.best_params_)
        print("========== [Elastic Net Regression Performance Report] ==========")
        y_train_pred = grid_search.predict(X_train)
        print("Training Set Error (MSE): ", mean_squared_error(y_train, y_train_pred))
        y_pred = grid_search.predict(X_test)
        print("Test Set Error     (MSE): ", mean_squared_error(y_test, y_pred))
    return grid_search


# ---------------- [Random Forest] ----------------
def rf_grid(X_train, X_test, y_train, y_test, report=True):

    param_grid = [{'n_estimators': np.arange(5, 15), 'max_depth': np.arange(1, 3),
                   'max_features': np.arange(20, 50)}]
    rf_reg = RandomForestRegressor(criterion='mse')
    grid_search = GridSearchCV(rf_reg, param_grid, cv=10, n_jobs=-1, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    if report:
        print("\n\n========== [Random Forest Grid Search Report] ==========")
        print("Best Param: ", grid_search.best_params_)
        print("========== [Random Forest Performance Report] ==========")
        y_train_pred = grid_search.predict(X_train)
        print("Training Set Error (MSE): ", mean_squared_error(y_train, y_train_pred))
        y_pred = grid_search.predict(X_test)
        print("Test Set Error     (MSE): ", mean_squared_error(y_test, y_pred))
    return grid_search


# ----------------- [Extra Tree] ------------------
def et_grid(X_train, X_test, y_train, y_test, report=True):

    param_grid = [{'n_estimators': np.arange(5, 15), 'max_depth': np.arange(1, 3),
                   'max_features': np.arange(20, 50)}]

    et_reg = ExtraTreesRegressor(criterion='mse')
    grid_search = GridSearchCV(et_reg, param_grid, cv=10, n_jobs=-1, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    if report:
        print("\n\n========== [Extra Tree Grid Search Report] ==========")
        print("Best Param: ", grid_search.best_params_)
        print("========== [Extra Tree Performance Report] ==========")
        y_train_pred = grid_search.predict(X_train)
        print("Training Set Error (MSE): ", mean_squared_error(y_train, y_train_pred))
        y_pred = grid_search.predict(X_test)
        print("Test Set Error     (MSE): ", mean_squared_error(y_test, y_pred))
    return grid_search


# ------------------- [XGBoost] -------------------
def xgb_grid(X_train, X_test, y_train, y_test, report=True):

    param_grid = [{'n_estimators': np.arange(5, 25), 'max_depth': np.arange(1, 3)}]
    xgb_reg = XGBRegressor()
    grid_search = GridSearchCV(xgb_reg, param_grid, cv=10, n_jobs=-1, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    if report:
        print("\n\n========== [Gradient Boosting Grid Search Report] ==========")
        print("Best Param: ", grid_search.best_params_)
        print("========== [Gradient Boosting Performance Report] ==========")
        y_train_pred = grid_search.predict(X_train)
        print("Training Set Error (MSE): ", mean_squared_error(y_train, y_train_pred))
        y_pred = grid_search.predict(X_test)
        print("Test Set Error     (MSE): ", mean_squared_error(y_test, y_pred))
    return grid_search


# ------------------[Neural Net] ------------------
def neural_net(X_train, X_test, y_train, y_test, report=True):

    model = Sequential()
    model.add(Dense(15, input_shape=(60, )))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='relu'))
    model.compile(optimizer='adam', loss=losses.mean_squared_error)

    model.fit(np.array(X_train), np.array(y_train), epochs=2200, verbose=0)

    if report:
        print("\n\n========== [Neural Network Performance Report] ==========")
        y_train_pred = model.predict(np.array(X_train))
        print("Training Set Error (MSE): ", mean_squared_error(y_train, y_train_pred))
        y_pred = model.predict(np.array(X_test))
        print("Test Set Error     (MSE): ", mean_squared_error(y_test, y_pred))
    return model


