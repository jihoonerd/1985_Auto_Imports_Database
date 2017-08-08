# -*- coding: utf-8 -*-
"""
Created on 8/4/17
Author: Jihoon Kim
"""


from data_loader import load_data, apply_task_condition, split_X_y
from sklearn.model_selection import train_test_split
from data_processor import full_pipeline_encoder
from modeling import ridge_grid, elastic_net_grid, rf_grid, et_grid, xgb_grid, neural_net, stacking_setup,\
                     stacking_average_blender, stacking_linear_regression_blender, stacking_neural_net_blender,\
                     plot_performance
from eda import *
from config import random_state

# -------------------- [Data Loading] --------------------
data = apply_task_condition(load_data())
X, y = split_X_y(data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

# -------------- [Exploratory Data Analysis] -------------
display_basic_info(data)
display_descriptive_statistics(data)
# plot_target_feature(data)

# ------------------ [Data Preparation] ------------------
X_train_stdby, X_test_stdby, y_train_stdby, y_test_stdby = full_pipeline_encoder(X, X_train, X_test, y_train, y_test)

# ---------------------- [Modeling] ----------------------
ridge_model = ridge_grid(X_train_stdby, y_train_stdby, X_test_stdby, y_test_stdby)
elastic_net_model = elastic_net_grid(X_train_stdby, y_train_stdby, X_test_stdby, y_test_stdby)
rf_model = rf_grid(X_train_stdby, y_train_stdby, X_test_stdby, y_test_stdby)
et_model = et_grid(X_train_stdby, y_train_stdby, X_test_stdby, y_test_stdby)
xgb_model = xgb_grid(X_train_stdby, y_train_stdby, X_test_stdby, y_test_stdby)
nn_model = neural_net(X_train_stdby, y_train_stdby, X_test_stdby, y_test_stdby)

# ------------------ [Stacking] ------------------

X_train_stacking, X_train_heldout, y_train_stacking, y_train_heldout = train_test_split(X_train_stdby, y_train_stdby,
                                                                                        test_size=0.3,
                                                                                        random_state=random_state)

train_blender_input, train_blender_output, test_blender_input, test_blender_output = stacking_setup(X_train_stacking,
                                                                                                    y_train_stacking,
                                                                                                    X_train_heldout,
                                                                                                    y_train_heldout,
                                                                                                    X_test_stdby,
                                                                                                    y_test_stdby)


stk_avg_y, stk_avg_y_pred = stacking_average_blender(train_blender_input, train_blender_output,
                                                     test_blender_input, test_blender_output, report=True)

stk_lr_y, stk_lr_y_pred = stacking_linear_regression_blender(train_blender_input, train_blender_output,
                                                             test_blender_input, test_blender_output, report=True)

stk_nn_y, stk_nn_y_pred = stacking_neural_net_blender(train_blender_input, train_blender_output,
                                                      test_blender_input, test_blender_output, report=True)

# ------------------- [Performance Plot] ------------------------
plot_performance(test_blender_input, test_blender_output, stk_avg_y_pred, stk_lr_y_pred, stk_nn_y_pred)