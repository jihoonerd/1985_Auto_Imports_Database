# -*- coding: utf-8 -*-
"""
Created on 8/5/17
Author: Jihoon Kim
"""

import numpy as np
import pandas as pd
import config

features = ['symboling',
            'normalized_losses',
            'make',
            'fuel_type',
            'aspiration',
            'num_of_doors',
            'body_style',
            'drive_wheels',
            'engine_location',
            'wheel_base',
            'length',
            'width',
            'height',
            'curb_weight',
            'engine_type',
            'num_of_cylinders',
            'engine_size',
            'fuel_system',
            'bore',
            'stroke',
            'compression_ratio',
            'horsepower',
            'peak_rpm',
            'city_mpg',
            'highway_mpg',
            'price']


def load_data():
    data = pd.read_csv(config.DATA_DIR+'imports-85.data', names=features)
    return data


def split_X_y(data):
    y = data["normalized_losses"]
    X = data.drop("normalized_losses", axis=1)
    return X, y


def apply_task_condition(data):
    """
    Missing values: denoted by quotation marks (‘?’). Skip data samples with missing values in the target.
    Features to ignore: ‘symboling’
    """
    not_null = data[data.normalized_losses.notnull()]
    conditioned = not_null.drop("symboling", axis=1)
    return conditioned


def transform_null_values(data):
    return data.replace('?', np.NaN)
