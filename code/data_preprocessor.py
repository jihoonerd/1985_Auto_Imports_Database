# -*- coding: utf-8 -*-
"""
Created on 8/5/17
Author: Jihoon Kim
"""

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