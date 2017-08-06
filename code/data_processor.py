# -*- coding: utf-8 -*-
"""
Created on 8/5/17
Author: Jihoon Kim
"""

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import Imputer, StandardScaler, LabelBinarizer
import pandas as pd

num_features = ['num_of_doors', 'wheel_base', 'length', 'width', 'height',
                'curb_weight', 'num_of_cylinders', 'engine_size', 'bore', 'stroke', 'compression_ratio',
                'horsepower', 'peak_rpm', 'city_mpg', 'highway_mpg', 'price']
cat_features = ['make', 'fuel_type', 'aspiration', 'body_style', 'drive_wheels', 'engine_location',
               'engine_type', 'fuel_system']


class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.loc[:, self.attribute_names].values


class ConvertWordToNum(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        mapping_num_of_doors = pd.Series([2, 4], index=['two', 'four'])
        mapping_num_of_cylinders = pd.Series([2, 3, 4, 5, 6, 8, 12], index=['two', 'three', 'four', 'five', 'six',
                                                                            'eight', 'twelve'])
        X.loc[:, 'num_of_doors'] = X.loc[:, 'num_of_doors'].map(mapping_num_of_doors)
        X.loc[:, 'num_of_cylinders'] = X.loc[:, 'num_of_cylinders'].map(mapping_num_of_cylinders)

        return X


class DtypeConverter(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        for col in X.columns:
            if col in num_features:
                X.loc[:, col] = pd.to_numeric(X.loc[:, col])

        return X


class Categorizer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()

        for col in X.columns:
            X.loc[:, col] = X.loc[:, col].astype('category')

        return X


num_pipeline = Pipeline([
    ('WordToNum', ConvertWordToNum()),
    ('DtypeCV', DtypeConverter()),
    ('selector', DataFrameSelector(num_features)),
    ('Imputer', Imputer(strategy="median")),
    ('StdScaler', StandardScaler()),
])


def full_pipeline_encoder(X, X_train, X_test, y_train, y_test):
    X_train_num = pd.DataFrame(num_pipeline.fit_transform(X_train), index=X_train.index,
                               columns=X_train[num_features].columns)
    X_test_num = pd.DataFrame(num_pipeline.transform(X_test), index=X_test.index,
                              columns=X_test[num_features].columns)

    X_OHE = pd.get_dummies(X[cat_features])
    X_train_ohe = X_OHE.loc[X_train.index]
    X_test_ohe = X_OHE.loc[X_test.index]

    X_train = pd.concat([X_train_num, X_train_ohe], axis=1)
    X_test = pd.concat([X_test_num, X_test_ohe], axis=1)

    return X_train, X_test, y_train, y_test



