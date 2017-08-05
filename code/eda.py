# -*- coding: utf-8 -*-
"""
Created on 8/5/17
Author: Jihoon Kim
"""

import matplotlib.pyplot as plt
import seaborn as sns


def display_basic_info(data):
    print("Number of Instances: ", data.shape[0])
    print("Number of Features : ", data.shape[1])
    print(data.dtypes)
    return None


def display_descriptive_statistics(data):
    print(data.describe())
    return None


def plot_target_feature(data):
    plt.figure()
    sns.distplot(pd.to_numeric(data.normalized_losses), rug=True, kde=True)
    plt.title("Normalized Losses")
    plt.show()
    return None


def plot_makers(data):
    plt.figure()
    plt.title("Makers")
    sns.countplot(x="make", data=data)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    return None


def plot_fuel_types(data):
    plt.figure()
    plt.title("Fuel Types")
    sns.countplot(x="fuel_type", data=data)
    plt.tight_layout()
    plt.show()
    return None


def plot_aspiration(data):
    plt.figure()
    plt.title("Aspiration")
    sns.countplot(x="aspiration", data=data)
    plt.tight_layout()
    plt.show()
    return None


def plot_num_of_doors(data):
    plt.figure()
    plt.title("Num of Doors")
    sns.countplot(x="num_of_doors", data=data)
    plt.tight_layout()
    plt.show()
    return None


def plot_body_style(data):
    plt.figure()
    plt.title("Body Style")
    sns.countplot(x="body_style", data=data)
    plt.tight_layout()
    plt.show()
    return None


def plot_drive_wheels(data):
    plt.figure()
    plt.title("Drive Wheels")
    sns.countplot(x="drive_wheels", data=data)
    plt.tight_layout()
    plt.show()
    return None


def plot_engine_location(data):
    plt.figure()
    plt.title("Engine Location")
    sns.countplot(x="engine_location", data=data)
    plt.tight_layout()
    plt.show()
    return None


def plot_wheel_base(data):
    plt.figure()
    sns.distplot(data.wheel_base, rug=True, kde=True)
    plt.title("Wheel Base")
    plt.show()
    return None


def plot_sizes(data):
    f, (ax1, ax2, ax3) = plt.subplots(3, 1)
    plt.suptitle("Size")
    sns.distplot(data.length, rug=True, kde=True, ax=ax1)
    sns.distplot(data.width, rug=True, kde=True, color='g', ax=ax2)
    sns.distplot(data.height, rug=True, kde=True, color='m', ax=ax3)
    plt.tight_layout()
    plt.show()
    return None


def plot_curb_weight(data):
    plt.figure()
    sns.distplot(data.curb_weight, rug=True, kde=True)
    plt.title("Curb Weight")
    plt.show()
    return None


def plot_engine_properties(data):
    f, ax = plt.subplots(3, 3, figsize=(12,10))
    plt.suptitle("Engine Properties")
    sns.countplot(x="engine_type", data=data, ax=ax[0, 0])
    sns.countplot(x="num_of_cylinders", data=data, ax=ax[0, 1])
    sns.distplot(data.engine_size, rug=True, kde=True, ax=ax[0, 2])
    sns.countplot(x="fuel_system", data=data, ax=ax[1, 0])
    sns.countplot(x="bore", data=data, ax=ax[1, 1])
    plt.setp(ax[1, 1].get_xticklabels(), rotation=90, fontsize=8)
    sns.countplot(x="stroke", data=data, ax=ax[1, 2])
    plt.setp(ax[1, 2].get_xticklabels(), rotation=90, fontsize=8)
    sns.distplot(data.compression_ratio, rug=True, kde=True, ax=ax[2, 0])
    sns.countplot(x="horsepower", data=data, ax=ax[2, 1])
    plt.setp(ax[2, 1].get_xticklabels(), rotation=90, fontsize=8)
    sns.countplot(x="peak_rpm", data=data, ax=ax[2, 2])
    plt.setp(ax[2, 2].get_xticklabels(), rotation=90, fontsize=8)
    plt.tight_layout()
    plt.show()
    return None


def plot_mpg_properties(data):
    f, ax = plt.subplots(2, 1)
    plt.suptitle("MPG")
    sns.distplot(data.city_mpg, rug=True, kde=True, ax=ax[0])
    sns.distplot(data.highway_mpg, rug=True, kde=True, ax=ax[1])
    plt.tight_layout()
    plt.show()
    return None


def plot_price(data):
    plt.figure()
    sns.distplot(pd.to_numeric(data.price), rug=True, kde=True)
    plt.title("Price")
    plt.show()
    return None
