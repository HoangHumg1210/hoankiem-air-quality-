import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import math

from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras.models import Sequential

TARGET = 'PM25'
LOOKBACK = 336
HORIZON = 8
USE_LOG_TARGET = True
DATA_PATH = 'data/processed/data2225_done.csv'

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

def load_and_clean_data(path):
    df = pd.read_csv(path)
    
    df['Local Time'] = pd.to_datetime(df['Local Time'])
    df = df.set_index('Local Time').sort_index()
    
    df = df[~df.index.duplicated(keep="last")]
    
    df = df.asfreq("1h")
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].interpolate(method="time").ffill().bfill()
    if "IsHoliday" in df.columns:
        df["IsHoliday"] = df["IsHoliday"].ffill().bfill().astype(int)
        
    return df

def add_time_features(df):
    idx = df.index
    df["day_of_week"] = idx.dayofweek
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    df["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * idx.month / 12)
    return df


def add_target_features(df, target=TARGET):
    for lag in [1, 3, 6, 12, 24, 48]:
        df[f"{target}_lag{lag}"] = df[target].shift(lag)

    df[f"{target}_roll_mean_24"] = df[target].rolling(24).mean()
    df[f"{target}_roll_std_24"] = df[target].rolling(24).std()
    df[f"{target}_roll_mean_72"] = df[target].rolling(72).mean()

    return df.dropna()


def split_data(df):
    train_df = df[: "2023-12-31"]
    val_df = df["2024-01-01":"2024-12-31"]
    test_df = df["2025-01-01":]
    return train_df, val_df, test_df


def transform_target(train_df, val_df, test_df, target=TARGET, use_log=USE_LOG_TARGET):
    def forward(y):
        y = np.asarray(y, dtype=np.float64)
        return np.log1p(np.clip(y, 0, None)) if use_log else y

    def inverse(y):
        y = np.asarray(y, dtype=np.float64)
        return np.expm1(y) if use_log else y

    y_train_raw = train_df[[target]].values
    y_val_raw = val_df[[target]].values
    y_test_raw = test_df[[target]].values

    y_train_t = forward(y_train_raw)
    y_val_t = forward(y_val_raw)
    y_test_t = forward(y_test_raw)

    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train_t)
    y_val = scaler_y.transform(y_val_t)
    y_test = scaler_y.transform(y_test_t)

    return y_train, y_val, y_test, scaler_y, inverse

def preprocess_features(train_df, val_df, test_df, target=TARGET):
    X_train_df = train_df.drop(columns=[target])
    X_val_df = val_df.drop(columns=[target])
    X_test_df = test_df.drop(columns=[target])

    num_cols = X_train_df.select_dtypes(include=[np.number, "bool"]).columns
    cat_cols = X_train_df.select_dtypes(include=["object"]).columns

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    X_train = preprocess.fit_transform(X_train_df)
    X_val = preprocess.transform(X_val_df)
    X_test = preprocess.transform(X_test_df)

    return X_train, X_val, X_test, preprocess


def create_sequences(X, y, lookback=LOOKBACK, horizon=HORIZON):
    y = np.asarray(y)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    X_seq, y_seq = [], []
    for i in range(lookback, len(X) - horizon + 1):
        X_seq.append(X[i - lookback:i])
        y_seq.append(y[i + horizon - 1])

    return np.array(X_seq), np.array(y_seq)

def main():
    set_seed()

    df = load_and_clean_data(DATA_PATH)
    df = add_time_features(df)
    df = add_target_features(df)

    train_df, val_df, test_df = split_data(df)

    y_train, y_val, y_test, scaler_y, inverse_target_transform = transform_target(
        train_df, val_df, test_df
    )
    X_train, X_val, X_test, _ = preprocess_features(train_df, val_df, test_df)

    X_train_seq, y_train_seq = create_sequences(X_train, y_train)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test)

    n_features = X_train_seq.shape[2]


if __name__ == "__main__":
    main()