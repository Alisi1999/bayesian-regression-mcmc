"""California Housing preprocessing and OLS baseline."""
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm


def load_and_prepare_data(test_size=0.2, random_state=42):
    frame = fetch_california_housing(as_frame=True).frame
    feature_names = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population"]
    X_raw = frame[feature_names].values
    y = frame["MedHouseVal"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = np.column_stack([np.ones(len(X_train)), X_train])
    X_test = np.column_stack([np.ones(len(X_test)), X_test])
    return X_train, X_test, y_train, y_test, ["Intercept"] + feature_names


def fit_ols_baseline(X_train, y_train):
    return sm.OLS(y_train, X_train).fit()
