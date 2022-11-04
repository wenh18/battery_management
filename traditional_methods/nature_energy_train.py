import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import numpy as np


def get_in_outputs(features_df, regression_type="discharge", model="regression"):
    # only three versions are allowed.
    assert regression_type in ["variance", "discharge"]

    # dictionary to hold the features indices for each model version.
    features = {
        # "full": [1, 2, 5, 6, 7, 9, 10, 11],
        "variance": [2],
        "discharge": [1, 2, 3, 4, 7, 8]
    }
    # get the features for the model version (full, variance, discharge)
    feature_indices = features[regression_type]
    # get all cells with the specified features
    model_features = features_df.iloc[:, feature_indices]
    # get last two columns (cycle life and classification)
    labels = features_df.iloc[:, -2:]
    # labels are (cycle life ) for regression other wise (0/1) for classsification
    labels = labels.iloc[:, 0] if model == "regression" else labels.iloc[:, 1]

    # # split data in to train/primary_test/and secondary test
    # train_cells = np.arange(1, 83, 2)
    # val_cells = np.arange(0, 84, 2)
    # test_cells = np.arange(84, 124, 1)

    # get cells and their features of each set and convert to numpy for further computations
    # x_train = np.array(model_features.iloc[train_cells])
    # x_val = np.array(model_features.iloc[val_cells])
    # x_test = np.array(model_features.iloc[test_cells])
    x = np.array(model_features)
    y = np.array(labels)

    return x, y


def split_train_val():
    features_train = pd.read_csv("./train.csv")
    features_test = pd.read_csv("./test.csv")
    train_x, train_y = get_in_outputs(features_train, regression_type="discharge", model="regression")
    test_x, test_y = get_in_outputs(features_test, regression_type="discharge", model="regression")
    return {"train": (train_x, train_y), "val": (test_x, test_y), "test": (test_x, test_y)}



def regression(datasets, pca=False, normalize=True, log_target=True,
               n_components=5, alpha=1e-4, l1_ratio=0.99, model="elastic"):
    # get three sets
    x_train, y_train = datasets.get("train")
    x_val, y_val = datasets.get("val")
    x_test, y_test = datasets.get("test")

    if pca:
        # create pca object with n_components (n_components is the reduced feature dimension)
        _pca = PCA(n_components=n_components)
        _pca.fit(x_train)
        # transfrom data to reduced features
        x_train = _pca.transform(x_train)
        x_val = _pca.transform(x_val)
        x_test = _pca.transform(x_test)
    if normalize:
        # create normalization object to normalize data
        scaler = StandardScaler()
        # find normalization params from train set, (mean, std)
        scaler.fit(x_train)
        # normalize sets (0 mean, 1 std)
        x_train = scaler.transform(x_train)
        x_val = scaler.transform(x_val)
        x_test = scaler.transform(x_test)

    if model == "elastic":
        print("Elastic net")
        regr = ElasticNet(random_state=4, alpha=alpha, l1_ratio=l1_ratio)
    else:
        print("Lasso net")
        regr = Lasso(alpha=alpha)
    # labels/ targets might be converted to log version based on choice
    targets = np.log(y_train) if log_target else y_train
    # fit regression model
    regr.fit(x_train, targets)

    # predict values/cycle life for all three sets
    pred_train = regr.predict(x_train)
    pred_val = regr.predict(x_val)
    pred_test = regr.predict(x_test)

    if log_target:
        # scale up the preedictions
        pred_train, pred_val, pred_test = np.exp(pred_train), np.exp(pred_val), np.exp(pred_test)

    # mean percentage error (same as paper)
    error_train = mean_absolute_percentage_error(y_train, pred_train) * 100
    error_val = mean_absolute_percentage_error(y_val, pred_val) * 100
    error_test = mean_absolute_percentage_error(y_test, pred_test) * 100

    print(f"Regression Error batch 3 (test (secondary)):  {error_test}%")
    print(f"Regression Error (Train):, {error_train}%")
    print(f"Regression Error (validation (primary) test): {error_val}%")


print("Discharge")
features = split_train_val()
import pdb;pdb.set_trace()
regression(features, pca=False, normalize=False, n_components=3, alpha=0.0005,
                 l1_ratio=1, log_target=False, model="elastic")


print("\nVariance")
features = split_train_val()
regression(features, pca=False, normalize=False, n_components=3, alpha=0.001,
                 l1_ratio=1, log_target=True, model="elastic")
