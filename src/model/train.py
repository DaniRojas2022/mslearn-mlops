# Import libraries
import argparse
import glob
import os
import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# define functions
def main(args):
    # TO DO: enable autologging
    mlflow.autolog()
    experiment_name = "MlOpsLearn"
    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(name=experiment_name)
    mlflow.set_tracking_uri('http://52.251.52.55:8080/')
    df = get_csvs_df(args.training_data)
    X_train, X_test, y_train, y_test = split_data(df)
    model = train_model(args.reg_rate, X_train, X_test, y_train, y_test)
    (acc, auc) = eval_metrics(model, X_test, y_test)


def eval_metrics(model, X_test, y_test):
    y_hat = model.predict(X_test)
    acc = np.average(y_hat == y_test)
    y_scores = model.predict_proba(X_test)
    auc = roc_auc_score(y_test, y_scores[:, 1])
    return acc, auc


def get_csvs_df(path):
    if not os.path.exists(path):
        raise RuntimeError(f"Cannot use non-existent path provided: {path}")
    csv_files = glob.glob(f"{path}/*.csv")
    if not csv_files:
        raise RuntimeError(f"No CSV files found in provided data path: {path}")
    return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)


def split_data(df):
    X, y = df[[
        'Pregnancies', 'PlasmaGlucose', 'DiastolicBloodPressure',
        'TricepsThickness', 'SerumInsulin', 'BMI', 'DiabetesPedigree',
        'Age']].values, df['Diabetic'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=0
        )
    return X_train, X_test, y_train, y_test


def train_model(
        reg_rate,
        X_train,
        X_test,
        y_train,
        y_test
        ):
    return LogisticRegression(C=1/reg_rate, solver="liblinear").fit(
        X_train,
        y_train
        )


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training_data", dest='training_data',
                        type=str)
    parser.add_argument("--reg_rate", dest='reg_rate',
                        type=float, default=0.01)

    # parse args
    args = parser.parse_args()

    # return args
    return args


if __name__ == "__main__":
    # parse args
    args = parse_args()
    # run main function
    main(args)
