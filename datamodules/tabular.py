from pathlib import Path

import numpy as np
import pandas as pd

from pytorch_lightning import LightningDataModule

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler


class MagicTelescope(LightningDataModule):
    name = 'magic'

    def __init__(self, seed=42, test_fraction=0.3, **kwargs):
        super().__init__()
        self.seed = seed
        self.test_fraction = test_fraction

    def prepare_data(self):
        fetch_openml(name='MagicTelescope')

    def setup(self, stage=None):
        dataset = fetch_openml(name='MagicTelescope')
        X = dataset.data
        y = LabelBinarizer().fit_transform(dataset.target).squeeze(-1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_fraction,
                                                            random_state=self.seed)
        ss = StandardScaler().fit(X_train)
        X_train, X_test = ss.transform(X_train), ss.transform(X_test)
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test


class Higgs(MagicTelescope):
    name = 'higgs'

    def prepare_data(self):
        fetch_openml(name='Higgs', version=2)

    def setup(self, stage=None):
        dataset = fetch_openml(name='Higgs', version=2)
        X = SimpleImputer().fit_transform(dataset.data)
        y = LabelBinarizer().fit_transform(dataset.target).squeeze(-1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_fraction,
                                                            random_state=self.seed)
        ss = StandardScaler().fit(X_train)
        X_train, X_test = ss.transform(X_train), ss.transform(X_test)
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test


class Adult(MagicTelescope):
    name = 'adult'

    def prepare_data(self):
        fetch_openml(name='Adult', version=2)

    def setup(self, stage=None):
        dataset = fetch_openml(name='Adult', version=2)
        X = SimpleImputer().fit_transform(dataset.data)
        y = LabelBinarizer().fit_transform(dataset.target).squeeze(-1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_fraction,
                                                            random_state=self.seed)
        ss = StandardScaler().fit(X_train)
        X_train, X_test = ss.transform(X_train), ss.transform(X_test)
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test


class FICO(MagicTelescope):
    name = 'fico'

    def __init__(self, data_dir, seed=42, test_fraction=0.3):
        super().__init__(seed=seed, test_fraction=test_fraction)
        self.data_dir = data_dir
        self.file_path = Path(data_dir) / 'heloc_dataset_v1.csv'

    def prepare_data(self):
        assert self.file_path.exists()

    @staticmethod
    def preprocess(df):
        # Details and preprocessing for FICO dataset

        # minimize dependence on ordering of columns in heloc data
        # x_cols, y_col = df.columns[0:-1], df.columns[-1]
        x_cols = list(df.columns.values)
        x_cols.remove('RiskPerformance')
        y_col = 'RiskPerformance'

        # Preprocessing the HELOC dataset
        # Remove all the rows containing -9 in the ExternalRiskEstimate column
        # df = df[df.ExternalRiskEstimate != -9]
        # add columns for -7 and -8 in the dataset
        for col in x_cols:
            # df[col][df[col].isin([-7, -8, -9])] = 0
            df.loc[df[col].isin([-7, -8, -9]), col] = -9
        # Get the column names for the covariates and the dependent variable
        # df = df[(df[x_cols].T != 0).any()]
        df = df[(df[x_cols].T != -9).any()]

        # minimize dependence on ordering of columns in heloc data
        # x = df.values[:, 0:-1]
        X = df[x_cols].values

        # encode target variable ('bad', 'good')
        cat_values = df[y_col].values
        enc = LabelEncoder()
        enc.fit(cat_values)
        y = enc.transform(cat_values)

        return X, y

    def setup(self, stage=None):
        df = pd.read_csv(self.file_path)
        X, y = self.preprocess(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_fraction,
                                                            random_state=self.seed)
        ss = StandardScaler().fit(X_train)
        X_train, X_test = ss.transform(X_train), ss.transform(X_test)
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test


class StumbleUpon(MagicTelescope):
    name = 'stumbleupon'

    def __init__(self, data_dir):
        super().__init__(seed=42, test_fraction=0.3)
        self.data_dir = data_dir
        self.file_path = Path(data_dir) / 'evergreen.npz'

    def prepare_data(self):
        assert self.file_path.exists()

    def setup(self, stage=None):
        data = np.load(self.file_path)
        X_train, X_test, y_train, y_test = (data['X_train'], data['X_test'], data['y_train'],
                                            data['y_test'])
        # ss = StandardScaler().fit(X_train)
        # X_train, X_test = ss.transform(X_train), ss.transform(X_test)
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test
