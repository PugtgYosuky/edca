from sklearn.base import BaseEstimator
import numpy as np
from edca.model import create_preprocessing_pipeline, instantiate_model, instantiate_data_augmentation
from sklearn.preprocessing import LabelEncoder
from sdv.metadata import SingleTableMetadata
import pandas as pd
import logging

logging.getLogger('sdv').setLevel(logging.CRITICAL)

class PipelineEstimator(BaseEstimator):
    """
    Class to execute a given individual/ pipeline

    """

    def __init__(self, individual_config, pipeline_config=None):
        """
        Initialization

        Assigns the given parameters

        Parameters:
        ----------
        individual_config : dict
            Configuration of an individual

        pipeline_config : dict
            Characteristics of the data to create the adquate pipeline

        use_sampling : bool
            Tell either should be use sampling to train the individual/pipeline or not

        use_feature_selection : bool
            Tell either should be use feature selection to train the individual/pipeline or not

        Returns:
        -------
            None
        """
        self.individual_config = individual_config
        self.pipeline_config = pipeline_config

    def fit(self, X, y):
        """
        Train the individual / pipeline based on the received data

        Selects the data to use, in case of applying sampling. If the pipeline_config parameters is null,
        it analysis the dataset to see which pipeline is required. It instantiate the pipeline and trains it.

        Parameters:
        ----------
        X : pandas.DataFrame
            Data / features to train the pipeline

        y : pandas.Series
            Target column

        Returns:
        -------
            self
        """
        self.selected_features = list(X.columns)
        # analyse dataset if pipeline config is null
        if self.pipeline_config is None:
            self.pipeline_config = dataset_analysis(X)
        self.X = X.copy()
        self.y = y.copy()
        self.X_train = X.copy()
        self.y_train = y.copy()

        # select samples to use in training
        if 'sample' in self.individual_config:
            # check if all values are 0 or 1 = binary representation
            if len(set(self.individual_config['sample'] + [0, 1])) == 2:
                indexes_vector = self.individual_config.get('sample')
                indexes_vector = np.array(indexes_vector).astype(bool)
                self.X_train = self.X_train.loc[indexes_vector].copy()
                self.y_train = self.y_train.loc[indexes_vector].copy()
            else:
                # integer representation
                self.X_train = self.X_train.iloc[self.individual_config['sample']].copy(
                )
                self.y_train = self.y_train.iloc[self.individual_config['sample']].copy(
                )

        # select features to use in training
        if 'features' in self.individual_config:
            # check if all values are 0 or 1 = binary representation
            if len(set(self.individual_config['features'] + [0, 1])) == 2:
                self.selected_features = self.X_train.columns[np.array(
                    self.individual_config['features']).astype(bool)]
                self.X_train = self.X_train[self.selected_features].copy()
            else:
                # integer representation
                self.selected_features = self.X_train.columns[self.individual_config['features']]
                self.X_train = self.X_train[self.selected_features].copy()

        # data augmentation
        if 'data_augmentation' in self.individual_config:
            logging.disable = True
            # create aux dataframe to train the data augmentation model 
            aux_data = self.X_train.copy()
            aux_data['target'] = self.y_train
            # create the metadata
            # Disable logging for the specific module
            metadata_detector = SingleTableMetadata()
            metadata_detector.detect_from_dataframe(aux_data)
            augmentation_config = self.individual_config['data_augmentation'].copy()   
            # create the data augmentation model
            augmentation_model, sample_percentage = instantiate_data_augmentation(
                augmentation_config=augmentation_config,
                metadata=metadata_detector
            )
            augmentation_model.fit(aux_data)
            # create the augmented data
            aux_x = augmentation_model.sample(int(sample_percentage * len(self.X_train))) # sample data size equal the sample_percentage parameter x the size of the training data (selected)
            aux_y = aux_x.pop('target')
            # add the augmented data to the training data
            self.X_train = pd.concat([self.X_train, aux_x], axis=0, ignore_index=True)
            self.y_train = pd.concat([self.y_train, aux_y], axis=0, ignore_index=True)
            logging.disable = False

        # create preprocessing pipeline
        self.pipeline = create_preprocessing_pipeline(
            selected_features=self.selected_features,
            pipeline_config=self.pipeline_config,
            individual=self.individual_config
        )
        self.X_training = self.pipeline.fit_transform(self.X_train).copy()

        # encode the target data
        self.y_encoder = LabelEncoder()
        self.y_train = self.y_encoder.fit_transform(self.y_train)

        # create the model
        self.model = instantiate_model(self.individual_config.get('model'))
        self.model.fit(self.X_training, self.y_train)
        return self

    def predict(self, X):
        """
        Predicts the output of a given data

        It preprocesses the given data and predicts the results of the model

        Parameters:
        ----------
        X : pandas.DataFrame
            Data to predict

        Returns:
        -------
            numpy.array
                predictions
        """
        X_test = X.copy()
        X_test = X_test[self.selected_features]
        X_test = self.pipeline.transform(X_test)
        preds = self.model.predict(X_test)
        return self.y_encoder.inverse_transform(preds)

    def predict_proba(self, X):
        """
        Predicts the probability of each output class of the given data

        It preprocesses the given data and predicts the results of the model

        Parameters:
        ----------
        X : pandas.DataFrame
            Data to predict

        Returns:
        -------
            numpy.array
                predictions
        """
        X_test = X.copy()
        X_test = X_test[self.selected_features]
        X_test = self.pipeline.transform(X_test)
        return self.model.predict_proba(X_test)

    def get_best_sample_data(self):
        if 'sample' in self.individual_config:
            # check if all values are 0 or 1 = binary representation
            if len(set(self.individual_config['sample'] + [0, 1])) == 2:
                indexes_vector = self.individual_config.get('sample')
                indexes_vector = np.array(indexes_vector).astype(bool)
                x = self.X.loc[indexes_vector].copy()
                y = self.y.loc[indexes_vector].copy()
            else:
                # integer representation
                x = self.X.iloc[self.individual_config['sample']].copy()
                y = self.y.iloc[self.individual_config['sample']].copy()
            return x, y
        else:
            return self.X, self.y


def dataset_analysis(df):
    """
    Analysis the given data

    It analysis the data types of the given columns of the dataframe

    Parameters:
    ----------
    df : pandas.DataFrame
        Data to analyse

    Returns:
    -------
        dict
            contains lists of features containing the different data types present in the dataset
    """

    print('>>> Dataset Analysis')
    # select null cols
    null_cols = list(df.columns[df.isnull().sum() == len(df)])
    # select columns with nan that not belong to null_cols
    columns_with_nans = [col for col in list(
        df.columns[df.isnull().any()]) if col not in null_cols]
    # select numerical columns dat not belong to null cols
    numerical_columns = [col for col in df.select_dtypes(
        include=['number']).columns.tolist() if col not in null_cols]
    # select other columns, i.e, binary and categorical columns
    other_columns = [
        col for col in df.columns if col not in numerical_columns and col not in null_cols]
    categorical_columns = []
    id_columns = []
    binary_columns = []

    # check bools with empty columns
    for column in other_columns:
        aux = df[column].dropna()
        if aux.nunique() == 2:
            binary_columns.append(column)
        elif aux.nunique() == len(aux):
            id_columns.append(column)
        else:
            categorical_columns.append(column)

    # combines the informations of the nans and type of features to calculate
    # the features seperated by type with nans
    numerical_with_nans = list(set(numerical_columns) & set(columns_with_nans))
    binary_with_nans = list(set(binary_columns) & set(columns_with_nans))
    categorical_with_nans = list(
        set(categorical_columns) & set(columns_with_nans))
    # create the pipeline config with the types of features separated by
    # characteristics
    pipeline_config = {
        'numerical_columns': numerical_columns,
        'categorical_columns': categorical_columns,
        'id_columns': id_columns,
        'binary_columns': binary_columns,
        'numerical_with_nans': numerical_with_nans,
        'categorical_with_nans': categorical_with_nans,
        'binary_with_nans': binary_with_nans,
        'null_cols': null_cols
    }
    print('<<< Dataset Analysis')
    return pipeline_config
