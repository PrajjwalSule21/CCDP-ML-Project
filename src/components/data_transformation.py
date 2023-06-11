import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def check_std_dev(self, train_df, test_df):
        """
        This Function will check if any of data feature consist 0 standard deviation or not.
        and if it finds any feature consist 0 standard deviation so it will simply drop that feature.
        """
        try:
            train_df_datafeatures = [
                i for i in train_df.columns if train_df[i].dtypes == 'int64' or train_df[i].dtypes == 'float64']
            test_df_datafeatures = [
                i for i in test_df.columns if test_df[i].dtypes == 'int64' or test_df[i].dtypes == 'float64']
            for feature in train_df_datafeatures:
                if train_df[feature].std() == 0:
                    train_df.drop(columns=feature, axis=1, inplace=True)

            for feature in test_df_datafeatures:
                if test_df[feature].std() == 0:
                    test_df.drop(columns=feature, axis=1, inplace=True)

            return (
                train_df, test_df
            )

        except Exception as e:
            raise CustomException(e, sys)

    def remove_duplicates(self, train_df, test_df):
        """
        This function will help us to remove the duplicate data from dataframe
        """
        try:
            train_df.drop_duplicates(keep='first', inplace=True)
            test_df.drop_duplicates(keep='first', inplace=True)

            return (
                train_df, test_df
            )
        except Exception as e:
            raise CustomException(e, sys)

    def feature_transformation(self, train_df, test_df):
        """
        This function will transform the feature into one important feature, as we already see in EDA.
        """
        try:
            train_df['TOTAL_BILL_AMT'] = train_df['BILL_AMT1'] + train_df['BILL_AMT2'] + \
                train_df['BILL_AMT3'] + train_df['BILL_AMT4'] + \
                train_df['BILL_AMT5'] + train_df['BILL_AMT6']
            train_df['TOTAL_PAY_AMT'] = train_df['PAY_AMT1'] + train_df['PAY_AMT2'] + \
                train_df['PAY_AMT3'] + train_df['PAY_AMT4'] + \
                train_df['PAY_AMT5'] + train_df['PAY_AMT6']

            test_df['TOTAL_BILL_AMT'] = test_df['BILL_AMT1'] + test_df['BILL_AMT2'] + \
                test_df['BILL_AMT3'] + test_df['BILL_AMT4'] + \
                test_df['BILL_AMT5'] + test_df['BILL_AMT6']
            test_df['TOTAL_PAY_AMT'] = test_df['PAY_AMT1'] + test_df['PAY_AMT2'] + \
                test_df['PAY_AMT3'] + test_df['PAY_AMT4'] + \
                test_df['PAY_AMT5'] + test_df['PAY_AMT6']

            train_df.drop(columns=['ID', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5',
                          'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'], inplace=True)
            
            test_df.drop(columns=['ID', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5',
                         'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'], inplace=True)

            return (
                train_df,
                test_df
            )

        except Exception as e:
            raise CustomException(e, sys)

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation

        '''
        try:
            numerical_columns = ['LIMIT_BAL' ,'AGE', 'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'TOTAL_BILL_AMT', 'TOTAL_PAY_AMT'
                                 ]

            categorical_columns = ['SEX', 'EDUCATION', 'MARRIAGE']

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())

                ]
            )

            cat_pipeline = Pipeline(

                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("scaler", StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipelines", cat_pipeline, categorical_columns)

                ]

            )

            logging.info('Pipeline for numerical and categorical completed')

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            logging.info(
                'Check if there is any feature which consist 0 standard deviation or not')
            train_df, test_df = self.check_std_dev(train_df, test_df)

            logging.info(
                'Check if there is any feature  consist duplicate value or not or not')
            train_df, test_df = self.remove_duplicates(train_df, test_df)

            logging.info(
                'Feature Transformation has been strated to transform the important featrue into one feature')
            train_df, test_df = self.feature_transformation(train_df, test_df)

            preprocessing_obj = self.get_data_transformer_object()
            logging.info("Obtaining preprocessing object")

            target_column_name = "DEFAULT_PAYMENT_NEXT_MONTH"

            input_feature_train_df = train_df.drop(
                columns=[target_column_name], axis=1)
            logging.info(
                'Gettting the X train dataframe consist all the features')
            target_feature_train_df = train_df[target_column_name]
            logging.info('Gettting the y train dataframe consist the label')

            input_feature_test_df = test_df.drop(
                columns=[target_column_name], axis=1)
            logging.info(
                'Gettting the X test dataframe consist all the features')
            target_feature_test_df = test_df[target_column_name]
            logging.info('Gettting the y test dataframe consist the label')

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

            # print(train_df.head())

            # print(input_feature_train_df.head())

            # print(input_feature_train_arr)

        except Exception as e:
            raise CustomException(e, sys)
