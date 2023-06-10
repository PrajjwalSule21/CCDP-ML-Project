import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
# import hydra
# from omegaconf import DictConfig, OmegaConf

# @hydra.main(config_path="../../conf", config_name="main", version_base=None)
# def path(conf:DictConfig):
#     train_path = conf.train_data_path
#     print(OmegaConf.to_yaml(conf))
#     return conf.new



@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join("artifacts","train.csv")
    test_data_path: str=os.path.join("artifacts","test.csv")
    raw_data_path: str=os.path.join("artifacts","data.csv")
   


class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered into the data ingestion method")
        try:
            path = r'data\CustomerCreditCard.csv'
            df = pd.read_csv(path)
            df['TOTAL_BILL_AMT'] = df['BILL_AMT1']+ df['BILL_AMT2'] + df['BILL_AMT3'] + df['BILL_AMT4'] + df['BILL_AMT5'] + df['BILL_AMT6']
            df['TOTAL_PAY_AMT'] = df['PAY_AMT1'] + df['PAY_AMT2'] + df['PAY_AMT3'] + df['PAY_AMT4'] + df['PAY_AMT5'] + df['PAY_AMT6']
            
            df.drop(columns=['ID','BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'], inplace=True)

            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            logging.info('Make a directory for training data')

            df.to_csv(self.ingestion_config.raw_data_path,index=False, header=True)
            logging.info('Save that dataframe into a csv file name as [data.csv] into the artifacts directory')

            logging.info('Train Test Split initiated')
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            logging.info('Save the train data as a csv file into artifacts folder')

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info('Save the test data as csv file into artifacts folder')
        
            logging.info('Ingestion of the data has been completed')
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)
        

if __name__ == "__main__":
    # a = path()
    # print(a)
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

#     data_transformation = DataTransformation()
#     train_array, test_array, other = data_transformation.initiate_data_transformation(train_data, test_data)

#     modeltrainer = ModelTrainer()
#     print(modeltrainer.initiate_model_trainer(train_array=train_array, test_array=test_array))





