import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
        
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(self,
        LIMIT_BAL: int,
        AGE: int,
        SEX: int,
        EDUCATION: int,
        MARRIAGE: int,
        PAY_1: int,
        PAY_2: int,
        PAY_3: int,
        PAY_4: int,
        PAY_5: int,
        PAY_6: int,
        lunch: str,
        TOTAL_BILL_AMT: int,
        TOTAL_PAY_AMT: int):

        self.LIMIT_BAL = LIMIT_BAL

        self.AGE = AGE

        self.SEX = SEX

        self.EDUCATION = EDUCATION

        self.MARRIAGE = MARRIAGE

        self.PAY_1 = PAY_1

        self.PAY_2 = PAY_2

        self.PAY_3 = PAY_3

        self.PAY_4 = PAY_4

        self.PAY_5 = PAY_5

        self.PAY_6 = PAY_6

        self.TOTAL_BILL_AMT = TOTAL_BILL_AMT

        self.TOTAL_PAY_AMT = TOTAL_PAY_AMT

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "LIMIT_BAL": [self.LIMIT_BAL],
                "AGE": [self.AGE],
                "SEX": [self.SEX],
                "EDUCATION": [self.EDUCATION],
                "MARRIAGE": [self.MARRIAGE],
                "PAY_1": [self.PAY_1],
                "PAY_2": [self.PAY_2],
                "PAY_3": [self.PAY_3],
                "PAY_4": [self.PAY_4],
                "PAY_5": [self.PAY_5],
                "PAY_6": [self.PAY_6],
                "TOTAL_BILL_AMT": [self.TOTAL_BILL_AMT],
                "TOTAL_PAY_AMT" : [self.TOTAL_PAY_AMT]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
