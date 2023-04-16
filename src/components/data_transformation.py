import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTETomek

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

class DataTransformConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformConfig()

    def get_data_transformer_object(self):
        try:
            columns=['hour_of_day', 'age', 'month', 'category', 'amount(usd)', 'gender', 'lat', 'long', 'is_fraud']
            
            # define SMOTETomek pipeline
            smote_tomek_pipeline = Pipeline([('smote_tomek', SMOTETomek())])

            logging.info("Oversampling done")

            preprocessor=ColumnTransformer(['smote_tomek_pipeline',smote_tomek_pipeline,columns])

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("the train and test read completed")

            logging.info("obtaining preprocessor")
            
            preprocessing_obj=self.get_data_transformer_object()

            target_column_name='is_fraud'
            columns=['hour_of_day', 'age', 'month', 'category', 'amount(usd)', 'gender', 'lat', 'long', 'is_fraud']

            smote_train_df=preprocessing_obj.fit_transform(train_df)
            
            input_feature_train_df=smote_train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=smote_train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]


            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            train_arr = np.c_[
                input_feature_train_df, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_df, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj)
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        
        
        except Exception as e:
            raise CustomException(e,sys)

