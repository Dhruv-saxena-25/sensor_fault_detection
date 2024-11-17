from sensor.utils.main_utils import load_numpy_array_data
from sensor.exception import SensorException
from sensor.logger import logging
from sensor.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from sensor.entity.config_entity import ModelTrainerConfig
import os
import sys
from sensor.ml.metric.classification_metric import get_classification_metric
from sensor.ml.model.estimator import SensorModel
from sensor.utils.main_utils import save_object, load_object

from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from dotenv import load_dotenv
load_dotenv()

import mlflow
from urllib.parse import urlparse 
import dagshub
dagshub.init(repo_owner='Dhruv-saxena-25', repo_name='sensor_fault_detection', mlflow=True)

os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI")
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")
class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise SensorException(e, sys)
    
    
    def finetune_train_model(self):
        try:
            pass
        except Exception as e:
            raise SensorException(e, sys)
        
    def track_mlflow(self, model, classificationmetric, label):
        try:
            mlflow.set_registry_uri("https://dagshub.com/Dhruv-saxena-25/sensor_fault_detection.mlflow")
            with mlflow.start_run():
                mlflow.set_tag("label", label)
                f1_score = classificationmetric.f1_score
                precision_score = classificationmetric.precision_score
                recall_score = classificationmetric.recall_score
                mlflow.log_metrics("f1_score", f1_score)
                mlflow.log_metrics("precision_score", precision_score)
                mlflow.log_metrics("recall_score", recall_score)
                mlflow.sklearn.log_model(model, "model")
        except Exception as e:
            raise SensorException(e, sys) 
    
    def train_model(self, x_train, y_train):
        try:
            xgb_clf = XGBClassifier()
            xgb_clf.fit(x_train, y_train)
            return xgb_clf
        except Exception as e:
            raise SensorException(e, sys)
    def initiate_model_trainer(self):
        logging.info("Starting model trainer!!!!")
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path
            
            # loading training array and testing array
            
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)
            
            x_train, y_train, x_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1],
            )
            
            model = self.train_model(x_train, y_train)
            
            y_train_pred = model.predict(x_train)
            
            classification_train_metric = get_classification_metric(y_true= y_train, y_pred= y_train_pred)
            ## Track the train experiements with mlflow
            self.track_mlflow(model, classification_train_metric, label= "Training")
            if classification_train_metric.f1_score <= self.model_trainer_config.expected_accuracy:
                raise Exception("Trained model is not to provide expected accuracy!!!!")
            
            y_test_pred = model.predict(x_test)
            classification_test_metric = get_classification_metric(y_true= y_test, y_pred= y_test_pred)
            ## Track the Test experiements with mlflow
            self.track_mlflow(model, classification_test_metric, label= "Testing")
            
            ## Checking for Overfitting and Underfitting 
            
            diff = abs(classification_train_metric.f1_score - classification_test_metric.f1_score)
            
            if diff > self.model_trainer_config.overfitting_underfitting_threshold:
                raise Exception("Model is not good try to do more experiments!!!")
            
            preprocessor= load_object(file_path= self.data_transformation_artifact.transformed_object_file_path)
            
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok= True)
            
            sensor_model = SensorModel(preprocessor= preprocessor, model = model)
            save_object(self.model_trainer_config.trained_model_file_path, obj= sensor_model)
            
            ## model trainer artifact
            
            model_trainer_artifact = ModelTrainerArtifact(trained_model_file_path= self.model_trainer_config.trained_model_file_path,
            train_mertic_artifact= classification_train_metric,
            test_metric_artifact= classification_test_metric,)
            
            logging.info(f"Model Trainer artifact {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise SensorException(e, sys)
        
        
        
        