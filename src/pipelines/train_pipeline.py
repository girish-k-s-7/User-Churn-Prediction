import sys
from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


def start_training_pipeline():
    try:
        logging.info("Starting training pipeline...")

        
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()

         
        transform = DataTransformation()
        X_train, X_test, y_train, y_test, _ = transform.initiate_data_transformation(
            train_path, test_path
        )

         
        trainer = ModelTrainer()
        acc, f1, model_path, model_name = trainer.initiate_model_trainer(
            X_train, X_test, y_train, y_test
        )

         
        logging.info("Training Pipeline Completed Successfully!")
        logging.info(f"Best Model: {model_name}, Accuracy: {acc}, F1: {f1}")

        
        print("\n================ PIPELINE RESULT =================")
        print(f"Best Model : {model_name}")
        print(f"Accuracy   : {acc:.4f}")
        print(f"F1 Score   : {f1:.4f}")
        print(f"Model Path : {model_path}")
        print("=================================================\n")

        return acc, f1, model_path, model_name

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    start_training_pipeline()
