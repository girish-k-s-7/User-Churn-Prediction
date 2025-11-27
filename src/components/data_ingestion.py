import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomException

class DataIngestion:
    def __init__(self):
        self.raw_data_path = os.path.join("artifacts", "raw_data.csv")
        self.train_data_path = os.path.join("artifacts", "train.csv")
        self.test_data_path = os.path.join("artifacts", "test.csv")

    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion process")

        try:
            # Path to your original dataset
            source_path = os.path.join(
                "Data Analysis", 
                "data", 
                "Telco_churn_after_DC.csv"
            )

            logging.info(f"Reading source data from: {source_path}")

            data = pd.read_csv(source_path)

            os.makedirs("artifacts", exist_ok=True)

            # Save raw data
            data.to_csv(self.raw_data_path, index=False)

            logging.info("Train-test split started")
            train_set, test_set = train_test_split(
                data, test_size=0.2, random_state=42
            )

            train_set.to_csv(self.train_data_path, index=False)
            test_set.to_csv(self.test_data_path, index=False)

            logging.info("Data ingestion completed successfully")

            return (
                self.train_data_path,
                self.test_data_path
            )

        except Exception as e:
            logging.error("Error during data ingestion", exc_info=True)
            raise CustomException(e, sys)


if __name__ == "__main__":
    ingestor = DataIngestion()
    train_path, test_path = ingestor.initiate_data_ingestion()
    print("Ingestion complete!")
    print("Train path:", train_path)
    print("Test path:", test_path)
