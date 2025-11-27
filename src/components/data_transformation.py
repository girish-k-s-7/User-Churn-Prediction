import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_preprocessor(self):
        try:
            logging.info("Preparing preprocessing pipeline...")

            # Identify column types
            numerical_columns = [
                "tenure",
                "MonthlyCharges",
                "TotalCharges"
            ]

            categorical_columns = [
                "gender",
                "SeniorCitizen",
                "Partner",
                "Dependents",
                "PhoneService",
                "MultipleLines",
                "InternetService",
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
                "Contract",
                "PaperlessBilling",
                "PaymentMethod"
            ]

            # Numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            # Combine both
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ]
            )

            logging.info("Preprocessor pipeline created successfully.")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)


    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Loading train and test data...")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and Test data loaded.")
            logging.info("Creating preprocessor object...")

            preprocessor = self.get_preprocessor()

            target_column = "Churn"

            # Split into input and output
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            logging.info("Applying preprocessing to training and testing datasets...")

            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            # Save the transformer
            save_object(
                file_path=self.config.preprocessor_obj_file_path,
                obj=preprocessor,
            )

            logging.info("Preprocessor saved.")

            return (
                X_train_transformed,
                X_test_transformed,
                y_train,
                y_test,
                self.config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
if __name__ == "__main__":
    transformer = DataTransformation()
    print("Running Data Transformation...")

    X_train, X_test, y_train, y_test, path = transformer.initiate_data_transformation(
        "artifacts/train.csv",
        "artifacts/test.csv"
    )

    print("Transformation Completed")
    print("Preprocessor saved at:", path)
