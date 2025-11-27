import os
import sys
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictionPipeline:
     

    def __init__(
        self,
        model_path: str = os.path.join("artifacts", "model.pkl"),
        preprocessor_path: str = os.path.join("artifacts", "preprocessor.pkl"),
    ):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path

        self._model = None
        self._preprocessor = None

    def _load_artifacts(self):
         
        try:
            if self._preprocessor is None:
                logging.info(f"Loading preprocessor from: {self.preprocessor_path}")
                self._preprocessor = load_object(self.preprocessor_path)

            if self._model is None:
                logging.info(f"Loading model from: {self.model_path}")
                self._model = load_object(self.model_path)

        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, input_df: pd.DataFrame) -> pd.DataFrame:
        
        try:
            self._load_artifacts()

            if not isinstance(input_df, pd.DataFrame):
                raise ValueError("input_df must be a pandas DataFrame")

            logging.info("Starting prediction on input data")
            logging.info(f"Input shape: {input_df.shape}")

            # Transform with preprocessor
            X_transformed = self._preprocessor.transform(input_df)

            # Class predictions ('Yes' / 'No')
            preds = self._model.predict(X_transformed)

            # Probabilities (robustly find column for 'Yes')
            if hasattr(self._model, "predict_proba"):
                class_index = list(self._model.classes_).index("Yes")
                proba = self._model.predict_proba(X_transformed)[:, class_index]
            else:
                proba = np.full(shape=(len(preds),), fill_value=np.nan)

            result = pd.DataFrame(
                {
                    "prediction": preds,
                    "churn_proba": proba
                },
                index=input_df.index
            )

            logging.info("Prediction completed successfully")
            return result

        except Exception as e:
            logging.error("Error in prediction pipeline", exc_info=True)
            raise CustomException(e, sys)


if __name__ == "__main__":
    print("This module is meant to be used via import.")
