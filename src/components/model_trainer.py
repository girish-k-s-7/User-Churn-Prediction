import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object


class ModelTrainer:

    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")

    def initiate_model_trainer(self, X_train, X_test, y_train, y_test):

        logging.info("Model training started...")

        try:
             
            models = {
                "LogisticRegression": LogisticRegression(max_iter=200),
                "RandomForest": RandomForestClassifier(),
                "GradientBoosting": GradientBoostingClassifier(),
                "SVC": SVC(probability=True)
            }

             
            params = {
                "LogisticRegression": {
                    "C": [0.1, 1, 10]
                },
                "RandomForest": {
                    "n_estimators": [50, 100],
                    "max_depth": [None, 10]
                },
                "GradientBoosting": {
                    "learning_rate": [0.01, 0.1],
                    "n_estimators": [100, 200]
                },
                "SVC": {
                    "C": [0.1, 1, 10]
                }
            }

             
            best_model = None
            best_score = -np.inf

            for model_name, model in models.items():
                logging.info(f"Training model: {model_name}")

                grid = GridSearchCV(model, params[model_name], cv=3)
                grid.fit(X_train, y_train)

                y_pred = grid.predict(X_test)

                # y_train/y_test are 'Yes'/'No', so we set pos_label="Yes"
                f1 = f1_score(y_test, y_pred, pos_label="Yes")

                logging.info(
                    f"{model_name} → Best Params: {grid.best_params_}, F1: {f1}"
                )

                if f1 > best_score:
                    best_score = f1
                    best_model = grid.best_estimator_

             
            save_object(self.model_path, best_model)
            logging.info(f"Best Model Saved at: {self.model_path}")

             
            final_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, final_pred)
            f1_final = f1_score(y_test, final_pred, pos_label="Yes")

            return (
                accuracy,
                f1_final,
                self.model_path,
                best_model.__class__.__name__
            )

        except Exception as e:
            logging.error("Error in Model Trainer", exc_info=True)
            raise CustomException(e, sys)


if __name__ == "__main__":
    print("Run this ONLY through the training pipeline.")
