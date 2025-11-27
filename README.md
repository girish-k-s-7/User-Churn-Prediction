# 🚀 Customer Churn Prediction System (End-to-End ML + MLOps)

This project implements a **production-ready Customer Churn Prediction System** using Machine Learning with a complete ML pipeline including data ingestion, data transformation, model training, evaluation, and deployment using Streamlit.

The model predicts whether a customer is likely to **churn (Yes / No)** based on historical usage patterns and customer attributes. The architecture follows MLOps best practices with modular code structure and reusable components.

---

## 🧠 Problem Statement

Customer churn directly affects business revenue and growth. Retaining existing customers is significantly cheaper than acquiring new ones.  
This project aims to build a supervised classification model that predicts customer churn using service usage data.

---

## 🏗 Project Architecture

User-Churn-Prediction
│
├── artifacts/ # Saved output files
│ ├── model.pkl # Final trained model
│ ├── preprocessor.pkl # Data transformation pipeline
│
├── data/ # Dataset
│ └── Telco_churn_data.csv
│
├── src/
│ ├── components/
│ │ ├── data_ingestion.py
│ │ ├── data_transformation.py
│ │ └── model_trainer.py
│ │
│ ├── pipelines/
│ │ ├── train_pipeline.py
│ │ └── prediction_pipeline.py
│ │
│ ├── utils.py # Utility functions
│ ├── logger.py # Logging system
│ └── exception.py # Custom exception handler
│
├── app.py # Streamlit web app
├── requirements.txt
├── setup.py
├── README.md
└── .gitignore


---

## ⚙️ Tech Stack

| Layer | Technologies |
|--------|-------------|
| Programming | Python |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Models | Logistic Regression, RandomForest, SVM, Gradient Boosting |
| Deployment | Streamlit |
| Version Control | Git, GitHub |
| Serialization | Pickle |
| Logging | Custom Logger |
| Error Handling | Custom Exceptions |

---

## 🔄 ML Pipeline Workflow

### ✅ Data Ingestion
Splits raw dataset into training and testing datasets.

### ✅ Data Transformation
Handles:
- Missing values
- One-hot encoding of categorical columns
- Feature scaling
- Saves preprocessing pipeline as `.pkl`

### ✅ Model Training
Trains and compares multiple models using GridSearchCV:
- Logistic Regression
- Random Forest
- Support Vector Machine
- Gradient Boosting

Evaluation Metrics:
- **F1-Score (primary)**  
- Accuracy (secondary)

Best model is automatically selected and saved.

---

## 📊 Model Performance

Best Model : LogisticRegression
Accuracy : 0.8190
F1 Score : 0.6341


---

## 🌐 Streamlit Web Application

The UI allows users to enter:

- Contract Type  
- Monthly Charges  
- Tenure  
- Internet Services  
- Payment Method  

The model returns:

✔ Predicted Churn Status  
✔ Confidence Score  

---

## 🏆 Key Highlights

✅ Complete ML pipeline  
✅ Feature engineering  
✅ Hyperparameter tuning  
✅ Modular codebase  
✅ Logging + exception handling  
✅ Real-time deployment with Streamlit  
✅ Production-grade project structure  

---

## 🚀 Future Improvements

- SHAP explainability dashboard  
- Dockerization  
- CI/CD pipeline  
- Cloud deployment  
- Model monitoring  

---


## 👨‍💻 Author

**Girish K S**  
Data Scientists

 
