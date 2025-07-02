# **PsychCareAI - Mental Health Analysis Platform**

PsychCareAI is an interactive web application designed to analyze mental health data and predict psychiatric disorders using machine learning models. The platform leverages advanced models like CatBoost, LightGBM, XGBoost, and Random Forest to provide predictions and risk assessments based on patient data. The app is built with Streamlit for real-time interaction and data visualization.

## **Features**
- **Patient Data Input**: Allows users to input patient details, including personal and clinical information.
- **Risk Assessment**: Calculates the risk of psychiatric disorders based on demographic, clinical, and behavioral risk factors.
- **Model Predictions**: Uses pre-trained machine learning models to predict psychiatric conditions.
- **Dashboard**: Displays key system metrics, model performance, and recent activities.
- **Chatbot**: Offers an interactive interface for patients to engage and get immediate responses on their mental health.

## **Tech Stack**
- **Streamlit**: For creating the interactive web application.
- **Scikit-learn**: For machine learning model evaluation.
- **CatBoost, LightGBM, XGBoost**: Models for psychiatric disorder prediction.
- **Pandas, NumPy**: Data manipulation and analysis.
- **SQLite**: For database storage.

## **File Structure**


```
PsychCareAI/
│
├── .streamlit/
│
├── app.py                               # Main application (Streamlit app)
│
├── MH.ipynb                              # Jupyter notebook for ML pipeline
│
├── psychcare.db                          # SQLite database used by the app
│
├── README.md                             # Project documentation (README)
│
├── attached_assets/                     # Assets
│   └── Cleaned mental health data.csv    # dataset used in the analysis
│
├── components/                          # Custom UI components for Streamlit app
│   ├── analysis_results.py               # Displays analysis results after input
│   ├── chatbot.py                       # Chatbot interface for user interaction
│   ├── dashboard.py                     # Displays system overview and model performance
│   ├── patient_input.py                  # User input form for patient data
│
├── database/                            # Database-related files and scripts
│   └── schema.py                        # Defines the schema for the database (tables, relationships)
│
├── models/                              # Pre-trained models and related files
│   ├── CatBoost_Optimized_model.pkl     # Pre-trained CatBoost model
│   ├── feature_info.pkl                 # Information about features used in models
│   ├── label_encoder.pkl                # Label encoder for categorical variables
│   ├── LightGBM_Original_model.pkl      # Pre-trained LightGBM model
│   ├── ml_models.py                     # Contains class for managing ML models
│   ├── model_evaluation.pkl             # Evaluation metrics for the models
│   ├── model_summary.pkl                # Summary of model architecture and parameters
│   ├── nlp_processor.py                 # Processes natural language data for the models
│   ├── pipeline_metadata.pkl            # Metadata for the ML pipeline
│   ├── Random_Forest_Original_model.pkl # Pre-trained Random Forest model
│   ├── scaler.pkl                       # Standard scaler used for feature scaling
│   ├── validation_report.pkl            # Validation report for model performance
│   └── XGBoost_Optimized_model.pkl      # Pre-trained XGBoost model
│
└── utils/                               # Utility functions for different tasks
    └── risk_assessment.py               # Risk assessment logic based on patient data
```

## **Installation**

### **1. Clone the repository**:
```
git clone https://github.com/yourusername/PsychCareAI.git
```

### **2. Install required dependencies**:

```
pip install -r requirements.txt
```

### **3. Run the Streamlit app**:

```
streamlit run app.py
```
Visit the application in your browser at http://localhost:5000.

## **Usage**

* **Input Patient Data**: Enter the patient's details in the provided form to initiate the analysis.
* **View Results**: Once data is entered, the app will display predictions, risk assessments, and relevant insights.
* **Dashboard**: View the overall system performance and model metrics in the dashboard.

## **Model Information**

* The system uses several machine learning models, including:
  * **CatBoost**
  * **LightGBM**
  * **XGBoost**
  * **Random Forest**

Each model is pre-trained and stored in the `models/` directory. These models are used to predict psychiatric disorders based on the patient data provided.

## **Contributing**

Feel free to open an issue or submit a pull request if you want to contribute to this project.

## **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
