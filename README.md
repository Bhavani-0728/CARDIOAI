# ❤️ CardioAI — AI Cardiovascular Risk Intelligence System

An AI-powered web application that predicts cardiovascular disease risk using machine learning. Users can input health metrics and receive a health score, risk classification, and AI-generated insights displayed through a modern dark-themed dashboard with an interactive circular gauge.

This project demonstrates an end-to-end machine learning pipeline with a deployment-ready UI using Streamlit.

## 🌟 Features

- **🧠 Cardiovascular Risk Prediction** – Predict probability of heart disease using Random Forest  
- **📊 Circular Health Score Gauge** – Interactive animated gauge (0–100 scale)  
- **🏷️ Dynamic Risk Classification** – Low / Moderate / High categories  
- **🔍 AI Risk Insights** – Context-aware lifestyle recommendations  
- **🌑 Premium Dark UI** – Black gradient SaaS-style interface  
- **⚡ Optimized Performance** – Cached model loading for smooth experience  
- **🧩 Modular ML Architecture** – Clean separation of preprocessing, feature engineering, and model training  
- **📁 GitHub-Ready Structure** – Production-style project organization  

## 📂 Project Structure

CardioAI/
│
├── data/
│ ├── raw/ # Original dataset
│ └── processed/ # Cleaned dataset
│
├── models/
│ ├── best_model.pkl # Trained ML model
│ ├── feature_columns.json # Feature order reference
│ └── feature_importance.png # Feature visualization
│
├── models/
│ ├── EDA.ipynb # Exploratory Data Analysis
│
├── src/
│ ├── data_preprocessing.py # Cleaning and preprocessing
│ ├── feature_engineering.py # BMI & pulse pressure creation
│ └── model_training.py # Model training logic
│
├── app.py # Streamlit frontend
├── train.py # Model training entry point
├── requirements.txt # Python dependencies
├── .gitignore # Git exclusions
└── README.md # Project documentation


## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Git
- pip

### 1️⃣ Clone the Repository

### Option 1: Clone with git
git clone <repository-url> 

### Option 2: Download and extract the ZIP file

### 2. Install Dependencies

### Install all required Python packages
pip install -r requirements.txt

### 3. 📥 Dataset Setup

This project uses the Cardiovascular Disease dataset.

If the dataset is not included in the repository:

1. Download the dataset from Kaggle (Cardiovascular Disease Dataset).

2. Place the file inside:

   - data/raw/cardio_train.csv

3. Make sure the file path matches the one used in `train.py`.

### 4. Train the model

python train.py

### 5. Run the Application

### Streamlit Frontend
streamlit run src/app.py

The app will open in your browser at `http://localhost:8501`

## 🧠 How It Works
1. User inputs health metrics in the sidebar

2. BMI and pulse pressure are calculated

3. Random Forest model predicts cardiovascular probability

4. Probability is converted into:

    - Health Score (0–100)

    - Risk Category

    - AI-based risk insights

5. Results are displayed via an interactive circular gauge

## 🛠Technical Details

### Technologies Used

- **Frontend**: Streamlit (Python web framework), Plotly (Visualize charts)
- **Machine Learning**: Scikit-learn (Random Forest)

- **Data Handling**: Pandas, NumPy

- **Model Persistence**: Joblib

- **Language**: Python 3.8+

## 🔧 Key Components

1. **`src/data_preprocessing.py`**: Data cleaning module – Handles missing values, removes invalid entries, filters outliers, and prepares the dataset for modeling.

2. **`src/feature_engineering.py`**: Feature engineering module – Computes derived features such as BMI (Body Mass Index) and pulse pressure to enhance model performance.

3. **`src/model_training.py`**: Model training module – Splits data, trains the Random Forest classifier, evaluates performance (Accuracy & ROC-AUC), and saves the trained model along with feature metadata.

4. **`train.py`**: Training entry script – Executes the full ML pipeline from preprocessing to model persistence.

5. **`app.py`**: Streamlit frontend – Premium dark-themed UI with circular health gauge, dynamic risk classification, and AI-generated health insights.

6. **`models/best_model.pkl`**: Serialized trained machine learning model used for real-time prediction.

7. **`models/feature_columns.json`**: Stores feature order to ensure correct alignment during prediction.

8. **`notebooks/eda.ipynb`**: Exploratory Data Analysis notebook – Includes correlation analysis, distribution plots, class imbalance study, and feature insights.

## ⚠️ Troubleshooting

## Common Issues

1. **"Model file not found" error**:
   - Make sure you have trained the model before running the app:
     ```
     python train.py
     ```
   - Confirm that `models/best_model.pkl` exists.

2. **"Module not found" error**:
   - Install dependencies:
     ```
     pip install -r requirements.txt
     ```
   - Ensure you are running commands from the project root directory.

3. **Feature mismatch / ValueError during prediction**:
   - Ensure `feature_columns.json` matches the trained model.
   - Retrain the model if feature order was modified.

4. **Streamlit not launching in browser**:
   - Try opening manually:
     ```
     http://localhost:8501
     ```
   - Check if another process is already using port 8501.

5. **Slow performance or app freezing**:
   - Ensure model caching is enabled (`@st.cache_resource`).
   - Close heavy background applications.
   - Restart Streamlit.

6. **Incorrect risk predictions**:
   - Verify that inputs are within realistic medical ranges.
   - Ensure BMI and pulse pressure calculations are correct.
   - Retrain the model if dataset was modified.

7. **Notebook not rendering properly (EDA issues)**:
   - Install Jupyter:
     ```
     pip install notebook
     ```
   - Run:
     ```
     jupyter notebook
     ```

## 🚀 Future Enhancements

📄 Downloadable AI health report (PDF)

📈 Lifestyle improvement simulator

🧠 SHAP explainability integration

🌐 Streamlit Cloud deployment

📊 Trend tracking dashboard

🔐 User authentication system

## ⚠ Disclaimer

- CardioAI provides AI-based cardiovascular risk estimation.
- It is not a medical diagnosis and should not replace professional medical consultation.

## 📞Support 

If you encounter any issues or have questions:
    Phone Number : +91 9063197036
    Email : bhavanibhavya77@gmail.com