# Multiclass Classification of Fetal Health using Cardiotocogram Data

<p align="center">

**An End-to-End Machine Learning System for Fetal Health Classification from Cardiotocogram (CTG) Data**

</p>

<p align="center">

Classifies CTG records into **Normal**, **Suspect**, and **Pathological** fetal health categories using supervised machine learning.

</p>

---

## 📌 Project Overview

Cardiotocography (CTG) is a commonly used technique for monitoring fetal heart rate and uterine activity. A CTG recording contains multiple numerical characteristics that can provide information about fetal condition.

This project develops an **end-to-end multiclass machine learning classification system** that uses CTG measurements to classify fetal health into three categories:

| Class | Fetal Health Status |
|:---:|---|
| **1** | Normal |
| **2** | Suspect |
| **3** | Pathological |

Seven supervised machine learning algorithms were trained and evaluated using the same dataset. Their performance was compared using **Accuracy, Macro F1-score, and Weighted F1-score**.

The **Gradient Boosting Classifier** achieved the strongest overall performance and was selected as the final model. The trained model and preprocessing scaler were saved and integrated into an interactive **Streamlit application**.

> ⚠️ **Academic Disclaimer:** This project is developed strictly for academic and educational purposes. It is not intended for clinical diagnosis, medical decision-making, or direct healthcare use.

---

# 🎯 Objectives

The main objectives of this project are:

1. To preprocess and clean Cardiotocogram data for machine learning.
2. To analyze the distribution of the fetal health classes.
3. To develop a multiclass classification pipeline using 21 CTG features.
4. To train and compare seven different supervised machine learning algorithms.
5. To evaluate model performance using multiple classification metrics.
6. To account for class imbalance using metrics such as Macro F1-score and per-class recall.
7. To identify the strongest-performing classification algorithm.
8. To perform hyperparameter tuning using cross-validation.
9. To evaluate the final model using a classification report, confusion matrix, and multiclass ROC-AUC.
10. To save the trained model and scaler for reproducible predictions.
11. To deploy the final model through a Streamlit web application.

---

# 📊 Dataset

The project uses the **UCI Cardiotocography (CTG) dataset**.

The dataset contains numerical measurements derived from Cardiotocography recordings together with fetal health classifications.

During preprocessing, invalid spreadsheet artifact rows without valid target labels were removed, followed by exact duplicate detection and removal.

## Final Dataset Statistics

| Property | Value |
|---|---:|
| Valid records before duplicate removal | 2,126 |
| Exact duplicate records removed | 12 |
| Final records | **2,114** |
| Input features | **21** |
| Target classes | **3** |
| Missing values after cleaning | **0** |
| Duplicate records after cleaning | **0** |

---

# 🧬 Input Features

The final classification model uses the following **21 CTG features**:

```text
LB
AC.1
FM.1
UC.1
DL.1
DS.1
DP.1
ASTV
MSTV
ALTV
MLTV
Width
Min
Max
Nmax
Nzeros
Mode
Mean
Median
Variance
Tendency

Target Variable
The target variable is:
NSP

with the following class mapping:
1 → Normal
2 → Suspect
3 → Pathological

⚖️ Class Distribution
The cleaned dataset is substantially imbalanced.
Class	Category	Records	Percentage
1	Normal	1,647	77.91%
2	Suspect	292	13.81%
3	Pathological	175	8.28%
	Total	2,114	100%


Class Distribution Visualization

Because the Normal class represents the majority of the dataset, Accuracy alone is not sufficient for evaluating the model. Therefore, Macro F1-score, per-class precision/recall, confusion matrix, and ROC-AUC were also considered.
🏗️ Machine Learning Workflow
The project follows a complete machine learning pipeline:
                         CTG Dataset
                              │
                              ▼
                       Data Selection
                              │
                              ▼
                        Data Cleaning
                              │
                  ┌───────────┴───────────┐
                  │                       │
           Invalid Rows              Duplicates
             Removed                  Removed
                  │                       │
                  └───────────┬───────────┘
                              ▼
                    Exploratory Analysis
                              │
                              ▼
                   Feature / Target Split
                              │
                              ▼
                    Stratified 80/20 Split
                              │
                              ▼
                       Feature Scaling
                              │
                              ▼
                  ┌───────────────────────┐
                  │   Seven ML Models     │
                  └───────────────────────┘
                              │
                              ▼
                     Model Comparison
                              │
                              ▼
                  Gradient Boosting Selected
                              │
                              ▼
                    Hyperparameter Tuning
                              │
                              ▼
                      Final Evaluation
                              │
              ┌───────────────┼────────────────┐
              ▼               ▼                ▼
       Classification     Confusion        ROC-AUC
          Report           Matrix          Analysis
              │               │                │
              └───────────────┼────────────────┘
                              ▼
                    Model + Scaler Saved
                              │
                              ▼
                    Streamlit Application


🧹 Data Preprocessing
The following preprocessing operations were performed.
1. Feature Selection
The 21 relevant CTG features were selected together with the NSP target variable.
2. Invalid Record Removal
Rows without a valid NSP target label were removed because supervised classification requires a known target.
This resulted in:
2,126 valid CTG records
3. Duplicate Detection
Exact duplicate rows were checked across the selected features and target.
A total of:
12 exact duplicate records
were identified.
4. Duplicate Removal
The duplicate records were removed, resulting in:
2,114 final records
5. Missing Value Verification
After cleaning:
Total missing values = 0
6. Train-Test Split
The dataset was divided into:
80% → Training set
20% → Testing set
Resulting dataset sizes:
Training samples = 1,691
Testing samples  =   423
Stratified sampling was used to preserve the relative distribution of the three target classes.
7. Feature Scaling
StandardScaler from Scikit-learn was used to standardize the numerical features.
The scaler was fitted only on the training data and then applied to both training and testing data to prevent test-data leakage.

🤖 Machine Learning Models
Seven supervised classification algorithms were trained and compared:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. K-Nearest Neighbors (KNN)
5. Support Vector Machine (SVM)
6. Gradient Boosting
7. Multi-Layer Perceptron (MLP)
Each model was evaluated on the same unseen test set.

📈 Model Comparison
The following results were obtained from the test set:
Rank	Model	Accuracy	Macro F1	Weighted F1
🥇	Gradient Boosting	95.74%	92.81%	95.57%
🥈	Random Forest	94.09%	90.09%	94.09%
🥉	Decision Tree	92.91%	87.92%	92.81%
4	MLP	92.67%	85.46%	92.28%
5	KNN	89.60%	78.83%	88.98%
6	SVM	89.36%	83.88%	90.18%
7	Logistic Regression	87.47%	79.67%	88.32%

The complete numerical comparison is available at:
results/metrics/model_comparison.csv

Model Selection
Gradient Boosting achieved the highest overall:
- Accuracy
- Macro F1-score
- Weighted F1-score
Therefore, Gradient Boosting was selected as the final classification model.

🏆 Final Model Performance
The final model is a:
Gradient Boosting Classifier

Test Set Results
Metric	Score
Accuracy	95.74%
Macro F1-score	92.81%
Weighted F1-score	95.57%
Macro ROC-AUC	97.75%


These results demonstrate strong classification performance on the held-out test set.

🔬 Classification Report
The final Gradient Boosting model produced the following class-wise performance:
Class	Precision	Recall	F1-score	Support
Normal	0.96	0.99	0.97	330
Suspect	0.94	0.76	0.84	58
Pathological	0.97	0.97	0.97	35
Macro Average	0.96	0.91	0.93	423
Weighted Average	0.96	0.96	0.96	423


Interpretation
The model performs strongly on the Normal and Pathological classes.
The primary area for improvement is the Suspect class, which achieved:
Recall = 0.76
F1-score = 0.84

This indicates that some Suspect cases were incorrectly classified, with the majority of these errors being predictions of the Normal class.

🔲 Confusion Matrix
The confusion matrix for the final Gradient Boosting model is:
                     Predicted
                 Normal  Suspect  Pathological

Actual Normal      327       3          0
Actual Suspect      13      44          1
Actual Pathological  1       0         34

Visualization

The largest classification error is:
Actual Suspect → Predicted Normal = 13 cases
This is consistent with the lower recall observed for the Suspect class.

📈 Multiclass ROC-AUC Analysis
A One-vs-Rest ROC analysis was performed for all three target classes.
Class	ROC-AUC
Normal	0.9783
Suspect	0.9547
Pathological	0.9993
Macro ROC-AUC	0.9775

ROC Curves
The final model achieved a Macro ROC-AUC of 0.9775, indicating strong class-separation performance on the test set.

🔧 Hyperparameter Tuning
The Gradient Boosting model was further evaluated using GridSearchCV with 3-fold cross-validation.
The following hyperparameter combinations were explored:
n_estimators:
100, 200

learning_rate:
0.05, 0.1

max_depth:
2, 3

Best Configuration
n_estimators = 200
learning_rate = 0.1
max_depth = 3

The best cross-validation Macro F1-score obtained during the search was:
0.9056

The selected configuration achieved a test Macro F1-score of:
0.9281

The tuning process therefore confirmed the selected Gradient Boosting configuration rather than producing a measurable improvement over the existing configuration.

💾 Saved Model Artifacts
The trained components required for prediction are stored in the models/ directory:
models/
├── fetal_health_gradient_boosting.pkl
└── fetal_health_scaler.pkl

Gradient Boosting Model
fetal_health_gradient_boosting.pkl
Contains the trained final Gradient Boosting classifier.
Feature Scaler
fetal_health_scaler.pkl
Contains the fitted StandardScaler used during model preprocessing.
Both artifacts are required by the Streamlit application.

🌐 Streamlit Application
The trained model has been integrated into an interactive Streamlit application.
The application allows a user to enter the 21 CTG measurements and obtain a predicted fetal health category.
Application Flow
21 CTG Measurements
        │
        ▼
Input DataFrame
        │
        ▼
Feature Ordering
        │
        ▼
Saved StandardScaler
        │
        ▼
Saved Gradient Boosting Model
        │
        ▼
Fetal Health Prediction
        │
        ▼
Normal / Suspect / Pathological

The application displays the result as:
🟢 Normal Fetal Health

🟡 Suspect Fetal Health

🔴 Pathological Fetal Health

The Streamlit application is intended only as an academic demonstration of the trained machine learning pipeline.

📁 Project Structure
fetal-health-ctg-classification/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   │   └── CTG.xls
│   │
│   └── processed/
│       └── fetal_health_clean.csv
│
├── models/
│   ├── fetal_health_gradient_boosting.pkl
│   └── fetal_health_scaler.pkl
│
├── notebooks/
│   └── 01_ctg_fetal_health_classification.ipynb
│
├── results/
│   ├── figures/
│   │   ├── class_distribution.png
│   │   ├── gradient_boosting_confusion_matrix.png
│   │   └── gradient_boosting_roc_curve.png
│   │
│   └── metrics/
│       └── model_comparison.csv
│
├── .gitignore
├── README.md
└── requirements.txt

🛠️ Technologies Used
Category	Technologies
Programming Language	Python
Data Processing	Pandas, NumPy
Machine Learning	Scikit-learn
Data Visualization	Matplotlib, Seaborn
Model Persistence	Joblib
Interactive Deployment	Streamlit
Development Environment	VS Code, Jupyter Notebook


🚀 Installation and Setup
Prerequisites
Make sure Python is installed on your system.
Python 3.12 was used during project development.
1. Clone the Repository
git clone https://github.com/kirthana17/Fetal-Health-CTG-Classification
cd fetal-health-ctg-classification

Replace git clone https://github.com/kirthana17/Fetal-Health-CTG-Classification with the URL of the project repository.

2. Create a Virtual Environment
python -m venv .venv

3. Activate the Virtual Environment
Windows PowerShell
.venv\Scripts\Activate.ps1

Windows Command Prompt
.venv\Scripts\activate

4. Install Dependencies
pip install -r requirements.txt

▶️ Running the Streamlit Application
From the project root directory:
streamlit run app/app.py

The Streamlit server will start locally.
Open the displayed local URL in your browser to access the application.

📓 Running the Machine Learning Notebook
The complete machine learning workflow is documented in:
notebooks/01_ctg_fetal_health_classification.ipynb

The notebook includes:
- Dataset loading
- Feature selection
- Data cleaning
- Missing-value verification
- Duplicate detection
- Class distribution analysis
- Train-test splitting
- Feature scaling
- Training of seven ML models
- Model comparison
- Gradient Boosting selection
- Classification report
- Confusion matrix
- Hyperparameter tuning
- ROC-AUC analysis
- Model persistence
- Prediction validation
To launch Jupyter Notebook:
jupyter notebook

Alternatively, open the notebook directly through VS Code using the configured Python environment.

📊 Results and Evaluation Artifacts
The project stores generated evaluation artifacts under:
results/

Figures
results/figures/
├── class_distribution.png
├── gradient_boosting_confusion_matrix.png
└── gradient_boosting_roc_curve.png

Metrics
results/metrics/
└── model_comparison.csv

These files provide reusable results for the project report, presentation, and analysis.

🧪 Model Validation
A real record from the held-out test set was passed through the saved preprocessing and final model pipeline as a validation check.
The validation demonstrated that the saved model and scaler can successfully process an unseen CTG record and produce a prediction.

⚠️ Limitations
Despite the strong test-set performance, the following limitations should be considered:
1. The dataset is substantially imbalanced, with Normal cases representing 77.91% of the final dataset.
2. The Suspect class has lower recall compared with the Normal and Pathological classes.
3. Evaluation was performed using the available CTG dataset and does not establish performance on an independent external dataset.
4. The model's performance may depend on the characteristics and distribution of the training dataset.
5. The Streamlit application is intended for demonstration and educational purposes only.
6. The predictions should not be interpreted as clinical diagnoses.
7. Additional clinical validation would be required before considering any real-world healthcare application.

🔮 Future Enhancements
Possible future improvements include:
- Broader hyperparameter optimization.
- Investigation of additional ensemble and boosting methods.
- More systematic class-imbalance handling.
- External dataset validation.
- More extensive cross-validation.
- Feature importance visualization.
- Explainable AI using techniques such as SHAP.
- Prediction probability visualization in the Streamlit interface.
- Improved input validation and user interface design.
- Robustness testing under different data distributions.
- Integration with additional clinical decision-support research workflows.

📌 Key Results at a Glance
Component	Result
Final Dataset	2,114 records
Input Features	21
Target Classes	3
Models Compared	7
Train/Test Split	80/20
Best Model	Gradient Boosting
Test Accuracy	95.74%
Macro F1-score	92.81%
Weighted F1-score	95.57%
Macro ROC-AUC	97.75%
Deployment	Streamlit

👥 Project Information
Information	Details
Course	UE24CS352A — Machine Learning
Project Title	Multiclass Classification of Fetal Health using Cardiotocogram Data
Team Member 1	Kithana S — PES1UG24AM136
Team Member 2	KeerthiShree R — PES1UG24AM132
Repository	Private GitHub Repository
Application	Streamlit
Programming Language	Python

✅ Conclusion
This project demonstrates a complete machine learning workflow for multiclass classification of fetal health from Cardiotocogram data.
Seven machine learning algorithms were trained and compared, with Gradient Boosting achieving the strongest overall test performance.
The final model achieved:
- 95.74% Test Accuracy
- 92.81% Macro F1-score
- 95.57% Weighted F1-score
- 97.75% Macro ROC-AUC
The trained model was persisted together with its preprocessing scaler and integrated into a Streamlit application, providing an end-to-end demonstration from CTG data preprocessing and model training to interactive prediction.