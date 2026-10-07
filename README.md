Multiclass Classification of Fetal Health using Cardiotocogram Data

An End-to-End Machine Learning System for Fetal Health Classification from CTG Data

──────────────────────────────────────────────────────────────────────────────────────────

Project Information

Team Members
1.Kirthana S (PES1UG24AM136)
2.Keerthishree R (PES1UG24AM132)

──────────────────────────────────────────────────────────────────────────────────────────

1. Project Overview

Cardiotocography (CTG) is a technique used to record fetal heart rate and uterine activity. CTG recordings contain multiple numerical characteristics that can provide information about fetal condition.

This project develops an end-to-end supervised machine learning system for classifying CTG records into three fetal health categories:

The project covers data preprocessing, exploratory analysis, model training, model comparison, evaluation, hyperparameter tuning, model persistence, and deployment through a Streamlit application.

──────────────────────────────────────────────────────────────────────────────────────────

2. Problem Statement

Given numerical features extracted from Cardiotocography recordings, the objective is to classify each CTG record into Normal, Suspect, or Pathological fetal health status.

This is formulated as a supervised multiclass classification problem. Because the dataset is imbalanced, evaluation considers accuracy together with class-wise performance and Macro F1-score.

──────────────────────────────────────────────────────────────────────────────────────────

3. Objectives

Perform preprocessing of CTG data.

Select relevant CTG features.

Analyze fetal health class distribution.

Train multiple supervised classification algorithms.

Compare models using appropriate evaluation metrics.

Select and tune the best-performing model.

Evaluate the final model using classification metrics, confusion matrix, and ROC-AUC.

Save the trained model and scaler for reuse.

Develop a Streamlit prediction interface.

Maintain a reproducible project structure.

──────────────────────────────────────────────────────────────────────────────────────────

4. Dataset

The project uses the UCI Cardiotocography dataset. The valid dataset contains 2,126 CTG records and 21 selected input features. The target variable is `NSP`.

Target Classes

Input Features

LB, AC.1, FM.1, UC.1, DL.1, DS.1, DP.1,

ASTV, MSTV, ALTV, MLTV, Width, Min, Max,

Nmax, Nzeros, Mode, Mean, Median, Variance, Tendency

──────────────────────────────────────────────────────────────────────────────────────────

5. Data Preprocessing

5.1 Feature Selection

The 21 selected CTG features were used together with the `NSP` target. The initial selected dataset contained 2,129 rows and 22 columns.

5.2 Invalid Record Removal

Three spreadsheet records did not contain a valid `NSP` target label. These records were removed because supervised learning requires a valid target.

Valid records after removal: 2,126.

5.3 Duplicate Detection and Removal

Exact duplicate records were checked across the selected features and target.

**12 duplicate records** were identified and removed.

Final cleaned records: 2,114.

5.4 Missing Value Verification

After cleaning:

Total missing values = 0

5.5 Train-Test Split

An 80:20 stratified train-test split was used.

5.6 Feature Scaling

`StandardScaler` from Scikit-learn was fitted only on the training data and then applied to both training and testing data to prevent test-set information leakage.

──────────────────────────────────────────────────────────────────────────────────────────

6. Class Distribution

The dataset is imbalanced, with Normal records forming the majority class. Macro F1-score was therefore emphasized during model comparison.

![Class Distribution](results/figures/class_distribution.png)

──────────────────────────────────────────────────────────────────────────────────────────

7. Machine Learning Models

Seven supervised classification algorithms were trained and compared:

1. Logistic Regression

2. Decision Tree

3. Random Forest

4. K-Nearest Neighbors (KNN)

5. Support Vector Machine (SVM)

6. Gradient Boosting

7. Multi-Layer Perceptron (MLP)

──────────────────────────────────────────────────────────────────────────────────────────

8. Model Comparison

Models were compared using Accuracy, Macro F1-score, and Weighted F1-score. The ranking below is based on Macro F1-score.

Gradient Boosting achieved the highest Macro F1-score and test accuracy and was selected as the final model.

──────────────────────────────────────────────────────────────────────────────────────────

9. Final Model

Gradient Boosting Classifier

Final configuration:

n_estimators = 200

learning_rate = 0.1

max_depth = 3

random_state = 42

──────────────────────────────────────────────────────────────────────────────────────────

10. Model Evaluation

Classification Report

The model performs strongly on Normal and Pathological records. Suspect recall is comparatively lower, indicating that some Suspect records are classified as Normal.

──────────────────────────────────────────────────────────────────────────────────────────

11. Confusion Matrix

                    Predicted

                 Normal  Suspect  Pathological

Actual Normal       327       3          0

       Suspect       13      44          1

       Pathological   1       0         34

The main classification error occurs between the Suspect and Normal classes.

![Gradient Boosting Confusion Matrix](results/figures/gradient_boosting_confusion_matrix.png)

──────────────────────────────────────────────────────────────────────────────────────────

12. Multiclass ROC-AUC

![Multiclass ROC Curve](results/figures/gradient_boosting_roc_curve.png)

──────────────────────────────────────────────────────────────────────────────────────────

13. Hyperparameter Tuning

`GridSearchCV` with 3-fold cross-validation was used with Macro F1-score as the optimization metric.

Search space:

n_estimators: 100, 200

learning_rate: 0.05, 0.1

max_depth: 2, 3

Best parameters:

Best Cross-Validation Macro F1 = 0.9056

The tuned model achieved the same test Macro F1-score of 0.9281, confirming the selected configuration on the held-out test set.

──────────────────────────────────────────────────────────────────────────────────────────

14. Model Persistence

The trained artifacts are stored using Joblib:

models/fetal_health_gradient_boosting.pkl

models/fetal_health_scaler.pkl

These artifacts allow the trained pipeline to be reused without retraining.

──────────────────────────────────────────────────────────────────────────────────────────

15. Streamlit Application

A Streamlit application provides an interactive prediction interface.

The application:

1. Loads the saved Gradient Boosting model.

2. Loads the saved feature scaler.

3. Accepts the 21 CTG input features.

4. Applies the same preprocessing used during training.

5. Generates a prediction.

6. Displays the predicted fetal health class.

Run the Application

streamlit run app/app.py

Prediction Outputs

──────────────────────────────────────────────────────────────────────────────────────────

16. Project Structure

Fetal-Health-CTG-Classification/

├── app/

│   └── app.py

├── data/

│   ├── processed/

│   │   └── fetal_health_clean.csv

│   └── raw/

│       └── CTG.xls

├── models/

│   ├── fetal_health_gradient_boosting.pkl

│   └── fetal_health_scaler.pkl

├── notebooks/

│   └── 01_ctg_fetal_health_classification.ipynb

├── results/

│   ├── figures/

│   │   ├── class_distribution.png

│   │   ├── gradient_boosting_confusion_matrix.png

│   │   └── gradient_boosting_roc_curve.png

│   └── metrics/

│       └── model_comparison.csv

├── .gitignore

├── README.md

└── requirements.txt

──────────────────────────────────────────────────────────────────────────────────────────

17. Technologies Used

**Python**

**Pandas**

**NumPy**

**Scikit-learn**

**Matplotlib**

**Seaborn**

**Joblib**

**Streamlit**

**Jupyter Notebook**

**Visual Studio Code**

**Git**

**GitHub**

──────────────────────────────────────────────────────────────────────────────────────────

18. Installation and Setup

Clone the Repository

git clone https://github.com/kirthana17/Fetal-Health-CTG-Classification.git

cd Fetal-Health-CTG-Classification

Create a Virtual Environment

Windows:

python -m venv .venv

.venv\Scripts\Activate.ps1

The virtual environment isolates this project's Python dependencies from other projects on the system.

Install Dependencies

pip install -r requirements.txt

──────────────────────────────────────────────────────────────────────────────────────────

19. Running the Notebook

The complete machine learning workflow is available at:

notebooks/01_ctg_fetal_health_classification.ipynb

The notebook includes:

Data loading

Data cleaning

Duplicate detection

Class distribution analysis

Train-test split

Feature scaling

Model training

Model comparison

Classification report

Confusion matrix

ROC-AUC evaluation

Hyperparameter tuning

Model persistence

Launch Jupyter Notebook:

jupyter notebook

──────────────────────────────────────────────────────────────────────────────────────────

20. Results and Artifacts

Figures

results/figures/class_distribution.png

results/figures/gradient_boosting_confusion_matrix.png

results/figures/gradient_boosting_roc_curve.png

Metrics

results/metrics/model_comparison.csv

Model Artifacts

models/fetal_health_gradient_boosting.pkl

models/fetal_health_scaler.pkl

──────────────────────────────────────────────────────────────────────────────────────────

21. Model Validation

A held-out test record was used to validate the saved model pipeline.

Actual class: **Normal**

Predicted class: **Normal**

This confirms that the saved model and preprocessing pipeline can be loaded and used for independent prediction.

──────────────────────────────────────────────────────────────────────────────────────────

22. Key Findings

1. The dataset is imbalanced toward the Normal class.

2. Macro F1-score is important because of the class imbalance.

3. Gradient Boosting achieved the strongest overall performance.

4. The final model achieved **95.74% test accuracy**.

5. The final model achieved **0.9281 Macro F1-score**.

6. The final model achieved **0.9775 Macro ROC-AUC**.

7. The main classification challenge was distinguishing Suspect from Normal records.

8. Hyperparameter tuning confirmed the selected Gradient Boosting configuration.

──────────────────────────────────────────────────────────────────────────────────────────

23. Limitations

The dataset is imbalanced, with Normal records forming the majority class.

Results are based on the available CTG dataset and do not constitute clinical validation.

The project is an academic machine learning demonstration.

The Streamlit application is not intended to provide clinical decision support or medical recommendations.

──────────────────────────────────────────────────────────────────────────────────────────

24. Future Enhancements

More extensive hyperparameter optimization.

Additional ensemble learning methods.

More systematic class-imbalance handling.

Feature importance and model interpretability analysis.

Explainable AI techniques such as SHAP.

Enhanced Streamlit visualization and prediction reporting.

Evaluation on additional independent datasets.

Docker-based containerization.

Cloud deployment.

──────────────────────────────────────────────────────────────────────────────────────────

25. Reproducibility

The repository preserves the source code, notebook, dataset files, processed dataset, evaluation results, visualization outputs, trained model, scaler, dependency specification, and Streamlit application.

The required Python dependencies are specified in:

requirements.txt

──────────────────────────────────────────────────────────────────────────────────────────

26. Conclusion

This project demonstrates an end-to-end machine learning workflow for multiclass fetal-health classification using Cardiotocography data.

Seven supervised machine learning algorithms were trained and evaluated. Gradient Boosting achieved the best overall performance based on Macro F1-score and test accuracy.

The final model achieved:

**95.74% Accuracy**

**0.9281 Macro F1-score**

**0.9557 Weighted F1-score**

**0.9775 Macro ROC-AUC**

The trained model and scaler were saved and integrated into a Streamlit application, providing a complete workflow from CTG data preprocessing to model-based prediction.

──────────────────────────────────────────────────────────────────────────────────────────

27. Repository
