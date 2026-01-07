# Machine Learning Zoomcamp Capstone Project
## Heart Failure Prediction

### Project Brief
Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worlwide.
Heart failure is a common event caused by CVDs and this dataset contains 12 features that can be used to predict mortality by heart failure.
Most cardiovascular diseases can be prevented by addressing behavioural risk factors such as tobacco use, unhealthy diet and obesity, physical inactivity and harmful use of alcohol using population-wide strategies.
People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management wherein a machine learning model can be of great help.

This machine learning project  predicts risk of death events for heart failure patients using clinical features. This project includes:

* A training script (train.py) that builds and evaluates a Logistic Regression pipeline.
* A FastAPI service (predict.py) that serves predictions.
* A client script (request.py) to test the API.

>⚠️ Disclaimer: This project is for educational purposes only and should not be used for clinical decision-making.

#### Objective: Predict the binary outcome DEATH_EVENT (1 = death during follow-up; 0 = alive) from clinical features.
#### Columns (overview): 
The Heart Failure Clinical Data dataset is compact (299 patients, 13 columns)
* Features: age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time

* Target: 
DEATH_EVENT → renamed to death_event.


### Notes:
time = follow‑up period (days).Can be treated as a standard feature for classification or do a survival analysis extension later.
Expect moderate class imbalance (~30–35% positive).


### Training
train.py loads the CSV, standardizes the target column name, builds a scikit-learn Pipeline:

DictVectorizer (transforms Python dicts to numeric feature arrays)
LogisticRegression (liblinear, penalty='l1', class_weight='balanced')

It then trains with a stratified 80/20 split, prints test metrics, and saves the pipeline to model.bin.
Run training:
Shellpython train.pyShow more lines
Output (example):

Confusion matrix, accuracy, ROC-AUC, classification report
Saved pipeline: model.bin

### API (FastAPI)
predict.py loads model.bin at startup and exposes:

GET / → service status
POST /predict → prediction endpoint


### Key Metrics (health context):
* ROC-AUC: separation quality between positive/negative classes
* Accuracy: overall correctness
* Confusion Matrix: counts TP/FP/TN/FN
* Classification Report: precision, recall, F1 per class


For healthcare framing, consider prioritizing recall (catching high-risk cases) and tune thresholds if desired.


