## Machine Learning Zoomcamp Capstone Project
### Heart Failure Prediction

#### Project Brief
Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worlwide.
Heart failure is a common event caused by CVDs and this dataset contains 12 features that can be used to predict mortality by heart failure.
Most cardiovascular diseases can be prevented by addressing behavioural risk factors such as tobacco use, unhealthy diet and obesity, physical inactivity and harmful use of alcohol using population-wide strategies.
People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management wherein a machine learning model can be of great help.

#### Objective: Predict the binary outcome DEATH_EVENT (1 = death during follow-up; 0 = alive) from clinical features.
#### Columns (overview): 
The Heart Failure Clinical Data dataset is compact (299 patients, 13 columns)
age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time, DEATH_EVENT

#### Notes:
time = follow‑up period (days).Can be treated as a standard feature for classification or do a survival analysis extension later.
Expect moderate class imbalance (~30–35% positive).

#### Key Metrics (health context):
* ROC‑AUC (general separation)
* PR‑AUC (more informative with imbalance)
* Recall @ fixed precision (catch high‑risk patients while controlling false alarms)
* Calibration/Brier score (are probabilities trustworthy?)
* Confusion matrix with threshold tuning