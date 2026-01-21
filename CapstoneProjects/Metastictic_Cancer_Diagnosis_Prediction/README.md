### Metastic Cancer Diagnosis Prediction
> Please note that the model.plk file was included in .gitignore because the file was too large and I was unable to push it to the repo.

### Overview: About the Dataset
Gilead Sciences providea a rich, real-world dataset which contains information about demographics, diagnosis and treatment options, and insurance provided about patients who were diagnosed with breast cancer. The dataset originated from Health Verity, one of the largest healthcare data ecosystems in the US. It was enriched with third party geo-demographic data to provide views into the socio economic aspects that may contribute to health equity. It is icluded in the repo as test.csv and train.csv and it can also be found at [Dataset](https://www.kaggle.com/competitions/widsdatathon2024-challenge2/data)

### Goal:
Predict the duration of time (number of days) it takes for patients to receive metastatic cancer diagnosis.

### Why is this important?
Metastatic TNBC is considered the most aggressive TNBC and requires urgent and timely treatment. Unnecessary delays in diagnosis and subsequent treatment can have devastating effects in these difficult cancers. Differences in the wait time to get treatment is a good proxy for disparities in healthcare access.

The primary goal of building these models is to detect relationships between demographics of the patient with the likelihood of getting timely treatment. The secondary goal is to see if climate patterns impact proper diagnosis and treatment.Gilead Sciences, Inc. is a biopharmaceutical company that specializes in the research, development, and commercialization of medicines, primarily in the fields of virology, infectious diseases, oncology, and other areas of unmet medical need.

### Metastatic Cancer Diagnosis Period Prediction (Regression, RMSE)
This project implements a clean, production‑style ML workflow (ML Zoomcamp pattern) to predict the number of days to metastatic cancer diagnosis for breast cancer patients, using a real‑world dataset (demographics, payer, geo‑demographics, and climate features).

> ⚠️ Educational use only. This model is not a clinical device and must not be used for medical decision‑making.

### Objective

**Task:** Predict metastatic_diagnosis_period (days) for each patient in test.csv.
**Metric:** RMSE (Root Mean Squared Error) on the validation set.
```Submission format:
patient_id,metastatic_diagnosis_period
372069,125
981264,78
```
> metastatic_diagnosis_period should be an integer number of days.

### Project Structure
```
.
├── app.py        # FastAPI service exposing /predict (by patient_id)
├── train.py      # Train pipeline & save model.pkl
├── predict.py    # Local single-patient inference by patient_id
├── request.py    # Client script to call the FastAPI endpoint
├── Dockerfile    # Container for serving the API
├── requirements.txt  # Python dependencies
├── model.pkl     # Trained pipeline artifact (created by train.py)
├── train.csv
└── test.csv

### Environment management

Initialize the project with 

```pip install uv 

    uv init 

    uv add numpy pandas scikit-learn fastapi uvicorn 

    uv lock

    uv run gunicorn -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:9696 predict:

 ```   
Creates new files
* main.py
* .python-version pyproject.toml 
* .venv with all the packages 

When you get a fresh copy of a project that already uses uv, you can install all the dependencies using the sync command:

```uv sync```

### Docker Containerisation
```
FROM python:3.13.5-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked


COPY predict.py model.pkl  ./ 

EXPOSE 9696

CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]
```


```docker build - # docker build -t cancer-predictor .```

```docker run -it --rm -p 9696:9696 cancer-predictor```