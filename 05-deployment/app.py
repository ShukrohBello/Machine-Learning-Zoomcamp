from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import uvicorn

# Load the pipeline (this will be pipeline_v2.bin in the base Docker image)
from fastapi import FastAPI
with open('pipeline_v1.bin', 'rb') as f:
    pipeline = pickle.load(f)

# Create FastAPI app
app = FastAPI()

class Lead(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

@app.post("/predict")
def predict(lead: Lead):
    # Create feature dictionary for DictVectorizer
    features = {
        'lead_source': lead.lead_source,
        'number_of_courses_viewed': lead.number_of_courses_viewed,
        'annual_income': lead.annual_income
    }
    
    # Make prediction
    probability = pipeline.predict_proba([features])[0][1]
    
    return {"probability": probability}

@app.get("/")
def root():
    return {"message": "Lead Scoring ML Model API"}

@app.get("/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
        uvicorn.run(app, host="0.0.0.0", port=8000)




# from typing import Dict, Any
# import pickle
# from fastapi import FastAPI
# import uvicorn

# app = FastAPI(title="lead-scoring-prediction")

# with open('pipeline_v1.bin', 'rb') as f_in:
#     pipeline = pickle.load(f_in)


# def predict_single(customer):
#     result = pipeline.predict_proba(customer)[0, 1]
#     return float(result)

# @app.post("/predict")
# def predict(customer: Dict[str, Any]):
#     prob = predict_single(customer)

#     return {
#         "lead_score_probability": prob,
#         "converted": bool(prob >= 0.5)
#     }

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=9696)




# import pickle
# from fastapi import FastAPI
# from typing import Dict, Any

# app = FastAPI(title="lead-score-prediction")

# # Load model when server starts
# with open('pipeline_v1.bin', 'rb') as f_in:
#     pipeline = pickle.load(f_in)


# def predict_single(customer):
#     result = pipeline.predict_proba(customer)[0, 1]
#     return float(result)


# @app.post("/predict")
# def predict(customer: Dict[str, Any]):
#     prob = predict_single(customer)
#     return {
#         "lead_score": prob
#     }

