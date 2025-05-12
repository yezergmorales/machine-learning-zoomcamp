import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict

# Assuming the customer data structure - adjust if needed
class Customer(BaseModel):
    gender: str
    seniorcitizen: int
    partner: str
    dependents: str
    tenure: int
    phoneservice: str
    multiplelines: str
    internetservice: str
    onlinesecurity: str
    onlinebackup: str
    deviceprotection: str
    techsupport: str
    streamingtv: str
    streamingmovies: str
    contract: str
    paperlessbilling: str
    paymentmethod: str
    monthlycharges: float
    totalcharges: float


model_file = 'model_C=1.0.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = FastAPI()

@app.get("/ping")
async def ping() -> Dict[str, str]:
    """pong"""
    return {"what??": "pong"}

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post('/predict')
def predict(customer: Customer):
    customer_dict = customer.dict() # Convert Pydantic model to dict
    X = dv.transform([customer_dict])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5

    result = {
        'churn_probability': float(y_pred),
        'churn': bool(churn)
    }

    return result
