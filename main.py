from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Literal
from model import predict_salary, display_metrics
import io
import sys

app = FastAPI(
    title="Employee Salary Prediction API",
    description="Predict employee salary using Linear Regression and XGBoost",
    version="1.0.0"
)


class EmployeeInput(BaseModel):
    experience_years: int
    education_level: Literal["High School", "Bachelor", "Master", "PhD"]
    skills_count: int
    industry: str
    company_size: Literal["Small", "Medium", "Large"]
    location: str
    remote_work: Literal["Yes", "No"]
    certifications: int
    job_title: str

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "experience_years": 8,
                "education_level": "Master",
                "skills_count": 10,
                "industry": "Finance",
                "company_size": "Large",
                "location": "Australia",
                "remote_work": "No",
                "certifications": 2,
                "job_title": "Data Analyst"
            }]
        }
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def get_metrics():
    buffer = io.StringIO()
    sys.stdout = buffer
    display_metrics()
    sys.stdout = sys.__stdout__
    return {"metrics": buffer.getvalue()}


@app.post("/predict")
def predict(
    employee: EmployeeInput,
    model_choice: Literal["lr", "xgb", "both"] = Query(default="both")
):
    result = predict_salary(employee.model_dump(), model_choice=model_choice)
    return {
        "input": employee.model_dump(),
        "model_choice": model_choice,
        "predictions": result
    }