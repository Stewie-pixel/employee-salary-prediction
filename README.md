<div align='center'>
    
# Employee Salary Prediction

![Docker CI](https://github.com/Stewie-pixel/employee-salary-prediction/actions/workflows/docker-image.yml/badge.svg)
[![Dataset](https://img.shields.io/badge/Dataset-250k%20rows-blue?logo=kaggle)](https://www.kaggle.com)
[![XGBoost R²](https://img.shields.io/badge/XGBoost%20R²-0.4454-success?logo=python)](https://github.com/Stewie-pixel/employee-salary-prediction)

A machine learning pipeline to predict employee salaries using Linear Regression and XGBoost, packaged as a Docker container for consistent and reproducible execution.

</div>

## Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [Pipeline](#pipeline)
5. [Models](#models)
6. [Getting Started](#getting-started)
7. [Running with Docker](#running-with-docker)
8. [CI/CD](#cicd)
9. [Tech Stack](#tech-stack)


## Overview

This project builds an end-to-end salary prediction pipeline covering data exploration, preprocessing, feature engineering, model training, evaluation and deployment. Two models are trained and compared — Linear Regression as the baseline and XGBoost as the comparison model.


## Dataset

| Property | Value |
|---|---|
| Source | `data/employee_salary_dataset.csv` |
| Rows | 250,000 |
| Columns | 10 |
| Target | `salary` |

**Features:**

| Feature | Type | Description |
|---|---|---|
| `experience_years` | Numerical | Years of work experience |
| `education_level` | Categorical | High School / Bachelor / Master / PhD |
| `skills_count` | Numerical | Number of skills |
| `industry` | Categorical | Industry sector |
| `company_size` | Categorical | Small / Medium / Large |
| `location` | Categorical | Country of employment |
| `remote_work` | Categorical | Yes / Hybrid / No |
| `certifications` | Numerical | Number of certifications held |
| `job_title` | Categorical | Employee job title |


## Project Structure

```
employee-salary-prediction/
├── .github/
│   └── workflows/
│       └── docker-ci.yml
├── data/
│   └── employee_salary_dataset.csv
├── models/
│   ├── lr_model.pkl
│   ├── xgb_model.pkl
│   ├── feature_names.pkl
│   ├── encoders.pkl
│   ├── metrics.pkl
│   └── split_data.pkl
├── src/
│   └── linear-regression/
│       └── prediction.ipynb
├── main.py
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── requirements.txt
└── .vscode/
    └── settings.json
```


## Pipeline

The full pipeline is documented in `src/linear-regression/prediction.ipynb` across 9 sections:

1. **Import Libraries & Load Data** — load and inspect the dataset
2. **Data Overview & Initial Inspection** — shape, dtypes, missing values, duplicates
3. **Exploratory Data Analysis** — salary distribution, correlations, boxplots, scatter plots
4. **Data Preprocessing & Cleaning** — encoding, outlier removal via IQR
5. **Feature Engineering** — derived features to improve model performance
6. **Modelling** — train Linear Regression and XGBoost
7. **Evaluation** — MAE, RMSE, R², residuals, cross validation, error distribution
8. **Save Pipeline Artifacts** — export all models and encoders to `/models` as `.pkl` files

**Engineered Features:**

| Feature | Description |
|---|---|
| `exp_per_skill` | Experience efficiency — years per skill |
| `cert_per_year` | Learning rate — certifications per year |
| `is_senior` | 1 if experience >= 10 years |
| `tech_role` | 1 if role is a technical position |
| `seniority_tier` | Junior / Mid / Senior / Lead / Principal |


## Models

### Linear Regression
Baseline model chosen for interpretability. Trained on all numeric features after preprocessing and feature engineering.

### XGBoost
Gradient boosting model trained on the same feature set as Linear Regression. Captures non-linear relationships that Linear Regression misses.

### Model Comparison

| Model | R² | MAE | RMSE |
|---|---|---|---|
| Linear Regression | 0.4163 | $21,561 | $27,500 |
| XGBoost | 0.4454 | $20,856 | $26,804 |

XGBoost outperforms Linear Regression across all metrics by capturing non-linear salary patterns.


## Getting Started

**Prerequisites:**
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- Docker Desktop

**Install dependencies:**
```bash
uv sync
```

**Run the model locally:**
```bash
python main.py
```

**Run the full notebook:**

Open `src/linear-regression/prediction.ipynb` in VS Code with the Jupyter extension and run all cells via `Kernel → Restart & Run All`.


## Running with Docker

**Build the image:**
```bash
docker build -t employee-salary-prediction .
```

**Run with Docker Compose:**
```bash
docker-compose up
```

**Run directly:**
```bash
docker run -v ./models:/app/models employee-salary-prediction
```

The `models/` directory is mounted as a volume so updated `.pkl` files are picked up without rebuilding the image.


## CI/CD

This project uses GitHub Actions for continuous integration and delivery.

**Workflow: `.github/workflows/docker-ci.yml`**

| Trigger | Action |
|---|---|
| Push to `main` | Build → Test → Push to Docker Hub |
| Pull Request to `main` | Build → Test only |

**Pipeline steps:**
1. Checkout repository
2. Set up Docker Buildx
3. Build Docker image tagged with `github.sha` and `latest`
4. Verify image exists
5. Run container and validate predictions output
6. Push to Docker Hub on merge to `main` only

**Docker Hub image:**
```
stewie-pixel/employee-salary-prediction:latest
```

**Required GitHub Secrets:**

| Secret | Description |
|---|---|
| `DOCKER_USERNAME` | Docker Hub username |
| `DOCKER_PASSWORD` | Docker Hub access token |


## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.11 | Core language |
| pandas | Data manipulation |
| numpy | Numerical computing |
| scikit-learn | Preprocessing and Linear Regression |
| XGBoost | Gradient boosting model |
| seaborn / matplotlib | Data visualization |
| Jupyter Notebook | Exploratory pipeline |
| Docker | Containerization |
| GitHub Actions | CI/CD pipeline |
| uv | Package management |
| VS Code | Development environment |
