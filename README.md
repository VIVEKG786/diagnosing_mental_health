# Mental Health Diagnosis

## Contributions
- [Kathanshi Jain](https://www.linkedin.com/in/kathanshi-jain/)
- [Utkarsh Sen](https://www.linkedin.com/in/utk-sen/)
## Model Link
[Diagnosing Mental Health](https://diagnosingmentalhealth.streamlit.app/)

## About
The dataset comprises 30 samples for each of the Normal, Mania Bipolar Disorder, Depressive Bipolar Disorder, and Major Depressive Disorder categories, totaling 120 patients. It contains 17 essential symptoms used by psychiatrists to diagnose the described disorders. These symptoms include levels of Sadness, Exhaustion, Euphoria, Sleep disorder, Mood swings, Suicidal thoughts, Anorexia, Anxiety, Try-explaining, Nervous breakdown, Ignore & Move-on, Admitting mistakes, Overthinking, Aggressive response, Optimism, Sexual activity, and Concentration in a Comma Separated Value (CSV) format.

The "Normal" category refers to individuals using therapy time for specialized counseling, personal development, and life skill enrichments. Although these individuals may also have minor mental problems, they differ from those suffering from Major Depressive Disorder and Bipolar Disorder.

## Problem Statement
Classify patients into Normal, Depressed, Bipolar Type-1, and Bipolar Type-2 based on the above mentioned features.

## Models Used
- Logistic Regression
- Support Vector Classifier (SVC)
- K Nearest Neighbors (KNN)
- Random Forest
- Multi-layer Perceptron

## Installation Instructions
Please refer to the `requirements.txt` file for installation instructions.

## Usage Guidelines
- The data file (`data.csv`) contains the dataset in CSV format.
- The `models` directory contains trained machine learning models.
- `classification.ipynb` is a Jupyter Notebook file for data analysis and model training.
- `streamlit_app.py` is a Python file for the Streamlit web application.
