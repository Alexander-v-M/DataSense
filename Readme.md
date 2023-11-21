# DataSense
Developed for an internship with Wetterskip Fryslân, this version of DataSense has disabled features and placeholder data due to confidentiality requirements.

## Overview

DataSense is a Python Time Series Forecasting (TSF) Framework is a versatile and customizable tool for time series forecasting and predictive modeling. This framework provides a set of machine learning and deep learning modules, functions, and tools for loading, preprocessing, training, evaluating, and making predictions on time series data.

## Features

- **Model Training:** Train a wide range of time series forecasting models, including ARIMA, SARIMA, RNN, LSTM \& GRU.

- **Model Evaluation:** Evaluate model performance using various metrics and visualization tools.

- **Predictions:** Make future predictions using trained models and save forecasts to different formats.

## Installation

Before usage, install Python v3.10+

1. Create Virtual Environment
```bash
python -m venv venv
```
2. Activate the Virtual Environment
```bash
venv\Scripts\activate
```
3. Install project requirements
```bash
pip install -r requirements.txt
```

## Usage

Navigate to the `DataSense` directory and access the `scripts` folder to find Python scripts for machine learning, deep learning, and baseline modules. These scripts are designed for training, testing, and generating predictions and forecasts based on the provided data. You can customize the parameters within the pipelines to suit your specific requirements for the data generation process.

`dl_main.py`: Train and use deep learning models for forecasting purposes. (RNN, LSTM, GRU)

`ml_main.py`: Train and use machine learning models for forecasting purposes. (ARIMA, SARIMA)

`ml_baseline_main.py`: Train a linear regression baseline on given data


## Contact

For questions, issues, and feedback, please reach out to Alexander van Meekeren at [alexandervanmeekeren@gmail.com]().