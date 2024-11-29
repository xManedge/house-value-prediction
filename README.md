# House Rent Value Prediction

This repository implements a project to predict **median house rent values** using historical time-series data. The project focuses on analyzing rental trends and using advanced time-series modeling techniques to forecast future values.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Analysis](#analysis)
- [Models](#models)
  - [LSTM](#lstm)
  - [SARIMA](#sarima)
- [Getting Started](#getting-started)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)
- [Acknowledgments](#acknowledgments)

---

## Overview

The project aims to predict median monthly house rent values by leveraging historical trends. By analyzing rental patterns, this tool provides valuable insights for property management companies, renters, and investors to plan ahead.

---

## Dataset
**The dataset is located in the folder  `Dataset`. The initial dataset was  `data.csv`, and the datasets `Philly.csv`, `new york.csv`, and `hoboken` were extracted from that dataset. This extraction was performed in `data analysis.ipynb`.
### Description

The dataset contains:

- **Median Rent Values**: Monthly median house rental prices.
- **Location**: Each location has its own monthly median house rental price for each month.
- **Time Period**: Spanning several years, capturing long-term trends.
- **Frequency**: Monthly records.

### Preprocessing
- Each region gets its own model, the dataset is split to retrieve the monthly median house rental values for each region. There were three regions chosen for this project, Hoboken, New York, and Philadelphia.
- **The file `data analysis.ipynb` has all the preprocessing and data analysis steps. Please check that file out** 
- Checked for missing months and imputed values if necessary.
- Normalized time-series data to improve model performance.
- Decomposed the data into **trend**, **seasonality**, and **residual** components for analysis.

---

## Analysis

Exploratory Data Analysis (EDA) was conducted to understand the data, including:

- **Trend Analysis**: Identification of long-term growth or decline in rental values.
- **Seasonality Detection**: Month-on-month variations and repeating patterns.
- **Visualization**: Time-series plots, decomposition graphs, and correlation heatmaps.

---

## Models

### LSTM (Long Short-Term Memory)

- **The main model used is LSTM. The code is at file  `LSTM.ipynb` please check that code out**. Each region (Hoboken, New York, Philadelphia) has its own LSTM model, and each model is seperately trained on the median house rent value of that region. 
- **Description**: LSTM, a type of recurrent neural network (RNN), excels at modeling sequential data and capturing long-term dependencies.
- **Key Steps**:
  - Prepared sequences using a sliding window approach.
  - Built a multi-layer LSTM network using a deep learning framework (TensorFlow).
  - Trained on the normalized median rent data.
- **Advantages**:
  - Handles complex temporal relationships.
  - Robust against nonlinearities and noise in the data.
- **Metrics**:
  - Root Mean Squared Error (RMSE):
    - Hoboken: 0.000434
    - New York: 2.416 e-5
    - Philadelphia: 0.000116
  - R2 Square:
    - Hoboken: 86.54%
    - New york: 94.2%
    - Philadelphia: 72.15%

### SARIMA (Seasonal Autoregressive Integrated Moving Average)

- **Description**: A traditional time-series model that accounts for trend, seasonality, and noise components.
- **Key Steps**:
  - Decomposed the data to identify seasonal and trend components.
  - Tuned ARIMA parameters (`p`, `d`, `q`) and seasonal parameters (`P`, `D`, `Q`, `s`).
  - Fit the SARIMA model using `statsmodels`.
- **Advantages**:
  - Simple and interpretable.
  - Suitable for data with strong seasonal patterns.

---

## Getting Started

### Requirements

- Python 3.8 or higher
- Libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `tensorflow` or `pytorch`
  - `statsmodels`
  - `scikit-learn`
  - `jupyterlab`

Install dependencies with:
```bash
pip install -r requirements.txt
```

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/house-rent-prediction.git
   cd house-rent-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. Run the analysis notebook to explore the data:
   ```bash
   jupyter notebook analysis.ipynb
   ```

2. Train models:
   - **LSTM**: Run `lstm_model.py`.
   - **SARIMA**: Run `sarima_model.py`.


## Results
The best model in is LSTM by a good margin. It works outstandingly well with the new york dataset and gives very good performance for hoboken and Philadelphia. It is near state of the art performance.

---

## Future Work

- Incorporate additional features such as demographic or economic data.
- Explore hybrid models combining LSTM and SARIMA for improved performance.
- Automate hyperparameter tuning for both LSTM and SARIMA.
- Develop a dashboard for visualizing predictions and trends in real-time.

---

## Acknowledgments

- **Libraries Used**: TensorFlow, PyTorch, Statsmodels, and Scikit-learn.

Feel free to contribute by submitting issues or pull requests. Together, let's make house rent prediction smarter and more efficient!

--- 
