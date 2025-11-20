# Time Series Forecasting Mini Project

![GitHub Repo Size](https://img.shields.io/github/repo-size/rezaghasemi/ts_mini_project)
![Python Version](https://img.shields.io/badge/python-3.10.19-blue)
![License](https://img.shields.io/github/license/rezaghasemi/ts_mini_project)
![MLflow](https://img.shields.io/badge/mlflow-tracking-orange)
![PyTorch](https://img.shields.io/badge/pytorch-2.1.0-red)
![Jupyter](https://img.shields.io/badge/jupyter-notebook-orange)
![Forecasting](https://img.shields.io/badge/forecasting-pytorch--forecasting-green)


A mini time series forecasting project using the classic **Sunspots** dataset from `statsmodels.datasets`. The goal is to implement and compare different forecasting models:  

- Temporal Fusion Transformer (TFT) ✅
- ARIMA
- DeepAR

---

## Project Overview

This project focuses on forecasting sunspot activity using multiple models:

1. **Temporal Fusion Transformer (TFT)** – a deep learning model for interpretable multi-horizon time series forecasting.
2. **ARIMA** – a classical statistical model for univariate time series.
3. **DeepAR** – a probabilistic forecasting model using autoregressive RNNs.

TFT is implemented, here is the result:

### Training Loss Figure (TFT)
<img src="imgs/tft_train_loss.png" alt="TFT Training Loss" width="450"/>

---

## Installation

Clone the repository:

```bash
git clone git@github.com:rezaghasemi/ts_mini_project.git
cd ts_mini_project
```

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Libraries

- **pytorch-forecasting**, **pytorch-lightning** – for deep learning models (TFT, DeepAR)  
- **statsmodels** – for ARIMA and dataset loading  
- **pandas**, **numpy**, **matplotlib** – data handling and visualization  
- **mlflow** – experiment tracking  
- **fastapi** / **flask** – for potential API deployment  


## Project Status

- **TFT implemented** ✅  
- **ARIMA implementation**: Pending  
- **DeepAR implementation**: Pending


## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
