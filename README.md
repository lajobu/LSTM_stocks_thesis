# LSTM_stocks_thesis:

This repository is part of the dissertation *Challenging the weak form of market efficiency: a univariate LSTM model to forecast the IBEX 35 index* by Jorge Bueno Perez (@lajobu). The related work is part of the master's thesis of the master studies in Data Science and Business Analytics at the University of Warsaw (Poland). 

# Summary:

The increase in the computational power together with the rapid development in the implementation of deep learning models enable to verify whether some hypotheses are still valid in the finance theory. A **LSTM (long short-term memory) univariate model**, based on recurrent neural networks, is used to forecast financial time series with the aim of predicting returns on the **IBEX 35 index high frequency hourly data for the last quarter of 2020**. A random walk with drift model is utilized for comparison purposes. The results of the statistical Diebold-Mariano test suggest that the weak form of efficient market hypothesis is accepted, due to the fact that the random walk with drift model has significantly better prediction accuracy than the LSTM model.

# Keywords:

deep learning, long short-term memory, recurrent neural networks, financial time series forecasting, IBEX 35 index, high frequency data, Diebold-Mariano test, efficient market hypothesis

# Some insights:

## 1) Model pipeline scheme:

<p align="center">
  <img src="https://github.com/lajobu/LSTM_stocks_thesis/blob/master/figures/Pipeline.png" width="500" height="200" />
</p>

## 2) Model summary:

<p align="center">
  <img src="https://github.com/lajobu/LSTM_stocks_thesis/blob/master/figures/Model1_summary.png" width="300" height="200" />
</p>

## 3) 

<p align="center">
  <img src="https://github.com/lajobu/LSTM_stocks_thesis/blob/master/figures/table_model.png" width="300" height="100" />
</p>

## 4) Training process. RMSE on validation and training samples

<p align="center">
  <img src="https://github.com/lajobu/LSTM_stocks_thesis/blob/master/figures/Model1_train.png" width="500" height="200" />
</p>

## 5) Real and predicted mid-price in test sample

<p align="center">
  <img src="https://github.com/lajobu/LSTM_stocks_thesis/blob/master/figures/Model_predict_test.png" width="500" height="200" />
</p>

## 6) Real and predicted mid-price in training and validation samples

<p align="center">
  <img src="https://github.com/lajobu/LSTM_stocks_thesis/blob/master/figures/Model_predict_val_train.png" width="500" height="400" />
</p>

# Licence

Copyright (c) 2021 Jorge Bueno Perez
