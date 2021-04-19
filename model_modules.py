# packages
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
import sklearn
from pathlib import Path
import time
import os
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
import matplotlib
import pydot
from keras.utils import plot_model
import matplotlib.dates as mdates
from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange 
import statsmodels
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
import statsmodels.tsa.api as tsa
from scipy.stats import probplot, moment
import statsmodels.api as sm
from arch.unitroot import PhillipsPerron, KPSS

# for general tuning
class prepare_LSTM:
    def __init__(self, raw_data_loc):
        self.raw_data_loc= raw_data_loc 
    def data_cleaning(self):
        xformatter = mdates.DateFormatter('%d.%b.%Y')
        data = pd.read_csv(self.raw_data_loc)
        # cleaning trading hours - normal days
        data["Local time"]= data["Local time"].astype(str).str[0:16]
        sel= [] 
        for i in range(len(data)):
            time= int(data['Local time'][i][11:13] + data['Local time'][i][14:16])
            if 900 <= time <= 1700:
                sel.append(i)  
        data= data.iloc[sel]
        data= data.reset_index(drop=True)
        # cleaning trading hours - special days
        spec_days = [] 
        for i in range(len(data)):
            if data['Local time'][i][0:5] == '31.12' or data['Local time'][i][0:5] == '24.12':
                time= int(data['Local time'][i][11:13] + data['Local time'][i][14:16])
                if 900 <= time <= 1300:
                    spec_days.append(i)
            else:
                spec_days.append(i)
        data= data.iloc[spec_days]
        data= data.reset_index(drop=True)
        # cleaning days
        data['date'] = pd.to_datetime(data["Local time"], format= '%d.%m.%Y %H:%M').dt.strftime('%d.%m.%Y')
        data_no_vol= data.groupby(['date'], as_index=False)['Volume'].sum()
        no_vol= data_no_vol.date[data_no_vol.Volume == 0].to_list()
        sel_= []
        for i in range(len(data)):
            if data.date[i] in no_vol:
                pass
            else:
                sel_.append(i)
        data= data.iloc[sel_]
        data= data.reset_index(drop=True)
        data= data.set_index(["Local time"])
        # deleting 2015-05-01
        data= data[(data.index < "01.05.2015 09:00") | (data.index > "01.05.2015 17:00")]
        # price transformation
        data["Price"]= (data.Low + data.High) / 2.0
        data= data[["Price"]] 
        return data
    def data_LSTM(self):
        scaler = MinMaxScaler(feature_range=(0, 1))
        data= prepare_LSTM(self.raw_data_loc).data_cleaning()
        dataset = scaler.fit_transform(data.values.astype('float32'))
        train= dataset[0:len(data[:'31.12.2019 13:00']),:]
        return dataset, train, data, scaler

class split_LSTM:
    def __init__(self, dataset, window_size):
        self.dataset= dataset
        self.window_size= window_size
    def create_dataset(self):
        dataX, dataY = [], []
        for i in range(len(self.dataset)-self.window_size):
            a = self.dataset[i:(i+self.window_size), 0]
            dataX.append(a)
            dataY.append(self.dataset[i + self.window_size, 0])
        return np.array(dataX), np.array(dataY)
    def model_SPLITS(self, data, train):
        validation= self.dataset[(len(data[:'31.12.2019 13:00']) - self.window_size):len(data[:'31.08.2020 17:00']),:]      
        trainX, y_train = split_LSTM(train, self.window_size).create_dataset()
        valX, y_val = split_LSTM(validation, self.window_size).create_dataset()
        X_train= np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        X_val = np.reshape(valX, (valX.shape[0], 1, valX.shape[1]))
        return y_train, y_val, X_train, X_val, trainX, valX, validation

class render_result:
    def __init__(self, data, save_loc):
        self.data= data
        self.save_loc= save_loc
    def create_table(self, col_width=2.0, row_height=0.6, font_size=16,
                     header_color='#760002', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
        colors = self.data.shape[0] * [self.data.shape[1] * ["#e9f1f7"]] 
        if ax is None:
            size = (np.array(self.data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
            fig, ax = plt.subplots(figsize= size)
            ax.axis('off')
        mpl_table = ax.table(cellText= self.data.values, bbox=bbox, colLabels= self.data.columns, **kwargs, cellLoc= "center", cellColours= colors)
        mpl_table.auto_set_font_size(False)
        mpl_table.set_fontsize(font_size)
        for k, cell in mpl_table._cells.items():
            cell.set_edgecolor(edge_color)
            if k[0] == 0 or k[1] < header_columns:
                cell.set_text_props(weight='bold', color='w', multialignment= 'center', fontname="Times New Roman")
                cell.set_facecolor(header_color)
            else:
                cell.set_text_props(color='black', horizontalalignment= 'center', fontname="Times New Roman")
        return ax.get_figure(), ax
        set_align_for_column(mpl_table, align="center")

class run_LSTM:
    def __init__(self, window_size, neurons= 38, batch= 84):
        self.window_size= window_size
        self.neurons= neurons
        self.batch= batch
    def model_BASE(self, scaler, i, y_train, y_val, X_train, X_val, trainX, valX, validation):
        result = pd.DataFrame(columns= ["Results RMSE", "Train", "Validation", "Windows size", "Neurons", "Batch size"]) # to store values
        tf.random.set_seed(1234)
        rnn = Sequential()
        rnn.add(LSTM(units= self.neurons, activation='relu', input_shape= (1, self.window_size), name='LSTM')) 
        rnn.add(Dense(1)) 
        rnn.compile(loss= 'mean_squared_error', 
        optimizer='rmsprop')   
        early_stopping = EarlyStopping(monitor= 'val_loss', 
                              patience= 20,
                              restore_best_weights= True)  
        lstm_training = rnn.fit(X_train,
                        y_train,
                        epochs= 150, 
                        batch_size= self.batch,
                        shuffle= True,
                        validation_data= (X_val, y_val),
                        callbacks= [early_stopping],
                        verbose= 0)    
        train_predict_scaled = rnn.predict(X_train)
        val_predict_scaled = rnn.predict(X_val)
        train_pred_ = scaler.inverse_transform(train_predict_scaled.reshape(train_predict_scaled.shape[0], train_predict_scaled.shape[1]))
        trainY = scaler.inverse_transform([y_train])
        val_pred_ = scaler.inverse_transform(val_predict_scaled.reshape(val_predict_scaled.shape[0], val_predict_scaled.shape[1]))
        valY = scaler.inverse_transform([y_val])   
        train_rmse = np.sqrt(mean_squared_error(train_pred_.reshape(-1,), trainY.reshape(-1,)))
        validation_rmse = np.sqrt(mean_squared_error(val_pred_.reshape(-1,), valY.reshape(-1,)))
        return [str("Model " + str(i)), str(f'{train_rmse:.3f}'), str(f'{validation_rmse:.3f}'), str(self.window_size), str(self.neurons), str(self.batch)]

# for final model, includes test sample

class split_final_LSTM:
    def __init__(self, dataset, window_size):
        self.dataset= dataset
        self.window_size= window_size  
    def create_dataset(self):
        dataX, dataY = [], []
        for i in range(len(self.dataset)-self.window_size):
            a = self.dataset[i:(i+self.window_size), 0]
            dataX.append(a)
            dataY.append(self.dataset[i + self.window_size, 0])
        return np.array(dataX), np.array(dataY)
    def model_final_SPLITS(self, data, train):
        validation= self.dataset[(len(data[:'31.12.2019 13:00']) - self.window_size):len(data[:'31.08.2020 17:00']),:]      
        test= self.dataset[(len(data[:'31.08.2020 17:00']) - self.window_size):,:] #
        trainX, y_train = split_LSTM(train, self.window_size).create_dataset()
        valX, y_val = split_LSTM(validation, self.window_size).create_dataset()
        testX, y_test = split_LSTM(test, self.window_size).create_dataset() #
        X_train= np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        X_val = np.reshape(valX, (valX.shape[0], 1, valX.shape[1]))
        X_test = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        return y_train, y_val, y_test, X_train, X_val, X_test, trainX, valX, testX, validation, test

class run_final_LSTM:
    def __init__(self, window_size= 15, neurons= 38, batch= 84):
        self.window_size= window_size
        self.neurons= neurons
        self.batch= batch
    def MAPE(self, Y_Predicted, Y_actual):
        mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
        return mape
    def model_final_BASE(self, scaler, i, y_train, y_val, y_test, X_train, X_val, X_test, trainX, valX, testX, validation, test):  
        result = pd.DataFrame(columns= ["Results RMSE", "Train", "Validation", "Windows size", "Neurons", "Batch size"]) # to store values
        tf.random.set_seed(1234)
        rnn = Sequential()
        rnn.add(LSTM(units= self.neurons, activation='relu', input_shape= (1, self.window_size), name='LSTM')) 
        rnn.add(Dense(1))        
        rnn.compile(loss= 'mean_squared_error', 
        optimizer='rmsprop')      
        early_stopping = EarlyStopping(monitor= 'val_loss', 
                              patience= 20,
                              restore_best_weights= True) 
        lstm_training = rnn.fit(X_train,
                        y_train,
                        epochs= 150, 
                        batch_size= self.batch,
                        shuffle= True,
                        validation_data= (X_val, y_val),
                        callbacks= [early_stopping],
                        verbose= 0)  
        train_predict_scaled = rnn.predict(X_train)
        val_predict_scaled = rnn.predict(X_val)
        test_predict_scaled = rnn.predict(X_test)
        train_pred_ = scaler.inverse_transform(train_predict_scaled.reshape(train_predict_scaled.shape[0], train_predict_scaled.shape[1]))
        trainY = scaler.inverse_transform([y_train])
        val_pred_ = scaler.inverse_transform(val_predict_scaled.reshape(val_predict_scaled.shape[0], val_predict_scaled.shape[1]))
        valY = scaler.inverse_transform([y_val])
        test_pred_ = scaler.inverse_transform(test_predict_scaled.reshape(test_predict_scaled.shape[0], test_predict_scaled.shape[1]))
        testY = scaler.inverse_transform([y_test])   
        train_rmse = np.sqrt(mean_squared_error(train_pred_.reshape(-1,), trainY.reshape(-1,)))
        validation_rmse = np.sqrt(mean_squared_error(val_pred_.reshape(-1,), valY.reshape(-1,)))
        test_rmse = np.sqrt(mean_squared_error(test_pred_.reshape(-1,), testY.reshape(-1,)))
        train_ic = spearmanr(y_train, train_predict_scaled)[0]
        val_ic = spearmanr(y_val, val_predict_scaled)[0]
        test_ic = spearmanr(y_test, test_predict_scaled)[0]
        train_mape = run_final_LSTM().MAPE(train_pred_.reshape(-1,), trainY.reshape(-1,))
        validation_mape = run_final_LSTM().MAPE(val_pred_.reshape(-1,), valY.reshape(-1,))
        test_mape = run_final_LSTM().MAPE(test_pred_.reshape(-1,), testY.reshape(-1,))
        return lstm_training, testY, trainY, valY, test_pred_, train_pred_, val_pred_, train_rmse, validation_rmse, test_rmse, train_mape, validation_mape, test_mape, train_ic, val_ic, test_ic

# for graphs of the final model
class graphs_final_model(run_final_LSTM):
    def __init__(self, window_size):
        super().__init__(window_size)
    def create_graphs(self, data, lstm_training, testY, trainY, valY, test_pred_, train_pred_, val_pred_):
        xformatter = mdates.DateFormatter('%d.%b.%Y')
        with sns.axes_style('whitegrid'):
            fig, ax = plt.subplots(figsize=(14, 6))
            plt.xticks(fontsize=16, fontname="Times New Roman")
            plt.xlabel("Epochs", fontsize=16, fontname="Times New Roman", fontdict=dict(weight='bold'))
            loss_history = pd.DataFrame(lstm_training.history).pow(.5)
            loss_history.index += 1
            best_rmse = loss_history.val_loss.min()
            best_epoch = loss_history.val_loss.idxmin()
            title = f'Best validation RMSE: {best_rmse:.4%})'
            loss_history.columns= ['Training RMSE', 'Validation RMSE']
            loss_history.rolling(5).mean().plot(logy=True, lw= 3, ax=ax, color= ['darkgreen', 'darkorange'])
            ax.set_title(title, fontsize=18, fontname="Times New Roman", fontdict=dict(weight='bold'))
            ax.axvline(best_epoch, ls='--', lw=3, c='red')
            sns.despine()
            ax.legend(prop={'size': 16, "family":"Times New Roman"})
            ax.set_xticks(np.arange(8, len(loss_history), 10).tolist() + [best_epoch, len(loss_history)])
            ax.set_facecolor('#E9F1F7')
            fig.tight_layout()
            plt.savefig('figures/Model_evaluation.png', dpi= 300)
        with sns.axes_style('whitegrid'):
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(16, 10))
            gs = gridspec.GridSpec(1, 1, wspace=0.1, hspace=0.1)
            ax1 = plt.subplot(gs[0])
            plt.xticks(fontsize=16, fontname="Times New Roman")
            plt.yticks(fontsize=16, fontname="Times New Roman")
            Real= pd.Series(testY.reshape(-1,), pd.to_datetime(data[len(data[:'31.08.2020 17:00']):].index, format= '%d.%m.%Y %H:%M'))
            Real.plot(c= 'darkred', legend= True, linewidth= 1, ax= ax1)
            Predicted= pd.Series(test_pred_.reshape(-1,), index= pd.to_datetime(data[len(data[:'31.08.2020 17:00']):].index, format= '%d.%m.%Y %H:%M'))
            Predicted.plot(c= 'blue', legend= True, linewidth= 2, ax= ax1)
            ax1.set_xlabel("Date", fontsize=16, fontname="Times New Roman", fontdict=dict(weight='bold'))
            ax1.set_ylabel("Mid price", fontsize=16, fontname="Times New Roman", fontdict=dict(weight='bold'))
            ax1.legend(['Real', 'Prediction on test sample'], prop={'size': 16, "family":"Times New Roman"})
            ax1.fmt_xdata = DateFormatter('%d.%b.%Y') 
            ax1.xaxis.set_major_formatter(xformatter)
            ax1.set_facecolor('#E9F1F7')
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation= 0, ha= 'center')
            plt.savefig('figures/Model_predict_test.png', dpi= 300)
        with sns.axes_style('whitegrid'):
            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16, 12))
            gs = gridspec.GridSpec(2, 1, wspace=0.4, hspace=0.3)
            ax1 = plt.subplot(gs[0])
            plt.xticks(fontsize=16, fontname="Times New Roman")
            plt.yticks(fontsize=16, fontname="Times New Roman")
            ax2 = plt.subplot(gs[1])
            plt.xticks(fontsize=16, fontname="Times New Roman")
            plt.yticks(fontsize=16, fontname="Times New Roman")
            Real1= pd.Series(trainY.reshape(-1,), pd.to_datetime(data[self.window_size:len(data[:'31.12.2019 13:00'])].index, format= '%d.%m.%Y %H:%M'))
            Real1.plot(c= '#760002', legend= True, linewidth= 1, ax= ax1)
            Predicted1= pd.Series(train_pred_.reshape(-1,), index= pd.to_datetime(data[self.window_size:len(data[:'31.12.2019 13:00'])].index, format= '%d.%m.%Y %H:%M'))
            Predicted1.plot(c= 'darkgreen', legend= True, linewidth= 2, ax= ax1)
            ax1.set_xlabel("Date", fontsize=16, fontname="Times New Roman", fontdict=dict(weight='bold'))
            ax1.set_ylabel("Mid price", fontsize=16, fontname="Times New Roman", fontdict=dict(weight='bold'))
            ax1.legend(['Real', 'Prediction on training sample'], prop={'size': 16, "family":"Times New Roman"})
            ax1.fmt_xdata = DateFormatter('%d.%b.%Y') 
            ax1.xaxis.set_major_formatter(xformatter)
            ax1.set_facecolor('#E9F1F7')
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation= 0, ha= 'center')
            Real2= pd.Series(valY.reshape(-1,), pd.to_datetime(data[len(data[:'31.12.2019 13:00']):len(data[:'31.08.2020 17:00'])].index, format= '%d.%m.%Y %H:%M'))
            Real2.plot(c= '#760002', legend= True, linewidth= 1, ax= ax2)
            Predicted2= pd.Series(val_pred_.reshape(-1,), index= pd.to_datetime(data[len(data[:'31.12.2019 13:00']):len(data[:'31.08.2020 17:00'])].index, format= '%d.%m.%Y %H:%M'))
            Predicted2.plot(c= 'darkorange', legend= True, linewidth= 2, ax= ax2)
            ax2.set_xlabel("Date", fontsize=16, fontname="Times New Roman", fontdict=dict(weight='bold'))
            ax2.set_ylabel("Mid price", fontsize=16, fontname="Times New Roman", fontdict=dict(weight='bold'))
            ax2.legend(['Real', 'Prediction on validation sample'], prop={'size': 16, "family":"Times New Roman"})
            ax2.fmt_xdata = DateFormatter('%d.%b.%Y') 
            ax2.xaxis.set_major_formatter(xformatter)
            ax2.set_facecolor('#E9F1F7')
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation= 0, ha= 'center')
            plt.savefig('figures/Model_predict_val_train.png', dpi= 300)
            plt.close()

# for econometric testing
class econ_test:
    def __init__(self, raw_data_loc, window_size):
        self.raw_data_loc= raw_data_loc
        self.window_size= window_size
    def data_prep(self):
        data= prepare_LSTM(self.raw_data_loc).data_cleaning()
        components = tsa.seasonal_decompose(data, model='additive', period= self.window_size)
        ts = (data
      .assign(Trend= components.trend)
      .assign(Seasonality= components.seasonal)
      .assign(Residual= components.resid))
        ts.index= pd.to_datetime(data.index, format= '%d.%m.%Y %H:%M')
        X= data.values
        return data, ts, components, X
    def graph_comp(self):
        data, ts, components, X= econ_test(self.raw_data_loc, self.window_size).data_prep()
        xformatter = mdates.DateFormatter('%d.%b.%Y')
        with sns.axes_style('whitegrid'):
            fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(16, 20))
            gs = gridspec.GridSpec(4, 1, wspace=0.4, hspace=0.3)
            ax1 = plt.subplot(gs[0])
            plt.xticks(fontsize=16, fontname="Times New Roman")
            plt.yticks(fontsize=16, fontname="Times New Roman")
            ts.Price.plot(ax= ax1, c= '#760002')
            ax1.set_title('Original, 2015-2020', fontsize=18, fontname="Times New Roman", fontdict=dict(weight='bold'))
            ax1.set_xlabel("Date", fontsize=12, fontname="Times New Roman")
            ax1.fmt_xdata = DateFormatter('%d.%b.%Y') 
            ax1.xaxis.set_major_formatter(xformatter)
            ax1.set_facecolor('#E9F1F7')
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation= 0, ha= 'center')
            ax2 = plt.subplot(gs[1])
            plt.xticks(fontsize=16, fontname="Times New Roman")
            plt.yticks(fontsize=16, fontname="Times New Roman")
            ts.Trend.plot(ax= ax2, c= 'blue')
            ax2.set_title('Trend component, 2015-2020', fontsize=18, fontname="Times New Roman", fontdict=dict(weight='bold'))
            ax2.set_xlabel("Date", fontsize=12, fontname="Times New Roman")
            ax2.fmt_xdata = DateFormatter('%d.%b.%Y') 
            ax2.xaxis.set_major_formatter(xformatter)
            ax2.set_facecolor('#E9F1F7')
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation= 0, ha= 'center')
            ax3 = plt.subplot(gs[2])
            plt.xticks(fontsize=16, fontname="Times New Roman")
            plt.yticks(fontsize=16, fontname="Times New Roman")
            ts.Seasonality["31.08.2020 18:00:00":].plot(ax= ax3, c= 'darkgreen')
            ax3.set_title('Seasonal component, second semester of 2020', fontsize=18, fontname="Times New Roman", fontdict=dict(weight='bold'))
            ax3.set_xlabel("Date", fontsize=12, fontname="Times New Roman")
            ax3.fmt_xdata = DateFormatter('%d.%b.%Y') 
            ax3.xaxis.set_major_formatter(xformatter)
            ax3.set_facecolor('#E9F1F7')
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation= 0, ha= 'center')
            ax4 = plt.subplot(gs[3])
            plt.xticks(fontsize=16, fontname="Times New Roman")
            plt.yticks(fontsize=16, fontname="Times New Roman")
            ts.Residual.plot(ax= ax4, c= 'darkorange')
            ax4.set_title('Remainder, 2015-2020', fontsize=18, fontname="Times New Roman", fontdict=dict(weight='bold'))
            ax4.set_xlabel("Date", fontsize=12, fontname="Times New Roman")
            ax4.fmt_xdata = DateFormatter('%d.%b.%Y') 
            ax4.xaxis.set_major_formatter(xformatter)
            ax4.set_facecolor('#E9F1F7')
            plt.setp(ax4.xaxis.get_majorticklabels(), rotation= 0, ha= 'center')
            fig.savefig('figures/seasonal_decomposition.png', dpi= 300)
            plt.close()
    def print_econ_mod(self, model, name):
        plt.rc('figure', figsize=(6, 3))
        plt.text(0, 0, str(model.summary()), {'fontsize': 16, 'fontname':"Times New Roman"})
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('figures/' + name + '.png', dpi= 300)
        plt.close()
    def pp_test(self):
        data, ts, components, X= econ_test(self.raw_data_loc, self.window_size).data_prep()
        pp = PhillipsPerron(X)
        pp.lags = self.window_size
        pp.trend = 'ct'
        econ_test(self.raw_data_loc, self.window_size).print_econ_mod(pp, 'pp_test')
    def kpss_test(self):
        data, ts, components, X= econ_test(self.raw_data_loc, self.window_size).data_prep()
        kpss = KPSS(X)
        kpss.lags= self.window_size
        kpss.trend = 'ct'
        econ_test(self.raw_data_loc, self.window_size).print_econ_mod(kpss, 'kpss_test')
    def autocorr_graph(self, name):
        data, ts, components, X= econ_test(self.raw_data_loc, self.window_size).data_prep()
        data_series= pd.Series(data.Price, index= data.index)
        data_log = np.log(data_series)
        data_log_diff = data_log.diff().dropna()
        data_log_diff.index= pd.to_datetime(data_log_diff.index, format= '%d.%m.%Y %H:%M')
        xformatter = mdates.DateFormatter('%d.%b.%Y')
        with sns.axes_style('whitegrid'):
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(16, 6))
            gs = gridspec.GridSpec(1, 1, wspace=0.4, hspace=0.3)
            ax1 = plt.subplot(gs[0])
            plt.xticks(fontsize=16, fontname="Times New Roman")
            plt.yticks(fontsize=16, fontname="Times New Roman")
            ax1.set_xticks(np.arange(0, 289, 36).tolist())
            plot_acf(data_log_diff, lags= self.window_size, zero=False, ax= ax1, c= '#760002')
            ax1.set_title("Autocorrelation", fontsize=18, fontname="Times New Roman", fontdict= dict(weight='bold'))
            ax1.set_xlabel("Lags", fontsize=16, fontname="Times New Roman", fontdict=dict(weight='bold'))
            ax1.set_facecolor('#E9F1F7')
            ax1.set_xticks(np.arange(1, self.window_size+1, 1).tolist())
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation= 0, ha= 'center')
            fig.savefig('figures/'+ name + '.png', dpi= 300)