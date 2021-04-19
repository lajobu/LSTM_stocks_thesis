from model_modules import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

window_size_= 15
neurons_= 50
batch_= 58

dataset, train, data, scaler= prepare_LSTM('data/ESP.IDXEUR_Candlestick_1_Hour_BID_01.01.2015-31.12.2020.csv').data_LSTM()
y_train, y_val, y_test, X_train, X_val, X_test, trainX, valX, testX, validation, test= split_final_LSTM(dataset, 15).model_final_SPLITS(data, train)
lstm_training, testY, trainY, valY, test_pred_, train_pred_, val_pred_, train_rmse, validation_rmse, test_rmse, train_mape, validation_mape, test_mape, train_ic, val_ic, test_ic= run_final_LSTM(window_size_, neurons_, batch_).model_final_BASE(scaler, 1, y_train, y_val, y_test, X_train, X_val, X_test, trainX, valX, testX, validation, test)
graphs_final_model(window_size_).create_graphs(data, lstm_training, testY, trainY, valY, test_pred_, train_pred_, val_pred_)

details = ["MODEL RESULTS", "RMSE", "MAPE", "IC"] 
training_ = ["Training sample:", str(f'{train_rmse:.2f}'), str(f'{train_mape:.2f}'), str(f'{train_ic:.4f}')] 
validation_ = ["Validation sample:", str(f'{validation_rmse:.2f}'), str(f'{validation_mape:.2f}'), str(f'{val_ic:.4f}')] 
test_ = ["Test sample:", str(f'{test_rmse:.2f}'), str(f'{test_mape:.2f}'), str(f'{test_ic:.4f}')] 
result = pd.DataFrame(data= [training_, validation_, test_],
                 columns= details)

fig,ax = render_result(result, "figures/table_final_results.png").create_table(header_columns= 0, col_width=2.6)
fig.savefig("figures/table_final_results.png", dpi= 300)