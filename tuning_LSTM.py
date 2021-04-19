from model_modules import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

result = pd.DataFrame(columns= ["Results RMSE", "Train", "Validation", "Windows size", "Neurons", "Batch size"])

# windows_size
window_size= np.arange(9, 82, 3)
dataset, train, data, scaler= prepare_LSTM('data/ESP.IDXEUR_Candlestick_1_Hour_BID_01.01.2015-31.12.2020.csv').data_LSTM()
i = 0
print("Tuning windows size, " + str(len(window_size)) + " models.")
while i < len(window_size):
    y_train, y_val, X_train, X_val, trainX, valX, validation= split_LSTM(dataset, window_size[i]).model_SPLITS(data, train)
    result.loc[i+1]= run_LSTM(window_size[i]).model_BASE(scaler, i+1, y_train, y_val, X_train, X_val, trainX, valX, validation)
    print(str(i+1), sep=' ', end=',', flush=True)
    i +=1
window_size_= int(result["Windows size"].iloc[[pd.to_numeric(result.Validation).idxmin()-1]])
print(" Best window size: " + str(window_size_))
y_train, y_val, X_train, X_val, trainX, valX, validation= split_LSTM(dataset, window_size_).model_SPLITS(data, train)

# neurons
result.drop(result.index, inplace=True)
neurons= np.arange(2, 51, 2)
i= 0
print("Tuninng neurons, " + str(len(neurons)) + " models.")
while i < len(neurons):
    result.loc[i+1]= run_LSTM(window_size_, neurons[i]).model_BASE(scaler, i+1, y_train, y_val, X_train, X_val, trainX, valX, validation)
    print(str(i+1), sep=' ', end=',', flush=True)
    i +=1
neurons_= int(result["Neurons"].iloc[[pd.to_numeric(result.Validation).idxmin()-1]])
print(" Best number of neurons: " + str(neurons_))

# batch
result.drop(result.index, inplace=True)
batch= np.arange(34, 84, 2)
i= 0
print("Tuning batch size, " + str(len(batch)) + " models.")
while i < len(batch):
    result.loc[i+1]= run_LSTM(window_size_, neurons_, batch[i]).model_BASE(scaler, i+1, y_train, y_val, X_train, X_val, trainX, valX, validation)
    print(str(i+1), sep=' ', end=',', flush=True)
    i +=1

fig,ax = render_result(result, "table_results.png").create_table()
fig.savefig("table_results.png", dpi= 300)

print(str(int(time.time() - start_time)/60) + " minutes")