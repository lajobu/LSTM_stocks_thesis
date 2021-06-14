library(forecast)

df <- read.csv("~/Desktop/Files/Thesis/Model/DF/df.csv")
df

dm.test(df$LSTM, df$Naive, alternative = "less", h = 1) # DM = -6.3632, Forecast horizon = 1, Loss function power = 2, p-value = 1.691e-10
dm.test(df$Drift, df$LSTM, alternative = "less", h = 1) # DM = 7.9558, Forecast horizon = 1, Loss function power = 2, p-value = 1
