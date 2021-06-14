library(forecast)

df <- read.csv("~/Desktop/Files/Thesis/Model/DF/df.csv")
df

dm.test(df$Drift, df$LSTM, alternative = "less", h = 1)
