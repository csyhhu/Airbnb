setwd('/Users/shangyu/Documents/Kaggle/Airbnb')
library('e1071')

trainX <- read.csv("trainX.csv")
trainY <- read.csv("trainY.csv")
trainX <- trainX[,c(2:dim(trainX)[2])]
trainY <- trainY[,c(2:dim(trainY)[2])]

model <- svm(trainX, trainY)
