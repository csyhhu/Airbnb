#setwd('/Users/shangyu/Documents/Kaggle/Airbnb')
setwd('/home/aurelius/Kaggle/Airbnb')
#library('e1071')
library('xgboost')
dataX <- read.csv("trainX.csv")
dataY <- read.csv("trainY2.csv")

sample_num = dim(dataX)[1]
Xfeature_num = dim(dataX)[2]
Yfeature_num = dim(dataY)[2]
train_num = floor(sample_num * 0.7)
test_num = sample_num - train_num

trainX <- dataX[c(1: train_num), c(2: feature_num)]
trainY <- dataY[c(1: train_num), c(2: Yfeature_num)]
testX <- dataX[c(train_num: sample_num), c(2: feature_num)]
testY <- dataY[c(train_num: sample_num), c(2: Yfeature_num)]

write.csv(trainX, 'trainX_2.csv')
write.csv(trainY, 'trainY_2.csv')
write.csv(testX, 'testX_2.csv')
write.csv(testY, 'testY_2.csv')

# trainX <- matrix(data = as.numeric(as.matrix(trainX)), nrow = trainX_num, ncol = trainX_fea)
# trainY <- c(data = as.matrix(trainY))

xgb_trainX <- xgb.DMatrix(data = trainX) 
xgb_trainY <- xgb.DMatrix(data = trainY)

bst <- xgboost(data = xgb_trainX, label = trainY, max.depth = 2,
               eta = 1, nthread = 2, nround = 2)
#model <- svm(trainX, trainY)
