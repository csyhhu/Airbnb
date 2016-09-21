setwd('/home/aurelius/Kaggle/Airbnb')
source('WorkSpace/myOneHot.R')

# Read in the dataset
train_users <- read.table('train_users_2.csv',sep = ',', header = TRUE)

# find the element of each features, and construct their mapping
# Signup method
signup_method <- unique(train_users$signup_method)
signup_method_hash <- c(1:length(signup_method))-1
names(signup_method_hash) <- signup_method
# gender
gender <- unique(train_users$gender)
gender_hash <- c(1:length(gender))-1
names(gender_hash) <- gender
# language
language <- unique(train_users$language)
language_hash <- c(1:length(language))-1
names(language_hash) <- language
# affiliate_channel
affiliate_channel <- unique(train_users$affiliate_channel)
affiliate_channel_hash <- c(1:length(affiliate_channel))-1
names(affiliate_channel_hash) <- affiliate_channel
# affiliate_provider
affiliate_provider <- unique(train_users$affiliate_provider)
affiliate_provider_hash <- c(1:length(affiliate_provider))-1
names(affiliate_provider_hash) <- affiliate_provider
# first_affiliate_tracked
first_affiliate_tracked <- unique(train_users$first_affiliate_tracked)
first_affiliate_tracked_hash <- c(1:length(first_affiliate_tracked))-1
names(first_affiliate_tracked_hash) <- first_affiliate_tracked
# signup_app
signup_app <- unique(train_users$signup_app)
signup_app_hash <- c(1:length(signup_app))-1
names(signup_app_hash) <- signup_app
# first_device_type
first_device_type <- unique(train_users$first_device_type)
first_device_type_hash <- c(1:length(first_device_type))-1
names(first_device_type_hash) <- first_device_type
# first_browser
first_browser <- unique(train_users$first_browser)
first_browser_hash <- c(1:length(first_browser))-1
names(first_browser_hash) <- first_browser

nulflag <- train_users[1,4]
# Attain the samples that don't have data_first_booking
train_users <- train_users[!train_users$date_first_booking == nulflag, ]
# Clean the age and divide into bucket
# Clean the age: fill the na with mean value, fill the age < 4 and > 94 with mean value
train_users$age[is.na(train_users$age)] <- mean(train_users$age, na.rm = TRUE)
train_users[train_users$age > 94 && train_users$age < 4, 'age'] <- mean(train_users$age)
train_users$age <- as.integer(floor(train_users$age))
# divide the age into [4:9],[10:14],[15:19]
train_users$age <- train_users$age %/% 5

# Create the train user's feature, slicing id and age as the initial
train_users_features <- as.matrix(train_users[c('id','age')]) 

# Do the one-hot coding for other paramaters
# For signup method
signup_method_onehot <- myOneHot(signup_method_hash[train_users$signup_method], length(signup_method))
# For gender
gender_onehot <- myOneHot(gender_hash[train_users$gender], length(gender))
# For language
language_onehot <- myOneHot(language_hash[train_users$language], length(language))
# For affiliate_channel
affiliate_channel_onehot <- myOneHot(affiliate_channel_hash[train_users$affiliate_channel], length(affiliate_channel))
# affiliate_provider
affiliate_provider_onehot <- myOneHot(affiliate_provider_hash[train_users$affiliate_provider], length(affiliate_provider))
# first_affiliate_tracked
first_affiliate_tracked_onehot <- myOneHot(first_affiliate_tracked_hash[train_users$first_affiliate_tracked], length(first_affiliate_tracked))
# signup_app
signup_app_onehot <- myOneHot(signup_app_hash[train_users$signup_app], length(signup_app))
# first_device_type
first_device_type_onehot <- myOneHot(first_device_type_hash[train_users$first_device_type], length(first_device_type))
# first_browser
first_browser_onehot <- myOneHot(first_browser_hash[train_users$first_browser], length(first_browser))

# Combine all the features
train_users_features <- cbind(train_users_features, signup_method_onehot, gender_onehot, language_onehot,
                              affiliate_channel_onehot, affiliate_provider_onehot, first_affiliate_tracked_onehot,
                              signup_app_onehot, first_device_type_onehot, first_browser_onehot)

