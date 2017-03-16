setwd('/Users/Shangyu/Documents/Kaggle/Airbnb/WorkSpace')

library(readr)
library(Hmisc)
#library(xgboost)
library(readr)
library(stringr)
library(caret)
library(car)
library(plyr)
library(dplyr)
library(tidyr)
library(data.table)
library(DescTools)
library(Matrix)
#library(glmnet)

df_train <- read_csv('../train_users_2.csv',
                     col_types = cols(
                      timestamp_first_active = col_character()))
df_test <- read_csv('../test_users.csv',
                    col_types = cols(
                      timestamp_first_active = col_character()))
labels <- df_train[, c('id', 'country_destination')]
df_test$country_destination = NA

# combine train and test data
df_train$dataset <- "train"
df_test$dataset <- "test"
df_all = rbind(df_train, df_test)

# **************************************
# clean age
# **************************************
df_all <- df_all %>%
  dplyr::mutate(
    age_cln = ifelse(age >= 1920, 2015 - age, age),
    age_cln2 = ifelse(age_cln < 14 | age_cln > 100, -1, age_cln),
    age_bucket = cut(age, breaks = c(Min(age_cln), 4, 9, 14, 19, 24,
                                     29, 34, 39, 44, 49, 54,
                                     59, 64, 69, 74, 79, 84,
                                     89, 94, 99, Max(age_cln)
    )),
    age_bucket = mapvalues(age_bucket,
                           from=c("(1,4]", "(4,9]", "(9,14]", "(14,19]",
                                  "(19,24]", "(24,29]", "(29,34]", "(34,39]",
                                  "(39,44]", "(44,49]", "(49,54]", "(54,59]",
                                  "(59,64]", "(64,69]", "(69,74]", "(74,79]",
                                  "(79,84]", "(84,89]", "(89,94]", "(94,99]", "(99,150]"),
                           to=c("0-4", "5-9", "10-14", "15-19",
                                "20-24", "25-29", "30-34", "35-39",
                                "40-44", "45-49", "50-54", "55-59",
                                "60-64", "65-69", "70-74", "75-79",
                                "80-84", "85-89", "90-94", "95-99", "100+"))
  )
