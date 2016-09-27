#convert categorical data into numerical form 
import pandas as pd
import numpy as np

# One Hot Encoding
def one_hot_encoding(df, selected_column):
	#find out all possible values of each column
    values = list(df[selected_column].drop_duplicates())

    for value in values:
        cat_name = str(value).replace(" ", "_").replace("/", "_").replace("-","").lower()
        col_name = selected_column + '_' + cat_name
        df[col_name] = 0
        df.loc[(df[selected_column] == value), col_name] = 1

    return df


train = pd.read_csv("E:\\NTU\\Academics\\SEM 1\\DM\Project\\Python Programs for project\\clean_train_data.csv")
selected_column = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 
                   'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']

for column in selected_column:
    train = one_hot_encoding(df=train, selected_column=column)
    train.drop(column, axis=1)

#print(train.head())

#output to a csv file
train.to_csv('train_data_after_one_hot_encoding.csv', sep=',');



