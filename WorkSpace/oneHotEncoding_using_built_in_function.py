import pandas as pd

train = pd.read_csv("E:\\NTU\\Academics\\SEM 1\\DM\Project\\Python Programs for project\\clean_train_data.csv")

selected_column = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 
                   'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']

for column in selected_column:
    one_hot = pd.get_dummies(train[column])
    train = train.join(one_hot, rsuffix="_r")
    train.drop(column, axis=1, inplace=True)
	 		   
print(train.head())

#output to a csv file
train.to_csv('train_data_after_one_hot_encoding.csv', sep=',');
