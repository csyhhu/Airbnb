import pandas as pd
import numpy as np
import csv

# Read Data
train = "E:\\NTU\\Academics\\SEM 1\\DM\Project\\train_users_2.csv"
df_train = pd.read_csv(train, header=0, index_col=None)
test = "E:\\NTU\\Academics\\SEM 1\\DM\Project\\test_users.csv"
df_test = pd.read_csv(test, header=0, index_col=None)

# Combine into one dataset
#df_both = pd.concat((df_train, df_test), axis=0, ignore_index=True)

# Change Dates to consistent format
print("Fixing timestamps...")
df_train['date_account_created'] = pd.to_datetime(df_train['date_account_created'], format='%Y-%m-%d')
df_test['date_account_created'] = pd.to_datetime(df_test['date_account_created'], format='%Y-%m-%d')
df_train['timestamp_first_active'] = pd.to_datetime(df_train['timestamp_first_active'], format='%Y%m%d%H%M%S')
df_test['timestamp_first_active'] = pd.to_datetime(df_test['timestamp_first_active'], format='%Y%m%d%H%M%S')
df_train['date_account_created'].fillna(df_train.timestamp_first_active, inplace=True)
df_test['date_account_created'].fillna(df_test.timestamp_first_active, inplace=True)

# Remove date_first_booking column
df_train.drop('date_first_booking', axis=1, inplace=True)
df_test.drop('date_first_booking', axis=1, inplace=True)


# Remove outliers function
def remove_outliers(df, column, min_val, max_val):
	col_values = df[column].values
	df[column] = np.where(np.logical_or(col_values<=min_val, col_values>=max_val), np.NaN, col_values)
	return df

# Fixing age column(Age is fixed by taking min value as 15 and max value as 90 - outliers are taken as -1)
print("Fixing age column...")
df_train = remove_outliers(df=df_train, column='age', min_val=15, max_val=90)
df_train['age'].fillna(-1, inplace=True)
df_test = remove_outliers(df=df_test, column='age', min_val=15, max_val=90)
df_test['age'].fillna(-1, inplace=True)


# Fill first_affiliate_tracked column
print("Filling first_affiliate_tracked column...")
df_train['first_affiliate_tracked'].fillna(-1, inplace=True)
df_test['first_affiliate_tracked'].fillna(-1, inplace=True)

#print(df_train.head())
#print(df_test.head())

print (df_train.apply(lambda x: sum(x.isnull()),axis=0))
print (df_test.apply(lambda x: sum(x.isnull()),axis=0))

#output to a csv file
df_train.to_csv('clean_train_data.csv', sep=',');
df_test.to_csv('clean_test_data.csv', sep=',');
