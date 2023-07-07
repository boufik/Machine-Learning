# *******************************************************************************************
# *******************************************************************************************
# 0. Function that evaluates a RandomForestRegressor model (with 100 trees) using MAE
# *******************************************************************************************
# *******************************************************************************************

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer


def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)





# *******************************************************************************************
# *******************************************************************************************
# 1. Training, validation (1st dataset) and testing dataset
# *******************************************************************************************
# *******************************************************************************************

import pandas as pd
from sklearn.model_selection import train_test_split

# 1a. Read the data from the 2 datasets
X_full = pd.read_csv('../input/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/test.csv', index_col='Id')

# 1b. Remove ROWS with MISSING TARGET, separate target from predictors
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

# 1c. To keep things simple, we'll use only numerical predictors
X = X_full.select_dtypes(exclude=['object'])
X_test = X_test_full.select_dtypes(exclude=['object'])

# 1d. Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
X_train.head()
# Shape of training data (num_rows, num_columns)
print(X_train.shape)


# *******************************************************************************************
# *******************************************************************************************
# 2. Preliminary investigation - Find which and how many columns contain missing data
# *******************************************************************************************
# *******************************************************************************************

# Number of missing values in each column of training data
missing_val_count_by_column = (X_train.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])

num_rows = 1168
num_cols_with_missing = 3
tot_missing = 212 + 6 + 58



# *******************************************************************************************
# *******************************************************************************************
# 3a. Drop missing values
# *******************************************************************************************
# *******************************************************************************************

# Get names of columns with missing values - The same as above DataFrame
cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]
print("Names of columns with missing data\n", cols_with_missing, '\n\n')

reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

print("MAE (Drop columns with missing values):")
print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))




# *******************************************************************************************
# *******************************************************************************************
# 3b. Imputation - Default strategy = mean
# *******************************************************************************************
# *******************************************************************************************

my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# Fill in the lines below: imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

print("MAE (Imputation):")
print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))

print('\n\n', "Given that thre are so few missing values in the dataset, we'd expect imputation to perform better than dropping columns entirely. However, we see that dropping columns performs slightly better! While this can probably partially be attributed to noise in the dataset, another potential explanation is that the imputation method is not a great match to this dataset. That is, maybe instead of filling in the mean value, it makes more sense to set every missing value to a value of 0, to fill in the most frequently encountered value, or to use some other method. For instance, consider the GarageYrBlt column (which indicates the year that the garage was built). It's likely that in some cases, a missing value could indicate a house that does not have a garage. Does it make more sense to fill in the median value along each column in this case? Or could we get better results by filling in the minimum value along each column? It's not quite clear what's best in this case, but perhaps we can rule out some options immediately - for instance, setting missing values in this column to 0 is likely to yield horrible results!", '\n\n')


# *******************************************************************************************
# *******************************************************************************************
# 3c. Imputation - Custom strategy = median
# *******************************************************************************************
# *******************************************************************************************

final_imputer = SimpleImputer(strategy='median')
final_X_train = pd.DataFrame(final_imputer.fit_transform(X_train))
final_X_valid = pd.DataFrame(final_imputer.transform(X_valid))
final_X_train.columns = X_train.columns
final_X_valid.columns = X_valid.columns

print("MAE (Imputation 2):")
print(score_dataset(final_X_train, final_X_valid, y_train, y_valid))



# *******************************************************************************************
# *******************************************************************************************
# 4. Define and fit model
# *******************************************************************************************
# *******************************************************************************************

model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(final_X_train, y_train)
preds_valid = model.predict(final_X_valid)

print("MAE (Your approach):")
print(mean_absolute_error(y_valid, preds_valid))



# *******************************************************************************************
# *******************************************************************************************
# 5. Apply the model in the 2nd dataset - for testing
# *******************************************************************************************
# *******************************************************************************************

# I have to also impute the columns of this DataFrame (X_test)
test_imputer = SimpleImputer(strategy='median')
final_X_test = pd.DataFrame(test_imputer.fit_transform(X_test))
final_X_test.columns = X_test.columns

preds_test = model.predict(final_X_test)

# Save test predictions to file
output = pd.DataFrame({'Id': X_test.index, 'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)
