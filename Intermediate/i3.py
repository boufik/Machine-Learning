import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder


# ********************************************************************************************************
# ********************************************************************************************************
# 0. Function - Compare scores between different models (train, validation datasets)
# ********************************************************************************************************
# ********************************************************************************************************

def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)





# ********************************************************************************************************
# ********************************************************************************************************
# 1. Basics - Read dataset and split it
# ********************************************************************************************************
# ********************************************************************************************************

# 1a. Read the 2 datasets
X = pd.read_csv('../input/train.csv', index_col='Id') 
X_test = pd.read_csv('../input/test.csv', index_col='Id')

# 1b. Remove rows with missing target
X.dropna(axis=0, subset=['SalePrice'], inplace=True)

# 1c. Separate target from predictors
y = X.SalePrice
X.drop(['SalePrice'], axis=1, inplace=True)

# 1d. Drop columns with missing values
cols_with_missing = [col for col in X.columns if X[col].isnull().any()] 
X.drop(cols_with_missing, axis=1, inplace=True)
X_test.drop(cols_with_missing, axis=1, inplace=True)

# 1e. Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# 1f. Preview the dataset
print(X_train.shape)
X_train.head()
X_train.describe()





# ********************************************************************************************************
# ********************************************************************************************************
# 2a. Drop columns with categorical data
# ********************************************************************************************************
# ********************************************************************************************************

# Drop columns in training and validation data (SUPPOSE THERE ARE NOT MISSING DATA COLUMNS TO DROP FIRST)
drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])
print("MAE from Approach 1 (Drop categorical variables):")
print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))





# ********************************************************************************************************
# ********************************************************************************************************
# 2b. Ordinal Encoding
# ********************************************************************************************************
# ********************************************************************************************************

# 2b1. A little search in the dataset
print("Unique values in 'Condition2' column in training data:", X_train['Condition2'].unique())
print("\nUnique values in 'Condition2' column in validation data:", X_valid['Condition2'].unique())
print("\nFitting an ordinal encoder to a column in the training data creates a corresponding integer-valued label for each unique value that appears in the training data. In the case that the validation data contains values that don't also appear in the training data, the encoder will throw an error, because these values won't have an integer assigned to them. Notice that the 'Condition2' column in the validation data contains the values 'RRAn' and 'RRNn', but these don't appear in the training data -- thus, if we try to use an ordinal encoder with scikit-learn, the code will throw an error.\n\n")

# 2b2. Categorical columns in the training data
object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

# 2b3. Columns that can be safely ordinal encoded
good_label_cols = [col for col in object_cols if set(X_valid[col]).issubset(set(X_train[col]))]
        
# 2b4. Problematic columns that will be dropped from the dataset
bad_label_cols = list(set(object_cols)-set(good_label_cols))

print('Categorical columns (ALL OF THEM):', object_cols, '\n')
print('Categorical columns that will be ordinal encoded:', good_label_cols, '\n')
print('\nCategorical columns that will be dropped from the dataset:', bad_label_cols, '\n')


# 2b5. Drop categorical columns that will not be encoded
label_X_train = X_train.drop(bad_label_cols, axis=1)
label_X_valid = X_valid.drop(bad_label_cols, axis=1)

# 2b6. Apply ordinal encoder 
ordinal_encoder = OrdinalEncoder()
label_X_train[good_label_cols] = ordinal_encoder.fit_transform(label_X_train[good_label_cols])
label_X_valid[good_label_cols] = ordinal_encoder.transform(label_X_valid[good_label_cols])
print("MAE from Approach 2 (Ordinal Encoding):") 
print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))






# ********************************************************************************************************
# ********************************************************************************************************
# 2c. One Hot Encoding
# ********************************************************************************************************
# ********************************************************************************************************

# 2c0. Get number of unique entries in each column with categorical data
object_nunique = list(map(lambda col: X_train[col].nunique(), object_cols))
d = dict(zip(object_cols, object_nunique))

# 2c1. Print number of unique entries by column, in ascending order
sorted(d.items(), key=lambda x: x[1])

high_cardinality_numcols = 3
num_cols_neighborhood = 25
OH_entries_added = 10000 * (100 - 1)
label_entries_added = 0

# 2c2. Low and high cardinality columns

low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < 10]
high_cardinality_cols = list(set(object_cols)-set(low_cardinality_cols))
print('Categorical columns that will be one-hot encoded:', low_cardinality_cols)
print('\nCategorical columns that will be dropped from the dataset:', high_cardinality_cols)



# 2c3. Drop high cardinality columns (It will contain numbers + low_cardinality)
OH_X_train = X_train.drop(high_cardinality_cols, axis=1)
OH_X_valid = X_valid.drop(high_cardinality_cols, axis=1)

# 2c4. Keep low cardinaliy columns (It will contain low_cardinality)
oh_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(oh_encoder.fit_transform(X_train[low_cardinality_cols]))
OH_cols_valid = pd.DataFrame(oh_encoder.transform(X_valid[low_cardinality_cols]))

# 2c5. Index was removed whie one-hot-encoded, I bring it back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# 2c6. Number columns in both training and validation X datasets (It will contain numbers, I dropped the old low_cordinality cols, because I will replace them with OH cols)
number_X_train = X_train.drop(object_cols, axis=1) 
number_X_valid = X_valid.drop(object_cols, axis=1)

# 2c7. Concatenate low cardinality and number columns
OH_X_train = pd.concat([OH_cols_train, number_X_train], axis=1)
OH_X_valid = pd.concat([OH_cols_valid, number_X_valid], axis=1)
print("MAE from Approach 3 (One-Hot Encoding):") 
print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))
