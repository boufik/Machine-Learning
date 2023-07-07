import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# ******************************************************************************************************
# ******************************************************************************************************
# 1. Read the 2 datasets and split the dataset - Choose 10 as maximum cardinality to drop some columns
# ******************************************************************************************************
# ******************************************************************************************************

# 1a. Read
X_full = pd.read_csv('../input/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/test.csv', index_col='Id')

# 1b. Remove rows with missing target, separate target from predictors
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

# 1c. Break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, train_size=0.8, test_size=0.2, random_state=0)

# 1d. Cardinality means the number of unique values in a column - OH Encoder is coming
categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and  X_train_full[cname].dtype == "object"]

# 1e. Select numerical columns
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

# 1f. Keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()

print(X_train.shape)
X_train.head()




# ******************************************************************************************************
# ******************************************************************************************************
# 2. Pipelines - Bundle a preprocessor and a model
# ******************************************************************************************************
# ******************************************************************************************************

# 2a. Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# 2b. Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# 2c. Bundle preprocessing for numerical and categorical data into a preprocessor
preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)])

# 2d. Define model
model = RandomForestRegressor(n_estimators=100, random_state=0)

# 2e. Bundle preprocessing and modeling code in a pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

# 2f. Preprocessing of training data, fit model, preprocessing of validation data, make predictions
clf.fit(X_train, y_train)
preds = clf.predict(X_valid)
print('MAE:', mean_absolute_error(y_valid, preds))






# ******************************************************************************************************
# ******************************************************************************************************
# 3. Improve performance - One solution here is to change the SimpleImputer of numerical columns and model
# ******************************************************************************************************
# ******************************************************************************************************

# 3a. Preprocessing for numerical data
numerical_transformer = SimpleImputer()

# 3b. Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
	('onehot', OneHotEncoder(handle_unknown='ignore'))])

# 3c. Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)])

# 3d. Define model
model = RandomForestRegressor()

# 3e. Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

# 3f. Preprocessing of training data, fit model, preprocessing of validation data, make predictions
my_pipeline.fit(X_train, y_train)
preds = my_pipeline.predict(X_valid)
score = mean_absolute_error(y_valid, preds)
print('MAE:', score)





# ******************************************************************************************************
# ******************************************************************************************************
# 4. Generate test predictions from the 2nd dataset using X_test
# ******************************************************************************************************
# ******************************************************************************************************

preds_test = my_pipeline.predict(X_test)
output = pd.DataFrame({'Id': X_test.index, 'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)
