#### 1. Read the dataset and separate target from predictors

# Read the dataset
X = pd.read_csv('../input/train.csv', index_col='Id')  
# Remove rows with missing target
X.dropna(axis=0, subset=['SalePrice'], inplace=True)
# Separate target from predictors
y = X.SalePrice
X.drop(['SalePrice'], axis=1, inplace=True)


#### 2. Split the dataset

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)


#### 3. Separate into numerical and categorical columns

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]
categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype == "object" and X_train_full[cname].nunique() < 10]		# for OH encoding
categorical_cols_ALL = [cname for cname in X_train_full.columns if X_train_full[cname].dtype == "object"]
print('Numerical columns:', numerical_cols, '\nCategorical columns:', categorical_cols, '\n\n')

my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()



#### 4. Cleaning - Drop cols with missing data - Φτιάχνω μεταβλητή με τα ονόματα των στηλών αυτών

cols_with_missing = [col for col in X.columns if X[col].isnull().any()] 
X.drop(cols_with_missing, axis=1, inplace=True)
X_test.drop(cols_with_missing, axis=1, inplace=True)


#### 5. Preprocessing - Drop columns with categorical data

drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])
print("MAE from Approach 1 (Drop categorical variables):")
print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))


#### 6. Preprocessing - Ordinal encoding ----> (object_cols = good_label_cols + bad_label_cols)
#### If I have some categorical variables in which there is the 'order attribute' (like "Beginner", "Amateur", "Advanced"), I can
#### make ordinal encoding. But first, I have to check if the values of these features are consistent through both training and
#### validation dataset. I can do so, by creating the following 2 variables named: good_label_cols and bad_label_cols

object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]
good_label_cols = [col for col in object_cols if set(X_valid[col]).issubset(set(X_train[col]))]
bad_label_cols = list(set(object_cols)-set(good_label_cols))
print('Categorical columns (ALL OF THEM):', object_cols, '\n')
print('Categorical columns that will be ordinal encoded:', good_label_cols, '\n')
print('\nCategorical columns that will be dropped from the dataset:', bad_label_cols, '\n')

label_X_train = X_train.drop(bad_label_cols, axis=1)
label_X_valid = X_valid.drop(bad_label_cols, axis=1)
 
ordinal_encoder = OrdinalEncoder()
label_X_train[good_label_cols] = ordinal_encoder.fit_transform(label_X_train[good_label_cols])
label_X_valid[good_label_cols] = ordinal_encoder.transform(label_X_valid[good_label_cols])



#### 7. Preprocessing - One Hot encoding ----> (object_cols = low_cardinality_cols + high_cardinality_cols)
#### For features that do not have the 'order-attribute' (Company: 'Microsoft', 'Amazon', 'Google'). OH Encoding creates new columns in
#### the dataset. The number of the columns created are equal to the number this categorical variable can take. In the above example, #### there are 3 companies in the 'Company' column, so OH Encoder will create 3 columns. But after that, we have to delete the old
#### column that contained this information. So, the new dataset has 2 more columns than the original one. Ofc, if there are more than
#### one columns that have this kind of values, we do the same thing.

object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]
object_nunique = list(map(lambda col: X_train[col].nunique(), object_cols))
d = dict(zip(object_cols, object_nunique))
sorted(d.items(), key=lambda x: x[1])
print(d)

low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < 10]
high_cardinality_cols = list(set(object_cols)-set(low_cardinality_cols))
print('Categorical columns that will be one-hot encoded:', low_cardinality_cols)
print('\nCategorical columns that will be dropped from the dataset:', high_cardinality_cols)
 
OH_X_train = X_train.drop(high_cardinality_cols, axis=1)
OH_X_valid = X_valid.drop(high_cardinality_cols, axis=1)

oh_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(oh_encoder.fit_transform(X_train[low_cardinality_cols]))
OH_cols_valid = pd.DataFrame(oh_encoder.transform(X_valid[low_cardinality_cols]))
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

number_X_train = X_train.drop(object_cols, axis=1) 
number_X_valid = X_valid.drop(object_cols, axis=1)
 
OH_X_train = pd.concat([OH_cols_train, number_X_train], axis=1)
OH_X_valid = pd.concat([OH_cols_valid, number_X_valid], axis=1)




#### 8. Pipelines - Preprocessor and Model

numerical_transformer = SimpleImputer(strategy='constant')
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])
 
preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)])
model = RandomForestRegressor(n_estimators=100, random_state=0)
 

clf = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
clf.fit(X_train, y_train)
preds = clf.predict(X_valid)
print('MAE:', mean_absolute_error(y_valid, preds))



#### 9. Output to CSV file - Requires a 2nd dataset ONLY FOR TESTING after your having validated your model
preds_test = my_pipeline.predict(X_test)
output = pd.DataFrame({'Id': X_test.index, 'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)
