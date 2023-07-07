import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


# *******************************************************************************************************
# *******************************************************************************************************
# 0. Function that returns the average score - Input = number of trees examined in the RFR model
# *******************************************************************************************************
# *******************************************************************************************************

def get_score(n_estimators):
    
    my_pipeline = Pipeline(steps=[
    ('preprocessor', SimpleImputer()), ('model', RandomForestRegressor(n_estimators=n_estimators, random_state=0))])
    scores = -1 * cross_val_score(my_pipeline, X, y, cv=3, scoring='neg_mean_absolute_error')
    avg = scores.mean()
    print("Average MAE score:", avg)
    return avg




# *******************************************************************************************************
# *******************************************************************************************************
# 1. Basics
# *******************************************************************************************************
# *******************************************************************************************************

# 1a. Read the 2 datasets
train_data = pd.read_csv('../input/train.csv', index_col='Id')
test_data = pd.read_csv('../input/test.csv', index_col='Id')

# 1b. Remove rows with missing target, separate target from predictors
train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = train_data.SalePrice              
train_data.drop(['SalePrice'], axis=1, inplace=True)

# 1c. Select numeric columns only (Not OH encoding below)
numeric_cols = [cname for cname in train_data.columns if train_data[cname].dtype in ['int64', 'float64']]
X = train_data[numeric_cols].copy()
X_test = test_data[numeric_cols].copy()
X.head()





# *******************************************************************************************************
# *******************************************************************************************************
# 2. Create a simple pipeline with a simple imputer for numerical columns that selected before
# *******************************************************************************************************
# *******************************************************************************************************

my_pipeline = Pipeline(steps=[
    ('preprocessor', SimpleImputer()), ('model', RandomForestRegressor(n_estimators=50, random_state=0))])





# *******************************************************************************************************
# *******************************************************************************************************
# 3. Create 5 folds in the dataset - Conduct 5 experiments with 5 different scores - Find the average
# *******************************************************************************************************
# *******************************************************************************************************

# Multiply by -1 since sklearn calculates *negative* MAE
scores = -1 * cross_val_score(my_pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')
print("Average MAE score:", scores.mean())




# *******************************************************************************************************
# *******************************************************************************************************
# 4. Test different parameter values - Variable n_estimators
# *******************************************************************************************************
# *******************************************************************************************************

# 4a. Create a dictionary of keys and values
n_estimators_list = list(range(50, 401, 50))
results = {n_estimators: get_score(n_estimators) for n_estimators in n_estimators_list}
print(results)

# 4b. Plot the results
%matplotlib inline
plt.plot(list(results.keys()), list(results.values()))
plt.show()
n_estimators_best = 200
