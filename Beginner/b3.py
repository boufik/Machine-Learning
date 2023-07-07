# *****************************************************************
# *****************************************************************
# 1. Read the dataframe
# *****************************************************************
# *****************************************************************

import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'
home_data = pd.read_csv(iowa_file_path)

y = home_data.SalePrice
feature_columns = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[feature_columns]

# Specify and Fit the Model in all the samples from the dataset (wrong methodology)
iowa_model = DecisionTreeRegressor()
iowa_model.fit(X, y)

print("First in-sample predictions:", iowa_model.predict(X.head()))
print("Actual target values for those homes:", y.head().tolist())



# *****************************************************************
# *****************************************************************
# 2. Split the dataset with a function from sklearn
# *****************************************************************
# *****************************************************************

from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)



# *****************************************************************
# *****************************************************************
# 3. Split and fit the model into the training data (correct methodology)
# *****************************************************************
# *****************************************************************

iowa_model = DecisionTreeRegressor(random_state=1)
iowa_model.fit(train_X, train_y)



# *****************************************************************
# *****************************************************************
# 4. Predictions with validation-testing data
# *****************************************************************
# *****************************************************************

val_predictions = iowa_model.predict(val_X)
# print the top few validation predictions
print(val_predictions, '\n')
# print the top few actual prices from validation data
val_y.head()



# *****************************************************************
# *****************************************************************
# 5. Mean Absolute Error
# *****************************************************************
# *****************************************************************

from sklearn.metrics import mean_absolute_error
val_mae = mean_absolute_error(val_predictions, val_y)
print(val_mae)
