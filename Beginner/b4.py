# *************************************************************
# *************************************************************
# 0. Function for evaluating a model according to max_leaf_nodes parameter
# *************************************************************
# *************************************************************

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)



# *************************************************************
# *************************************************************
# 1. Dataset, y, X, split, predict without parameters chosen
# *************************************************************
# *************************************************************

# Code you have previously used to load data
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'
home_data = pd.read_csv(iowa_file_path)

y = home_data.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
iowa_model = DecisionTreeRegressor(random_state=1)
iowa_model.fit(train_X, train_y)

# Make validation predictions and calculate mean absolute error
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE: {:,.0f}".format(val_mae))



# *************************************************************
# *************************************************************
# 2. Compare different tree sizes
# *************************************************************
# *************************************************************

candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
min_mae = 10**10
index = -1
for i in range(len(candidate_max_leaf_nodes)):
    max_leaf_nodes = candidate_max_leaf_nodes[i]
    mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print(max_leaf_nodes, mae)
    if mae < min_mae:
        min_mae = mae
        index = i
    

# Store the best value of max_leaf_nodes (it will be either 5, 25, 50, 100, 250 or 500)
best_tree_size = candidate_max_leaf_nodes[index]
print('\n', best_tree_size)



# *************************************************************
# *************************************************************
# 3. Fit model using ALL THE DATA with the right value of parameter max_leaf_nodes
# *************************************************************
# *************************************************************

# Fill in argument to make optimal size
final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=0)
final_model.fit(X, y)
