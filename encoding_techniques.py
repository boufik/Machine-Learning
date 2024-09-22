# 0. Basics
import pandas as pd

companies = ["AlfaRomeo", "Audi", "BMW", "Audi", "AlfaRomeo", "Mercedes", "Mercedes", "Audi", "BMW", "BMW"]
prices = [30000, 45000, 55000, 48000, 32000, 60000, 64000, 51000, 56000, 57000]
years = [2020, 2021, 2018, 2017, 2020, 2021, 2016, 2019, 2015, 2017]
regions = ["Milan", "Warsaw", "Munich", "Frankfurt", "Athens", "Berlin", "Munich", "Frankfurt", "Munich", "Berlin"]
dic = {"Company":companies, "Region":regions, "Year":years, "Price":prices}
df = pd.DataFrame(dic)
print(f"\n\n******** DataFrame Size = {df.shape} ********\n\n", df)
print(df.select_dtypes(["object"]))
print(df.select_dtypes(["object"]).nunique())



# 1. One-hot encoding - Creates N extra cols categorical per feature (N=unique values of this feature)
X1 = df.copy()
categorical_cols = [colname for colname in X1.columns if X1[colname].dtype in ["object", "category"]]
print(f"Categorical cols = {categorical_cols}")
for colname in categorical_cols:
    n_unique = X1[colname].nunique()
    unique_values = list(X1[colname].unique())
    print(f"----> There are {n_unique} unique values in column named {colname}:\n{unique_values}")


for colname in categorical_cols:
    dummies_temp_df = pd.get_dummies(X1[colname], prefix=colname)
    X1 = pd.concat([X1, dummies_temp_df], axis=1)

X1 = X1.drop(categorical_cols, axis=1)
print(f"\n\n******** DataFrame Size = {X1.shape} ********\n\n", X1)



# 2. Label Encoding - Does not create any extra columns
X2 = df.copy()
for colname in X2.select_dtypes("object"):
    X2[colname], _ = X2[colname].factorize()
discrete_features = X2.dtypes == int
print(f"******** DataFrame Size = {X2.shape} ********\n\n", X2, '\n\n', discrete_features)



# 3. Target Encoding - Creates 1 extra column per categorical feature
# I will "connect" each company with its mean price
X3 = df.copy()
X3["MeanPrice (Label_for_Company)"] = X2.groupby("Company")["Price"].transform("mean")
print(f"******** DataFrame Size = {X3.shape} ********\n\n", X3)
