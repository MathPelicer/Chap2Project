#%% Download and Decompress Archive

import os
import tarfile
from six.moves import urllib

downloadRoot = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
housingPath = os.path.join("datasets", "housing")
housingURL = downloadRoot + "datasets/housing/housing.tgz"

def fetching_housing_data(housing_url = housingURL, housing_path = housingPath):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")

    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


fetching_housing_data()

#%% Loading data into Pandas

import pandas as pd

def load_housing_data(housing_path = housingPath):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

#%%Looking at the data

housing = load_housing_data()
housing.head()

#%% Looking at data pt. 2

housing.info()

# %% Ocean_proximity values

housing["ocean_proximity"].value_counts()


# %% Describing

housing.describe()


# %% Histogram

import matplotlib.pyplot as plt

housing.hist(bins = 50, figsize=(20, 15))
plt.show()


# %% Setting the test set function

import numpy as np

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# %% Creating the test set

train_set, test_set = split_train_test(housing, 0.2)
print(len(train_set), "train +", len(test_set), "test")


# %% Limiting the categories of the house incoming

housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

housing["income_cat"].hist(bins = 50, figsize=(20, 15))
plt.show()

# %% sampling based on house incoming

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

strat_test_set["income_cat"].value_counts() / len(strat_test_set)


# %% Removing income_cat

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)


# %% Creating a copy of the data set for training

housing = strat_train_set.copy()


# %% Dispertion diagram creation

housing.plot(kind = "scatter", x="longitude", y = "latitude", alpha=0.4,
             s = housing["population"] / 100, label = "population", figsize=(10, 7),
             c = "median_house_value", cmap = plt.get_cmap("jet"), colorbar = True)

plt.legend()


# %% Correlation between attributes

corr_matrix = housing.corr()

corr_matrix["median_house_value"].sort_values(ascending=False)


# %% Correlation plotted between some attributes

from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]

scatter_matrix(housing[attributes], figsize=(12, 8))

# %% Plotting median_income x median_house_value

housing.plot(kind = "scatter", x = "median_income", y = "median_house_value", alpha=0.1)


# %% Creating new attributes

housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]

corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


# %% Copying strat_train_test again

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()


# %% Cleaning the data set

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")

housing_num = housing.drop("ocean_proximity", axis=1)

imputer.fit(housing_num)
print(imputer.statistics_)

X = imputer.transform(housing_num)

housing_tr = pd.DataFrame(X, columns=housing_num.columns)




# %% TRansforming Ocean Proximity Strings to numbers

housing_cat = housing["ocean_proximity"]

housing_cat_encoded, housing_categories = housing_cat.factorize()
housing_cat_encoded[:10]

housing_categories


# %% Turning category (ocean_proximity) into one-hot vector (1 or 0 vector with only one 1 per line)

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()

housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1, 1))
housing_cat_1hot


# %% Customizing transformers

from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinatedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y = None):
        return self
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinatedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

# %% Transformation Pipelines

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinatedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)


# %% Custom transformator to deal with pandas dataframes in scikit-learn

from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y = None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values


# %% Categorical attribs pipeline

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinatedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])


# %% Sticking those pipelines together

from sklearn.compose import ColumnTransformer

full_pipeline = ColumnTransformer([
        ('num_pipeline', num_pipeline, num_attribs),
        ('cat_pipeline', OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)
housing_prepared.shape


# %%Training a linear regression

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions: ", lin_reg.predict(some_data_prepared))
print("Labels: ", list(some_labels))


# %% Measuring Errors

from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# %% Training the model with a new method (Tree Regression)

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)


# %% Using the new method

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse


# %% K-fold Validation

from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)

tree_rmse_scores = np.sqrt(-scores)

# %% Displaying results 

def DisplayResults(scores):
    print("Scores: ", scores)
    print("Mean:", scores.mean())
    print("Standart deviation: ", scores.std())

DisplayResults(tree_rmse_scores)

# %% Cross validation for the linear regression

lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                            scoring="neg_mean_squared_error", cv=10)

lin_rmse_scores = np.sqrt(-lin_scores)

DisplayResults(lin_rmse_scores)


# %% Random forests

from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)

housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse


# %% Displaying results for random forests

forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                scoring="neg_mean_squared_error", cv=10)

forest_rmse_scores = np.sqrt(-forest_scores)

DisplayResults(forest_rmse_scores)


# %% Best hiperparemeter values

from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv = 5, scoring='neg_mean_squared_error')

grid_search.fit(housing_prepared, housing_labels)

# %% Best params

grid_search.best_params_

# %% Best estimator

grid_search.best_estimator_


# %% Scores of the avaliation

cvres = grid_search.cv_results_

for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


# %% Relative importance

feature_importances = grid_search.best_estimator_.feature_importances_

feature_importances


# %%    More importances with their attribs

extra_attribs = ["roooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_["cat_pipeline"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse = True)


# %% Evaluationg the system

final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis = 1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)


# %%
