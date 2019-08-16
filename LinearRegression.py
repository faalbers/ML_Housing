"""
    Training Data using Linear regression
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

# local dataset path to extracted data to
HOUSING_PATH = os.path.join("datasets", "housing")

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True): # no *args or **karg
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        try:
            X = check_array(X, accept_sparse='csc', dtype=FLOAT_DTYPES,
                            force_all_finite=True, copy=True)
        except ValueError as ve:
            if "could not convert" in str(ve):
                raise ValueError("Cannot use {0} strategy with non-numeric "
                                 "data. Received datatype :{1}."
                                 "".format(self.strategy, X.dtype.kind))
            else:
                raise ve
        rooms_per_household = X[:, 3] / X[:, 6]
        population_per_household = X[:, 5] / X[:, 6]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, 4] / X[:, 3]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

    def addedAttributes(self):
        attrNames = ["rooms_per_household", "population_per_household"]
        if self.add_bedrooms_per_room:
            attrNames.append("bedrooms_per_room")
        return attrNames

def load_housing_data(housing_path=HOUSING_PATH):
    """ Load data from database file and return as Panda DataFrame
    """
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

def split_train_test_by_cat_strat(data, test_ratio, cat_attribute):
    """Create train and test set based on category stratified sampling.
    Test set ratio (0.0 to 1.0) determins of all data that goes to test set.
    """

    split = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=42)
    for train_index, test_index in split.split(data, data[cat_attribute]):
        return data.loc[train_index], data.loc[test_index]

def showModelResults(model, training_data_prepared, training_labels, length):
    # Compare prediction
    model_predict = model.predict(training_data_prepared[:length])
    i = 0
    for label in training_labels[:length]:
        diff = 100.0 * (model_predict[i] - label) / label
        print("%f --> %f = %f%s" % (label, model_predict[i], diff, '%'))
        i += 1

def showModelRMSE(model, training_data_prepared, training_labels):
    rmse = mean_squared_error(training_labels, model.predict(training_data_prepared))
    rmse = np.sqrt(rmse)
    print("Training data RMSE = %f" % rmse)

def showModelRMSE_CrossVal(model, training_data_prepared, training_labels, cv):
    scores = cross_val_score(model, training_data_prepared, training_labels,
        scoring="neg_mean_squared_error", cv=cv)
    rmse_scores = np.sqrt(-scores)
    print("Validation Set RMSE =", end='')
    for score in rmse_scores:
        print(" %.2f" % score, end='')
    print("\nValidation Set RMSE Mean = %.f" % rmse_scores.mean())
    print("Validation Set RMSE Deviation = %.f" % rmse_scores.std())

def main():
    # Retrieve data from database file as Panda DataFrame
    housing = load_housing_data()

    # We can do random sampling of data to split between train and test sets
    # But that could introduce a bias in the median_income data.
    # To avoid that, we will applying stratified sampling on median income

    # We will create the homogeneous subgroups, aka strata
    # There should be enough strata , but not too many so we don't introduce
    # bias . We will limit to 5 income categories by deviding data by 1.5 and
    # ceil to round numbers
    housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
    housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace = True)

    # Create train and test set based on category stratified sampling.
    # Test set will be 20% and Train set 80% of all data
    train_set, test_set = split_train_test_by_cat_strat(housing, 0.2, "income_cat")

    #remove income_cat attribute on both sets
    for set_ in (train_set, test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    #prepare train set for ML Algorithms

    #seperate median_house_value attribute from training data and create labels out of it
    housing = train_set.drop("median_house_value", axis=1)
    housing_labels = train_set["median_house_value"].copy()

    #remove ocean proximity attribute so we can fill missing data on float only values
    housing_num = housing.drop("ocean_proximity", axis=1)
    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]

    # pipeline to handle numeric data
    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder(add_bedrooms_per_room=False)),
        ('std_scaler', StandardScaler()),
    ])

    # pipeline to categorize text data
    cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('cat_encoder', OneHotEncoder(sparse=False)),
    ])

    # prepare full data
    full_pipeline = FeatureUnion(transformer_list=[
        ('num_pipeline', num_pipeline),
        ('cat_pipeline', cat_pipeline),
    ])

    # final prepared training set
    housing_prepared = full_pipeline.fit_transform(housing)
    housing_prepared_attribs = num_attribs.copy() + list(cat_pipeline.named_steps["cat_encoder"].categories_[0])

    # lets train using linear regression
    print("\nTesting Linear Regression Model on partial training data...")
    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)
    showModelResults(lin_reg, housing_prepared, housing_labels, 5)
    showModelRMSE(lin_reg, housing_prepared, housing_labels)
    showModelRMSE_CrossVal(lin_reg, housing_prepared, housing_labels, 10)

if __name__== "__main__":
    main()
