# Import necessary packages
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle

# Load the training data
train_df = pd.read_csv("../Dataset/data.csv")

# Changing the data types to appropriate values
train_df['price']     = train_df['price'].astype('int64')
train_df['bedrooms']  = train_df['bedrooms'].astype('int64')
train_df['bathrooms'] = train_df['bathrooms'].astype('int64')
train_df['floors']    = train_df['floors'].astype('int64')
train_df['street']    = train_df['street'].astype('string')
train_df['city']      = train_df['city'].astype('string')
train_df['statezip']  = train_df['statezip'].astype('string')
train_df['country']   = train_df['country'].astype('string')

# Drop the duplicates
train_df.drop_duplicates(inplace=True)

# Missing value treatment
(train_df.price == 0).sum()
train_df['price'].replace(0, np.nan, inplace = True)
train_df.dropna(inplace=True)

# Change the date column to DateTime
train_df['date'] = pd.to_datetime(train_df['date'])

# Extract the year from Date Column
train_df.insert(1, "year", train_df.date.dt.year)

# Creating new columns
train_df['Age'] = 2023 - train_df['yr_built']
train_df['AgeRenovated'] = 2023 - train_df['yr_renovated']

# Drop the un-necessary columns
train_df = train_df.drop(['date','year', 'street', 'statezip', 'country','city',
                          'yr_built','yr_renovated'], axis = 1)

# Scale the datasets
columns = train_df.columns
scaler = MinMaxScaler(feature_range = (0, 1))
normal = pd.DataFrame(scaler.fit_transform(train_df), columns = columns)

# Split the target feature
y = train_df["price"]
train_df = train_df.drop('price', axis=1)

# Split the dataset in to train and test splits
X_train, X_test, y_train, y_test = train_test_split(train_df, y, test_size=0.3, random_state=42)

# Train the model
# Initialize the Linear Regression model
linear_model = LinearRegression()

# Train the model
linear_model.fit(X_train, y_train)

# Make predictions on the training set
y_train_pred = linear_model.predict(X_train)

# Make predictions on the test set
y_test_pred = linear_model.predict(X_test)

# Evaluate the model
mse_train = mean_squared_error(y_train, y_train_pred)
print(f'Mean Squared Error (MSE) for Training Set: {mse_train}')

r2_train = r2_score(y_train, y_train_pred)
print(f'R-squared Score for Training Set: {r2_train}')

mse_test = mean_squared_error(y_test, y_test_pred)
print(f'Mean Squared Error (MSE) for Test Set: {mse_test}')

r2_test = r2_score(y_test, y_test_pred)
print(f'R-squared Score for Test Set: {r2_test}')

# Save the model
with open("../Output/lr.pkl", 'wb') as model_file:
        pickle.dump(linear_model, model_file)