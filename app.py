import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

data = pd.read_csv('E:\DATA Science\Machine Learning\CAR DETAILS.csv')
data.head()

print(data.describe())
data.info()

print(data.isnull().sum())

sns.boxplot(data['selling_price'])
plt.title('selling price over the years')
plt.xticks(rotation=90)
plt.xlabel('Year')
plt.ylabel('selling price')
plt.show()

data['fuel'] = data['fuel'].str.strip().str.lower()

data['year'] = data['year'].astype(int)
data['selling_price'] = data['selling_price'].astype(float)
data['km_driven'] = data['km_driven'].astype(float)

data['car_age'] = 2024 - data['year']

sns.histplot(data['selling_price'], kde=True, bins=30, color='blue')
plt.title('Distribution of Selling Price')
plt.xlabel('Selling_Price')
plt.ylabel('Frequency')
plt.show()

sns.countplot(x='year', data=data, hue="year", palette='viridis')
plt.title('Count of Cars by Year')
plt.xticks(rotation=90)
plt.xlabel('Year of Manufacture')
plt.ylabel('Number of Cars')
plt.show()

fuel_counts = data['fuel'].value_counts()
fuel_counts.plot(kind='bar', color='orange', figsize=(8, 6))
plt.title('Distribution of Fuel Types')
plt.xlabel('Fuel')
plt.ylabel('Number of Cars')
plt.show()

sns.boxplot(x='year', y='selling_price', data=data, hue="year", palette='coolwarm')
plt.title('Selling Price vs Year')
plt.xlabel('Year of Manufacture')
plt.ylabel('Selling Price')
plt.xticks(rotation=90)
plt.show()

sns.boxplot(x='fuel', y='selling_price', data=data, hue="fuel", palette='muted')
plt.title('Selling Price vs Fuel Type')
plt.xlabel('Fuel Type')
plt.ylabel('Selling Price')
plt.show()

plt.figure(figsize=(8, 6))
sns.relplot(x = 'km_driven', y = 'selling_price', data=data, hue='fuel', palette='deep', alpha=0.7)
plt.title('Selling Price vs Kilometer Driven')
plt.xlabel('Km Driven')
plt.ylabel('Selling Price')
plt.xticks(rotation=90)
plt.show()

sns.catplot(x='fuel', y='selling_price', hue='transmission', kind='box', data=data, height=6, aspect=1.5, palette='Set2')
plt.title('Selling Price by Fuel Type and Transmission')
plt.show()

sns.barplot(x='owner', y='selling_price', data=data, hue="seller_type", palette='dark')
plt.title('Selling Price by Ownership Type')
plt.xlabel('Owner Type')
plt.ylabel('Selling Price')
plt.show()

corr = data[['selling_price', 'km_driven', 'year']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

sns.pairplot(data, vars=['selling_price', 'km_driven', 'year'], hue='fuel', palette='husl')
plt.title('Pair Plot of Numerical Variables')
plt.show()

plt.savefig('graph_name.png')

data.drop(columns=['name'], inplace=True)

print(data['seller_type'].unique())

print(data['owner'].unique())

data.replace({'fuel':{'petrol':0, 'diesel':1, 'cng':2, 'electric':3, 'lpg':4}}, inplace=True)
data.replace({'transmission':{'Manual':0, 'Automatic':1}}, inplace=True)

data.replace({'seller_type':{'Individual':0, 'Dealer':1, 'Trustmark Dealer':2}}, inplace=True)
data.replace({'owner':{'First Owner':0, 'Second Owner':1, 'Third Owner':2, 'Fourth & Above Owner':3, 'Test Drive Car':4}}, inplace=True)

correlation_matrix = data.corr()
print(correlation_matrix['selling_price'].sort_values(ascending=False))

categorical_columns = ['fuel', 'transmission', 'owner']

X = data.drop(columns=['selling_price'])  # Features
y = data['selling_price']                # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Create a preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(), categorical_columns)
    ])

# Apply transformations
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

Q1 = data['km_driven'].quantile(0.25)
Q3 = data['km_driven'].quantile(0.75)
IQR = Q3 - Q1

# Apply capping
data['km_driven'] = np.where(data['km_driven'] < Q1 - 1.5 * IQR, Q1 - 1.5 * IQR,
                                    np.where(data['km_driven'] > Q3 + 1.5 * IQR, Q3 + 1.5 * IQR,
                                             data['km_driven']))

print(X_train.shape, X_test.shape)
print(X_train[:5])  # View a sample of the transformed training data

np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate_model(model, X_train, y_train, X_test, y_test):
    # Train the model
    model.fit(X_train, y_train)

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Metrics for training and testing data
    metrics = {
        'Model': model.__class__.__name__,
        'Train_R2': r2_score(y_train, y_pred_train),
        'Test_R2': r2_score(y_test, y_pred_test),
        'MAE': mean_absolute_error(y_test, y_pred_test),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
    }
    return metrics

lr = LinearRegression()
lr_metrics = evaluate_model(lr, X_train, y_train, X_test, y_test)
print(lr_metrics)

ridge = Ridge(alpha=1.0)
ridge_metrics = evaluate_model(ridge, X_train, y_train, X_test, y_test)
print(ridge_metrics)

lasso = Lasso(alpha=0.1)
lasso_metrics = evaluate_model(lasso, X_train, y_train, X_test, y_test)
print(lasso_metrics)

dt = DecisionTreeRegressor(max_depth=10, random_state=42)
dt_metrics = evaluate_model(dt, X_train, y_train, X_test, y_test)
print(dt_metrics)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf_metrics = evaluate_model(rf, X_train, y_train, X_test, y_test)
print(rf_metrics)

gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gbr_metrics = evaluate_model(gbr, X_train, y_train, X_test, y_test)
print(gbr_metrics)

bagging = BaggingRegressor(base_estimator=DecisionTreeRegressor(), n_estimators=50, random_state=42)
bagging_metrics = evaluate_model(bagging, X_train, y_train, X_test, y_test)
print(bagging_metrics)

import pandas as pd

# Collect metrics
results = pd.DataFrame([lr_metrics, ridge_metrics, lasso_metrics, dt_metrics, rf_metrics, gbr_metrics, bagging_metrics])
print(results.sort_values(by='Test_R2', ascending=False))  # Sort by Test R2 Score

import joblib

# Save the model
joblib.dump(rf, 'best_car_price_model.pkl')

# Load the model
loaded_model = joblib.load('best_car_price_model.pkl')

import joblib

# Assume 'rf' is the best model (e.g., Random Forest Regressor)
best_model = rf

# Save the model to a file
joblib.dump(best_model, 'best_car_price_model.pkl')

print("Model saved as 'best_car_price_model.pkl'")

# Load the model from the file
loaded_model = joblib.load('best_car_price_model.pkl')

print("Model loaded successfully.")

# Assume `X_test` contains test data features
predictions = loaded_model.predict(X_test)

# Display predictions
print("Predictions for the test set:")
print(predictions[:20])  # Show the first 20 predictions

# Save the model with versioning
joblib.dump(best_model, 'best_car_price_model_v1.pkl')

# Load a specific version
loaded_model_v1 = joblib.load('best_car_price_model_v1.pkl')

# Save in `.sav` format
joblib.dump(best_model, 'best_car_price_model.sav')

# Load the `.sav` model
loaded_model_sav = joblib.load('best_car_price_model.sav')








