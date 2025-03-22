**Car Selling Price Prediction**


**Overview**

This project aims to build a machine learning model to predict car selling prices based on various features such as mileage, year, fuel type, and more. The model is evaluated using standard regression metrics to assess its accuracy and reliability.

**Dataset**

The dataset consists of multiple features influencing car prices, including:

Year: Manufacturing year of the car.

Present Price: Current market price of the car.

Kms Driven: Distance traveled by the car in kilometers.

Fuel Type: Petrol, Diesel, or CNG.

Seller Type: Individual or Dealer.

Transmission: Manual or Automatic.

Owner: Number of previous owners.

Selling Price: The target variable representing the actual selling price.

**Project Workflow**

Data Preprocessing

Handling missing values.

Encoding categorical variables.

Scaling numerical features.

Exploratory Data Analysis (EDA)

Understanding data distribution.

Checking for outliers using boxplots.

Correlation analysis between features.

Model Training & Evaluation

Splitting the data into training and testing sets.

Training models such as Linear Regression, Decision Tree, and Random Forest.

Evaluating models using:

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

R² Score

Comparing performance to select the best model.

**Model Deployment**

Saving the trained model using Joblib.

Loading the saved model for making predictions on new data.

**Random Sample Predictions**

Selecting 20 random data points from the dataset.

Predicting selling prices and comparing them with actual values.

Assessing model performance on this subset.

**Results & Insights**

The best-performing model was identified based on evaluation metrics.

The predicted selling prices were close to actual values, proving the model's effectiveness.

The model can be further improved with hyperparameter tuning and additional features.

**How to Use the Model**

**Clone this repository:**

git clone https://github.com/your-repo/car-price-prediction.git

**Install dependencies:**

pip install -r requirements.txt

**Run the script to train the model:**

python train_model.py

**Save the model using Joblib:**

import joblib
joblib.dump(model, 'car_price_model.pkl')

**Load the model and make predictions:**

model = joblib.load('car_price_model.pkl')
predictions = model.predict(new_data)

**Future Improvements**

Hyperparameter tuning for better performance.

Incorporating more features like brand, maintenance cost, and insurance history.

Deploying the model using Flask or FastAPI for real-time predictions.




Shweta Sonkar (Email: shwetasonkar108@gmail.com)
