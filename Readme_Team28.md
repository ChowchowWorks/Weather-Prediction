# Weather Humidity Prediction

## Running the Project

This project code is submitted in the Colab format. Please follow these steps to execute the code:

1. Data File: Ensure that the weather_data.csv file is uploaded to your Colab environment.

2. Section 1 - CSV Handling:

   - Running this section will install the essential libraries (if not already installed) and load the weather_data.csv file into a Pandas DataFrame.

3. Section 2 - Helper Functions:

   - Running this section will initialize the helper functions used throughout the code for data preprocessing and window creation.

4. Section 3 - Unsupervised Learning:

   - This section performs unsupervised learning analysis on the weather data.
   - Running each subsection (a), (b), etc., will produce different outputs like correlation matrices, ACF/PACF plots, and stationarity test results.

5. Section 4 - Train-Test Split:

   - This section splits the data into training and testing sets.
   - Important: This section must be executed for the rest of the code to function correctly, as it prepares the data for model training and evaluation.

6. Section 5 - Baseline Model - Random Forest:
   - You will have to run the model 3 times, once for each hour prediction by following the above bullet point:
   - Before running the model please change the .shift(-1) to -1 for 1h, -6 for 6h and -24 for 2h predictions in the `def prepare_features_and_target` method of the `RandomForestWeatherPredictionModel` class:
     over here:
   ```
   self.df['target'] = self.df['relative_humidity'].shift(-1)
   ```
   - Finally run as follows:
   ```
   model = RandomForestWeatherPredictionModel(data_path='weather_data.csv', model_path='rf_1h.joblib')
   model.run()
   ```
7. Section 6 - XGBoost Model:

   - This section contains the XGBoost model implementation.
   - The code defines an XGBoost class and demonstrates how to train and evaluate the model.
   - You will have to run the below model 3x each time changing the self.Y = self.df[['humidity_1h']] in the split_data(self) each time as shown below:
   - The code has been set to predicting for 1hr forward under split_data(self) as shown below.
   - eg: self.Y = self.df[['humidity_1h']]
   - As for the 6h and 24 hour predictions, the df should be changed as such before running the code.
   - 6-hour predictions: self.Y = self.df[['humidity_6h']]
   - 24-hour predictions: self.Y = self.df[['humidity_24h']]

8. Section 7 - LSTM Model:

   - This section contains the LSTM model implementation and is divided into three segments:
     - Segment (a): Defines the LSTM class. Running this code initializes the class for use in later segments.
     - Segment (b): Performs hyperparameter tuning for the LSTM model using Optuna. Note: This segment may take a considerable amount of time to run.
     - Segment (c): Builds and trains the LSTM models using either the tuned hyperparameters or pre-defined ones (commented out at the top of the segment).

9. Section 8 - LSTM_XGB hybrid:

   - This section implements the hybrid LSTM-XGBoost model and is divided into eight segments:
     - Segment (a): Initializes the LSTM class used in the hybrid model.
     - Segment (b): Copies the layers and weights from the trained LSTM models in Section 7. Important: Ensure Section 7 has been executed before running this segment.
     - Segment (c): Creates feature matrices by passing the training and testing data through the LSTM model. Important: Ensure you have six different feature matrix arrays generated at the end of this segment.
     - Segment (d): Initializes the XGBoost model used in the hybrid model.
     - Segment (e): Splits the feature matrices into X (features) and y (target) values.
     - Segment (f): Contains the hyperparameter tuning function for the XGBoost part of the hybrid model.

10. Section 9 - Predictions and Evaluation

- This section performs predictions using the implemented models and evaluates their performance using suitable metrics.
