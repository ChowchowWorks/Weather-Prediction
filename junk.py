# Train XGBoost model on 1-hour, 6-hour, and 24-hour features
def train_xgb_model(X_train, y_train, X_test, y_test):
    model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=7, objective='reg:squarederror')
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Evaluate performance
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    
    return predictions, mae, mse, rmse

# Train models for different time horizons and evaluate
predictions_1hr, mae_1hr, mse_1hr, rmse_1hr = train_xgb_model(X_train_1hr, y_train_1hr, X_test_1hr, y_test_1hr)
predictions_6hr, mae_6hr, mse_6hr, rmse_6hr = train_xgb_model(X_train_6hr, y_train_6hr, X_test_6hr, y_test_6hr)
predictions_24hr, mae_24hr, mse_24hr, rmse_24hr = train_xgb_model(X_train_24hr, y_train_24hr, X_test_24hr, y_test_24hr)


lstm_1hr.compile(optimizer = 'rmsprop', loss = mean_absolute_error)
lstm_6hr.compile(optimizer = 'rmsprop', loss = mean_absolute_error)
lstm_24hr.compile(optimizer = 'rmsprop', loss = mean_absolute_error)