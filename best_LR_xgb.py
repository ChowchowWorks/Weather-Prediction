def find_best_learning_rate(X_train_1h, y_train_1h, X_test_1h, y_test_1h,
                            X_train_6h, y_train_6h, X_test_6h, y_test_6h,
                            X_train_24h, y_train_24h, X_test_24h, y_test_24h):
    learning_rates = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3]
    test_mae_scores = []

    for lr in learning_rates:
        model_1h = XGBRegressor(n_estimators=300, learning_rate=lr, max_depth=8, objective='reg:squarederror')
        model_6h = XGBRegressor(n_estimators=300, learning_rate=lr, max_depth=8, objective='reg:squarederror')
        model_24h = XGBRegressor(n_estimators=300, learning_rate=lr, max_depth=8, objective='reg:squarederror')
        
        model_1h.fit(X_train_1h, y_train_1h)
        model_6h.fit(X_train_6h, y_train_6h)
        model_24h.fit(X_train_24h, y_train_24h)
        
        test_predictions_1h = model_1h.predict(X_test_1h)
        test_predictions_6h = model_6h.predict(X_test_6h)
        test_predictions_24h = model_24h.predict(X_test_24h)
        
        test_mae_1h = mean_absolute_error(y_test_1h, test_predictions_1h)
        test_mae_6h = mean_absolute_error(y_test_6h, test_predictions_6h)
        test_mae_24h = mean_absolute_error(y_test_24h, test_predictions_24h)
        
        total_test_mae = test_mae_1h + test_mae_6h + test_mae_24h
        test_mae_scores.append(total_test_mae)
    
    # Plot Learning Rate vs Total Test MAE
    plt.figure(figsize=(8, 5))
    plt.plot(learning_rates, test_mae_scores, marker='s', linestyle='-', color='r', label='Total Test MAE')
    plt.xlabel("Learning Rate")
    plt.ylabel("Total Mean Absolute Error (MAE)")
    plt.title("Learning Rate vs Total Test MAE")
    plt.legend()
    plt.grid()
    plt.show()
    
    best_lr = learning_rates[np.argmin(test_mae_scores)]
    print("Best Learning Rate (Based on Total Test MAE):", best_lr)
    return best_lr
