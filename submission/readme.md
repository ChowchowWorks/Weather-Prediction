This project code is submitted in the Colab format. 
1. Ensure that you have the weather_data.csv file is in your files
2. Running Section 1 will download the essential libraries and load the csv file into a pandas dataframe. 
3. Running Section 2 will initialse the helper functions used throughout the code. 
4. Section 3 begins the Unsupervised learning segment, running each part (a), (b), etc will result in different outputs
5. Section 4 contains the Train-Test split, this section must run for the rest of the code to work.
6. Section 5 contains our baseline Random Forest model, it has only one section, this block of code will only build the model. 
7. Section 6 contains the XGboost model. This block of code shows the XGBoost class that we have implemented. 
8. Section 7 contains the LSTM model. There are 3 segments in this section.
    - Segment (a): This contains the LSTM class that we have implemented, running this code initialises the LSTM class. 
    - Segment (b): This contains the Hyperparameter tuning function of the model. Running this code will take a long time. 
    - Segment (c): In the event that segment(b) exhausts the run time, use the hyperparameters commented out at the top of segment (c).
9. Section 8 contains the LSTM_XGB class. There are 8 segments in this section. 
    - Segment (a): This initialises the LSTM class used in LSTM-XGB
    - Segment (b): This segment copies over the layers and weights from the LSTM models in Section 7. Ensure that Section 7 is been initialised before this. 
    - Segment (c): This segment passes through the training and testing data through the LSTM model. Ensure that at the end of this segment, you have 6 different feature matrix arrays.
    - Segment (d): This segment intialises the XGB model used in LSTM-XGB
    - Segment (e): This segment splits the feature matrix into X values and y values. 
    - Segment (f): This segment contains the hyperparamter tuning functions for Bayesian Optimisation. Run this function before conducting the next segment. 
    - Segment (g): This segment runs the hyperparameter tuning for each model. 
    - Segment (h): This segment build the XGB models that are used to predict relative humidity. 
10. Section 9 contains the results and evaluation codes for each model. There are 4 segments to this section. 
    - Segment (a): This segment runs the predictions for Random Forest and displays the evaluation metrics as the output. 
    - Segment (b): This segment runs the predictions for XGBoost and displays the evaluation metrics as the output. 
    - Segment (c): This segment runs the predictions and evaluations for the LSTM model and displays the evaluation metrics as the output. 
    - Segment (d): This segment runs the predictions and evaluations for the LSTM-XGB model and displays the evalutions metrics as the output. 