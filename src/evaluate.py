import numpy as np  # 'import' is used to bring in 'numpy', a library for numerical operations in Python. 'np' is the alias used to refer to numpy functions.
import matplotlib.pyplot as plt  # Imports 'matplotlib.pyplot' under the alias 'plt'. This library is used for creating plots and visualizing data.
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # Imports specific metrics ('mean_absolute_error', 'mean_squared_error', 'r2_score') from scikit-learn, which are used to evaluate model performance.
import logging  # Imports 'logging' to keep track of the model evaluation steps with log messages.

# Set up logging to track each step in model evaluation
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  # Configures logging to display time, log level, and message.


# Major Task: Define a function to evaluate the performance of the model
def evaluate_model(model, x_test, y_test, x_train=None, y_train=None):  # 'def' defines the function named 'evaluate_model'. It takes a trained model and test data, with optional training data for cross-validation.
    """
    Evaluates the given model on test data and computes various performance metrics. Optionally, performs cross-validation on training data.  # A docstring explaining what the function does.
    Parameters:
    - model: The trained machine learning model to evaluate.
    - x_test, y_test: The test data (input features and labels).
    - x_train=None, y_train=None: Optional training data (default is None).

    Returns: None.
    """

    # Major Task: Predict the values for the test data and calculate evaluation metrics
    logging.info("Starting model evaluation...")  # Logs a message indicating that evaluation is starting.
    y_pred = model.predict(x_test)  # 'model.predict()' uses the trained model to make predictions ('y_pred') on the test data ('x_test').

    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_test,y_pred)  # Computes the Mean Absolute Error, which measures the average absolute difference between predicted and actual values.
    logging.info(f"MAE: {mae}")  # Logs the MAE value.

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_test,y_pred)  # Computes the Mean Squared Error, which squares the difference between predicted and actual values, thus magnifying larger errors.
    logging.info(f"MSE: {mse}")  # Logs the MSE value.

    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)  # Takes the square root of the MSE to compute the RMSE, which is in the same units as the target variable (house prices).
    logging.info(f"RMSE: {rmse}")  # Logs the RMSE value.

    # Calculate R-squared (R²)
    r2 = r2_score(y_test,y_pred)  # Computes the R² score, which represents how well the model explains the variance in the target variable.
    logging.info(f"R²: {r2}")  # Logs the R² value.

    # Major Task: Perform cross-validation on training data if provided
    if x_train is not None and y_train is not None:  # Checks if both training data and labels are provided.
        logging.info("Performing cross-validation on training data...")  # Logs that cross-validation is being performed.
        y_train_pred = model.predict(x_train)  # Makes predictions on the training data ('x_train').
        cv_mse = mean_squared_error(y_train, y_train_pred)  # Calculates the Mean Squared Error on the training data (cross-validation MSE).
        logging.info(f"Cross-validated MSE: {cv_mse}")  # Logs the cross-validation MSE value.

    # Major Task: Plot the residuals (difference between actual and predicted values)
    residuals = y_test - y_pred  # Computes the residuals (the difference between actual and predicted values for the test data).
    plt.scatter(y_pred, residuals)  # Creates a scatter plot of the predicted values ('y_pred') against the residuals.
    plt.axhline(y=0, color='r', linestyle='-')  # Draws a horizontal red line at 0 to represent where residuals should ideally be if predictions were perfect.
    plt.title("Residual Plot")  # Sets the title of the plot as 'Residual Plot'.
    plt.xlabel("Predicted Values")  # Labels the x-axis as 'Predicted Values'.
    plt.ylabel("Residuals")  # Labels the y-axis as 'Residuals'.
    plt.show()  # Displays the plot on the screen.
    logging.info("Displayed residual plot.")  # Logs that the residual plot was displayed.
