from sklearn.linear_model import Ridge, Lasso  # 'from' imports specific classes (Ridge and Lasso) from 'sklearn.linear_model'. These classes represent two types of linear regression models with regularization (Ridge uses L2 regularization, Lasso uses L1 regularization).
from sklearn.model_selection import GridSearchCV  # Imports 'GridSearchCV', a tool for searching over a grid of hyperparameters to find the best ones for the model.
from sklearn.exceptions import ConvergenceWarning  # Imports 'ConvergenceWarning' from 'sklearn.exceptions', used to handle warnings when a model fails to converge during training.
import logging  # Imports 'logging' for logging messages about the progress and status of the program.
import joblib  # Imports 'joblib', which is used to save and load Python objects, such as trained models, to/from a file.
import warnings  # Imports the 'warnings' module to manage and filter warning messages (used later to catch and ignore convergence warnings).

# Set up logging to track each step in model training
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  # Sets up logging configuration to display time, log level, and the message.


# Major Task: Train a model using Ridge and Lasso regression, and perform hyperparameter tuning
def train_model(x_train, y_train):  # Changed 'X_train' and 'Y_train' to 'x_train' and 'y_train' to follow Python naming conventions for function parameters.
    """
    Trains a Ridge and Lasso regression model using GridSearchCV to find the best hyperparameters.  # Describes what this function does.
    Parameters:
    - x_train (numpy.ndarray): The features for training the model.  # Updated to lowercase 'x_train' to follow Python conventions.
    - y_train (numpy.ndarray): The target (house prices) for training.  # Updated to lowercase 'y_train' to follow Python conventions.

    Returns:
    - model: The best-performing trained model.
    - best_params: The best hyperparameters found by GridSearchCV.
    - best_score: The best score achieved by the model during cross-validation.
    """

    # Major Task: Set up the models and hyperparameter grids
    ridge = Ridge()  # Creates an instance of the 'Ridge' regression model, which uses L2 regularization.
    lasso = Lasso()  # Creates an instance of the 'Lasso' regression model, which uses L1 regularization.

    # Major Task: Define hyperparameters to search over for Ridge and Lasso
    param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}  # A dictionary containing the hyperparameters to search. 'alpha' controls the strength of regularization. The values [0.01, 0.1, 1, 10, 100] will be tested to find the best one.

    # Major Task: Perform hyperparameter tuning using GridSearchCV for Ridge
    logging.info("Performing grid search for Ridge Regression.")  # Logs a message to indicate that GridSearchCV is being run for Ridge regression.
    ridge_cv = GridSearchCV(ridge, param_grid, scoring='neg_mean_squared_error', cv=5)  # 'GridSearchCV' searches for the best 'alpha' by using 5-fold cross-validation ('cv=5'). The scoring method is negative mean squared error, where lower is better.
    ridge_cv.fit(x_train, y_train)  # Fits the 'Ridge' model to the training data ('x_train' and 'y_train') and performs the grid search to find the best hyperparameters.
    best_ridge_alpha = ridge_cv.best_params_  # Retrieves the best 'alpha' found by GridSearchCV for Ridge regression.
    logging.info(f"Best Ridge alpha: {best_ridge_alpha}")  # Logs the best alpha value found for Ridge regression.

    # Major Task: Perform hyperparameter tuning using GridSearchCV for Lasso
    logging.info("Performing grid search for Lasso Regression.")  # Logs a message to indicate that GridSearchCV is being run for Lasso regression.
    lasso_cv = GridSearchCV(lasso, param_grid, scoring='neg_mean_squared_error', cv=5)  # 'GridSearchCV' searches for the best 'alpha' by using 5-fold cross-validation. The scoring method is also negative mean squared error.

    # Major Task: Handle potential convergence warnings for Lasso regression
    with warnings.catch_warnings():  # 'with' starts a context that catches any warnings during Lasso fitting.
        warnings.simplefilter('ignore', ConvergenceWarning)  # Ignores convergence warnings that can happen if Lasso doesn't converge to a solution.
        lasso_cv.fit(x_train, y_train)  # Fits the 'Lasso' model to the training data and performs the grid search to find the best hyperparameters.
    best_lasso_alpha = lasso_cv.best_params_  # Retrieves the best 'alpha' found by GridSearchCV for Lasso regression.
    logging.info(f"Best Lasso alpha: {best_lasso_alpha}")  # Logs the best alpha value found for Lasso regression.

    # Major Task: Choose the model based on cross-validation performance
    if ridge_cv.best_score_ > lasso_cv.best_score_:  # Compares the best scores of Ridge and Lasso models. If Ridge has a better score:
        logging.info("Ridge Regression chosen.")  # Logs that Ridge performed better and was selected.
        model = ridge_cv.best_estimator_  # 'best_estimator_' gives the best model (Ridge) with the best parameters found by GridSearchCV.
    else:  # If Lasso performed better:
        logging.info("Lasso Regression chosen.")  # Logs that Lasso performed better and was selected.
        model = lasso_cv.best_estimator_  # 'best_estimator_' gives the best model (Lasso) with the best parameters found by GridSearchCV.

    # Major Task: Save the best model to a file using joblib
    joblib.dump(model, 'best_model.pkl')  # 'joblib.dump' saves the best model to a file named 'best_model.pkl' so it can be loaded later without retraining.
    logging.info("Best model saved as 'best_model.pkl'.")  # Logs a message indicating that the model has been saved.

    return model, ridge_cv.best_params_ if ridge_cv.best_score_ > lasso_cv.best_score_ else lasso_cv.best_params_, max(ridge_cv.best_score_, lasso_cv.best_score_)  # Returns the best model, its best parameters, and its best cross-validation score.
