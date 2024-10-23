import pandas as pd  # 'import' is used to include the 'pandas' library, which is commonly used for data manipulation. 'pd' is an alias for pandas.
from sklearn.preprocessing import StandardScaler  # Imports 'StandardScaler' from scikit-learn, which is used to standardize numerical data (scale features to have a mean of 0 and a standard deviation of 1).
import logging  # Imports the 'logging' module to log messages about the progress of the program.

# Set up logging to track each step in preprocessing
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  # Sets up logging to display time, log level, and the message.

# Major Task: Define a function to load and preprocess the dataset
def load_and_preprocess_data(
        file_path):  # 'def' defines a function named 'load_and_preprocess_data'. It takes 'file_path' (a string representing the location of the dataset) as an argument.
    """
    Loads the dataset, handles missing values, creates new features, and standardizes the numerical data.  # A docstring explaining what this function does.
    Parameters:
    - file_path (str): Path to the dataset file.  # Explains the parameter expected by this function.

    Returns:
    X_scaled (numpy.ndarray): Scaled features.  # The preprocessed features (input data) after scaling.
    y (pandas.Series): Target variable (SalePrice).  # The target variable (house prices) that the model will predict.
    """

    # Major Task: Load the dataset and check for errors
    try:  # 'try' allows us to attempt something that might fail (like loading a file). If it fails, the program won't crash; we'll handle the error.
        df = pd.read_csv(file_path)  # 'pd.read_csv' reads the CSV file at 'file_path' and loads it into a pandas DataFrame ('df').
        logging.info(f"Loaded data from {file_path} successfully.")  # Logs a message indicating the dataset was loaded successfully.
    except FileNotFoundError:  # 'except' handles the case where the file does not exist (this would cause a 'FileNotFoundError').
        logging.error(f"File not found at {file_path}.")  # Logs an error message if the file isn't found.
        return None, None  # Returns 'None' values to indicate that loading failed, and exits the function early.

    # Major Task: Display the dataset columns for confirmation
    logging.info(f"Columns in dataset: {df.columns}")  # Logs the names of the columns in the dataset for verification.

    # Major Task: Ensure the 'SalePrice' column is present (this is the target variable we want to predict)
    if 'SalePrice' not in df.columns:  # Checks if 'SalePrice' is missing from the dataset.
        logging.error("'SalePrice' column not found in the dataset.")  # Logs an error message if 'SalePrice' isn't found.
        return None, None  # Returns 'None' values to indicate the data is incomplete, and exits the function.

    # Major Task: Handle columns with too many missing values
    missing_threshold = 0.3  # Sets a threshold of 30%. Columns with more than 30% missing data will be dropped.
    missing_fraction = df.isnull().sum() / len(df)  # Calculates the fraction of missing values for each column.
    columns_to_drop = missing_fraction[missing_fraction > missing_threshold].index  # Identifies columns that exceed the threshold for missing data.
    df = df.drop(columns=columns_to_drop, axis=1)  # Drops the identified columns from the dataset.
    logging.info(f"Dropped columns with more than {missing_threshold * 100}% missing values: {columns_to_drop}")  # Logs the names of the columns that were dropped.

    # Major Task: Fill missing values for numerical columns
    numeric_columns = df.select_dtypes(
        include=['float64', 'int64']).columns  # Selects only the numerical columns from the dataset.
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())  # Fills any missing values in numerical columns with the median value of each column.
    logging.info("Filled missing values for numerical columns with median.")  # Logs that missing values in numerical columns were filled.

    # Major Task: Fill missing values for categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns  # Selects only the categorical columns (text-based columns).
    df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])  # Fills missing values in categorical columns with the mode (most frequent value).
    logging.info("Filled missing values for categorical columns with mode.")  # Logs that missing values in categorical columns were filled.

    # Major Task: Create new feature: 'Total_SqFt'
    try:  # 'try' is used to ensure the necessary columns exist before creating the new feature.
        if 'Total Bsmt SF' in df.columns and '1st Flr SF' in df.columns and '2nd Flr SF' in df.columns:  # Checks if all necessary columns are present to create 'Total_SqFt'.
            df['Total_SqFt'] = df['Total Bsmt SF'] + df['1st Flr SF'] + df['2nd Flr SF']  # Creates a new column, 'Total_SqFt', by adding up the square footage of the basement, first, and second floors.
            logging.info("Created new feature: Total_SqFt")  # Logs that the new feature was created.
        else:
            raise KeyError("Required columns for square footage calculation not found.")  # Raises a KeyError if any of the required columns are missing.
    except KeyError as e:  # Catches the KeyError if the necessary columns aren't found.
        logging.error(e)  # Logs the error message.
        return None, None  # Returns 'None' values to indicate failure and exits the function early.

    # Major Task: Convert categorical variables into dummy variables (one-hot encoding)
    df = pd.get_dummies(df, drop_first=True)  # Converts categorical variables into numerical 'dummy' variables. 'drop_first=True' avoids creating redundant columns.
    logging.info("Performed one-hot encoding on categorical variables.")  # Logs that one-hot encoding was performed.

    # Major Task: Ensure numerical columns remain after preprocessing
    if df.select_dtypes(include=['float64', 'int64']).empty:  # Checks if there are no numerical columns left after preprocessing.
        logging.error("No numerical columns available after preprocessing.")  # Logs an error if no numerical columns are left.
        return None, None  # Returns 'None' values to indicate failure and exits the function early.

    # Major Task: Split the data into features (X) and target (y)
    X = df.drop('SalePrice', axis=1)  # 'X' contains all columns except 'SalePrice' (these are the features the model will use to make predictions).
    y = df['SalePrice']  # 'y' contains the 'SalePrice' column (the target variable the model will predict).
    logging.info("Split the data into features (X) and target (y).")  # Logs that the data was split successfully.

    # Major Task: Standardize the features (scaling the data)
    scaler = StandardScaler()  # Creates an instance of the 'StandardScaler' class to standardize the features.
    X_scaled = scaler.fit_transform(X)  # 'fit_transform' scales the features. The result ('X_scaled') has a mean of 0 and a standard deviation of 1 for each feature.
    logging.info("Standardized numerical features.")  # Logs that the features were standardized.

    return X_scaled, y  # Returns the standardized features ('X_scaled') and the target variable ('y').
