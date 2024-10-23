import logging
from src.preprocess import load_and_preprocess_data
from src.model import train_model
from src.evaluate import evaluate_model
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Step 1: Load and preprocess data
print("Starting data preprocessing...")
X, y = load_and_preprocess_data('data/AmesHousing.csv')
if X is None or y is None:
    print("Data loading failed. Please check the file path and try again.")
    exit(1)


# Step 2: Split data into training and test sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train the model
print("Training the model...")
model, best_params, best_score = train_model(X_train, y_train)
print(f"Best parameters: {best_params}, Best score: {best_score}")

# Step 4: Evaluate the model
print("Evaluating the model...")
evaluate_model(model, X_test, y_test, X_train, y_train)
