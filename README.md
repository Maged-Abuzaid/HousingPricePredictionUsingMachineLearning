
# House Price Prediction

## Key Learnings

Throughout the development of the house price prediction model, several important lessons and insights were gained:

1. **Data Preprocessing is Critical**: Handling missing values and scaling features played a crucial role in ensuring the model's performance. By strategically dropping columns with excessive missing data and filling missing values in numeric and categorical columns, the dataset was made more robust for model training.

2. **Feature Engineering**: Creating new features, such as the 'Total_SqFt', allowed the model to capture more complex relationships in the data. This was especially helpful in improving predictive accuracy by summarizing key aspects of the housing data.

3. **Model Regularization**: The use of Ridge (L2 regularization) and Lasso (L1 regularization) regression helped in controlling model complexity and preventing overfitting. Hyperparameter tuning with `GridSearchCV` enabled the selection of optimal alpha values for both models.

4. **Model Evaluation**: Evaluation metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R²) were instrumental in comparing model performance. Additionally, plotting residuals helped visualize prediction errors, offering insights into how the model could be improved further.

5. **Logging and Error Handling**: The integration of comprehensive logging throughout data loading, preprocessing, model training, and evaluation was key to tracking the flow of operations and diagnosing issues promptly, such as missing critical columns or convergence warnings during model training.

By carefully following these steps, the project demonstrated how well-designed preprocessing and thoughtful model selection can lead to a more reliable machine learning pipeline for predicting house prices.

## Project Overview
This project aims to predict house prices using the Ames Housing Dataset. The project demonstrates how to build a machine learning pipeline that includes data preprocessing, model training (using Ridge and Lasso regression), and model evaluation. The goal is to accurately predict the sale price of houses based on a variety of features.

## Features
- Data preprocessing: Handles missing values, creates new features, and performs one-hot encoding.
- Model training: Uses Ridge and Lasso regression models with hyperparameter tuning using GridSearchCV.
- Model evaluation: Evaluates the model's performance using key metrics like MAE, MSE, RMSE, and R², and visualizes residual plots.
- Cross-validation: Optionally performs cross-validation on training data to assess model performance.
- Model persistence: Saves the best model for later use.

## Technologies Used
- Python: The programming language used for the project.
- Pandas: For data manipulation and analysis.
- NumPy: For numerical operations.
- Scikit-learn: For machine learning, including Ridge and Lasso regression, GridSearchCV, and evaluation metrics.
- Matplotlib: For visualizing residual plots.
- Joblib: For saving and loading the trained model.

## Project Structure
```
House_Price_Prediction/
|
├── data/
│   └── AmesHousing.csv          # The dataset used for training and evaluation.
├── src/
│   ├── preprocess.py            # Data loading and preprocessing.
│   ├── model.py                 # Model training and hyperparameter tuning.
│   ├── evaluate.py              # Model evaluation and performance metrics.
├── house_price_prediction.py    # Main script that orchestrates the pipeline.
└── requirements.txt             # List of dependencies (e.g., numpy, pandas, scikit-learn).
```

## Files:
- **house_price_prediction.py**: The main script that runs the entire machine learning pipeline from data preprocessing to model evaluation.
- **preprocess.py**: Handles the data loading, cleaning, and preprocessing steps (e.g., handling missing values, feature creation, one-hot encoding).
- **model.py**: Contains the logic for training Ridge and Lasso models, including hyperparameter tuning using GridSearchCV.
- **evaluate.py**: Evaluates the model using performance metrics such as MAE, MSE, RMSE, and R², and generates residual plots.
- **AmesHousing.csv**: The dataset that contains information on houses in Ames, Iowa, and their sale prices.
- **requirements.txt**: Lists the libraries and versions required to run the project.

## Installation and Setup
To run this project locally, follow these steps:

### Prerequisites:
Make sure you have Python installed on your system. You can download it from the official Python website (https://www.python.org/).

### Step 1: Clone the repository
```bash
git clone https://github.com/Maged-Abuzaid/Predictive_Real_Estate_Valuation_ML
cd House_Price_Prediction
```

### Step 2: Set up a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
```

### Step 3: Install the required dependencies
Use the following command to install the necessary packages listed in requirements.txt:
```bash
pip install -r requirements.txt
```

### Step 4: Run the project
You can now run the pipeline by executing the main script:
```bash
python house_price_prediction.py
```

## Usage
Once the script is executed, the following steps will occur:
1. The dataset is loaded and preprocessed (handling missing values, creating features, one-hot encoding).
2. Ridge and Lasso regression models are trained, and the best hyperparameters are selected through GridSearchCV.
3. The best model is saved as `best_model.pkl`.
4. The model is evaluated using key metrics (MAE, MSE, RMSE, R²), and residual plots are displayed.

## Output
- **Best Ridge Alpha**: The best hyperparameter value found for Ridge regression.
- **Best Lasso Alpha**: The best hyperparameter value found for Lasso regression.
- **Model Evaluation**: Metrics including MAE, MSE, RMSE, and R².
- **Residual Plot**: A visual representation of residuals (difference between predicted and actual house prices).
- **Saved Model**: The best model is saved to `best_model.pkl` for future use.

## Future Improvements
- Experiment with additional models, such as Random Forest or Gradient Boosting, to potentially improve performance.
- Implement more advanced feature engineering techniques (e.g., interaction terms, polynomial features).
- Add more robust hyperparameter tuning (e.g., RandomizedSearchCV or Bayesian Optimization).
- Use pipelines to streamline the workflow even more.

## Contact Information
Feel free to reach out if you have any questions or suggestions:
- **Email**: MagedM.Abuzaid@gmail.com
- **LinkedIn**: https://www.linkedin.com/in/magedabuzaid/
- **GitHub**: https://github.com/Maged-Abuzaid

## License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
