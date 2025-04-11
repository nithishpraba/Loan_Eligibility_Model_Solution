import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from utils.logger import get_logger
from utils.preprocessing import preprocess_data

logger = get_logger(__name__)

def train_model():
    # 1. Load the dataset from the data folder
    data_path = os.path.join("data", "credit.csv")  # Ensure the CSV is here
    try:
        df = pd.read_csv(data_path)
        logger.info("Dataset loaded. Shape: %s", df.shape)
    except Exception as e:
        logger.error("Error loading data from %s: %s", data_path, e)
        return None

    # 2. Preprocess the data to get X_train and y_train
    try:
        X_train, y_train = preprocess_data(df)
        logger.info("Data preprocessing complete.")
    except Exception as e:
        logger.error("Error during preprocessing: %s", e)
        return None

    # 3. Train the model (Logistic Regression as an example)
    try:
        # Increase max_iter if you see a convergence warning
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        logger.info("Model training complete.")
    except Exception as e:
        logger.error("Error during model training: %s", e)
        return None

    return model

def main():
    model = train_model()
    if model is not None:
        os.makedirs("models", exist_ok=True)
        model_path = os.path.join("models", "model.pkl")
        joblib.dump(model, model_path)
        logger.info("Model saved to %s", model_path)
        print("Training complete and model saved at", model_path)
    else:
        print("Training failed. See log for details.")

if __name__ == "__main__":
    main()
