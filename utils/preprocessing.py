import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    """
    Preprocess the DataFrame for the Loan Eligibility model:
    1. Drop rows with missing values.
    2. Remove the "Loan_ID" column if it exists (not useful for training).
    3. Convert the target column "Loan_Approved" from Y/N to 1/0.
    4. Label-encode categorical features like Gender, Married, Dependents, etc.
    5. Separate features (X) and target (y).
    """

    # 1. Drop missing values
    df_clean = df.dropna().copy()


    # 2. Drop Loan_ID if present
    if "Loan_ID" in df_clean.columns:
        df_clean.drop("Loan_ID", axis=1, inplace=True)

    # 3. Target column must be "Loan_Approved" as determined in your CSV
    target_column = "Loan_Approved"
    if target_column not in df_clean.columns:
        raise Exception(
            f"Target column '{target_column}' not found. "
            f"Available columns: {df_clean.columns.tolist()}"
        )

    # Map Y/N -> 1/0
    df_clean[target_column] = df_clean[target_column].map({"Y": 1, "N": 0})

    # 4. Identify categorical columns that need encoding
    #    (Adjust this list based on your CSV)
    cat_cols = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]

    for col in cat_cols:
        if col in df_clean.columns:
            # Convert all data to string just in case of mixed types
            df_clean[col] = df_clean[col].astype(str)
            # Create and apply a LabelEncoder for each column
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col])

    # 5. Separate features (X) and target (y)
    X = df_clean.drop(target_column, axis=1).values
    y = df_clean[target_column].values

    return X, y
