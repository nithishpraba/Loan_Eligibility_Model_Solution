# Loan Eligibility Prediction App

This project is part of the CST2216 Individual Term Project. It predicts loan eligibility using a Logistic Regression model trained on credit data. The project is built in a modular fashion and is deployed as a Streamlit app.

## Project Structure

Loan_Eligibility_Model_Solution/ ├── data/
│ └── credit.csv # Input dataset; ensure this file is placed here ├── models/
│ └── model.pkl # Trained model will be saved here after running the training script ├── scripts/
│ └── train.py # Script to load data, preprocess, train the model, and save it ├── utils/
│ ├── init.py # (empty file to mark folder as a package) │ ├── logger.py # Utility for logging messages │ └── preprocessing.py # Data cleaning, encoding, and splitting into features and target ├── app.py # Streamlit app for user prediction interface ├── requirements.txt # List of project dependencies ├── .gitignore # Files and folders to be excluded from Git └── README.md # This documentation file

bash
Copy

## Setup Instructions

1. **Clone the Repository:**

   ```bash
   git clone <your-repo-url>
   cd Loan_Eligibility_Model_Solution
Create and Activate a Virtual Environment:

Linux/Mac:

bash
Copy
python -m venv venv
source venv/bin/activate
Windows:

bash
Copy
python -m venv venv
venv\Scripts\activate
Install Dependencies:

bash
Copy
pip install -r requirements.txt
Place Your Dataset:

Ensure the CSV file (credit.csv) is located in the data/ folder.

Training the Model
Run the training script to preprocess data, train the Logistic Regression model, and save it as models/model.pkl:

bash
Copy
python -m scripts.train
If successful, you should see log messages indicating data loading, preprocessing, and model training are complete. The trained model will be stored in models/model.pkl.

Running the App
Launch the Streamlit app to interact with your model:

bash
Copy
streamlit run app.py
Your web browser should open with the prediction interface. Enter the required details to get the loan eligibility prediction.

Usage
Input Fields:
The app asks for the following details:

Gender: Male / Female

Married: Yes / No

Dependents: (0, 1, 2, 3+)

Education: Graduate / Not Graduate

Self Employed: Yes / No

Applicant Income: Numeric value

Coapplicant Income: Numeric value

Loan Amount: Numeric value

Loan Amount Term: Numeric value (often 360)

Credit History: 1 (good) / 0 (bad)

Property Area: Urban / Semiurban / Rural

Prediction:
Once you fill in all fields, click Predict. The app will display the predicted loan approval status ("Y" for approved, "N" for not approved).

Troubleshooting
Prediction Errors:
Make sure you enter all required values. The app uses a dummy target column to run preprocessing, but all feature fields must be provided.

Model Training Issues:
Verify your dataset (credit.csv) is in the data/ folder and that you’ve activated your environment with dependencies installed.

Environment Issues:
Ensure you’ve installed all dependencies from requirements.txt.

Deployment (Optional)
To deploy this app on Streamlit Cloud:

Push your code to GitHub.

Create a free account on Streamlit Cloud.

Link your GitHub repository and set app.py as the main file.

Share the app’s URL for others to test.

Git Configuration
A sample .gitignore is provided in this repository to exclude files such as virtual environments, Python caches, and local editor configurations.

Contact
For any questions or issues, please contact Nithish at nithishpraba23@gmail.com