import pathlib
import os
import prediction_model

# To get the Root directory 
PACKAGE_ROOT = pathlib.Path(prediction_model.__file__).resolve().parent

# Path to the datasets
DATAPATH = os.path.join(PACKAGE_ROOT,'datasets')

# TRAIN File
TRAIN_FILE = 'train.csv'

# TEST File
TEST_FILE = 'test.csv'

MODEL_NAME = 'classification.pkl'
# Saved model path
SAVE_MODEL_PATH = os.path.join(PACKAGE_ROOT, 'trained_models')

# Target Variable
TARGET = 'Loan_Status'

#Final features used in the model
FEATURES = ['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Property_Area']

# Numerical features used in the model
NUM_FEATURES = ['ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

# Categorical   features used in the model
CAT_FEATURES = ['Gender',
 'Married',
 'Dependents',
 'Education',
 'Self_Employed',
 'Credit_History',
 'Property_Area']

# Features to Encode
# in our case it is same as Categorical features
FEATURES_TO_ENCODE = ['Gender',
 'Married',
 'Dependents',
 'Education',
 'Self_Employed',
 'Credit_History',
 'Property_Area']

FEATURE_TO_MODIFY = ['ApplicantIncome']
FEATURE_TO_ADD = 'CoapplicantIncome'

DROP_FEATURES = ['CoapplicantIncome']

LOG_FEATURES = ['ApplicantIncome', 'LoanAmount'] # taking log of numerical columns