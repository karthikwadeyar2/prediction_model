import pytest

from pathlib import Path
import os
import sys

# Adding the below path to avoid module not found error
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))
from prediction_model.config import config
from prediction_model.processing.data_handling import load_dataset
from prediction_model.predict import generate_predictions




# output from predict script should not be null
# output from predict script is str data type 
# The output is Y for an example data


# Fixtures --> function which runs before each test function --> ensure single_prediction runs first

@pytest.fixture
def single_prediction():
    test_dataset = load_dataset(file_name=config.TEST_FILE)
    single_row = test_dataset[0:1]
    result = generate_predictions(single_row)
    return result

# Output is not None
def test_single_pred_not_none(single_prediction):
    assert single_prediction is not None

# Data type is string
def test_single_pred_str_type(single_prediction):
    assert isinstance(single_prediction.get('prediction')[0], str)

# Check the output is Y
def test_single_pred_validate(single_prediction):
    assert single_prediction.get('prediction')[0] == 'Y'