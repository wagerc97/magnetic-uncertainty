"""Utility functions"""
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import normalize
from datetime import datetime

pd.set_option('display.max_columns', None)


# ======================================[ Constants ]===============================================

BASE_PATH = Path(__file__).parent.parent
MODEL_PATH = (BASE_PATH / "models").absolute()
print(f"BASE_PATH: {BASE_PATH}")
print(f"MODEL_PATH: {MODEL_PATH}")


# ======================================[ File structure ]=========================================

# if no models folder exists in BASE_PATH, create it    
if not MODEL_PATH.exists():
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    print(f"Created MODEL_PATH at: {MODEL_PATH}")

# write a .gitignore file in MODEL_PATH to ignore all files
gitignore_path = MODEL_PATH / ".gitignore"
if not gitignore_path.exists():
    with open(gitignore_path, "w") as f:
        f.write("*\n")
    print(f"Created .gitignore in MODEL_PATH to ignore all files.")


# ======================================[ Utility functions ]======================================


DATE_TIME_STR_FORMAT = "%Y-%m-%d__%H-%M-%S"
def get_timestamp_as_string() -> str:
    current_timestamp = datetime.now()
    str_format = DATE_TIME_STR_FORMAT
    formatted_timestamp = current_timestamp.strftime(str_format)
    return formatted_timestamp


# DATA LOADING
def load_experimental(file_name: str, dir_name: str=None, features: list=None) -> pd.DataFrame:
    """
    Read the CSV file into a DataFrame

    :param file_name:
    :return:
    """
    print(f"Load data from '{file_name}'")
    if dir_name is None:
        dir_name = DATA_PATH
    df = pd.read_csv(dir_name / file_name)
    df = keep_one_per_feature_set(df, features=features)
    df = sphere_transform_df(df)
    df = scale_temp(df)
    return df


# DATA PROCESSING
def keep_one_per_feature_set(
    df: pd.DataFrame,
    features: list = None
) -> pd.DataFrame:
    """
    Keep the first occurrence of each unique feature set and
    drop all subsequent duplicates.
    """
    if features is None:
        features = ['Nd', 'Ce', 'La', 'Pr', 'Y', 'Tb', 'Dy', 'Fe', 'Co', 'Ni', 'B', 'C', 'temp']
        print(f"No features provided for duplicate removal. Using default features: {features}")
    orig_num_rows = df.shape[0]
    df = df.drop_duplicates(subset=features, keep='first').copy()
    new_num_rows = df.shape[0]
    num_dropped = orig_num_rows - new_num_rows
    print(f"Dropped {num_dropped} duplicates.")
    return df


# PREPROCESSING
def sphere_transform_df(df):
    """
    Apply sphere transform to 12 X input elements:
     - apply the L2 norm (also known as the Euclidean norm) to each group.
     - add "_s" suffix to column names for "scaled"

    SOURCES:
     - park, kernel methods for radial transformed compositional data, section 4.1
    """
    df = df.copy()
    df[['Nd_s', 'Ce_s', 'La_s', 'Pr_s', 'Y_s', 'Tb_s', 'Dy_s']] = normalize(df[['Nd', 'Ce', 'La', 'Pr', 'Y', 'Tb', 'Dy']])
    df[['Fe_s', 'Co_s', 'Ni_s']] = normalize(df[['Fe', 'Co', 'Ni']])
    df[['B_s', 'C_s']] = normalize(df[['B', 'C']])
    print("Applied sphere transform to elemental features.")
    return df


# PREPROCESSING
def scale_temp(df):
    df = df.copy()
    temp_max = df['temp'].max()
    df['temp_s'] = df['temp'] / temp_max
    print(f"Scaled 'temp' feature to 'temp_s' by dividing by max temp.")
    return df
