# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/sahilsingla/toursim-purchase-prediction/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")


# ------------------------------------------------------
# Step 1: Drop the unique identifier columns
# ------------------------------------------------------
# Unique ID or index columns don't provide any predictive value
unique_cols = [col for col in df.columns if df[col].is_unique]
if unique_cols:
    df.drop(columns=unique_cols, inplace=True)
    print(f"Dropped unique columns: {unique_cols}")

# ------------------------------------------------------
# Step 2: Separate numeric and categorical columns
# ------------------------------------------------------
num_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = df.select_dtypes(exclude="number").columns.tolist()

# ------------------------------------------------------
# Step 3: Handle missing values
# ------------------------------------------------------
# - For numeric: replace missing with median
# - For categorical: replace missing with most frequent value
df[num_cols] = df[num_cols].apply(lambda x: x.fillna(x.median()).astype(x.dtype))
df[cat_cols] = df[cat_cols].apply(lambda x: x.fillna(x.mode()[0]))

print("Missing values imputed if any.")

# ------------------------------------------------------
# Step 4: Outlier treatment (IQR method)
# ------------------------------------------------------
# Any value outside 1.5 * IQR range will be capped at boundary
# Apply only to continuous numeric columns (not discrete/categorical ints)
continuous_cols = [
    col for col in num_cols
    if df[col].nunique() > 10  # heuristic: treat only if column has >10 unique values
]

for col in continuous_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Preserve dtype (int stays int, float stays float)
    dtype = df[col].dtype
    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    df[col] = df[col].astype(dtype)

print("Outliers treated with IQR capping on continuous columns only.")

# ------------------------------------------------------
# Step 5: Encode categorical variables
# ------------------------------------------------------
# LabelEncoder assigns integer values to string categories
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

print("Categorical variables encoded using LabelEncoder.")

# ------------------------------------------------------
# Step 6: Normalize numerical columns
# ------------------------------------------------------
# StandardScaler: mean = 0, variance = 1
#if num_cols:
#    scaler = StandardScaler()
#    df[num_cols] = scaler.fit_transform(df[num_cols])
#    print("Numerical columns normalized.")

# ------------------------------------------------------
# Step 7: Separate target variable
# ------------------------------------------------------
# Replace "Target" with the actual target column of tourism dataset
target_col = 'ProdTaken'  # <-- Update this with correct target column name
if target_col in df.columns:
    X = df.drop(columns=[target_col])  # Features
    y = df[target_col]                 # Target
else:
    raise ValueError("Target column not found. Please update 'target_col'.")

# ------------------------------------------------------
# Step 8: Train-test split
# ------------------------------------------------------
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y
)

print("ytrain -> 1 =", (ytrain == 1).sum())
print("ytrain -> 0 =", (ytrain == 0).sum())
print("ytest -> 1  =", (ytest == 1).sum())
print("ytest -> 0  =", (ytest == 0).sum())

# ------------------------------------------------------
# Step 9: Save processed data locally
# ------------------------------------------------------
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Train-test data prepared and saved locally as CSVs.")

# ------------------------------------------------------
# Step 10: Upload processed files to Hugging Face Hub
# ------------------------------------------------------
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="sahilsingla/toursim-purchase-prediction",
        repo_type="dataset",
    )

print("Processed data uploaded successfully to Hugging Face Hub.")
