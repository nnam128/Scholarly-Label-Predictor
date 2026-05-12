from pathlib import Path

# BASE DIRECTORY
BASE_DIR = Path(__file__).resolve().parent.parent

# RAW DATA PATHS
TRAIN_PATH_STAGE1 = BASE_DIR / "data" / "raw" / "Stage1" / "Stage_1_publcitrain.csv"
TEST_PATH_STAGE1 = BASE_DIR / "data" / "raw" / "Stage1" / "test (2).csv"

SAMPLE_SUBMISSION_PATH_STAGE1 = BASE_DIR / "data" / "raw" / "Stage1" / "sample_submission_DM252.csv"


# EXTERNAL DATA
CRAWLED_ABSTRACT_PATH = BASE_DIR / "data" / "external" / "crawled_abstracts.csv"


# INTERIM DATA (MERGED + CLEANED)
MERGED_TRAIN_PATH = BASE_DIR / "data" / "interim" / "merged_train.csv"
MERGED_TEST_PATH = BASE_DIR / "data" / "interim" / "merged_test.csv"
CLEANED_TRAIN_PATH = BASE_DIR / "data" / "interim" / "cleaned_train.csv"
CLEANED_TEST_PATH = BASE_DIR / "data" / "interim" / "cleaned_test.csv"


# FINAL FEATURES (FEATURE READY)
FINAL_TRAIN_SVM_PATH = BASE_DIR / "data" / "processed" / "train_SVM_features.npz"
FINAL_TEST_SVM_PATH = BASE_DIR / "data" / "processed" / "test_SVM_features.npz"

TRAIN_LABEL_PATH = BASE_DIR / "data" / "processed" / "train_labels.csv"

#SVM MODEL PATH
SVM_MODEL_PATH = BASE_DIR / "models" / "saved" / "svm_model.pkl"

#SUBMISSION
SUBMISSION_SVM_PATH = BASE_DIR / "data" / "submission" / "submission_baseline.csv"