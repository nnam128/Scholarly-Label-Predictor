from pathlib import Path

# BASE DIRECTORY
BASE_DIR = Path(__file__).resolve().parent.parent

# RAW DATA PATHS
TRAIN_PATH_STAGE1 = BASE_DIR / "data" / "raw" / "Stage1" / "Stage_1_publcitrain.csv"

TEST_PATH_STAGE1 = BASE_DIR / "data" / "raw" / "Stage1" / "test (2).csv"

SAMPLE_SUBMISSION_PATH_STAGE1 = BASE_DIR / "data" / "raw" / "Stage1" / "sample_submission_DM252.csv"


# EXTERNAL DATA
CRAWLED_ABSTRACT_PATH = BASE_DIR / "data" / "external" / "crawled_abstracts.csv"
