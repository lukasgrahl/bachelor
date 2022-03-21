import os

PROJECT_DIR = os.path.dirname(__file__)
RAW_DATA_DIR = os.path.join(PROJECT_DIR, "data", "1_raw")
WORK_DATA_DIR = os.path.join(PROJECT_DIR, "data", "2_work")
OUT_DATA_DIR = os.path.join(PROJECT_DIR, "data", "3_out")
OUT_MODEL_DIR = os.path.join(PROJECT_DIR, "data", "3_out", "models")

random_state = 101