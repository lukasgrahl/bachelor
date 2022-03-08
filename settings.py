import os

PROJECT_DIR = os.path.dirname(__file__)
RAW_DATA_DIR = os.path.join(PROJECT_DIR, "data", "raw")
WORK_DATA_DIR = os.path.join(PROJECT_DIR, "data", "work")
OUT_DATA_DIR = os.path.join(PROJECT_DIR, "data", "out")
OUT_MODEL_DIR = os.path.join(PROJECT_DIR, "data", "out", "models")

random_state = 101