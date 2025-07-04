import os
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH", "data/")
FIGURES_PATH = os.getenv("FIGURES_PATH", "figures/")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

