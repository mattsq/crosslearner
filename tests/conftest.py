import os
import sys

# Add repository root to PYTHONPATH for tests
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
SRC_DIR = os.path.join(ROOT_DIR, "src")
if os.path.isdir(SRC_DIR) and SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
