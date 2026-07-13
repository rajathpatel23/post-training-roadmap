"""
Every scratch_nanogpt script imports its siblings as top-level modules
(`from config import ...`, `from data import ...`) — this only works when
scratch_nanogpt/ itself is on sys.path, which happens automatically when you
run e.g. `python train.py` from inside this directory. Tests need the same
thing explicitly, since pytest is invoked from the repo root.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
