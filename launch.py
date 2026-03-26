#!/usr/bin/env python3

import os
import sys

TARGET_DIR = os.path.expanduser("~/CRAFTtests")
PYTHON = "/opt/homebrew/opt/python@3.11/bin/python3.11"
SCRIPT = os.path.join(TARGET_DIR, "run_gui.py")

env = os.environ.copy()
env["PATH"] = "/opt/homebrew/opt/python@3.11/bin:/usr/bin:/bin:/usr/sbin:/sbin"
env["PYTHONHOME"] = ""
env["PYTHONPATH"] = ""

# 🔥 THIS IS THE KEY CHANGE
os.chdir(TARGET_DIR)
os.execve(PYTHON, [PYTHON, SCRIPT], env)