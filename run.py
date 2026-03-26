#!/usr/bin/env python3

import sys
from app.hard_pipeline import run_hard_pipeline
from app.real_pipeline import run_real_pipeline


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Usage: run.py [mode] [input.tsv] [testname optional]")
        print("mode = hard | real")
        sys.exit(1)

    mode = sys.argv[1].lower()
    input_tsv = sys.argv[2]

    # ---------------------------------------------------------
    # OPTIONAL TEST NAME (ALWAYS PREFIX WITH MODE)
    # ---------------------------------------------------------
    if len(sys.argv) >= 4:
        base_name = sys.argv[3]
    else:
        base_name = ""

    if base_name:
        testname = f"{mode}-{base_name}"
    else:
        testname = mode

    # ---------------------------------------------------------
    # PIPELINE DISPATCH
    # ---------------------------------------------------------
    if mode == "hard":
        run_hard_pipeline(input_tsv, testname)

    elif mode == "real":
        run_real_pipeline(input_tsv, testname)

    else:
        print("❌ Invalid mode. Use 'hard' or 'real'.")
        sys.exit(1)