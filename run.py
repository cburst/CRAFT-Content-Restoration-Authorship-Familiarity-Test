#!/usr/bin/env python3

import sys
import os
from app.hard_pipeline import run_hard_pipeline
from app.real_pipeline import run_real_pipeline


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Usage: run.py [mode] [input.tsv] [testname optional]")
        print("mode = hard | real")
        sys.exit(1)

    mode = sys.argv[1].lower()
    input_tsv = os.path.abspath(sys.argv[2])

    if not os.path.exists(input_tsv):
        print(f"❌ Input file not found: {input_tsv}")
        sys.exit(1)

    # ---------------------------------------------------------
    # OPTIONAL TEST NAME (ALWAYS PREFIX WITH MODE)
    # ---------------------------------------------------------
    if len(sys.argv) >= 4:
        base_name = sys.argv[3].strip()
    else:
        base_name = ""

    testname = f"{mode}-{base_name}" if base_name else mode

    # ---------------------------------------------------------
    # PIPELINE DISPATCH
    # ---------------------------------------------------------
    if mode == "hard":
        pdf, key, log, ts = run_hard_pipeline(input_tsv, testname)

    elif mode == "real":
        pdf, key, log, ts = run_real_pipeline(input_tsv, testname)

    else:
        print("❌ Invalid mode. Use 'hard' or 'real'.")
        sys.exit(1)

    # ---------------------------------------------------------
    # FINAL OUTPUT
    # ---------------------------------------------------------
    print("\n✅ DONE")
    print(f"PDF: {pdf}")
    print(f"KEY: {key}")
    print(f"LOG: {log}")