#!/usr/bin/env python3

import shutil
import os
from datetime import datetime
import sys
import importlib.util
import traceback
import io
from contextlib import redirect_stdout, redirect_stderr

# -------------------------------------------------------------
# PATHS (FINAL ARCHITECTURE)
# -------------------------------------------------------------

if hasattr(sys, "_MEIPASS"):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))

APP_DIR = os.path.join(BASE_DIR, "app")

WORKSPACE = os.path.join(BASE_DIR, "workspace")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

ARCHIVE_DIR = os.path.join(BASE_DIR, "archive")
ARCHIVE_LOGS = os.path.join(ARCHIVE_DIR, "logs")
ARCHIVE_TESTS = os.path.join(ARCHIVE_DIR, "test-materials")

PY = sys.executable  # ✅ cross-platform safe

# -------------------------------------------------------------
# LOGGING
# -------------------------------------------------------------

def setup_log(mode):
    timestamp = datetime.now().strftime("%b%d-%H%M")
    os.makedirs(WORKSPACE, exist_ok=True)
    return timestamp, os.path.join(WORKSPACE, f"{timestamp}-{mode}.log")

def log(msg, log_file):
    print(msg, flush=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def run_script(script, log_file, *args):

    script_path = os.path.join(APP_DIR, script)

    # --- emulate old subprocess-style command logging ---
    cmd_str = f"{os.path.basename(script)} {' '.join(args)}"
    log(cmd_str, log_file)

    old_argv = sys.argv.copy()
    old_cwd = os.getcwd()

    try:
        # 🔥 match subprocess cwd behavior
        os.chdir(WORKSPACE)

        # 🔥 simulate CLI args
        sys.argv = [script_path] + list(args)

        # 🔥 tee output (console + file)
        class Tee(io.TextIOBase):
            def __init__(self, logfile):
                self.logfile = logfile

            def write(self, s):
                sys.__stdout__.write(s)
                sys.__stdout__.flush()

                self.logfile.write(s)
                self.logfile.flush()  # 🔥 important
                return len(s)

            def flush(self):
                self.logfile.flush()

        with open(log_file, "a", encoding="utf-8") as logfile:
            tee = Tee(logfile)

            with redirect_stdout(tee), redirect_stderr(tee):

                spec = importlib.util.spec_from_file_location("__main__", script_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # 🔥 run main if present
                if hasattr(module, "main"):
                    module.main()

        log(f"✔ Finished {script}", log_file)

    except Exception as e:
        log(f"❌ Error in {script}: {e}", log_file)
        log(traceback.format_exc(), log_file)
        raise

    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)


# -------------------------------------------------------------
# WORKSPACE RESET
# -------------------------------------------------------------

def reset_workspace():
    if os.path.exists(WORKSPACE):
        shutil.rmtree(WORKSPACE)
    os.makedirs(WORKSPACE, exist_ok=True)


# -------------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------------

def run_real_pipeline(input_tsv, testname):
    """
    Runs the REAL (handwritten) pipeline.

    Returns:
        final_pdf_path
        answer_key_path
        log_path
        timestamp
    """

    reset_workspace()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(ARCHIVE_LOGS, exist_ok=True)

    timestamp, log_file = setup_log("real")

    log(f"▶ Starting REAL pipeline: {testname}", log_file)

    # ---------------------------------------------------------
    # 1. Copy TSV into workspace
    # ---------------------------------------------------------
    students_path = os.path.join(WORKSPACE, "students.tsv")
    shutil.copy(input_tsv, students_path)
    log(f"✔ Copied TSV → {students_path}", log_file)

    # ---------------------------------------------------------
    # 2. Run hybrid generator
    # ---------------------------------------------------------
    log("▶ Running hybrid generator...", log_file)
    run_script("hybrid-intruder-synonym.py", log_file, "students.tsv", "answer_key_hybrid_synonym_intruders.tsv")

    if os.path.exists(students_path):
        os.remove(students_path)

    # ---------------------------------------------------------
    # 3. Rename outputs (workspace-safe)
    # ---------------------------------------------------------
    old_pdf_dir = os.path.join(WORKSPACE, "PDFs-hybrid-synonym-intruders")
    new_pdf_dir = os.path.join(WORKSPACE, "real_PDFs")

    old_key = os.path.join(WORKSPACE, "answer_key_hybrid_synonym_intruders.tsv")
    new_key = os.path.join(WORKSPACE, "real_answer_key.tsv")

    if os.path.exists(old_pdf_dir):
        shutil.move(old_pdf_dir, new_pdf_dir)
        log("✔ PDF folder renamed", log_file)
    else:
        raise RuntimeError("❌ PDF folder not found")

    if os.path.exists(old_key):
        shutil.move(old_key, new_key)
        log("✔ Answer key renamed", log_file)
    else:
        raise RuntimeError("❌ Answer key not found")

    # ---------------------------------------------------------
    # 4. Fix long PDFs
    # ---------------------------------------------------------
    log("▶ Fixing long PDFs...", log_file)
    run_script("long.py", log_file, new_pdf_dir)

    # ---------------------------------------------------------
    # 5. Merge PDFs
    # ---------------------------------------------------------
    log("▶ Merging PDFs...", log_file)
    run_script("merge_pdfs.py", log_file, new_pdf_dir)

    merged_pdf = os.path.join(
        new_pdf_dir,
        f"{os.path.basename(new_pdf_dir)}.pdf"
    )

    if not os.path.exists(merged_pdf):
        raise RuntimeError("❌ Merged PDF not found")

    # ---------------------------------------------------------
    # 6. Move final outputs (clean naming)
    # ---------------------------------------------------------
    safe_name = "".join(
        c for c in testname if c.isalnum() or c in (" ", "-", "_")
    ).strip() or "test"

    final_pdf = os.path.join(
        OUTPUT_DIR,
        f"{safe_name}-{timestamp}.pdf"
    )

    final_key = os.path.join(
        OUTPUT_DIR,
        f"{safe_name}-answerkey-{timestamp}.tsv"
    )

    shutil.copy(merged_pdf, final_pdf)
    shutil.copy(new_key, final_key)

    log(f"✔ Final PDF → {final_pdf}", log_file)
    log(f"✔ Answer key → {final_key}", log_file)

    # ---------------------------------------------------------
    # 7. Archive log only
    # ---------------------------------------------------------
    archived_log = os.path.join(
        ARCHIVE_LOGS,
        os.path.basename(log_file)
    )

    shutil.move(log_file, archived_log)

    print(f"✔ Log archived → {archived_log}")

    # ---------------------------------------------------------
    # 8. Archive full test materials
    # ---------------------------------------------------------
    os.makedirs(ARCHIVE_TESTS, exist_ok=True)

    archive_run_dir = os.path.join(
        ARCHIVE_TESTS,
        f"{safe_name}-{timestamp}"
    )

    os.makedirs(archive_run_dir, exist_ok=True)

    # --- PDFs folder ---
    if os.path.exists(new_pdf_dir):
        shutil.copytree(
            new_pdf_dir,
            os.path.join(archive_run_dir, "pdfs")
        )

    # --- answer key ---
    if os.path.exists(new_key):
        shutil.copy(
            new_key,
            os.path.join(archive_run_dir, "answer_key.tsv")
        )

    # --- log (already moved, so copy from archive/logs) ---
    if os.path.exists(archived_log):
        shutil.copy(
            archived_log,
            os.path.join(archive_run_dir, "log.txt")
        )

    print(f"✔ Archived full test materials → {archive_run_dir}")

    # ---------------------------------------------------------
    # DONE
    # ---------------------------------------------------------
    return final_pdf, final_key, archived_log, timestamp