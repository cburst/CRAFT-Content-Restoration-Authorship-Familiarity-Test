#!/usr/bin/env python3
import shutil
import subprocess
import os
import glob
import time
from datetime import datetime

# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
DOWNLOADS = "/Users/rescreen/Downloads"
TESTDIR = "/Users/rescreen/Downloads/CRAFTtests"
ARCHIVE_DIR = os.path.join(TESTDIR, "old")
PY = "/opt/homebrew/opt/python@3.11/bin/python3.11"

TIMESTAMP = datetime.now().strftime("%b%d-%H%M")

# -------------------------------------------------------------
# LOGGING SETUP
# -------------------------------------------------------------
LOG_DIR = os.path.join(TESTDIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, f"{TIMESTAMP}-pipeline.log")

def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def run_script(script, *args):
    cmd = [PY, "-u", script] + list(args)  # -u = unbuffered Python
    log(" ".join(cmd))

    with open(LOG_FILE, "a", encoding="utf-8") as logfile:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        for line in iter(process.stdout.readline, ''):
            print(line, end='', flush=True)
            logfile.write(line)
            logfile.flush()

        process.stdout.close()
        process.wait()

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)

def safe_move_pdfs(src_pattern, dst_dir):
    files = glob.glob(src_pattern)
    if not files:
        log(f"⚠ No PDFs in {src_pattern}, skipping.")
        return
    os.makedirs(dst_dir, exist_ok=True)
    for f in files:
        base = os.path.basename(f)
        dst = os.path.join(dst_dir, base)
        try:
            if os.path.exists(dst):
                os.remove(dst)
            shutil.move(f, dst)
        except Exception as e:
            log(f"❌ Failed to move {f} → {dst}: {e}")
        else:
            log(f"Moved {f} → {dst}")
    log(f"✔ Moved {len(files)} files into {dst_dir}")

# -------------------------------------------------------------
# PRE-STEP: RUN LLM TEXT GENERATOR USING MOST RECENT TSV
# -------------------------------------------------------------
log(f"Finding most recent TSV in {DOWNLOADS} for LLM generation...")

tsv_files = glob.glob(os.path.join(DOWNLOADS, "*.tsv"))
if not tsv_files:
    raise SystemExit("❌ No TSV files found in Downloads!")

latest_tsv = max(tsv_files, key=os.path.getmtime)
log(f"Latest TSV: {latest_tsv}")

tsv_dir = os.path.dirname(latest_tsv)
orig_name = os.path.basename(latest_tsv)

os.chdir(tsv_dir)
log(f"Changed directory to TSV location: {tsv_dir}")

if orig_name != "students.tsv":
    if os.path.exists("students.tsv"):
        raise SystemExit("❌ students.tsv already exists — aborting")
    os.rename(orig_name, "students.tsv")
    log(f"Renamed {orig_name} → students.tsv")
else:
    log("Latest TSV is already named students.tsv")

log("▶ Running llmtextgenerator.py ...")
run_script(os.path.join(TESTDIR, "llmtextgenerator.py"))

time.sleep(1)
log("✔ LLM generation complete")

os.chdir(TESTDIR)
log(f"Returned to working directory: {TESTDIR}")

# -------------------------------------------------------------
# FIND TWO MOST RECENT TSV FILES
# -------------------------------------------------------------
log(f"Finding 2 most recent TSVs in {DOWNLOADS} ...")

tsv_files = sorted(
    glob.glob(os.path.join(DOWNLOADS, "*.tsv")),
    key=os.path.getmtime,
    reverse=True
)

if len(tsv_files) < 2:
    raise SystemExit("❌ Need at least two TSV files!")

tsv1 = tsv_files[0]
tsv2 = tsv_files[1]

log(f"TSV #1: {tsv1}")
log(f"TSV #2: {tsv2}")

dest1 = os.path.join(TESTDIR, "students1.tsv")
dest2 = os.path.join(TESTDIR, "students2.tsv")

shutil.copy(tsv1, dest1)
shutil.copy(tsv2, dest2)

log(f"Copied TSV 1 → {dest1}")
log(f"Copied TSV 2 → {dest2}")

os.chdir(TESTDIR)
log(f"Working directory: {os.getcwd()}")

# -------------------------------------------------------------
# PROCESS FUNCTION
# -------------------------------------------------------------
def process_tsv_run(order_number, tsv_file):

    log(f"\n===== RUN {order_number}: Processing {tsv_file} =====")

    if not os.path.exists(tsv_file):
        raise SystemExit(f"❌ TSV missing: {tsv_file}")

    if os.path.exists("students.tsv"):
        os.remove("students.tsv")

    shutil.copy(tsv_file, "students.tsv")
    log(f"✔ Copied {tsv_file} → students.tsv")

    log(f"▶ Running hybrid-intruder-synonym.py for run {order_number} ...")
    run_script("hybrid-intruder-synonym.py")

    if os.path.exists("students.tsv"):
        os.remove("students.tsv")
        log("✔ Deleted temporary students.tsv")

    old_pdf_dir = "PDFs-hybrid-synonym-intruders"
    new_pdf_dir = f"{order_number}_PDFs-hybrid-synonym-intruders"

    old_key = "answer_key_hybrid_synonym_intruders.tsv"
    new_key = f"{order_number}_answer_key_hybrid_synonym_intruders.tsv"

    if os.path.exists(old_pdf_dir):
        if os.path.exists(new_pdf_dir):
            shutil.rmtree(new_pdf_dir)
        shutil.move(old_pdf_dir, new_pdf_dir)
        log(f"✔ Renamed {old_pdf_dir} → {new_pdf_dir}")

    if os.path.exists(old_key):
        if os.path.exists(new_key):
            os.remove(new_key)
        shutil.move(old_key, new_key)
        log(f"✔ Renamed {old_key} → {new_key}")

    log(f"===== RUN {order_number} COMPLETE =====\n")

# -------------------------------------------------------------
# RUN BOTH
# -------------------------------------------------------------
process_tsv_run(1, "students1.tsv")
process_tsv_run(2, "students2.tsv")

# -------------------------------------------------------------
# LONG FIX
# -------------------------------------------------------------
log("Running long.py on run 1 ...")
run_script("long.py", "1_PDFs-hybrid-synonym-intruders")
safe_move_pdfs("long_fixed/*.pdf", "1_PDFs-hybrid-synonym-intruders")
shutil.rmtree("long", ignore_errors=True)

log("Running long.py on run 2 ...")
run_script("long.py", "2_PDFs-hybrid-synonym-intruders")
safe_move_pdfs("long_fixed/*.pdf", "2_PDFs-hybrid-synonym-intruders")
shutil.rmtree("long", ignore_errors=True)

# -------------------------------------------------------------
# MERGE
# -------------------------------------------------------------
log("Merging matching PDFs ...")
run_script(
    "merge_matchingPDFs.py",
    "1_PDFs-hybrid-synonym-intruders/",
    "2_PDFs-hybrid-synonym-intruders/"
)

log("Running merge_pdfs.py ...")
run_script("merge_pdfs.py", "merged/")

final_pdf = os.path.join(TESTDIR, "merged", "merged.pdf")
dest_pdf = os.path.join(DOWNLOADS, f"{TIMESTAMP}-merged.pdf")

if not os.path.exists(final_pdf):
    raise SystemExit("❌ merged.pdf not found!")

shutil.copy(final_pdf, dest_pdf)
log(f"Copied merged.pdf → {dest_pdf}")

# -------------------------------------------------------------
# ARCHIVE EVERYTHING
# -------------------------------------------------------------
RUN_DIR = os.path.join(ARCHIVE_DIR, f"{TIMESTAMP}-run")
os.makedirs(RUN_DIR, exist_ok=True)
log(f"✔ Created run archive folder: {RUN_DIR}")

def safe_archive(path):
    if os.path.exists(path):
        shutil.move(path, os.path.join(RUN_DIR, os.path.basename(path)))
        log(f"✔ Archived {path}")

safe_archive("1_PDFs-hybrid-synonym-intruders")
safe_archive("2_PDFs-hybrid-synonym-intruders")
safe_archive("merged")
safe_archive("1_answer_key_hybrid_synonym_intruders.tsv")
safe_archive("2_answer_key_hybrid_synonym_intruders.tsv")
safe_archive("students1.tsv")
safe_archive("students2.tsv")

# Archive llm output
safe_archive(os.path.join(DOWNLOADS, "llmoutput.tsv"))

# Archive log file
shutil.copy(LOG_FILE, os.path.join(RUN_DIR, os.path.basename(LOG_FILE)))
log("✔ Archived pipeline log")

log("✔ All tasks complete!")