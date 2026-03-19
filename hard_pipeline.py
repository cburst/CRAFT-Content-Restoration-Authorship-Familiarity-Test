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

# -------------------------------------------------------------
# LOGGING SETUP (LIVE + FILE)
# -------------------------------------------------------------
TIMESTAMP = datetime.now().strftime("%b%d-%H%M")
LOG_FILE = os.path.join(TESTDIR, f"{TIMESTAMP}-real-run.log")

def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def run_script(script, *args):
    cmd = [PY, "-u", script] + list(args)
    log(" ".join(cmd))

    with open(LOG_FILE, "a", encoding="utf-8") as logfile:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in process.stdout:
            print(line, end="", flush=True)
            logfile.write(line)

        process.wait()

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)

# -------------------------------------------------------------
# Helper: move PDFs safely
# -------------------------------------------------------------
def safe_move_pdfs(src_pattern, dst_dir):
    files = glob.glob(src_pattern)
    if not files:
        log(f"⚠ No PDFs matching {src_pattern}")
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
# 1. Find most recent TSV → students.tsv
# -------------------------------------------------------------
log(f"Finding most recent TSV in {DOWNLOADS} ...")

tsv_files = sorted(
    glob.glob(os.path.join(DOWNLOADS, "*.tsv")),
    key=os.path.getmtime,
    reverse=True
)

if not tsv_files:
    raise SystemExit("❌ No TSV files found!")

latest = tsv_files[0]
log(f"Found latest TSV: {latest}")

dest_students = os.path.join(TESTDIR, "students.tsv")

if os.path.exists(dest_students):
    os.remove(dest_students)

shutil.copy(latest, dest_students)
log(f"✔ Copied → {dest_students}")

# -------------------------------------------------------------
# 1b. Generate LLM-authored TSV
# -------------------------------------------------------------
log("▶ Preparing students.tsv for LLM generation...")

students_path = os.path.join(TESTDIR, "students.tsv")
llm_output = os.path.join(TESTDIR, "llmoutput.tsv")
downloads_llm = os.path.join(DOWNLOADS, "llmoutput.tsv")

# Recreate students.tsv from the latest TSV
if os.path.exists(students_path):
    os.remove(students_path)

shutil.copy(latest, students_path)
log(f"✔ Copied {latest} → {students_path}")

# Run the LLM generator
log("▶ Generating LLM-authored TSV with llmtextgenerator.py...")

cwd = os.getcwd()
os.chdir(TESTDIR)

run_script("llmtextgenerator.py")

os.chdir(cwd)

# Ensure LLM output was produced
if not os.path.exists(llm_output):
    raise SystemExit("❌ llmoutput.tsv was not produced!")

# Replace students.tsv with the LLM-generated text
os.replace(llm_output, students_path)
log("✔ Replaced students.tsv with LLM-generated text")

# Optional: keep a copy in Downloads for inspection
if os.path.exists(downloads_llm):
    os.remove(downloads_llm)

shutil.copy(students_path, downloads_llm)
log(f"✔ Copied LLM TSV → {downloads_llm}")

# -------------------------------------------------------------
# 2. Change working directory
# -------------------------------------------------------------
os.chdir(TESTDIR)
log(f"Working directory: {os.getcwd()}")

# -------------------------------------------------------------
# 3. Run hybrid generator (ONE RUN ONLY)
# -------------------------------------------------------------
log("▶ Running hybrid-intruder-synonym.py...")
run_script("hybrid-intruder-synonym.py")

# Delete temporary students.tsv
if os.path.exists("students.tsv"):
    os.remove("students.tsv")
    log("✔ Deleted temporary students.tsv")

# -------------------------------------------------------------
# 4. Rename outputs to isolate from test pipeline
# -------------------------------------------------------------
old_pdf_dir = "PDFs-hybrid-synonym-intruders"
new_pdf_dir = "real_PDFs-hybrid-synonym-intruders"

old_key = "answer_key_hybrid_synonym_intruders.tsv"
new_key = "real_answer_key_hybrid_synonym_intruders.tsv"

if os.path.exists(old_pdf_dir):
    if os.path.exists(new_pdf_dir):
        shutil.rmtree(new_pdf_dir)
    shutil.move(old_pdf_dir, new_pdf_dir)
    log(f"✔ Renamed {old_pdf_dir} → {new_pdf_dir}")
else:
    log(f"⚠ No folder {old_pdf_dir} found.")

if os.path.exists(old_key):
    if os.path.exists(new_key):
        os.remove(new_key)
    shutil.move(old_key, new_key)
    log(f"✔ Renamed {old_key} → {new_key}")
else:
    log(f"⚠ No answer key {old_key} found.")

# -------------------------------------------------------------
# 5. Run long.py
# -------------------------------------------------------------
log(f"▶ Running long.py on {new_pdf_dir} ...")
run_script("long.py", new_pdf_dir)

safe_move_pdfs("long_fixed/*.pdf", new_pdf_dir)

shutil.rmtree("long_fixed", ignore_errors=True)
shutil.rmtree("long", ignore_errors=True)

# -------------------------------------------------------------
# 6. Merge PDFs
# -------------------------------------------------------------
log("▶ Merging real PDFs into one ...")
run_script("merge_pdfs.py", new_pdf_dir)

folder_basename = os.path.basename(new_pdf_dir)
final_pdf = os.path.join(new_pdf_dir, f"{folder_basename}.pdf")
output_pdf = os.path.join(DOWNLOADS, f"{TIMESTAMP}-real.pdf")

if not os.path.exists(final_pdf):
    raise SystemExit(f"❌ {final_pdf} not found!")

shutil.copy(final_pdf, output_pdf)
log(f"✔ Copied merged real PDF → {output_pdf}")

# -------------------------------------------------------------
# 7. Archive run contents
# -------------------------------------------------------------
RUN_DIR = os.path.join(ARCHIVE_DIR, f"{TIMESTAMP}-real-run")
os.makedirs(RUN_DIR, exist_ok=True)
log(f"✔ Created archive folder: {RUN_DIR}")

def safe_archive(path):
    if os.path.exists(path):
        shutil.move(path, os.path.join(RUN_DIR, os.path.basename(path)))
        log(f"✔ Archived {path}")
    else:
        log(f"⚠ {path} not found — skipping")

safe_archive(new_pdf_dir)
safe_archive(new_key)
safe_archive(LOG_FILE)          # 🔵 archive log file

log("✔ Real pipeline complete!")