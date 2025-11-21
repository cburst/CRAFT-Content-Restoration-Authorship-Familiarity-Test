#!/usr/bin/env python3
import shutil
import subprocess
import os
import glob
from datetime import datetime

# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
DOWNLOADS = "/Users/your-username/Downloads"
TESTDIR = "/Users/your-username/Downloads/surprise tests"
ARCHIVE_DIR = os.path.join(TESTDIR, "old")
PY = "/opt/homebrew/opt/python@3.11/bin/python3.11"

def log(msg):
    print(msg, flush=True)

# -------------------------------------------------------------
# 1. Find most recent TSV and copy to students.tsv
# -------------------------------------------------------------
log(f"Finding most recent TSV in {DOWNLOADS} ...")
tsv_files = sorted(glob.glob(os.path.join(DOWNLOADS, "*.tsv")), key=os.path.getmtime, reverse=True)

if not tsv_files:
    raise SystemExit("❌ No TSV files found!")

latest = tsv_files[0]
log(f"Found: {latest}")

dest_students = os.path.join(TESTDIR, "students.tsv")
shutil.copy(latest, dest_students)
log(f"Copied to {dest_students}")

# -------------------------------------------------------------
# 2. Change working directory
# -------------------------------------------------------------
os.chdir(TESTDIR)
log(f"Working directory: {os.getcwd()}")

# Timestamp like Jan26-1750
TIMESTAMP = datetime.now().strftime("%b%d-%H%M")

# -------------------------------------------------------------
# Helper to run Python scripts
# -------------------------------------------------------------
def run_script(script, *args):
    cmd = [PY, script] + list(args)
    log(" ".join(cmd))
    subprocess.run(cmd, check=True)

# -------------------------------------------------------------
# Helper to safely move PDFs if they exist
# -------------------------------------------------------------
def safe_move_pdfs(src_pattern, dst_dir):
    files = glob.glob(src_pattern)
    if not files:
        log(f"⚠ No PDFs in {src_pattern}, skipping.")
        return

    os.makedirs(dst_dir, exist_ok=True)

    for f in files:
        base = os.path.basename(f)
        dst = os.path.join(dst_dir, base)

        # Overwrite behavior like `mv -f`
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
# 3. Run synonym replacer
# -------------------------------------------------------------
log("Running hybrid-assembler-replacer.py ...")
run_script("hybrid-assembler-replacer.py")

log("Running long.py on PDFs-hybrid-assembler-replacer ...")
run_script("long.py", "PDFs-hybrid-assembler-replacer")

log("Automatically overwriting PDFs-hybrid-assembler-replacer with long_fixed ...")
safe_move_pdfs("long_fixed/*.pdf", "PDFs-hybrid-assembler-replacer")
shutil.rmtree("long", ignore_errors=True)

# -------------------------------------------------------------
# 4. Run sentence intruders
# -------------------------------------------------------------
log("Running hybrid-intruders.py ...")
run_script("hybrid-intruders.py")

log("Running long.py on PDFs-hybrid-sentence-intruders ...")
run_script("long.py", "PDFs-hybrid-sentence-intruders")

log("Automatically overwriting PDFs-hybrid-sentence-intruders with long_fixed ...")
safe_move_pdfs("long_fixed/*.pdf", "PDFs-hybrid-sentence-intruders")
shutil.rmtree("long", ignore_errors=True)

# -------------------------------------------------------------
# 5. Merge matching PDFs
# -------------------------------------------------------------
log("Merging synonym + intruder PDFs ...")
run_script("merge_matchingPDFs.py", "PDFs-hybrid-sentence-intruders/", "PDFs-hybrid-assembler-replacer/")

# -------------------------------------------------------------
# 6. Merge all PDFs into one PDF
# -------------------------------------------------------------
log("Running merge_pdfs.py on merged/")
run_script("merge_pdfs.py", "merged/")

# -------------------------------------------------------------
# 7. Copy merged result & archive EVERYTHING in one run folder
# -------------------------------------------------------------
final_pdf = os.path.join(TESTDIR, "merged", "merged.pdf")
dest_pdf = os.path.join(DOWNLOADS, f"{TIMESTAMP}-merged.pdf")

if not os.path.exists(final_pdf):
	log("❌ merged/merged.pdf not found!")
	raise SystemExit()

log(f"Copying merged.pdf to {dest_pdf} ...")
shutil.copy(final_pdf, dest_pdf)

# Create single run archive folder inside old/
RUN_DIR = os.path.join(ARCHIVE_DIR, f"{TIMESTAMP}-run")
os.makedirs(RUN_DIR, exist_ok=True)
log(f"✔ Created run archive folder: {RUN_DIR}")

def safe_move_into_run(src):
	if os.path.exists(src):
		dst = os.path.join(RUN_DIR, os.path.basename(src))
		shutil.move(src, dst)
		log(f"✔ Archived {src} → {dst}")
	else:
		log(f"⚠ {src} not found, skipping.")

# Archive folders
safe_move_into_run("PDFs-hybrid-assembler-replacer")
safe_move_into_run("PDFs-hybrid-sentence-intruders")
safe_move_into_run("merged")

# Archive answer keys
for f in [
	"answer_key_hybrid_assembler.tsv",
	"answer_key_hybrid_sentence_intruders.tsv",
	"answer_key_hybrid_synonyms.tsv",
]:
	safe_move_into_run(f)

# Archive students.tsv from this run
safe_move_into_run("students.tsv")

# Clean up temporary folders
shutil.rmtree("long_fixed", ignore_errors=True)
shutil.rmtree("long", ignore_errors=True)

log("✔ All tasks complete!")