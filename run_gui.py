#!/usr/bin/env python3

import ttkbootstrap as tb
from ttkbootstrap.constants import *
import tkinter as tk
from tkinter import filedialog
import threading
import os
import sys
import time
import subprocess

# -----------------------------
# FIX IMPORT PATH
# -----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from app.hard_pipeline import run_hard_pipeline
from app.real_pipeline import run_real_pipeline

os.environ["PATH"] = (
    "/opt/homebrew/bin:/opt/homebrew/sbin:"
    "/usr/local/bin:"
    + os.environ.get("PATH", "")
)

# -----------------------------
# APP SETUP
# -----------------------------

app = tb.Window(themename="flatly")
app.title("CRAFT Test Generator")
app.minsize(540, 420)

# -----------------------------
# ICON (STABLE)
# -----------------------------

icon_path = os.path.expanduser("~/CRAFTtests/app/data/icon.png")

def apply_icon():
    try:
        app.icon_img = tk.PhotoImage(file=icon_path)
        app.iconphoto(True, app.icon_img)
    except Exception as e:
        print("❌ icon failed:", e)

app.after_idle(apply_icon)

# -----------------------------
# SAFE UI HELPER
# -----------------------------

def safe_ui(func, *args):
    app.after(0, lambda: func(*args))

# -----------------------------
# API KEY
# -----------------------------

def ensure_api_key():

    key = os.getenv("OPENAI_API_KEY")
    if key:
        return True

    key_path = os.path.expanduser("~/.craft_api_key")

    if os.path.exists(key_path):
        try:
            with open(key_path) as f:
                key = f.read().strip()
                if key:
                    os.environ["OPENAI_API_KEY"] = key
                    return True
        except:
            pass

    dialog = tb.dialogs.Querybox.get_string(
        "Enter your OpenAI API Key:\n\nhttps://platform.openai.com/api-keys",
        title="Setup API Key"
    )

    if not dialog:
        tb.dialogs.Messagebox.show_error(
            "API key is required.",
            title="Missing API Key"
        )
        return False

    key = dialog.strip().strip('"').strip("'")

    if not key:
        return False

    try:
        with open(key_path, "w") as f:
            f.write(key)
        os.chmod(key_path, 0o600)
    except:
        pass

    os.environ["OPENAI_API_KEY"] = key
    return True

# -----------------------------
# LAYOUT
# -----------------------------

content = tb.Frame(app, padding=20)
content.pack(fill=BOTH, expand=YES)

bottom_bar = tb.Frame(app, padding=(20, 10))
bottom_bar.pack(fill=X)

# --------------------------------------------------
# INFO PANEL (TEXT)
# --------------------------------------------------

info_text = (
    "CRAFT Test Generator:\n"
    "This tool generates Content Restoration Authorship Familiarity Tests (CRAFT).\n"
    "The output includes test PDFs and answer keys. The test PDFs include LLM-generated\n"
    "additional sentences and synonym replacements to check authorship familiarity.\n\n"
    "Input format:\n"
    "- Tab Separated Value (TSV) file with no header row\n"
    "- 3 columns: student_number, name, text"
)

info_label = tb.Label(
    content,
    text=info_text,
    justify=LEFT,
    anchor="w",
    wraplength=540
)
info_label.pack(fill=X, pady=(0, 0))


# --------------------------------------------------
# EXAMPLE TABLE (NO HEADERS)
# --------------------------------------------------

tb.Label(
    content,
    text="Example (TSV, no header):",
).pack(anchor="w", pady=(5, 5))

table_frame = tb.Frame(content)
table_frame.pack(fill=X, pady=(0, 15))

def make_cell(parent, text):
    return tb.Label(
        parent,
        text=text,
        borderwidth=1,
        relief="solid",
        padding=5,
        anchor="w",
        justify="left",     # 🔥 allows multi-line alignment
        wraplength=380,     # 🔥 forces line wrapping
        font=("Segoe UI", 10)
    )

make_cell(table_frame, "N6MAA10816").grid(row=0, column=0, sticky="nsew")
make_cell(table_frame, "Roy Batty").grid(row=0, column=1, sticky="nsew")
make_cell(
    table_frame,
    "I've seen things you people wouldn't believe. "
    "Attack ships on fire off the shoulder of Orion. "
    "I watched C-beams glitter in the dark near the Tannhäuser Gate. "
    "All those moments will be lost in time, like tears in rain."
).grid(row=0, column=2, sticky="nsew")

table_frame.columnconfigure(0, weight=1)
table_frame.columnconfigure(1, weight=1)
table_frame.columnconfigure(2, weight=4)


# --------------------------------------------------
# FOOTER (GITHUB + CONTACT)
# --------------------------------------------------

footer_text = (
    "GitHub:\n"
    "https://github.com/cburst/CRAFT-Content-Restoration-Authorship-Familiarity-Test\n\n"
    "Developed by Richard Rose (HUFS)\n"
    "Contact: richard.rose@hufs.ac.kr"
)

footer_label = tb.Label(
    content,
    text=footer_text,
    justify=LEFT,
    anchor="w",
    wraplength=540
)

footer_label.pack(fill=X, pady=(0, 15))

# -----------------------------
# FILE PICKER
# -----------------------------

file_var = tk.StringVar()

def browse():
    f = filedialog.askopenfilename(filetypes=[("TSV files", "*.tsv")])
    if f:
        file_var.set(f)

tb.Label(content, text="Input TSV").pack(anchor="w")

row = tb.Frame(content)
row.pack(fill=X, pady=5)

tb.Entry(row, textvariable=file_var).pack(side=LEFT, fill=X, expand=YES, padx=(0, 5))
tb.Button(row, text="Browse", command=browse).pack(side=RIGHT)

# -----------------------------
# MODE
# -----------------------------

mode_var = tk.StringVar(value="real")

tb.Label(content, text="Mode").pack(anchor="w", pady=(10, 0))

tb.Radiobutton(
    content,
    text="Generate the CRAFT test for transcribed handwritten text",
    variable=mode_var,
    value="real"
).pack(anchor="w")

tb.Radiobutton(
    content,
    text="Generate the CRAFT test for LLM-generated text based on transcribed text",
    variable=mode_var,
    value="hard"
).pack(anchor="w")

tb.Radiobutton(
    content,
    text="Create both CRAFT tests (to compare performance)",
    variable=mode_var,
    value="both"
).pack(anchor="w")

# -----------------------------
# TEST NAME
# -----------------------------

name_var = tk.StringVar()

tb.Label(content, text="Test name").pack(anchor="w", pady=(10, 0))
tb.Entry(content, textvariable=name_var).pack(fill=X)

# -----------------------------
# STATUS + TIMER
# -----------------------------

status_var = tk.StringVar(value="Idle")
timer_var = tk.StringVar(value="")

tb.Label(content, textvariable=status_var).pack(pady=(15, 0))
tb.Label(content, textvariable=timer_var).pack()

# -----------------------------
# PROGRESS
# -----------------------------

# --- STYLE (run once, safe to keep here) ---
style = tb.Style()

style.configure(
    "Custom.Horizontal.TProgressbar",
    troughcolor="#1f1f1f",     # darker trough
    background="#4A90E2",
    lightcolor="#7EC8FF",      # brighter highlight
    darkcolor="#1C5FA8",       # deeper shadow
    bordercolor="#1f1f1f",
    thickness=16               # 🔥 thicker = much nicer
)

# --- PROGRESS BAR ---
progress = tb.Progressbar(
    content,
    mode="indeterminate",
    bootstyle="info-striped",   # 🔥 adds animated stripes
    length=500
)
progress.pack(fill=X, pady=12)

# --- TIMER ---
timer_running = False

def update_timer(start):
    if not timer_running:
        return

    elapsed = int(time.time() - start)

    mins = elapsed // 60
    secs = elapsed % 60

    if mins > 0:
        timer_var.set(f"Running... {mins}m {secs}s")
    else:
        timer_var.set(f"Running... {secs}s")

    app.after(1000, update_timer, start)

# -----------------------------
# RUN
# -----------------------------
def open_file(path):
    try:
        if not path or not os.path.exists(path):
            return

        if sys.platform == "darwin":
            subprocess.call(["open", path])
        elif os.name == "nt":
            os.startfile(path)
        else:
            subprocess.call(["xdg-open", path])
    except Exception:
        pass


def run_pipeline():

    global timer_running

    if not ensure_api_key():
        return

    f = file_var.get()
    mode = mode_var.get()
    name = name_var.get().strip() or "test"

    if not f:
        status_var.set("❌ Select a file")
        return

    safe_ui(status_var.set, "Running...")
    safe_ui(timer_var.set, "")

    def task():
        global timer_running

        try:
            start = time.time()
            timer_running = True

            safe_ui(lambda: progress.start(10))
            safe_ui(update_timer, start)

            # ---------------- REAL ----------------
            if mode == "real":
                safe_ui(status_var.set, "Running REAL pipeline...")
                real_pdf, *_ = run_real_pipeline(f, name)
                safe_ui(open_file, real_pdf)

            # ---------------- HARD ----------------
            elif mode == "hard":
                safe_ui(status_var.set, "Running HARD pipeline...")
                hard_pdf, *_ = run_hard_pipeline(f, name)
                safe_ui(open_file, hard_pdf)

            # ---------------- BOTH ----------------
            else:
                safe_ui(status_var.set, "Running REAL pipeline...")
                real_pdf, *_ = run_real_pipeline(f, name + "-real")
                safe_ui(open_file, real_pdf)

                safe_ui(status_var.set, "Running HARD pipeline...")
                hard_pdf, *_ = run_hard_pipeline(f, name + "-hard")
                safe_ui(open_file, hard_pdf)

            # ---------------- DONE ----------------
            timer_running = False
            safe_ui(progress.stop)
            safe_ui(status_var.set, "✔ Done")

        except Exception as e:
            timer_running = False
            safe_ui(progress.stop)
            safe_ui(status_var.set, f"❌ {e}")

    threading.Thread(target=task, daemon=True).start()

# -----------------------------
# BUTTON
# -----------------------------

tb.Button(
    bottom_bar,
    text="Run",
    command=run_pipeline,
    bootstyle="success",
    width=20
).pack(pady=5)

# -----------------------------
# START
# -----------------------------

app.mainloop()