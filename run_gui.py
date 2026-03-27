#!/usr/bin/env python3

import os
import sys
import time
import threading
import subprocess
import gc

import ttkbootstrap as tb
from ttkbootstrap.constants import *
from tkinter import filedialog
import tkinter as tk

from app.hard_pipeline import run_hard_pipeline
from app.real_pipeline import run_real_pipeline


def load_shell_env():
    try:
        result = subprocess.run(
            ["/bin/zsh", "-c", "source ~/.zshrc && printenv OPENAI_API_KEY"],
            capture_output=True,
            text=True
        )

        key = result.stdout.strip()
        if key:
            os.environ["OPENAI_API_KEY"] = key
            print("🔑 OPENAI_API_KEY loaded from .zshrc")

    except Exception as e:
        print("⚠️ Failed to load shell environment:", e)

# 🔥 run immediately
load_shell_env()

def ensure_api_key():

    # -----------------------------
    # 1. Check environment (shell already loaded)
    # -----------------------------
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return True

    # -----------------------------
    # 2. Try config file (~/.craft_api_key)
    # -----------------------------
    key_path = os.path.expanduser("~/.craft_api_key")

    if os.path.exists(key_path):
        try:
            with open(key_path) as f:
                key = f.read().strip()
                if key:
                    os.environ["OPENAI_API_KEY"] = key
                    print("🔑 API key loaded from ~/.craft_api_key")
                    return True
        except Exception as e:
            print("⚠️ Failed to read API key file:", e)

    # -----------------------------
    # 3. Prompt user (final fallback)
    # -----------------------------
    dialog = tb.dialogs.Querybox.get_string(
        "Enter your OpenAI API Key.\n\n"
        "Get one here:\n"
        "https://platform.openai.com/api-keys\n\n"
        "Paste it here. It will be saved securely on your computer.",
        title="Setup API Key"
    )

    if not dialog:
        tb.dialogs.Messagebox.show_error(
            "API key is required to run this application.",
            title="Missing API Key"
        )
        return False

    # -----------------------------
    # 4. Clean + validate input
    # -----------------------------
    key = dialog.strip().strip('"').strip("'")

    if not key:
        tb.dialogs.Messagebox.show_error(
            "Invalid API key.",
            title="Error"
        )
        return False

    # -----------------------------
    # 5. Save to file (secure permissions)
    # -----------------------------
    try:
        with open(key_path, "w") as f:
            f.write(key)
        os.chmod(key_path, 0o600)  # 🔒 user-only access
        print("🔑 API key saved to ~/.craft_api_key")
    except Exception as e:
        print("⚠️ Failed to save API key:", e)

    # -----------------------------
    # 6. Set environment
    # -----------------------------
    os.environ["OPENAI_API_KEY"] = key

    return True

# --------------------------------------------------
# THREAD-SAFE UI HELPER
# --------------------------------------------------

def safe_ui(func, *args, **kwargs):
    app.after(0, lambda: func(*args, **kwargs))


# --------------------------------------------------
# PLATFORM-SAFE FILE OPEN
# --------------------------------------------------

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


# --------------------------------------------------
# GUI SETUP
# --------------------------------------------------

app = tb.Window(themename="flatly")

# 🔥 Check API key after GUI starts
app.after(0, lambda: ensure_api_key())

ICON_PATH = os.path.expanduser("~/CRAFTtests/icon.png")

# 🔥 Window icon
try:
    icon = tk.PhotoImage(file=ICON_PATH)
    app.iconphoto(True, icon)
except Exception as e:
    print("Window icon failed:", e)

app.title("CRAFT Test Generator")
app.minsize(540, 420)

default_font = ("Segoe UI", 11)
app.option_add("*Font", default_font)

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
    wraplength=480
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
        wraplength=320,     # 🔥 forces line wrapping
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
    wraplength=480
)

footer_label.pack(fill=X, pady=(0, 15))

# --------------------------------------------------
# FILE PICKER
# --------------------------------------------------

file_var = tb.StringVar()

def browse_file():
    file = filedialog.askopenfilename(filetypes=[("TSV files", "*.tsv")])
    if file:
        file_var.set(file)

tb.Label(content, text="Input TSV").pack(anchor=W)

file_row = tb.Frame(content)
file_row.pack(fill=X, pady=5)

tb.Entry(file_row, textvariable=file_var).pack(side=LEFT, fill=X, expand=YES, padx=(0, 5))
tb.Button(file_row, text="Browse", command=browse_file, bootstyle="primary-outline").pack(side=RIGHT)


# --------------------------------------------------
# MODE  (🔥 ORIGINAL TEXT RESTORED)
# --------------------------------------------------

mode_var = tb.StringVar(value="real")

tb.Label(content, text="Mode").pack(anchor=W, pady=(10, 0))

tb.Radiobutton(
    content,
    text="Generate the CRAFT test for transcribed handwritten text",
    variable=mode_var,
    value="real"
).pack(anchor=W)

tb.Radiobutton(
    content,
    text="Generate the CRAFT test for LLM-generated text based on transcribed text",
    variable=mode_var,
    value="hard"
).pack(anchor=W)

tb.Radiobutton(
    content,
    text="Create both CRAFT tests (to compare performance)",
    variable=mode_var,
    value="both"
).pack(anchor=W)


# --------------------------------------------------
# TEST NAME
# --------------------------------------------------

name_var = tb.StringVar()

tb.Label(content, text="Test name (optional)").pack(anchor=W, pady=(10, 0))
tb.Entry(content, textvariable=name_var).pack(fill=X)


# --------------------------------------------------
# STATUS + TIMER
# --------------------------------------------------

status_var = tb.StringVar(value="Idle")
timer_var = tb.StringVar(value="")

status_label = tb.Label(content, textvariable=status_var)
status_label.pack(pady=(15, 0))

tb.Label(content, textvariable=timer_var).pack()


# --------------------------------------------------
# PROGRESS BAR
# --------------------------------------------------

progress = tb.Progressbar(content, mode="indeterminate", bootstyle="success")
progress.pack(fill=X, pady=10)


# --------------------------------------------------
# TIMER CONTROL
# --------------------------------------------------

timer_running = False

def update_timer(start_time):
    if not timer_running:
        return
    elapsed = int(time.time() - start_time)
    timer_var.set(f"Running... {elapsed}s")
    app.after(1000, update_timer, start_time)


# --------------------------------------------------
# RUN PIPELINE
# --------------------------------------------------

def run_pipeline():

    # 🔥 Ensure API key before running
    if not ensure_api_key():
        return

    input_file = file_var.get()
    mode = mode_var.get()
    testname = name_var.get().strip() or "test"

    if not input_file:
        status_var.set("❌ Please select a TSV file")
        status_label.configure(foreground="#d9534f")
        return

    run_button.config(state="disabled")

    def task():
        global timer_running

        try:
            start_time = time.time()
            timer_running = True

            safe_ui(progress.start, 10)
            safe_ui(status_label.configure, foreground="black")
            safe_ui(status_var.set, "Running pipeline...")

            update_timer(start_time)

            # ---------------- RUN MODES ----------------

            real_pdf = None
            hard_pdf = None

            if mode == "real":
                real_pdf, key, log, ts = run_real_pipeline(input_file, testname)

            elif mode == "hard":
                hard_pdf, key, log, ts = run_hard_pipeline(input_file, testname)

            elif mode == "both":

                safe_ui(status_var.set, "Running REAL pipeline...")
                real_pdf, _, _, _ = run_real_pipeline(
                    input_file, testname + "-real"
                )

                safe_ui(status_var.set, "Running HARD pipeline...")
                hard_pdf, _, _, _ = run_hard_pipeline(
                    input_file, testname + "-hard"
                )

            # ---------------- DONE ----------------

            elapsed = int(time.time() - start_time)
            timer_running = False

            safe_ui(progress.stop)
            safe_ui(progress.configure, value=0)

            outputs = []
            if real_pdf:
                outputs.append(os.path.basename(real_pdf))
            if hard_pdf:
                outputs.append(os.path.basename(hard_pdf))

            output_text = " | ".join(outputs) if outputs else "No output"

            safe_ui(status_label.configure, foreground="#2b7cff")
            safe_ui(status_var.set, f"✔ Done in {elapsed}s")
            safe_ui(timer_var.set, f"Output: {output_text}")

            if real_pdf:
                safe_ui(open_file, real_pdf)
            if hard_pdf:
                safe_ui(open_file, hard_pdf)

            time.sleep(0.2)
            gc.collect()

        except Exception as e:
            timer_running = False

            safe_ui(progress.stop)
            safe_ui(progress.configure, value=0)

            safe_ui(status_label.configure, foreground="#d9534f")
            safe_ui(status_var.set, "❌ Error")
            safe_ui(timer_var.set, str(e))

        finally:
            safe_ui(run_button.config, state="normal")

    threading.Thread(target=task, daemon=True).start()


# --------------------------------------------------
# RUN BUTTON
# --------------------------------------------------

run_button = tb.Button(
    bottom_bar,
    text="Run",
    command=run_pipeline,
    bootstyle="success",
    width=22
)

run_button.pack(pady=5)


# --------------------------------------------------
# START
# --------------------------------------------------

app.mainloop()