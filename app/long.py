#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============================================================
# IMPORTS
# ============================================================

import os
import sys
import shutil

from pdf2image import convert_from_path
from PyPDF2 import PdfReader
from PIL import Image, ImageChops

# ============================================================
# BASE DIR (PyInstaller-safe)
# ============================================================

if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_DIR = os.path.join(BASE_DIR, "app", "data")
# ============================================================
# POPPLER (required for pdf2image)
# ============================================================

POPPLER_BIN = os.path.join(DATA_DIR, "poppler", "Library", "bin")

# Windows needs PATH injection
if sys.platform == "win32":
    os.environ["PATH"] = (
        POPPLER_BIN + os.pathsep +
        os.environ.get("PATH", "")
    )

# ============================================================
# CONSTANTS
# ============================================================

# A4 page at 200 DPI
PAGE_WIDTH  = 1654
PAGE_HEIGHT = 2339
DPI = 200

# Margins
LEFT_MARGIN   = 40
RIGHT_MARGIN  = 40
TOP_MARGIN    = 100
BOTTOM_MARGIN = 100

# Spacing between page1 and page2
GAP = 10


# ============================================================
# WHITESPACE TRIMMING
# ============================================================

def trim_top_whitespace(img):
    bg = Image.new(img.mode, img.size, "white")
    diff = ImageChops.difference(img, bg)
    bbox = diff.getbbox()
    if not bbox:
        return img
    top = bbox[1]
    return img.crop((0, top, img.width, img.height))


def trim_bottom_whitespace(img):
    bg = Image.new(img.mode, img.size, "white")
    diff = ImageChops.difference(img, bg)
    bbox = diff.getbbox()
    if not bbox:
        return img
    bottom = bbox[3]
    return img.crop((0, 0, img.width, bottom))


# ============================================================
# PHASE 1: COLLECT MULTIPAGE PDFs
# ============================================================

def collect_long_pdfs(src_folder, long_folder):
    os.makedirs(long_folder, exist_ok=True)

    for fname in os.listdir(src_folder):
        if not fname.lower().endswith(".pdf"):
            continue

        fpath = os.path.join(src_folder, fname)

        try:
            reader = PdfReader(fpath)
            if len(reader.pages) > 1:
                shutil.copy(fpath, os.path.join(long_folder, fname))
        except Exception:
            pass


# ============================================================
# PHASE 2: PROCESS SINGLE PDF
# ============================================================

def process_pdf(path, outpath):
    try:
        if sys.platform == "win32":
            pages = convert_from_path(
                path,
                dpi=DPI,
                poppler_path=POPPLER_BIN
            )
        else:
            pages = convert_from_path(
                path,
                dpi=DPI
            )
    except Exception as e:
        print(f"❌ Failed to convert PDF → {path}: {e}")
        return

    # Only process exactly 2-page PDFs
    if len(pages) != 2:
        print(f"⚠️ Skipping (not 2 pages): {path}")
        return

    page1, page2 = pages

    # Trim whitespace
    page1 = trim_top_whitespace(page1)
    page1 = trim_bottom_whitespace(page1)

    page2 = trim_top_whitespace(page2)
    page2 = trim_bottom_whitespace(page2)

    # Normalize widths
    target_w = min(page1.width, page2.width)
    page1 = page1.resize((target_w, int(page1.height * target_w / page1.width)))
    page2 = page2.resize((target_w, int(page2.height * target_w / page2.width)))

    # Combine vertically
    combined_h = page1.height + GAP + page2.height
    combined = Image.new("RGB", (target_w, combined_h), "white")

    combined.paste(page1, (0, 0))
    combined.paste(page2, (0, page1.height + GAP))

    # Fit to A4 with margins
    avail_w = PAGE_WIDTH  - LEFT_MARGIN - RIGHT_MARGIN
    avail_h = PAGE_HEIGHT - TOP_MARGIN  - BOTTOM_MARGIN

    scale_w = avail_w / combined.width
    scale_h = avail_h / combined.height
    scale = min(scale_w, scale_h)

    scaled_w = int(combined.width  * scale)
    scaled_h = int(combined.height * scale)

    resized = combined.resize((scaled_w, scaled_h), Image.LANCZOS)

    # Place on page
    final_img = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), "white")

    offset_x = LEFT_MARGIN + (avail_w - scaled_w) // 2
    offset_y = TOP_MARGIN

    final_img.paste(resized, (offset_x, offset_y))

    # Save
    final_img.save(outpath, "PDF", resolution=DPI)


# ============================================================
# PHASE 3: PROCESS FOLDER
# ============================================================

def process_long_folder(long_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for fname in os.listdir(long_folder):
        if fname.lower().endswith(".pdf"):
            process_pdf(
                os.path.join(long_folder, fname),
                os.path.join(output_folder, fname)
            )


# ============================================================
# MAIN PIPELINE FUNCTION (USED BY YOUR APP)
# ============================================================

def process_long_pdfs(input_dir, temp_dir, output_dir):
    """
    Process multipage PDFs and overwrite originals.

    input_dir: where PDFs currently live (workspace/pdfs)
    temp_dir: temporary working directory (workspace/tmp)
    output_dir: usually same as input_dir
    """

    long_dir = os.path.join(temp_dir, "long")
    fixed_dir = os.path.join(temp_dir, "long_fixed")

    print(f"\n📂 Processing long PDFs")
    print(f"Input: {input_dir}")
    print(f"Temp: {temp_dir}")

    # Clean temp dirs
    shutil.rmtree(long_dir, ignore_errors=True)
    shutil.rmtree(fixed_dir, ignore_errors=True)

    # Step 1: collect multipage PDFs
    collect_long_pdfs(input_dir, long_dir)

    # Step 2: process them
    process_long_folder(long_dir, fixed_dir)

    # Step 3: overwrite originals
    if os.path.exists(fixed_dir):
        for fname in os.listdir(fixed_dir):
            src = os.path.join(fixed_dir, fname)
            dst = os.path.join(output_dir, fname)

            try:
                if os.path.exists(dst):
                    os.remove(dst)
                shutil.move(src, dst)
                print(f"✔ Replaced {fname}")
            except Exception as e:
                print(f"❌ Failed to replace {fname}: {e}")

    # Cleanup (only if it's clearly a temp folder)
    if os.path.basename(temp_dir).startswith("_tmp"):
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("✅ Long PDF processing complete\n")


# ============================================================
# CLI SUPPORT (OPTIONAL)
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python long.py <input_dir>")
        sys.exit(1)

    input_dir = sys.argv[1]

    process_long_pdfs(
        input_dir=input_dir,
        temp_dir=os.path.join(input_dir, "_tmp_long"),
        output_dir=input_dir
    )