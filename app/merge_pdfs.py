#!/usr/bin/env python3
import os
from PyPDF2 import PdfMerger


def merge_pdfs(input_dir, output_path):
    """
    Merge all PDFs in input_dir into output_path.
    Sorted alphabetically.
    Automatically excludes the output file if it already exists.
    """

    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    output_name = os.path.basename(output_path)

    pdf_files = sorted(
        f for f in os.listdir(input_dir)
        if f.lower().endswith(".pdf") and f != output_name
    )

    if not pdf_files:
        raise RuntimeError(f"No PDFs found in {input_dir}")

    merger = PdfMerger()

    for pdf in pdf_files:
        full_path = os.path.join(input_dir, pdf)

        if not os.path.isfile(full_path):
            continue  # extra safety

        print(f"➕ Adding {full_path}")
        merger.append(full_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merger.write(output_path)
    merger.close()

    print(f"📄 Merged PDF saved → {output_path}")
    return output_path


# --------------------------------------------------
# CLI SUPPORT
# --------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python merge_pdfs.py <input_dir> [output_pdf]")
        sys.exit(1)

    input_dir = sys.argv[1]

    if len(sys.argv) >= 3:
        output_pdf = sys.argv[2]
    else:
        output_pdf = os.path.join(
            input_dir,
            f"{os.path.basename(input_dir)}.pdf"
        )

    merge_pdfs(input_dir, output_pdf)