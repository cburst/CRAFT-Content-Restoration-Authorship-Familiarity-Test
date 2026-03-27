import csv
import re
import os
from collections import Counter
from weasyprint import HTML
import html

# -----------------------------
# 0. One pager
# -----------------------------

from PyPDF2 import PdfReader, PdfWriter

def remove_blank_last_page(pdf_path):
    reader = PdfReader(pdf_path)
    writer = PdfWriter()
    if len(reader.pages) > 1:
        # Keep all but the last page
        for i in range(len(reader.pages) - 1):
            writer.add_page(reader.pages[i])
        with open(pdf_path, "wb") as f:
            writer.write(f)
        # print(f"🧹 Trimmed extra page from: {pdf_path}")

# -----------------------------
# 1. Load frequency dictionary
# -----------------------------
def load_wiki_frequency_dict(path="wiki_freq.txt"):
    freq = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                word, count_str = line.rsplit(maxsplit=1)
                freq[word.lower()] = int(count_str)
            except ValueError:
                continue
    return freq


# -----------------------------
# 2. Tokenization
# -----------------------------
def tokenize(text):
    raw_tokens = re.findall(r"[A-Za-z']+", str(text).lower())
    clean_tokens = [t.strip("'") for t in raw_tokens if t.strip("'")]
    return clean_tokens


# -----------------------------
# 3. Find most obscure words
# -----------------------------
def find_most_obscure_words(text, freq_dict, top_n=10):
    tokens = tokenize(text)
    counts = Counter(tokens)
    candidate_words = [w for w in counts if len(w) > 2]
    results = [(w, freq_dict.get(w, 0), counts[w]) for w in candidate_words]
    results.sort(key=lambda x: x[1])  # sort by frequency ascending
    obscure_words = [w for w, _, _ in results[:top_n]]
    while len(obscure_words) < top_n:
        obscure_words.append("")
    return obscure_words


# -----------------------------
# 4. Generate per-student PDF
# -----------------------------
def sanitize_filename(name):
    forbidden = r'\/:*?"<>|'
    return "".join(c for c in name if c not in forbidden).strip() or "student"

def generate_pdf_for_student(student_id, name, obscure_words, out_dir="PDFs-sentence-creator"):
    os.makedirs(out_dir, exist_ok=True)
    safe_name = sanitize_filename(name)
    pdf_path = os.path.join(out_dir, f"{safe_name}.pdf")

    # Escape HTML special chars just in case
    esc_name = html.escape(name)
    esc_number = html.escape(student_id)

    # Build HTML with your provided CSS and structure
    html_parts = [
        "<html>",
        "<head>",
        "<meta charset='utf-8'>",
        "<style>",
        "  @page { margin: 1cm; }",
        "  body { font-family: Arial, sans-serif; font-size: 15pt; line-height: 1.25; }",
        "  .header { font-weight: bold; margin-bottom: 0.5em; }",
        "  .text { white-space: pre-wrap; }",
        "</style>",
        "</head>",
        "<body>",
        f'  <div class="header">',
        f'    Name: {esc_name}<br>',
        f'    Student Number: {esc_number}',
        f'  </div>',
    ]

    # Add numbered obscure words
    for i, w in enumerate(obscure_words, start=1):
        if not w:
            continue
        html_parts.append(
            f"<div class='text'>{i}. {html.escape(w)}<br>"
            f"{'_' * 60}<br>{'_' * 60}.</div><br>"
        )

    html_parts.append("</body></html>")
    html_doc = "\n".join(html_parts)

    HTML(string=html_doc).write_pdf(pdf_path)
    remove_blank_last_page(pdf_path)
    print(f"📝 PDF created: {pdf_path}")


# -----------------------------
# 5. TSV Processor
# -----------------------------
def process_tsv(input_tsv, output_tsv, freq_file="wiki_freq.txt", top_n=10, pdf_dir="PDFs-sentence-creator"):
    freq_dict = load_wiki_frequency_dict(freq_file)

    with open(input_tsv, newline="", encoding="utf-8") as infile, \
         open(output_tsv, "w", newline="", encoding="utf-8") as outfile:
        reader = csv.reader(infile, delimiter="\t")
        writer = csv.writer(outfile, delimiter="\t")

        for row in reader:
            if len(row) < 3:
                continue
            student_id, name, text = row[0], row[1], row[2]
            obscure_words = find_most_obscure_words(text, freq_dict, top_n=top_n)
            writer.writerow(row + obscure_words)
            generate_pdf_for_student(student_id, name, obscure_words, out_dir=pdf_dir)

    print(f"✅ TSV and PDFs complete! Output TSV: {output_tsv}")

# -----------------------------
# 6. Run example
# -----------------------------
if __name__ == "__main__":
    INPUT_TSV = "students.tsv"
    OUTPUT_TSV = "word_list_sentence-creator.tsv"
    WIKI_FREQ = "wiki_freq.txt"
    PDF_DIR = "PDFs-sentence-creator"

    process_tsv(INPUT_TSV, OUTPUT_TSV, WIKI_FREQ, top_n=10, pdf_dir=PDF_DIR)