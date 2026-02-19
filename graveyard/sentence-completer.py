#!/usr/bin/env python3
import csv
import os
import re
import html
from weasyprint import HTML

# ---------- Configuration ---------- #
INPUT_TSV = "students.tsv"               # Input TSV with 3 columns, no headers
OUTPUT_TSV = "answer_key_sentence_completer.tsv"   # Output TSV with 10 extra columns
QUIZ_PDF_DIR = "quiz"                    # Output directory for quiz PDFs
KEY_PDF_DIR = "key"                      # Output directory for key PDFs
TOP_N = 10                               # Number of words to pick
FREQ_FILE = "wiki_freq.txt"             # Word frequency list: "word count" per line

# Words to never use as targets (lowercased)
AVOID_WORDS = {"hufs", "macalister", "minerva"}
# ----------------------------------- #

# Match "words" (letters, digits, underscore) in Unicode
WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)


def load_frequency_dict(path=FREQ_FILE):
    """
    Load a frequency dictionary from a file with lines like:
    the 186631452
    of 88349543
    ...
    Returns: dict[word_lower] = count (int)
    """
    freq = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                word = parts[0].lower()
                try:
                    count = int(parts[-1])
                except ValueError:
                    continue
                # if duplicates, keep the first or last; doesn't matter much
                freq[word] = count
    except FileNotFoundError:
        print(f"⚠️ Frequency file not found: {path}. All words treated as if not in list.")
    return freq


def get_top_words(text, freq_dict, n=TOP_N):
    """
    Return up to n most *obscure* words from text, based on freq_dict.

    - Only consider words that appear in freq_dict (i.e., in your word list).
    - Exclude very short words (len <= 2).
    - Exclude AVOID_WORDS (e.g. 'hufs', 'macalister', 'minerva').
    - Sort by frequency ascending (rarest first).
    - Final list is ordered by first appearance in the text.
    """
    tokens = WORD_RE.findall(text)
    first_pos = {}   # lower-word -> first index in tokens
    originals = {}   # lower-word -> original form as first seen

    for idx, tok in enumerate(tokens):
        key = tok.lower()
        if key not in first_pos:
            first_pos[key] = idx
            originals[key] = tok

    # Collect candidate words that are in freq_dict and not in avoid list
    candidates = []
    for key in first_pos.keys():
        if len(key) <= 2:
            continue
        if key in AVOID_WORDS:
            continue
        if key not in freq_dict:
            # Since you said "from the word list", we skip words not in freq_dict
            continue
        count = freq_dict.get(key, None)
        if count is None:
            continue
        candidates.append((count, key))

    # Sort by frequency ascending (smaller count = rarer)
    candidates.sort(key=lambda x: x[0])

    # Take top N by rarity
    selected_keys = [key for _, key in candidates[:n]]

    # Order selected words by first appearance in the text
    ordered = sorted(selected_keys, key=lambda k: first_pos[k])

    # Return original-casing words
    return [originals[k] for k in ordered]


def blank_out_words(text, words):
    """
    Replace all occurrences of the given words (case-insensitive, whole words)
    with "__" repeated for each character of the matched word.
    """
    if not words:
        return text

    escaped = [re.escape(w) for w in words]
    pattern = r"\b(" + "|".join(escaped) + r")\b"
    regex = re.compile(pattern, flags=re.IGNORECASE | re.UNICODE)

    def repl(match):
        word = match.group(0)
        return "__" * len(word)

    return regex.sub(repl, text)


def underline_words_html(text, words):
    """
    Return an HTML-safe string where all occurrences of the given words
    (case-insensitive, whole words) are wrapped in <u>...</u>,
    and everything else is properly HTML-escaped.
    """
    if not words:
        # Just escape the whole text
        return html.escape(text)

    escaped_words = [re.escape(w) for w in words]
    pattern = r"\b(" + "|".join(escaped_words) + r")\b"
    regex = re.compile(pattern, flags=re.IGNORECASE | re.UNICODE)

    result_parts = []
    last_end = 0

    for match in regex.finditer(text):
        start, end = match.start(), match.end()
        # Add the text before the match, escaped
        result_parts.append(html.escape(text[last_end:start]))
        # Add the matched word, underlined and escaped
        result_parts.append("<u>" + html.escape(match.group(0)) + "</u>")
        last_end = end

    # Add the remaining tail
    result_parts.append(html.escape(text[last_end:]))

    return "".join(result_parts)


def sanitize_filename(name, student_number):
    """
    Make a safe filename:
    - Keep Unicode characters (e.g., Korean names).
    - Replace path separators and colon with underscore.
    - If the result is empty, fall back to the student_number.
    """
    s = name.strip()
    s = s.replace("/", "_").replace("\\", "_").replace(":", "_")

    if not s:
        s = student_number.strip()
        s = s.replace("/", "_").replace("\\", "_").replace(":", "_")

    if not s:
        s = "student"

    return s


def make_quiz_pdf(student_number, name, gapped_text, outdir):
    """
    Generate the QUIZ PDF (target words replaced by "__" sequences).
    """
    os.makedirs(outdir, exist_ok=True)

    safe_name = sanitize_filename(name, student_number)
    pdf_path = os.path.join(outdir, f"{safe_name}.pdf")

    esc_name = html.escape(name)
    esc_number = html.escape(student_number)
    esc_text = html.escape(gapped_text)  # plain text → escaped

    html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{esc_name} - Quiz</title>
<style>
  @page {{
    margin: 2cm;
  }}
  body {{
    font-family: Arial, sans-serif;
    font-size: 14pt;
    line-height: 1.5;
  }}
  .header {{
    font-weight: bold;
    margin-bottom: 0.5em;
  }}
  .text {{
    white-space: pre-wrap;  /* preserve line breaks in the original text */
  }}
</style>
</head>
<body>
  <div class="header">
    Name: {esc_name}<br>
    Student Number: {esc_number}<br>
    Sentence Completer
  </div>
  <div class="text">
{esc_text}
  </div>
</body>
</html>
"""

    HTML(string=html_content).write_pdf(pdf_path)
    print(f"✅ Generated QUIZ PDF: {pdf_path}")


def make_key_pdf(student_number, name, key_text_html, outdir):
    """
    Generate the KEY PDF (target words underlined, visible).
    key_text_html must already be HTML-safe with <u> tags inserted.
    """
    os.makedirs(outdir, exist_ok=True)

    safe_name = sanitize_filename(name, student_number)
    pdf_path = os.path.join(outdir, f"{safe_name}.pdf")

    esc_name = html.escape(name)
    esc_number = html.escape(student_number)

    html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{esc_name} - Key</title>
<style>
  @page {{
    margin: 2cm;
  }}
  body {{
    font-family: Arial, sans-serif;
    font-size: 15pt;
    line-height: 1.5;
  }}
  .header {{
    font-weight: bold;
    margin-bottom: 0.5em;
  }}
  .text {{
    white-space: pre-wrap;
  }}
</style>
</head>
<body>
  <div class="header">
    Name: {esc_name}<br>
    Student Number: {esc_number}
  </div>

  <br>

  <div class="text">
{key_text_html}
  </div>
</body>
</html>
"""

    HTML(string=html_content).write_pdf(pdf_path)
    print(f"✅ Generated KEY PDF: {pdf_path}")


def main():
    # Load frequency dictionary once
    freq_dict = load_frequency_dict(FREQ_FILE)

    # Create output directories if they don't exist
    os.makedirs(QUIZ_PDF_DIR, exist_ok=True)
    os.makedirs(KEY_PDF_DIR, exist_ok=True)

    with open(INPUT_TSV, "r", encoding="utf-8", newline="") as infile, \
         open(OUTPUT_TSV, "w", encoding="utf-8", newline="") as outfile:

        reader = csv.reader(infile, delimiter="\t")
        writer = csv.writer(outfile, delimiter="\t")

        for row in reader:
            # Expect at least 3 columns: student_number, name, text
            if len(row) < 3:
                continue

            student_number, name, text = row[0], row[1], row[2]

            # 1) Get TOP_N most obscure words (based on freq_dict)
            top_words = get_top_words(text, freq_dict, TOP_N)

            # 2) Append them as TOP_N extra columns in the output TSV
            extra_cols = top_words + [""] * (TOP_N - len(top_words))
            writer.writerow([student_number, name, text] + extra_cols)

            # 3) Create gapped text for QUIZ
            gapped_text = blank_out_words(text, top_words)

            # 4) Create underlined HTML text for KEY
            key_text_html = underline_words_html(text, top_words)

            # 5) Make per-student PDFs
            make_quiz_pdf(student_number, name, gapped_text, QUIZ_PDF_DIR)
            make_key_pdf(student_number, name, key_text_html, KEY_PDF_DIR)


if __name__ == "__main__":
    main()