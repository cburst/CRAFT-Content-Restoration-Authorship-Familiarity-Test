#!/usr/bin/env python3
import csv
import os
import re
import random
import time
import html
import json
import requests
from weasyprint import HTML

import nltk
from nltk.tokenize import sent_tokenize

# Ensure punkt is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# ===============================
# CONFIGURATION
# ===============================

INPUT_TSV = "students.tsv"  # student_id, name, text
PDF_DIR = "PDFs-sentence-intruder"
ANSWER_KEY = "answer_key_sentence_intruder.tsv"

# 🔑 Put your DeepSeek API key here
DEEPSEEK_API_KEY = "YOUR-API-KEY-HERE"

DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

MAX_DEEPSEEK_RETRIES = 5


# ===============================
# Utilities
# ===============================

def split_into_sentences(text):
    if not text:
        return []
    sentences = sent_tokenize(str(text))
    return [s.strip() for s in sentences if s.strip()]

def tokenize_words(text):
    return re.findall(r"[A-Za-z']+", str(text).lower())

def normalize_sentence(text):
    return " ".join(tokenize_words(text)).strip()

def sanitize_filename(name):
    forbidden = r'\/:*?"<>|'
    return "".join(c for c in name if c not in forbidden).strip() or "student"


# ===============================
# DeepSeek generation
# ===============================

def generate_inserted_sentence_with_deepseek(essay_text,
                                             prev_sentence,
                                             next_sentence,
                                             existing_sentences,
                                             model=DEEPSEEK_MODEL,
                                             max_retries=MAX_DEEPSEEK_RETRIES):
    """
    Ask DeepSeek to write ONE additional sentence that could naturally appear
    between prev_sentence and next_sentence, matching style/tone.

    Prints all DeepSeek output to the terminal.
    Ensures the sentence is not identical (after normalization) to any existing sentence.
    """
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DeepSeek API key not set. Please fill DEEPSEEK_API_KEY.")

    system_prompt = (
        "You are a careful writing assistant. Given a student's essay and two "
        "adjacent sentences from it, you will write exactly one additional sentence "
        "that could naturally appear between them.\n\n"
        "Requirements:\n"
        "1) The new sentence must be on the same topic as the essay.\n"
        "2) It should match the student's style, tone, and level (non-native but academic).\n"
        "3) It must connect logically between the previous and next sentence.\n"
        "4) It should be 10–30 words long.\n"
        "5) Do NOT copy any existing sentence.\n"
        "6) Output only the new sentence, no explanations or numbering."
    )

    # We can truncate essay_text if very long, but usually it's fine as context.
    user_prompt = (
        "Here is the student's essay:\n"
        f"{essay_text}\n\n"
        "You will insert one new sentence between the following two sentences:\n"
        f"Previous sentence:\n\"{prev_sentence}\"\n\n"
        f"Next sentence:\n\"{next_sentence}\"\n\n"
        "Write one sentence that could naturally appear between them, following the rules."
    )

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 200,
        "temperature": 0.7,
        "stream": False,
    }

    existing_norms = {normalize_sentence(s) for s in existing_sentences}

    attempt = 0
    while attempt <= max_retries:
        try:
            print("\n----------------------------")
            print("Context for inserted sentence:")
            print("Previous:", prev_sentence)
            print("Next    :", next_sentence)
            print("Calling DeepSeek... (attempt", attempt + 1, ")")

            resp = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=30)

            if resp.status_code >= 400:
                print(f"⚠️ DeepSeek HTTP {resp.status_code}")
                print("Response body:")
                print(resp.text)
                raise requests.HTTPError(f"HTTP {resp.status_code}: {resp.text}")

            job = resp.json()

            # Extract text content
            text = None
            if "choices" in job and job["choices"]:
                choice = job["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    text = choice["message"]["content"]
                elif "text" in choice:
                    text = choice["text"]
            if text is None:
                text = str(job)

            print("DeepSeek raw content:")
            print(text)
            print("----------------------------")

            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            if not lines:
                attempt += 1
                print("⚠️ Empty DeepSeek response lines; retrying...")
                time.sleep(1.0 * attempt)
                continue

            candidate = lines[0]
            # Strip outer quotes if present
            if (candidate.startswith('"') and candidate.endswith('"')) or \
               (candidate.startswith("'") and candidate.endswith("'")):
                candidate = candidate[1:-1].strip()

            norm_candidate = normalize_sentence(candidate)
            if not norm_candidate:
                print("⚠️ Empty candidate after normalization; retrying...")
                attempt += 1
                time.sleep(1.0 * attempt)
                continue

            if norm_candidate in existing_norms:
                print("⚠️ Candidate matches an existing sentence; retrying...")
                attempt += 1
                time.sleep(1.0 * attempt)
                continue

            print("Accepted inserted sentence:")
            print(candidate)
            print("----------------------------")
            return candidate

        except Exception as e:
            print(f"⚠️ DeepSeek error: {e}")
            attempt += 1
            time.sleep(1.0 * attempt)

    # Fallback: if DeepSeek fails repeatedly, make a simple bridging sentence.
    print("⚠️ Using local fallback sentence (DeepSeek failed).")
    fallback = "In addition, this point connects closely to the previous idea."
    return fallback


# ===============================
# Build augmented text per student
# ===============================

def build_augmented_text_with_intruder(text):
    """
    Given a student's full text, return:
      augmented_sentences (list of strings),
      intruder_index (0-based),
      intruder_sentence (string)

    The intruder sentence is generated via DeepSeek and inserted somewhere
    in the middle of the essay.
    """
    sentences = split_into_sentences(text)
    if len(sentences) < 2:
        # Too short to sensibly insert in the middle
        return None, None, None

    # Choose an insertion index somewhere inside.
    # If 2 sentences, insert between them (index 1).
    # If more, pick from 1..len-1 (not before first, not after last).
    if len(sentences) == 2:
        insert_idx = 1
    else:
        insert_idx = random.randint(1, len(sentences) - 1)

    prev_sentence = sentences[insert_idx - 1]
    next_sentence = sentences[insert_idx] if insert_idx < len(sentences) else ""

    intruder = generate_inserted_sentence_with_deepseek(
        essay_text=text,
        prev_sentence=prev_sentence,
        next_sentence=next_sentence,
        existing_sentences=sentences,
    )

    augmented = sentences[:insert_idx] + [intruder] + sentences[insert_idx:]
    return augmented, insert_idx, intruder


# ===============================
# PDF generation
# ===============================

def generate_pdf_for_student(student_id, name, augmented_sentences,
                             intruder_index, out_dir=PDF_DIR):
    """
    augmented_sentences: list of sentences (including the added one)
    intruder_index: 0-based index of the inserted sentence
    """
    os.makedirs(out_dir, exist_ok=True)
    safe_name = sanitize_filename(name)
    pdf_path = os.path.join(out_dir, f"{safe_name}.pdf")

    esc_name = html.escape(name)
    esc_number = html.escape(student_id)

    html_parts = [
        "<html><head><meta charset='utf-8'><style>",
        "@page { margin: 1.5cm; size: A4; }",
        "body { font-family: Arial, sans-serif; font-size: 13pt; line-height: 1.3; }",
        ".header { font-weight: bold; margin-bottom: 0.5em; }",
        ".text { white-space: pre-wrap; margin: 0.3em 0; }",
        "</style></head><body>",
        f"<div class='header'>Name: {esc_name}<br>Student Number: {esc_number}</div>",
        "<div class='header'>Sentence Intruder </div>",
        "<div class='text'>Below is a version of your essay with <b>one extra sentence</b> added.</div>",
        "<div class='text'>Identify the sentence that was <b>not</b> in your original essay.</div>",
        "<br>",
    ]

    # Number and display each sentence on its own line
    for i, sent in enumerate(augmented_sentences, start=1):
        esc_sent = html.escape(sent)
        html_parts.append(
            f"<div class='text'><b>{i}.</b> {esc_sent}</div>"
        )

    html_parts.append("</body></html>")
    html_doc = "\n".join(html_parts)

    HTML(string=html_doc).write_pdf(pdf_path)
    print(f"📝 Intruder-sentence PDF created: {pdf_path}")


# ===============================
# Main processing
# ===============================

def process_tsv_for_intruder_sentence(input_tsv=INPUT_TSV,
                                      pdf_dir=PDF_DIR,
                                      answer_key_tsv=ANSWER_KEY):
    """
    Input TSV: student_id, name, text

    For each student:
      - Build an augmented version of their essay with one inserted sentence.
      - Generate a PDF immediately.
      - Append answer-key row with intruder sentence info.
    """
    with open(answer_key_tsv, "w", newline="", encoding="utf-8") as keyfile:
        keywriter = csv.writer(keyfile, delimiter="\t")
        keywriter.writerow([
            "student_id",
            "name",
            "intruder_sentence_number",  # 1-based index in augmented text
            "intruder_sentence",
            "original_text",
            "augmented_text"
        ])

        with open(input_tsv, newline="", encoding="utf-8") as infile:
            reader = csv.reader(infile, delimiter="\t")

            for row in reader:
                if len(row) < 3:
                    continue
                student_id, name, text = row[0], row[1], row[2]

                print(f"\n=== Processing {student_id} / {name} ===")
                augmented_sentences, intruder_index, intruder_sentence = \
                    build_augmented_text_with_intruder(text)

                if augmented_sentences is None:
                    print(f"⚠️ Text too short for {student_id}; skipping.")
                    continue

                # Generate PDF for this student right away
                generate_pdf_for_student(
                    student_id,
                    name,
                    augmented_sentences,
                    intruder_index,
                    out_dir=pdf_dir
                )

                intruder_number = intruder_index + 1  # 1-based

                augmented_text_str = " ".join(augmented_sentences)

                keywriter.writerow([
                    student_id,
                    name,
                    intruder_number,
                    intruder_sentence,
                    text,
                    augmented_text_str
                ])
                keyfile.flush()

                print(f"✅ Finished {student_id} / {name}")

    print(f"\n🎯 Done. Answer key saved to {answer_key_tsv}")


# ===============================
# MAIN
# ===============================

if __name__ == "__main__":
    random.seed()  # or set a fixed seed for reproducibility
    process_tsv_for_intruder_sentence()