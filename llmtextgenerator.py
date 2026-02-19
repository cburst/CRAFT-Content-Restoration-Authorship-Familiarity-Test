#!/usr/bin/env python3
import csv
import time
import requests
import nltk
import os
from nltk.tokenize import sent_tokenize, word_tokenize

# =========================
# CONFIG (OpenAI)
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

OPENAI_URL = "https://api.openai.com/v1/responses"
MODEL = "gpt-4.1"

INPUT_TSV = "students.tsv"
OUTPUT_TSV = "llmoutput.tsv"

MIN_SENTENCES = 16
MIN_WORDS = 250
MAX_RETRIES = 5

# =========================
# OpenAI helper (future-proof)
# =========================
def llm_chat(prompt, temperature=0.8, max_tokens=1000):
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "input": prompt,
        "temperature": temperature,
        "max_output_tokens": max_tokens,
    }

    r = requests.post(OPENAI_URL, headers=headers, json=payload, timeout=60)
    r.raise_for_status()

    data = r.json()

    # Robust extraction
    if "output_text" in data:
        return data["output_text"].strip()

    return data["output"][0]["content"][0]["text"].strip()


# =========================
# Validation utilities (NLTK)
# =========================
def count_sentences(text):
    return len(sent_tokenize(text))

def count_words(text):
    return len(word_tokenize(text))

# =========================
# Main idea extraction
# =========================
def extract_main_idea(original_text):
    prompt = (
        "Read the following student text and state its main idea "
        "in ONE clear academic sentence.\n\n"
        f"TEXT:\n{original_text}"
    )
    return llm_chat(prompt, temperature=0.3, max_tokens=80)

# =========================
# Text generation (strict)
# =========================
def generate_valid_text(original_text, main_idea):
    base_prompt = (
        "Write a completely NEW academic-style text at approximately CEFR B2 level, "
        "suitable for an upper-intermediate university student.\n\n"
        "TOPIC (main idea):\n"
        f"{main_idea}\n\n"
        "LANGUAGE REQUIREMENTS (CEFR B2):\n"
        "- Clear topic sentences and paragraph development\n"
        "- Some complex sentences, but limited embedding\n"
        "- Mostly high-frequency academic vocabulary\n"
        "- Avoid highly technical or specialist terminology\n\n"
        "STRUCTURAL REQUIREMENTS:\n"
        f"- At least {MIN_SENTENCES} sentences\n"
        f"- At least {MIN_WORDS} words total\n"
        "- Do NOT reuse wording, sentence structure, or phrasing from the original\n"
        "- Maintain an academic register appropriate for B2 writers\n\n"
        f"ORIGINAL TEXT (for context only):\n{original_text}"
    )

    best_text = None
    best_score = -1
    best_sents = 0
    best_words = 0

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"    → Generation attempt {attempt}...")
        text = llm_chat(base_prompt)

        sents = count_sentences(text)
        words = count_words(text)

        passes = (sents >= MIN_SENTENCES and words >= MIN_WORDS)

        print(
            f"      sentences: {sents} | words: {words} "
            f"{'✓ PASS' if passes else '✗ FAIL'}"
        )

        # Scoring: how close are we?
        score = min(sents / MIN_SENTENCES, 1.0) + min(words / MIN_WORDS, 1.0)

        if score > best_score:
            best_score = score
            best_text = text
            best_sents = sents
            best_words = words

        if passes:
            print("      ✓ Constraints satisfied\n")
            return text

        time.sleep(1)

    # ---- FALLBACK MODE ----
    print(
        f"⚠️ Using best candidate after {MAX_RETRIES} attempts "
        f"(sentences={best_sents}, words={best_words})\n"
    )

    return best_text

# =========================
# TSV processing
# =========================
def process_tsv(input_tsv, output_tsv):
    with open(input_tsv, newline="", encoding="utf-8") as infile, \
         open(output_tsv, "w", newline="", encoding="utf-8") as outfile:

        reader = csv.reader(infile, delimiter="\t")
        writer = csv.writer(outfile, delimiter="\t")

        for row in reader:
            if len(row) < 3:
                continue

            student_id, name, original_text = row[0], row[1], row[2]

            print(f"\n=== Processing {student_id} / {name} ===")

            try:
                main_idea = extract_main_idea(original_text)
                print("  Main idea sentence:")
                print(f"  → {main_idea}\n")

                generated_text = generate_valid_text(original_text, main_idea)

            except Exception as e:
                print(f"⚠️ Hard failure for {student_id}: {e}")
                continue

            writer.writerow([
                student_id,
                name,
                generated_text,
            ])

    print(f"\n✅ Done. Output written to {output_tsv}")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    process_tsv(INPUT_TSV, OUTPUT_TSV)