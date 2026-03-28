#!/usr/bin/env python3
import csv
import time
import requests
import nltk
import os
import sys
from nltk.tokenize import sent_tokenize, word_tokenize

# ============================================================
# NLTK SETUP (PACKAGING-SAFE)
# ============================================================

def ensure_nltk():
    """
    Ensure required NLTK tokenizers are available.
    Works in packaged environments.
    """
    resources = ["punkt", "punkt_tab"]

    for res in resources:
        try:
            nltk.data.find(f"tokenizers/{res}")
        except LookupError:
            print(f"⬇️ Downloading {res}...")
            nltk.download(res, quiet=True)

ensure_nltk()

# ============================================================
# CONFIG (OPENAI)
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

OPENAI_URL = "https://api.openai.com/v1/responses"
MODEL = "gpt-4.1"

# Defaults (CLI fallback only)
DEFAULT_INPUT_TSV = "students.tsv"
DEFAULT_OUTPUT_TSV = "llmoutput.tsv"

MIN_SENTENCES = 16
MIN_WORDS = 250
MAX_RETRIES = 5

# ============================================================
# OPENAI HELPER
# ============================================================

def llm_chat(prompt, temperature=0.8, max_tokens=1000):
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "input": prompt,
        "temperature": temperature,
        "max_output_tokens": max(16, max_tokens),
    }

    transient = {408, 429, 500, 502, 503, 504, 520, 522, 524}
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        start_time = time.perf_counter()
        print(f"\n⏱ LLM call attempt {attempt}/{MAX_RETRIES}")

        try:
            r = requests.post(
                OPENAI_URL,
                headers=headers,
                json=payload,
                timeout=60
            )

            if r.status_code in transient:
                raise requests.HTTPError(f"Transient error {r.status_code}")

            r.raise_for_status()
            data = r.json()

            elapsed = time.perf_counter() - start_time
            print(f"⏱ Completed in {elapsed:.2f}s")

            if data.get("output_text"):
                return data["output_text"].strip()

            for item in data.get("output", []):
                for block in item.get("content", []):
                    if block.get("type") == "output_text":
                        return block.get("text", "").strip()

            raise RuntimeError("No text returned")

        except Exception as e:
            elapsed = time.perf_counter() - start_time
            print(f"⚠️ Attempt failed ({elapsed:.2f}s): {e}")
            last_error = e

            if attempt < MAX_RETRIES:
                wait = 2 ** attempt
                print(f"⏳ Retrying in {wait}s...")
                time.sleep(wait)

    raise RuntimeError(f"LLM failed after retries: {last_error}")


# ============================================================
# TEXT METRICS
# ============================================================

def count_sentences(text):
    return len(sent_tokenize(text))

def count_words(text):
    return len(word_tokenize(text))


# ============================================================
# MAIN IDEA
# ============================================================

def extract_main_idea(original_text):
    prompt = (
        "Read the following student text and state its main idea "
        "in ONE clear academic sentence.\n\n"
        f"TEXT:\n{original_text}"
    )
    return llm_chat(prompt, temperature=0.3, max_tokens=80)


# ============================================================
# TEXT GENERATION
# ============================================================

def generate_valid_text(original_text, main_idea):
    base_prompt = (
        "Write a completely NEW academic-style text at approximately CEFR B2 level.\n\n"
        "TOPIC:\n"
        f"{main_idea}\n\n"
        f"REQUIREMENTS:\n"
        f"- At least {MIN_SENTENCES} sentences\n"
        f"- At least {MIN_WORDS} words\n"
        "- Clear academic structure\n"
    )

    best_text = None
    best_score = -1

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"    → Generation attempt {attempt}")
        text = llm_chat(base_prompt)

        sents = count_sentences(text)
        words = count_words(text)

        passes = (sents >= MIN_SENTENCES and words >= MIN_WORDS)

        print(f"      sentences={sents} words={words} {'✓' if passes else '✗'}")

        score = min(sents / MIN_SENTENCES, 1.0) + min(words / MIN_WORDS, 1.0)

        if score > best_score:
            best_score = score
            best_text = text

        if passes:
            return text

        time.sleep(1)

    print("⚠️ Using best fallback text")
    return best_text


# ============================================================
# TSV PROCESSING
# ============================================================

def process_tsv(input_tsv, output_tsv):
    with open(input_tsv, newline="", encoding="utf-8") as infile, \
         open(output_tsv, "w", newline="", encoding="utf-8") as outfile:

        reader = csv.reader(infile, delimiter="\t")
        writer = csv.writer(outfile, delimiter="\t")

        for row in reader:
            if len(row) < 3:
                continue

            student_id, name, original_text = row[0], row[1], row[2]

            print(f"\n=== {student_id} / {name} ===")

            try:
                main_idea = extract_main_idea(original_text)
                generated = generate_valid_text(original_text, main_idea)
            except Exception as e:
                print(f"⚠️ Failed: {e}")
                continue

            writer.writerow([student_id, name, generated])

    print(f"\n✅ Output → {output_tsv}")


# ============================================================
# PIPELINE ENTRY (IMPORTANT)
# ============================================================

def generate_llm_tsv(input_tsv, output_tsv):
    """
    Called from your pipeline (hard_pipeline)
    """
    print("\n🚀 Generating LLM TSV")
    print(f"Input: {input_tsv}")
    print(f"Output: {output_tsv}")

    process_tsv(input_tsv, output_tsv)

    return output_tsv


# ============================================================
# CLI SUPPORT
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) == 3:
        generate_llm_tsv(sys.argv[1], sys.argv[2])
    else:
        generate_llm_tsv(DEFAULT_INPUT_TSV, DEFAULT_OUTPUT_TSV)