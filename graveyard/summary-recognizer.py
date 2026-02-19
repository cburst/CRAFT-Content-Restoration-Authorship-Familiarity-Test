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
PDF_DIR = "PDFs-summary-recognizer"
ANSWER_KEY = "answer_key_summary_recognizer.tsv"

# 🔑 Put your DeepSeek API key here
DEEPSEEK_API_KEY = "YOUR-API-KEY-HERE"
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"
MAX_DEEPSEEK_RETRIES = 5

# Words that should not be altered or omitted in the summaries
AVOID_WORDS = {
    "hufs", "macalister", "minerva", "students", "learners",
    "student", "learner", "Hankuk", "University", "Foreign", "Studies"
}

NUM_SUMMARIES = 3   # total summaries per student (1 correct + 2 distractors)
SUMMARY_MIN_WORDS = 70
SUMMARY_MAX_WORDS = 130

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

def sanitize_filename(name):
    forbidden = r'\/:*?"<>|'
    return "".join(c for c in name if c not in forbidden).strip() or "student"

def count_words(text):
    return len(tokenize_words(text))


# ===============================
# DeepSeek: generate 1 correct + 2 slightly wrong summaries
# ===============================

def generate_summaries_with_deepseek(essay_text,
                                     avoid_words=AVOID_WORDS,
                                     model=DEEPSEEK_MODEL,
                                     max_retries=MAX_DEEPSEEK_RETRIES):
    """
    Generates:
      - correct summary
      - two distractor summaries (each with 1–3 subtle errors)

    Returns:
        (correct, [d1, d2])
    Or:
        (None, None) if DeepSeek fails after retries.
    """

    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DeepSeek API key not set.")

    avoid_list = sorted(avoid_words)
    avoid_str = ", ".join(avoid_list) if avoid_list else "none"

    # ----- SYSTEM PROMPT (triple-quoted, no escaped quotes!) -----
    system_prompt = f"""
You are designing a multiple-choice reading test for advanced EFL students.

Given a student's essay, produce three summaries:

1) A fully accurate summary.
2) A mostly accurate summary with 1–3 SMALL incorrect details.
3) Another mostly accurate summary with 1–3 SMALL incorrect details.

RULES:

- The following words must NOT be changed or paraphrased:
  {avoid_str}

- All summaries must:
  • Be between 70 and 130 words.
  • Use neutral academic tone.
  • Match B2-level L2-writing style.

- Distractors must contain subtle factual inaccuracies 
  (number change, reversed detail, swapped order, minor cause/effect shift)
  but must remain coherent and plausible.

- Output MUST be valid JSON ONLY, with keys:

{{
  "correct": "...",
  "distractor1": "...",
  "distractor2": "..."
}}

Do NOT include comments, markdown fences, or explanations.
"""

    # ----- USER PROMPT -----
    user_prompt = f"""
Here is the student's essay:

{essay_text}

Now output ONLY a JSON object:

{{
  "correct": "...",
  "distractor1": "...",
  "distractor2": "..."
}}

Nothing else.
"""

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
        "max_tokens": 800,
        "temperature": 0.7,
        "stream": False,
    }

    # ----- RETRIES -----
    attempt = 0
    while attempt < max_retries:
        try:
            print("\n----------------------------")
            print(f"Calling DeepSeek for summaries (attempt {attempt+1})")

            resp = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=90)

            if resp.status_code >= 400:
                print(f"⚠️ DeepSeek HTTP {resp.status_code}")
                print(resp.text)
                raise requests.HTTPError(resp.status_code)

            job = resp.json()

            # Extract content
            raw = None
            if "choices" in job and job["choices"]:
                choice = job["choices"][0]
                msg = choice.get("message", {})
                raw = msg.get("content") or choice.get("text")

            if raw is None:
                raw = str(job)

            print("DeepSeek raw:")
            print(raw)
            print("----------------------------")

            cleaned = raw.strip()
            cleaned = re.sub(r"^```[a-zA-Z]*", "", cleaned).strip()
            cleaned = re.sub(r"```$", "", cleaned).strip()

            # Parse JSON
            data = json.loads(cleaned)

            correct = data.get("correct", "").strip()
            d1 = data.get("distractor1", "").strip()
            d2 = data.get("distractor2", "").strip()

            if not correct or not d1 or not d2:
                raise ValueError("Missing one or more JSON fields.")

            print("Accepted summaries.")
            return correct, [d1, d2]

        except Exception as e:
            print(f"⚠️ Error generating summaries: {e}")
            attempt += 1
            time.sleep(1.0 * attempt)

    print("❌ DeepSeek failed after all retries.")
    return None, None


# ===============================
# PDF generation
# ===============================

def generate_pdf_for_student(student_id, name, summaries,
                             correct_index, out_dir=PDF_DIR):
    """
    summaries: list of 3 summaries (already shuffled)
    correct_index: 0-based index of the accurate summary in 'summaries'
    """
    os.makedirs(out_dir, exist_ok=True)
    safe_name = sanitize_filename(name)
    pdf_path = os.path.join(out_dir, f"{safe_name}.pdf")

    esc_name = html.escape(name)
    esc_number = html.escape(student_id)

    html_parts = [
        "<html><head><meta charset='utf-8'><style>",
        "@page { margin: 1.5cm; size: A4; }",
        "body { font-family: Arial, sans-serif; font-size: 12pt; line-height: 1.4; }",
        ".header { font-weight: bold; margin-bottom: 0.5em; }",
        ".text { white-space: pre-wrap; margin: 0.5em 0; }",
        "</style></head><body>",
        f"<div class='header'>Name: {esc_name}<br>Student Number: {esc_number}<br>Summary Recognizer</div>",
        "<div class='text'>Below are three summaries of your essay.</div>",
        "<div class='text'>Exactly ONE summary is fully accurate. The other two contain small incorrect details.</div>",
        "<div class='text'>Choose the summary that best represents your original text.</div>",
        "<br>",
    ]

    # Number and display each summary
    for i, summ in enumerate(summaries, start=1):
        esc_summ = html.escape(summ)
        html_parts.append(
            f"<div class='text'><b>Summary {i}.</b><br>{esc_summ}</div><br>"
        )

    html_parts.append("</body></html>")
    html_doc = "\n".join(html_parts)

    HTML(string=html_doc).write_pdf(pdf_path)
    print(f"📝 Summary-test PDF created: {pdf_path}")


# ===============================
# Main processing
# ===============================

def process_tsv_for_summary_triplets(input_tsv=INPUT_TSV,
                                     pdf_dir=PDF_DIR,
                                     answer_key_tsv=ANSWER_KEY):
    """
    Input TSV: student_id, name, text

    For each student:
      - Ask DeepSeek for 1 correct + 2 subtly incorrect summaries.
      - Randomly order the three summaries.
      - Generate a PDF listing the three summaries.
      - Append answer-key row with correct index and all three summaries.
    """
    with open(answer_key_tsv, "w", newline="", encoding="utf-8") as keyfile:
        keywriter = csv.writer(keyfile, delimiter="\t")
        keywriter.writerow([
            "student_id",
            "name",
            "correct_summary_number",  # 1-based index in shown order
            "summary_1",
            "summary_2",
            "summary_3",
            "original_text"
        ])

        with open(input_tsv, newline="", encoding="utf-8") as infile:
            reader = csv.reader(infile, delimiter="\t")

            for row in reader:
                if len(row) < 3:
                    continue
                student_id, name, text = row[0], row[1], row[2]

                print(f"\n=== Processing {student_id} / {name} ===")
                correct_summary, distractors = generate_summaries_with_deepseek(text, AVOID_WORDS)

                # Build list and shuffle
                all_summaries = [correct_summary] + distractors
                if len(all_summaries) != 3:
                    print("⚠️ Unexpected number of summaries; skipping student.")
                    continue

                indices = list(range(3))
                random.shuffle(indices)
                shuffled_summaries = [all_summaries[i] for i in indices]

                # Identify where the correct summary ended up
                correct_index = indices.index(0)  # because original index 0 was the correct one
                correct_number = correct_index + 1  # 1-based for answer key

                # Generate PDF
                generate_pdf_for_student(
                    student_id,
                    name,
                    shuffled_summaries,
                    correct_index,
                    out_dir=pdf_dir
                )

                keywriter.writerow([
                    student_id,
                    name,
                    correct_number,
                    shuffled_summaries[0],
                    shuffled_summaries[1],
                    shuffled_summaries[2],
                    text
                ])
                keyfile.flush()

                print(f"✅ Finished {student_id} / {name}")

    print(f"\n🎯 Done. Answer key saved to {answer_key_tsv}")


# ===============================
# MAIN
# ===============================

if __name__ == "__main__":
    random.seed()  # or set a fixed seed for reproducibility
    process_tsv_for_summary_triplets()