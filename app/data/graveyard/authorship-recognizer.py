#!/usr/bin/env python3
import csv
import re
import os
import random
import time
import html
import json
import requests
from weasyprint import HTML

# NLTK sentence tokenizer
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

# 🔑 Hardcode your DeepSeek API key here
DEEPSEEK_API_KEY = "YOUR-API-KEY-HERE"

# DeepSeek chat API endpoint & model
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

# ===============================
# 1. Sentence utilities
# ===============================

def split_into_sentences(text):
    if not text:
        return []
    sentences = sent_tokenize(str(text))
    return [s.strip() for s in sentences if s.strip()]

def tokenize_words(text):
    """
    Lowercase and split into word tokens consisting of letters/apostrophes.
    This means capitalization differences are ignored in token space.
    """
    return re.findall(r"[A-Za-z']+", str(text).lower())

def pick_longest_sentences(text, max_sentences=5, min_words=8):
    sentences = split_into_sentences(text)
    scored = []
    for s in sentences:
        words = tokenize_words(s)
        wc = len(words)
        if wc >= min_words:
            scored.append((wc, s))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored[:max_sentences]]

# ===============================
# 2. Casing helpers (original-based capitalization)
# ===============================

def _build_casing_map_from_original(real_sentence):
    """
    Build a mapping from lowercase word -> canonical casing based on the original sentence.

    Rules:
    - If the first word of the sentence is repeated, use the capitalization of the
      SECOND appearance as the canonical form for that word.
      Otherwise, use the first appearance.
    - For any other word that appears multiple times, use the capitalization of the
      FIRST appearance.
    """
    # Surface-form words from original (with their original casing)
    surface_tokens = re.findall(r"[A-Za-z']+", str(real_sentence))
    if not surface_tokens:
        return {}

    occ = {}  # lower -> list of surface forms in order
    for surf in surface_tokens:
        lower = surf.lower()
        occ.setdefault(lower, []).append(surf)

    casing_map = {}

    first_lower = surface_tokens[0].lower()
    for lower, forms in occ.items():
        if lower == first_lower:
            # First word: use 2nd occurrence if it exists, else 1st
            if len(forms) >= 2:
                canonical = forms[1]
            else:
                canonical = forms[0]
        else:
            # Other words: use 1st occurrence
            canonical = forms[0]
        casing_map[lower] = canonical

    return casing_map

def _apply_original_casing_and_period(real_sentence, decoy_sentence):
    """
    Take a decoy sentence (already in plain text) and:
    - Apply capitalization rules based on the original sentence:
      * Any word repeated from the original uses the canonical capitalization
        from _build_casing_map_from_original.
      * If it doesn't appear in the original:
         - First word: capitalize first letter.
         - Others: lowercase.
    - Ensure the sentence ends with a period.
    """
    casing_map = _build_casing_map_from_original(real_sentence)
    decoy_lower_tokens = tokenize_words(decoy_sentence)

    if not decoy_lower_tokens:
        result = decoy_sentence.strip()
        if result and not result.endswith("."):
            result += "."
        return result

    new_tokens = []
    for i, lw in enumerate(decoy_lower_tokens):
        if lw in casing_map:
            # Use canonical casing from original
            new_tokens.append(casing_map[lw])
        else:
            # Word not in original sentence
            if i == 0:
                # First word: capitalize
                new_tokens.append(lw.capitalize())
            else:
                # Others: lowercase
                new_tokens.append(lw.lower())

    result = " ".join(new_tokens).strip()

    # Ensure final period
    if result and not result.endswith("."):
        result += "."

    return result

# ===============================
# 3. Helpers for decoys (first & last word fixed)
# ===============================

def _force_same_ends(real_sentence, decoy_sentence, n_end_words=1):
    """
    Enforce that decoy has the same first/last word as real_sentence
    in lowercase token space (n_end_words=1 → first & last).
    Then apply original-based capitalization and final period.
    """
    real_tokens = tokenize_words(real_sentence)
    decoy_tokens = tokenize_words(decoy_sentence)

    if len(real_tokens) < 2 * n_end_words + 1:
        # Too short to enforce sensible middle; just re-cased/period-fixed decoy
        return _apply_original_casing_and_period(real_sentence, decoy_sentence)

    # Ensure decoy has some middle; if not, copy original middle
    if len(decoy_tokens) < 2 * n_end_words + 1:
        middle = real_tokens[n_end_words:-n_end_words]
        decoy_tokens = real_tokens[:n_end_words] + middle + real_tokens[-n_end_words:]

    # Force same ends (in lowercase token space)
    decoy_tokens[:n_end_words] = real_tokens[:n_end_words]
    decoy_tokens[-n_end_words:] = real_tokens[-n_end_words:]

    # Reconstruct lowercase sentence from tokens
    lowered_sentence = " ".join(decoy_tokens)
    # Now apply original-based casing and period
    return _apply_original_casing_and_period(real_sentence, lowered_sentence)

def _simple_middle_variant(real_sentence, flavor_word):
    """
    Cheap local variant: insert a 'flavor' word into the middle,
    then enforce same first & last word and apply casing/period.
    """
    real_tokens = tokenize_words(real_sentence)
    if len(real_tokens) < 3:  # too short for meaningful middle
        return _apply_original_casing_and_period(real_sentence, f"{real_sentence} {flavor_word}")

    # Middle is everything except first and last token
    middle = real_tokens[1:-1]
    if not middle:
        return _apply_original_casing_and_period(real_sentence, f"{real_sentence} {flavor_word}")

    mid = middle[:]
    pos = max(0, len(mid) // 2)
    mid.insert(pos, flavor_word.lower())
    candidate_tokens = real_tokens[:1] + mid + real_tokens[-1:]
    lowered = " ".join(candidate_tokens)
    return _force_same_ends(real_sentence, lowered, n_end_words=1)

def _normalize_for_comparison(sentence):
    """Normalize sentence for equality checks (lowercase tokens joined)."""
    return " ".join(tokenize_words(sentence)).strip()

def _middle_tokens(tokens, n_end_words=1):
    """
    Extract the 'middle' of a sentence (everything except first and last word).
    If too short, just return the whole token list.
    """
    if len(tokens) > 2 * n_end_words:
        return tokens[n_end_words:-n_end_words]
    return tokens[:]

# ===============================
# 4. DeepSeek decoy generation
# ===============================

def generate_decoys_with_deepseek(real_sentence, n_decoys=2,
                                 model=DEEPSEEK_MODEL, max_retries=5):
    """
    Generate n_decoys that:
      - keep same first and last word as real_sentence (in token space)
      - have a middle that is NOT exactly identical (as a word sequence) to the original middle
      - are distinct from each other and from the original
      - follow original-based capitalization rules and end with a period

    Capitalization-only changes do NOT count as changes, because we compare in lowercased token space.
    """
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DeepSeek API key not found! Add it at the top of the script.")

    system_prompt = (
        "You are a careful writing assistant. Given an original sentence, "
        "you will create decoy sentences for a quiz.\n\n"
        "Each decoy sentence must:\n"
        "1) Express essentially the SAME meaning as the original.\n"
        "2) Make SMALL changes in the middle of the sentence (even 1–2 words is okay).\n"
        "3) KEEP THE FIRST WORD EXACTLY THE SAME as the original.\n"
        "4) KEEP THE LAST WORD EXACTLY THE SAME as the original.\n"
        "5) NOT be identical to the original sentence.\n"
        "6) Be a single sentence, 8–25 words."
    )

    user_prompt = (
        f'Original sentence:\n"{real_sentence}"\n\n'
        f"Produce {n_decoys} decoy sentences satisfying the rules above. "
        "Return them as a JSON array of strings only."
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

    real_tokens = tokenize_words(real_sentence)
    real_mid_tokens = _middle_tokens(real_tokens, n_end_words=1)
    norm_real = _normalize_for_comparison(real_sentence)

    attempt = 0
    while attempt <= max_retries:
        try:
            print("\n----------------------------")
            print("Original sentence:")
            print(real_sentence)
            print("Calling DeepSeek... (attempt", attempt + 1, ")")

            resp = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=30)

            if resp.status_code >= 400:
                print(f"⚠️ DeepSeek HTTP {resp.status_code}")
                print("Response body:")
                print(resp.text)
                raise requests.HTTPError(f"HTTP {resp.status_code}: {resp.text}")

            job = resp.json()

            # Extract text
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

            # Try JSON array
            raw_decoys = []
            try:
                parsed = json.loads(text.strip())
                if isinstance(parsed, list):
                    raw_decoys = [str(s).strip() for s in parsed][:n_decoys]
            except Exception:
                lines = [ln.strip("-• \t") for ln in text.splitlines() if ln.strip()]
                raw_decoys = [ln for ln in lines if len(ln.split()) >= 5][:n_decoys]

            final = []
            seen_norm = set()

            for d in raw_decoys:
                if not d:
                    continue
                # Force same first/last word and apply casing/period
                forced = _force_same_ends(real_sentence, d, n_end_words=1)
                norm_forced = _normalize_for_comparison(forced)
                if not norm_forced or norm_forced == norm_real:
                    continue

                # Compare middles as *words*, ignoring capitalization
                cand_tokens = tokenize_words(forced)
                cand_mid = _middle_tokens(cand_tokens, n_end_words=1)
                if cand_mid == real_mid_tokens:
                    # Middle is exactly the same sequence of words; reject
                    continue

                if norm_forced in seen_norm:
                    continue
                final.append(forced.strip())
                seen_norm.add(norm_forced)
                if len(final) >= n_decoys:
                    break

            # If we have enough good decoys, return them
            if len(final) >= n_decoys:
                print("Processed decoys:")
                for i, dec in enumerate(final, start=1):
                    print(f"  Decoy {i}: {dec}")
                print("----------------------------")
                return final[:n_decoys]

            # Try local augmentation before retrying
            print("Not enough distinct decoys from DeepSeek; filling with local variants.")
            flavor_words = ["also", "indeed", "in particular", "for instance", "in fact"]
            idx_flavor = 0
            while len(final) < n_decoys and idx_flavor < len(flavor_words) * 2:
                flavor = flavor_words[idx_flavor % len(flavor_words)]
                idx_flavor += 1
                candidate = _simple_middle_variant(real_sentence, flavor)
                norm_c = _normalize_for_comparison(candidate)
                if not norm_c or norm_c == norm_real or norm_c in seen_norm:
                    continue
                cand_tokens = tokenize_words(candidate)
                cand_mid = _middle_tokens(cand_tokens, n_end_words=1)
                if cand_mid == real_mid_tokens:
                    # middle words identical, ignore
                    continue
                final.append(candidate.strip())
                seen_norm.add(norm_c)

            if len(final) >= n_decoys:
                print("Processed decoys (DeepSeek + local):")
                for i, dec in enumerate(final, start=1):
                    print(f"  Decoy {i}: {dec}")
                print("----------------------------")
                return final[:n_decoys]

            # If we reach here, this attempt didn't yield enough; try again
            attempt += 1
            print(f"⚠️ Not enough usable decoys on attempt {attempt}. Retrying...")
            time.sleep(1.0 * attempt)

        except Exception as e:
            print(f"⚠️ DeepSeek error: {e}")
            attempt += 1
            time.sleep(1.0 * attempt)

    # After all retries, last-resort local variants only
    print("Using local fallback decoys after maximum retries.")
    final = []
    seen_norm = set()
    flavor_words = ["also", "indeed", "in particular", "for instance", "in fact"]
    for flavor in flavor_words:
        candidate = _simple_middle_variant(real_sentence, flavor)
        norm_c = _normalize_for_comparison(candidate)
        if not norm_c or norm_c == norm_real or norm_c in seen_norm:
            continue
        cand_tokens = tokenize_words(candidate)
        cand_mid = _middle_tokens(cand_tokens, n_end_words=1)
        if cand_mid == real_mid_tokens:
            continue
        final.append(candidate.strip())
        seen_norm.add(norm_c)
        if len(final) >= n_decoys:
            break

    if not final:
        # Absolute last fallback: just return two copies (not ideal, but avoids crash)
        final = [real_sentence, real_sentence]

    print("Processed fallback decoys:")
    for i, dec in enumerate(final, start=1):
        print(f"  Decoy {i}: {dec}")
    print("----------------------------")
    return final[:n_decoys]

# ===============================
# 5. Build items per student
# ===============================

def build_source_id_items(text, n_items=5):
    items = []
    candidates = pick_longest_sentences(text, max_sentences=n_items)
    for real_sentence in candidates:
        if len(items) >= n_items:
            break
        decoys = generate_decoys_with_deepseek(real_sentence, n_decoys=2)
        options = [real_sentence] + decoys
        random.shuffle(options)
        correct_index = options.index(real_sentence)
        items.append((
            "Which sentence actually appears in your essay?",
            options,
            correct_index,
            real_sentence
        ))
    return items

# ===============================
# 6. PDF generation
# ===============================

def sanitize_filename(name):
    return "".join(c for c in name if c not in r'\/:*?"<>|').strip() or "student"

def generate_pdf_for_student(student_id, name, items, out_dir="pdfs_source_id"):
    os.makedirs(out_dir, exist_ok=True)
    safe_name = sanitize_filename(name)
    pdf_path = os.path.join(out_dir, f"{safe_name}.pdf")

    esc_name = html.escape(name)
    esc_number = html.escape(student_id)

    html_parts = [
        "<html><head><meta charset='utf-8'><style>",
        "@page {",
        "  size: A4;",
        "  margin-top: 3.5cm;",   # generous top space
        "  margin-right: 2cm;",
        "  margin-bottom: 2cm;",
        "  margin-left: 2cm;",
        "  @top-left { content: element(page-header); }",
        "}",
        "body {",
        "  font-family: Arial, sans-serif;",
        "  font-size: 14pt;",
        "  line-height: 1.5;",
        "  margin: 0;",
        "  padding: 0;",
        "}",
        # Generously sized, bold header matching your other scripts
        ".page-header {",
        "  position: running(page-header);",
        "  font-size: 14pt;",
        "  font-weight: bold;",
        "  line-height: 1.5;",
        "  text-align: left;",
        "  margin: 0;",
        "  padding: 0;",
        "}",
        ".title {",
        "  font-weight: bold;",
        "  margin-top: 1em;",
        "  margin-bottom: 0.8em;",
        "}",
        ".text {",
        "  white-space: pre-wrap;",
        "  margin: 0.3em 0;",
        "}",
        "</style></head><body>",

        # Repeated header (flush-left, bold, full-size)
        f"<div class='page-header'>"
        f"Name: {esc_name}<br>"
        f"Student Number: {esc_number}<br>"
        f"Authorship Recognizer"
        f"</div>",
    ]

    labels = ["A", "B", "C"]

    for q_num, (q_text, options, _correct_index, _real_sentence) in enumerate(items, start=1):
        html_parts.append(f"<div class='text'><b>Q{q_num}.</b> {html.escape(q_text)}</div>")
        for idx, opt in enumerate(options):
            html_parts.append(
                f"<div class='text' style='margin-left:1em'>{labels[idx]}. {html.escape(opt)}</div>"
            )
        html_parts.append("<br>")

    html_parts.append("</body></html>")
    html_doc = "\n".join(html_parts)

    HTML(string=html_doc).write_pdf(pdf_path)
    print(f"📝 PDF created: {pdf_path}")
    
# ===============================
# 7. Process TSV sequentially
# ===============================

def process_tsv_for_source_id(input_tsv, pdf_dir="PDFs-authorship-recognizer", answer_key_tsv="answer_key_authorship_recognizer.tsv"):
    labels = ["A", "B", "C"]
    with open(answer_key_tsv, "w", newline="", encoding="utf-8") as keyfile:
        keywriter = csv.writer(keyfile, delimiter="\t")
        keywriter.writerow(["student_id", "name", "question_number", "correct_option", "real_sentence"])

        with open(input_tsv, newline="", encoding="utf-8") as infile:
            reader = csv.reader(infile, delimiter="\t")

            for row in reader:
                if len(row) < 3:
                    continue
                student_id, name, text = row[0], row[1], row[2]

                print(f"\n=== Processing {student_id} / {name} ===")
                items = build_source_id_items(text, n_items=5)
                if not items:
                    print(f"⚠️ No suitable sentences for {student_id}; skipping.")
                    continue

                generate_pdf_for_student(student_id, name, items, out_dir=pdf_dir)

                for q_num, (_q_text, _options, correct_index, real_sentence) in enumerate(items, start=1):
                    keywriter.writerow([student_id, name, q_num, labels[correct_index], real_sentence])

                keyfile.flush()
                print(f"✅ Finished {student_id} / {name}")

    print(f"\n🎯 All done (or stopped). Answer key saved to {answer_key_tsv}")

# ===============================
# MAIN
# ===============================

if __name__ == "__main__":
    INPUT_TSV = "students.tsv"   # student_id, name, text
    PDF_DIR = "PDFs"
    ANSWER_KEY = "answer_key_authorship_recognizer.tsv"
    process_tsv_for_source_id(INPUT_TSV, pdf_dir=PDF_DIR, answer_key_tsv=ANSWER_KEY)