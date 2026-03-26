#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ====================================================
# IMPORTS & GLOBAL CONFIG
# ====================================================

import csv
import os
import re
import random
import time
import html
import json
import requests
import math
from collections import Counter
import statistics
from weasyprint import HTML
import nltk
from nltk.stem import SnowballStemmer
from nltk.tokenize import sent_tokenize
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Ensure NLTK punkt tokenizer
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# -----------------------------
# FILE PATHS & CONSTANTS
# -----------------------------
INPUT_TSV = "students.tsv"   # student_id, name, text
PDF_DIR = "PDFs-hybrid-synonym-intruders"
ANSWER_KEY = "answer_key_hybrid_synonym_intruders.tsv"
FREQ_FILE = "wiki_freq.txt"  # "word count" per line

NUM_INTRUDERS = 4            # one per quarter
INTRUDER_SIMILARITY_BASE = 0.6	# Global intruder similarity base threshold
NUM_WORDS_TO_REPLACE = 5     # target number of synonym replacements
NUM_CANDIDATE_OBSCURE = 20   # how many rare words to consider for synonym replacement
MIN_QUARTER_SIZE = 2          # minimum sentences per segment for intruder generation
MAX_PARAGRAPH_RETRIES = 4     # max placement attempts before fallback
MAX_INTRUDER_DETECTION_RATE = 0.25  # max acceptable intruder detectability
MIN_DISPERSION_SCORE = 0.5  # minimum irregularity required for acceptance

# Use environment variable for safety
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

OPENAI_URL = "https://api.openai.com/v1/responses"
OPENAI_MODEL = "gpt-4.1"
OPENAI_MAX_RETRIES = 4

AVOID_WORDS = {
    "hufs", "macalister", "minerva", "students", "learners",
    "student", "learner", "Hankuk", "University", "Foreign", "Studies"
}

STOPWORDS = {
    "the","a","an","and","or","but","if","than","then","therefore","so","because",
    "of","to","in","on","for","at","by","from","with","as","about","into","through",
    "after","over","between","out","against","during","without","before","under","around",
    "among","is","am","are","was","were","be","been","being","have","has","had","do","does",
    "did","can","could","will","would","shall","should","may","might","must","i","you",
    "he","she","it","we","they","me","him","her","us","them","my","your","his","their",
    "our","its","this","that","these","those","there","here","up","down","very","also",
    "just","only","not","no","yes","than","such","many","much","few","several","some",
    "any","all","each","every","both","either","neither","one","two","three","four",
    "five","first","second","third"
}

stemmer = SnowballStemmer("english")

# ====================================================
# GENERAL UTILITIES
# ====================================================

def split_into_sentences(text):
    """Split text into sentences using NLTK sent_tokenize and strip whitespace."""
    if not text:
        return []
    return [s.strip() for s in sent_tokenize(str(text)) if s.strip()]


def tokenize_words_lower(text):
    """Return list of lowercased word tokens (A–Z and apostrophe)."""
    return re.findall(r"[A-Za-z']+", str(text).lower())


def sanitize_filename(name):
    """Remove forbidden characters for file names."""
    forbidden = r'\/:*?"<>|'
    safe = "".join(c for c in name if c not in forbidden).strip()
    return safe or "student"


def normalize_sentence(sent):
    """Normalize sentence to a simple lowercased token string."""
    return " ".join(tokenize_words_lower(sent)).strip()


def load_frequency_ranks(freq_file):
    """
    Load frequency ranks from a file with 'word count' per line.
    Lower rank = more frequent. Unseen words get very large rank.
    """
    freq_ranks = {}
    try:
        with open(freq_file, encoding="utf-8") as f:
            rank = 1
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                word = parts[0].lower()
                if word not in freq_ranks:
                    freq_ranks[word] = rank
                    rank += 1
    except FileNotFoundError:
        print(f"⚠️ Frequency file {freq_file} not found — all words treated equally.")
    return freq_ranks


def levenshtein(a, b):
    """Compute Levenshtein edit distance between strings a and b."""
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]

    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j

    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + cost
            )
    return dp[m][n]

def build_unified_paragraph(sentences):
    """
    Convert a list of sentences into one continuous paragraph with
    bracketed 2-digit labels: [ 01 ], [ 02 ], ...
    All sentences appear in one unified paragraph with single spaces between them.
    """
    parts = []
    for i, sent in enumerate(sentences, start=1):
        label = f"[ {i:02d} ]"
        parts.append(f"{label} {sent}")
    return " ".join(parts)

def same_order_of_magnitude(r1, r2, multiplier_cap=5):
    if r1 <= 0 or r2 <= 0:
        return False

    m1 = int(math.log10(r1))
    m2 = int(math.log10(r2))

    # ±1 order-of-magnitude check
    if abs(m1 - m2) > 1:
        return False

    # Linear multiplier cap (prevents extreme rarity inflation)
    if r2 > r1 * multiplier_cap:
        return False

    return True
    
def levenshtein_similarity(a, b):
    """
    Normalized Levenshtein similarity in [0,1].
    1.0 = identical, 0.0 = completely different
    """
    if not a or not b:
        return 0.0
    dist = levenshtein(a, b)
    return 1.0 - dist / max(len(a), len(b))

def synonym_rejection_score(original_rank, candidate_rank):
    """
    Lower is better.
    Score = order-of-magnitude distance between original and candidate.
    """
    original_rank = max(1, int(original_rank))
    candidate_rank = max(1, int(candidate_rank))

    orig_mag = int(math.log10(original_rank))
    cand_mag = int(math.log10(candidate_rank))

    return abs(cand_mag - orig_mag)

def extract_surface_phrases(sentence):
    """
    Extract surface spans, but return ONLY content-bearing phrases:
      - first word (only if not a stopword)
      - start → first comma
      - internal comma clauses
      - last comma → end
    Any phrase must contain ≥2 NON-stopwords to be kept.
    """
    sent = sentence.strip().lower()
    if not sent:
        return []

    phrases = []

    def content_phrase(text):
        tokens = tokenize_words_lower(text)
        content = [t for t in tokens if t not in STOPWORDS]
        if len(content) >= 2:
            return " ".join(content)
        return None

    # ---- First word (rarely useful, but keep if content-bearing) ----
    m = re.match(r"\s*([a-z']+)", sent)
    if m:
        fw = m.group(1)
        if fw not in STOPWORDS:
            phrases.append(fw)

    # ---- Comma-based spans ----
    if "," in sent:
        parts = [p.strip() for p in sent.split(",")]

        # start → first comma
        p = content_phrase(parts[0])
        if p:
            phrases.append(p)

        # internal clauses
        for mid in parts[1:-1]:
            p = content_phrase(mid)
            if p:
                phrases.append(p)

        # last comma → end
        p = content_phrase(parts[-1])
        if p:
            phrases.append(p)

    return phrases


def intruder_too_similar(candidate, existing_intruders, threshold=INTRUDER_SIMILARITY_BASE):
    """
    Reject candidate if a content-bearing surface phrase
    is too similar to a previous sentence.

    Threshold is globally tunable.
    """

    cand_phrases = extract_surface_phrases(candidate)

    if not cand_phrases:
        return False

    for prev in existing_intruders:
        prev_phrases = extract_surface_phrases(prev)

        for c in cand_phrases:
            for p in prev_phrases:
                sim = levenshtein_similarity(c, p)
                if sim >= threshold:
                    print(
                        f"⚠️ Intruder rejected — content similarity {sim:.2f}\n"
                        f"    '{c}' ~ '{p}'"
                    )
                    return True

    return False

def llm_chat(system_prompt, user_prompt, temperature=0.7, max_tokens=200):
    """
    Robust OpenAI Responses API wrapper with:
      - safe error printing
      - exponential backoff
      - minimum token floor (>=16 required)
      - API timing instrumentation
    """

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": OPENAI_MODEL,
        "input": f"{system_prompt}\n\n{user_prompt}",
        "temperature": temperature,
        "max_output_tokens": max(16, max_tokens),
    }

    last_error = None

    for attempt in range(1, OPENAI_MAX_RETRIES + 1):
        resp = None
        try:
            print(f"\n⏱ LLM API attempt {attempt} start: {time.strftime('%H:%M:%S')}")
            api_start = time.perf_counter()

            resp = requests.post(
                OPENAI_URL,
                headers=headers,
                json=payload,
                timeout=60
            )

            elapsed = time.perf_counter() - api_start
            print(
                f"⏱ LLM API attempt {attempt} end: {time.strftime('%H:%M:%S')} "
                f"(elapsed {elapsed:.2f}s)"
            )

            # Retry only on transient errors
            if resp.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(
                    f"Transient API error {resp.status_code}: {resp.text}",
                    response=resp
                )

            resp.raise_for_status()
            data = resp.json()

            # Shortcut field
            if data.get("output_text"):
                return data["output_text"].strip()

            # Structured extraction
            for item in data.get("output", []):
                for block in item.get("content", []):
                    if block.get("type") == "output_text":
                        return block.get("text", "").strip()

            raise RuntimeError(f"No text returned from OpenAI response: {data}")

        except Exception as e:
            elapsed = time.perf_counter() - api_start
            print(
                f"⏱ LLM API attempt {attempt} failed after {elapsed:.2f}s"
            )

            last_error = e
            wait = 2 ** attempt

            print(f"⚠️ LLM API attempt {attempt} failed: {e}")
            if resp is not None:
                try:
                    print(resp.text)
                except:
                    pass

            if attempt < OPENAI_MAX_RETRIES:
                print(f"⏳ Retrying in {wait} seconds...")
                time.sleep(wait)

    raise RuntimeError(
        f"LLM call failed after {OPENAI_MAX_RETRIES} retries: {last_error}"
    )

def api_place_intruders(original_sentences, intruder_sentences):
    """
    Stable index-based intruder placement.

    The LLM returns ONLY insertion indices.
    We perform deterministic insertion locally.

    Guarantees:
        - No sentence loss
        - No duplication
        - Original order preserved
        - Exact final length
    """

    n_original = len(original_sentences)
    n_intruders = len(intruder_sentences)

    # Label originals O1..On
    numbered_original = "\n".join(
        f"O{i+1}: {s}"
        for i, s in enumerate(original_sentences)
    )

    # Label intruders I1..Ik
    numbered_intruders = "\n".join(
        f"I{i+1}: {s}"
        for i, s in enumerate(intruder_sentences)
    )

    system_prompt = (
        "You are selecting insertion points for intruder sentences.\n\n"
        "Return insertion positions using exactly this format:\n"
        "I1 -> after O3\n"
        "I2 -> after O7\n\n"
        "Goal:\n"
        "Place the intruders so that test-takers cannot guess them based on regular spacing.\n\n"
        "Spacing rules:\n"
        "- Avoid equal or evenly spaced gaps\n"
        "- Prefer highly unequal distances between insertions\n"
        "- Either cluster some intruders close together OR spread them very far apart\n"
        "- The distances between insertions should not form a simple pattern\n\n"
        "Return ONLY the mapping lines. Do NOT explain."
    )

    user_prompt = (
        "ORIGINAL SENTENCES:\n"
        f"{numbered_original}\n\n"
        "INTRUDER SENTENCES:\n"
        f"{numbered_intruders}"
    )

    response = llm_chat(
        system_prompt,
        user_prompt,
        temperature=0.8,
        max_tokens=200
    )

    # --------------------------------------------------
    # Parse mapping lines: I1 -> after O3
    # --------------------------------------------------

    pattern = r"I(\d+)\s*->\s*after\s*O(\d+)"
    matches = re.findall(pattern, response)

    if len(matches) != n_intruders:
        raise RuntimeError(
            f"Placement mapping invalid. "
            f"Expected {n_intruders} mappings, got {len(matches)}.\nResponse:\n{response}"
        )

    # Convert to structured list
    placements = []

    used_positions = set()

    for intr_idx_str, orig_idx_str in matches:
        intr_idx = int(intr_idx_str)
        orig_idx = int(orig_idx_str)

        if not (1 <= intr_idx <= n_intruders):
            raise RuntimeError("Invalid intruder index returned by LLM.")

        if not (1 <= orig_idx <= n_original):
            raise RuntimeError("Invalid original index returned by LLM.")

        if orig_idx in used_positions:
            raise RuntimeError("LLM reused same insertion point.")

        used_positions.add(orig_idx)
        placements.append((intr_idx, orig_idx))

    # --------------------------------------------------
    # Perform deterministic insertion
    # --------------------------------------------------

    augmented = list(original_sentences)

    # Sort by original index descending so insertion is stable
    placements.sort(key=lambda x: x[1], reverse=True)

    for intr_idx, orig_idx in placements:
        intruder_text = intruder_sentences[intr_idx - 1]
        augmented.insert(orig_idx, intruder_text)

    # Final structural validation
    expected_length = n_original + n_intruders

    if len(augmented) != expected_length:
        raise RuntimeError(
            f"Post-insertion length mismatch: "
            f"{len(augmented)} vs expected {expected_length}"
        )

    return augmented

def spacing_irregularity_score(intruder_positions, total_sentences):
    """
    Returns a dispersion score in [0,1].

    Combines:
      - entropy of gap distribution
      - coefficient of variation (CV) of gaps
      - clustering reward
      - asymmetry reward
      - penalty for modular patterns
    """

    if not intruder_positions or len(intruder_positions) < 2:
        return 0.0

    positions = sorted(intruder_positions)

    # -------------------------------------------------
    # GAP ANALYSIS
    # -------------------------------------------------
    gaps = [positions[i+1] - positions[i] for i in range(len(positions)-1)]

    if not gaps:
        return 0.0

    mean_gap = sum(gaps) / len(gaps)

    # --- CV ---
    variance = sum((g - mean_gap)**2 for g in gaps) / len(gaps)
    std = variance ** 0.5
    cv = std / mean_gap if mean_gap > 0 else 0.0
    cv_score = min(1.0, cv)

    # --- ENTROPY ---
    from math import log2
    total_gap = sum(gaps)
    probs = [g / total_gap for g in gaps if g > 0]

    entropy = -sum(p * log2(p) for p in probs)
    max_entropy = log2(len(gaps)) if len(gaps) > 1 else 1
    entropy_score = entropy / max_entropy if max_entropy > 0 else 0.0

    # -------------------------------------------------
    # CLUSTER REWARD (small local clusters)
    # -------------------------------------------------
    small_gaps = sum(1 for g in gaps if g <= mean_gap * 0.6)
    cluster_score = small_gaps / len(gaps)

    # -------------------------------------------------
    # ASYMMETRY REWARD
    # -------------------------------------------------
    midpoint = total_sentences / 2
    left = sum(1 for p in positions if p < midpoint)
    right = len(positions) - left
    asymmetry_score = abs(left - right) / len(positions)

    # -------------------------------------------------
    # MODULAR PENALTY (e.g., every 4 sentences)
    # -------------------------------------------------
    mod_penalty = 0.0
    for base in (3, 4, 5):
        remainders = [p % base for p in positions]
        most_common = max(remainders.count(r) for r in set(remainders))
        if most_common / len(positions) > 0.75:
            mod_penalty = 0.5
            break

    # -------------------------------------------------
    # FINAL SCORE
    # -------------------------------------------------
    score = (
        0.30 * entropy_score +
        0.30 * cv_score +
        0.20 * cluster_score +
        0.20 * asymmetry_score
    )

    score -= mod_penalty
    score = max(0.0, min(1.0, score))

    return score

def api_detect_outlier_sentences(hybrid_paragraph):
    """
    Ask the LLM open-ended which sentence numbers stand out.

    Returns:
        list of integers (sentence numbers it flagged)
    """

    system_prompt = (
        "You are a careful academic reader.\n\n"
        "Some sentences in the paragraph below may not fit naturally "
        "with the surrounding discourse.\n\n"
        "List the sentence numbers (e.g., 3, 7, 12) that stand out as "
        "possibly unrelated or inserted.\n"
        "If none stand out, return: NONE\n"
        "Return ONLY a comma-separated list of numbers or NONE."
    )

    user_prompt = hybrid_paragraph

    raw = llm_chat(system_prompt, user_prompt, temperature=0.2, max_tokens=80)

    raw = raw.strip()

    if raw.upper() == "NONE":
        return []

    nums = re.findall(r"\d+", raw)
    return [int(n) for n in nums]

def compute_detection_rate(flagged_numbers, true_intruder_numbers):
    flagged = set(flagged_numbers)
    true_intruders = set(true_intruder_numbers)

    correct_hits = flagged.intersection(true_intruders)

    if not true_intruders:
        return 0.0

    return len(correct_hits) / len(true_intruders)


# ====================================================
# SYNONYM REPLACEMENT LOGIC
# ====================================================

def find_obscure_words(text, freq_ranks, num_candidates=NUM_CANDIDATE_OBSCURE):
    """
    Return up to num_candidates obscure words (rarest first).

    Improvements:
      - Skip likely misspellings (must exist in freq_ranks)
      - Skip ultra-rare tokens (stability filter)
      - Log skipped words for transparency
    """

    tokens = tokenize_words_lower(text)
    counts = Counter(tokens)

    candidates = []
    skipped_misspell = 0
    skipped_ultrarare = 0

    ULTRA_RARE_CAP = 500_000  # adjust if needed

    for w, c in counts.items():

        if len(w) < 4:
            continue

        if w in STOPWORDS or w in AVOID_WORDS:
            continue

        if "'" in w:
            continue  # skip possessives / contractions

        rank = freq_ranks.get(w, None)

        # --- Skip likely misspellings ---
        if rank is None:
            skipped_misspell += 1
            continue

        # --- Skip ultra-rare unstable tokens ---
        if rank > ULTRA_RARE_CAP:
            skipped_ultrarare += 1
            continue

        candidates.append((rank, w))

    # Sort by rarity (highest rank = rarest first)
    candidates.sort(key=lambda x: x[0], reverse=True)

    result = []
    for _, w in candidates:
        if w not in result:
            result.append(w)
        if len(result) >= num_candidates:
            break

    if skipped_misspell > 0:
        print(f"🔎 Skipped {skipped_misspell} likely misspellings.")
    if skipped_ultrarare > 0:
        print(f"🔎 Skipped {skipped_ultrarare} ultra-rare tokens.")

    return result


def find_sentence_and_surface_word(text, word_lower):
    """
    Find the first sentence containing word_lower (case-insensitive),
    and return (sentence, surface_form_as_it_appears).
    """
    sentences = split_into_sentences(text)
    pattern = re.compile(r"\b" + re.escape(word_lower) + r"\b", re.IGNORECASE)

    for sent in sentences:
        m = pattern.search(sent)
        if m:
            return sent, m.group(0)  # surface form in that sentence

    return None, None


def get_synonym_from_llm(surface_word, sentence, all_words_in_text, freq_ranks):
    """
    LLM synonym generator with FULL diagnostic logging.
    Strict first. If all attempts fail, choose best rejected synonym (by closeness of difficulty).
    Preserves: repeat-reject abort logic.
    """

    # ---- skip capitalized originals ----
    if any(c.isupper() for c in surface_word):
        print(f"⚠️ Skipping '{surface_word}' — contains uppercase letters.")
        return None

    surface_lower = surface_word.lower()
    original_rank = freq_ranks.get(surface_lower, 10**9)

    system_prompt = (
        "You are a precise lexical substitution assistant. Given an English word "
        "as it appears inside a sentence, produce exactly one lowercase synonym "
        "that can replace the original word without breaking grammar.\n\n"
        "Requirements:\n"
        "1) lowercase only\n"
        "2) same part of speech and inflection\n"
        "3) must preserve the same argument structure (no new prepositions)\n"
        "4) similar or slightly higher difficulty, but not drastically harder\n"
        "5) output ONLY the word\n"
        "6) if unsure, output #FAIL"
    )

    user_prompt = (
        f"Sentence:\n{sentence}\n\n"
        f"Original word: {surface_word}"
    )

    original_threshold = max(1, int(len(surface_lower) * 0.30))

    # ---- compute allowed magnitude range (±1) ----
    if original_rank <= 0:
        original_rank = 1

    orig_mag = int(math.log10(original_rank))
    min_allowed_mag = orig_mag - 1
    max_allowed_mag = orig_mag + 1

    last_rejected = None

    # Store rejected synonyms ONLY for end-of-loop fallback
    # Each item: (synonym, syn_rank)
    rejected_synonyms = []

    for attempt in range(1, OPENAI_MAX_RETRIES + 1):

        print("\n----------------------------")
        print(f"🔍 LLM synonym attempt {attempt} for '{surface_word}'")
        print("Sentence:", sentence)

        try:
            raw = llm_chat(system_prompt, user_prompt, temperature=0.45, max_tokens=10)
            print("🧠 LLM raw output:", raw)

            candidate = raw.strip("'\"").strip().lower()

            if candidate == "#fail":
                print("⚠️ LLM returned #FAIL — aborting.")
                return None

            if not re.fullmatch(r"[a-z]+", candidate):
                print("❌ Rejected — synonym must be a single unhyphenated word.")
                if candidate == last_rejected:
                    print("⚠️ Same rejected candidate twice in a row — aborting.")
                    return None
                last_rejected = candidate
                continue

            synonym = candidate

            if synonym == surface_lower:
                print("❌ Rejected — identical to original.")
                if synonym == last_rejected:
                    print("⚠️ Same rejected candidate twice in a row — aborting.")
                    return None
                last_rejected = synonym
                continue

            syn_rank = freq_ranks.get(synonym, 10**9)

            print(f"📊 Frequency ranks — original: {original_rank}, candidate: {syn_rank}")

            # ---- ±1 ORDER-OF-MAGNITUDE CHECK ----
            if syn_rank <= 0:
                syn_rank = 1

            syn_mag = int(math.log10(syn_rank))

            if not (min_allowed_mag <= syn_mag <= max_allowed_mag):
                print(
                    f"❌ Rejected — outside ±1 order of magnitude "
                    f"(orig_mag={orig_mag}, cand_mag={syn_mag})"
                )
                rejected_synonyms.append((synonym, syn_rank))
                if synonym == last_rejected:
                    print("⚠️ Same rejected candidate twice in a row — aborting.")
                    return None
                last_rejected = synonym
                continue

            # ---- STEM REJECTION ----
            if stemmer.stem(surface_lower) == stemmer.stem(synonym):
                print("❌ Rejected — same morphological stem.")
                rejected_synonyms.append((synonym, syn_rank))
                if synonym == last_rejected:
                    print("⚠️ Same rejected candidate twice in a row — aborting.")
                    return None
                last_rejected = synonym
                continue

            # ---- LEVENSHTEIN SIMILARITY CHECK ----
            dist_orig = levenshtein(surface_lower, synonym)
            if dist_orig <= original_threshold:
                print(
                    f"❌ Rejected — too similar to original "
                    f"(dist={dist_orig}, threshold={original_threshold})"
                )
                rejected_synonyms.append((synonym, syn_rank))
                if synonym == last_rejected:
                    print("⚠️ Same rejected candidate twice in a row — aborting.")
                    return None
                last_rejected = synonym
                continue

            # ---- COLLISION WITH OTHER WORDS ----
            conflict = False
            for w in all_words_in_text:
                if w == surface_lower:
                    continue
                threshold_other = max(1, int(len(w) * 0.30))
                if levenshtein(w, synonym) <= threshold_other:
                    print(f"❌ Rejected — too similar to existing word '{w}'")
                    conflict = True
                    break

            if conflict:
                rejected_synonyms.append((synonym, syn_rank))
                if synonym == last_rejected:
                    print("⚠️ Same rejected candidate twice in a row — aborting.")
                    return None
                last_rejected = synonym
                continue

            # ---- GRAMMAR REGRESSION CHECK ----
            modified_sentence = re.sub(
                rf"\b{re.escape(surface_word)}\b",
                synonym,
                sentence,
                count=1
            )

            grammar_prompt = (
                "Compare the two sentences below.\n\n"
                "Sentence A (original):\n"
                f"{sentence}\n\n"
                "Sentence B (modified):\n"
                f"{modified_sentence}\n\n"
                "Does sentence B introduce a new grammatical error "
                "that was not already present in sentence A?\n\n"
                "Answer YES or NO only."
            )

            grammar_response = llm_chat(
                system_prompt="You are a strict grammar evaluator.",
                user_prompt=grammar_prompt,
                temperature=0,
                max_tokens=5
            ).strip().upper()

            print("🧠 Grammar comparison:", grammar_response)

            if grammar_response == "YES":
                print("❌ Rejected — introduces new grammatical error.")
                rejected_synonyms.append((synonym, syn_rank))
                if synonym == last_rejected:
                    print("⚠️ Same rejected candidate twice in a row — aborting.")
                    return None
                last_rejected = synonym
                continue

            print(f"✅ Accepted synonym for '{surface_word}': {synonym}")
            return synonym

        except Exception as e:
            print(f"⚠️ LLM error on attempt {attempt}: {e}")

    # ---- If we got here, strict acceptance failed ----
    print(f"⚠️ No suitable synonym found for '{surface_word}' after retries.")

    # -------------------------------------------------
    # Global fallback: choose best rejected synonym
    # -------------------------------------------------
    if rejected_synonyms:

        # Remove duplicates while preserving order
        seen = set()
        unique_rejected = []
        for syn, rank in rejected_synonyms:
            if syn not in seen:
                seen.add(syn)
                unique_rejected.append((syn, rank))

        best_synonym = None
        best_score = float("inf")
        best_rank = None

        for syn, rank in unique_rejected:
            score = synonym_rejection_score(original_rank, rank)

            # Lower score = closer difficulty match
            if score < best_score:
                best_score = score
                best_synonym = syn
                best_rank = rank

        if best_synonym:
            print(
                f"⚠️ Global fallback activated — "
                f"using best rejected synonym: {best_synonym} "
                f"(rank={best_rank}, difficulty_score={best_score})"
            )
            return best_synonym

    return None
    
def get_pos_from_llm(surface_word, sentence):
    """
    Ask LLM for the part of speech of a word as used in the sentence.
    """

    system_prompt = (
        "You are an expert at identifying the part of speech of English words "
        "based on their usage in context. Return ONLY the POS label such as "
        "'noun', 'verb', 'adjective', or 'adverb'."
    )

    user_prompt = (
        f"Sentence:\n{sentence}\n\n"
        f"Target word: {surface_word}"
    )

    try:
        raw = llm_chat(system_prompt, user_prompt, temperature=0.2, max_tokens=16)
        pos_clean = raw.lower().split()[0]
        return pos_clean
    except Exception as e:
        print("⚠️ POS LLM error:", e)
        return "?"

def apply_synonym_case_preserving(text, original_lower, synonym):
    """
    Replace all occurrences of original_lower in text (case-insensitive),
    preserving case pattern of each occurrence.
    """
    pattern = re.compile(r"\b" + re.escape(original_lower) + r"\b", re.IGNORECASE)
    result_parts = []
    last_end = 0

    for m in pattern.finditer(text):
        result_parts.append(text[last_end:m.start()])
        orig = m.group(0)
        if orig.isupper():
            rep = synonym.upper()
        elif orig[0].isupper():
            rep = synonym.capitalize()
        else:
            rep = synonym.lower()
        result_parts.append(rep)
        last_end = m.end()

    result_parts.append(text[last_end:])
    return "".join(result_parts)


def transform_text_with_synonyms(text, freq_ranks):
    """
    1) Find up to NUM_CANDIDATE_OBSCURE rare words.
    2) For each, find a sentence and the surface form of the word.
    3) Ask LLM for a synonym that matches POS + inflection.
    4) Ask LLM for POS of the original word.
    5) Reject synonyms that:
         - are too similar to the original word
         - OR too similar to ANY other word in the text
         - OR contain capitalization
    6) Replace up to NUM_WORDS_TO_REPLACE words in the text.

    Returns:
      modified_text,
      replacements_list = list of (original_surface, synonym, pos_label)
    """

    all_words = set(tokenize_words_lower(text))

    candidate_words = find_obscure_words(
        text, freq_ranks, num_candidates=NUM_CANDIDATE_OBSCURE
    )

    modified_text = text
    replacements = []

    for w_lower in candidate_words:
        if len(replacements) >= NUM_WORDS_TO_REPLACE:
            break

        sentence, surface_word = find_sentence_and_surface_word(modified_text, w_lower)
        if not sentence or not surface_word:
            continue

        # --- (A) SYNONYM GENERATION ---
        synonym = get_synonym_from_llm(surface_word, sentence, all_words, freq_ranks)
        if not synonym:
            continue

        # Ensure the word still exists in text
        pattern = re.compile(r"\b" + re.escape(w_lower) + r"\b", re.IGNORECASE)
        if not pattern.search(modified_text):
            continue

        # --- (B) POS LABELING ---
        pos_label = get_pos_from_llm(surface_word, sentence)
        if not pos_label:
            pos_label = "?"

        # --- (C) APPLY REPLACEMENT ---
        modified_text = apply_synonym_case_preserving(modified_text, w_lower, synonym)

        # Store (original_surface, synonym, part_of_speech)
        replacements.append((surface_word, synonym, pos_label))

    return modified_text, replacements


# ====================================================
# INTRUDER GENERATION LOGIC
# ====================================================

def generate_intruder_sentence(essay_section_sentences, existing_sentences, intruder_index):
    """
    Robust intruder generator with:

      - Progressive relaxation
      - Global similarity threshold (via intruder_too_similar)
      - Strong discourse marker suppression
      - Best-candidate tracking
      - Never crashes
      - Returns best observed candidate if no strict pass
    """

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing — cannot generate intruders.")

    # --------------------------------------------------
    # Section statistics
    # --------------------------------------------------
    lengths, densities = [], []
    for s in essay_section_sentences:
        tokens = tokenize_words_lower(s)
        if not tokens:
            continue
        lengths.append(len(tokens))
        content = [t for t in tokens if t not in STOPWORDS]
        densities.append(len(content) / len(tokens))

    avg_len = max(8, int(sum(lengths) / max(1, len(lengths))))
    avg_density = sum(densities) / max(1, len(densities))

    min_len_base = max(6, int(avg_len * 0.6))
    max_len_base = int(avg_len * 1.4)
    min_density_base = max(0.0, avg_density - 0.20)

    # --------------------------------------------------
    # Prompt
    # --------------------------------------------------
    system_prompt = (
        "You are a careful academic writing assistant.\n\n"
        "Write ONE sentence that could plausibly continue the following section.\n\n"
        "The sentence must:\n"
        "- Add a concrete detail, clarification, or implication\n"
        "- Match academic tone and informational density\n"
        "- Avoid discourse markers (e.g., Furthermore, In conclusion, Overall, Moreover)\n"
        "- Avoid copying wording from the section\n"
        "- Stand alone grammatically\n"
        "Output ONE sentence only."
    )

    user_prompt = "Essay section:\n" + " ".join(essay_section_sentences)

    existing_norms = {normalize_sentence(s) for s in existing_sentences}

    best_candidate = None
    best_score = -1

    MAX_ATTEMPTS = 8

    # --------------------------------------------------
    # Attempt loop with progressive relaxation
    # --------------------------------------------------
    for attempt in range(1, MAX_ATTEMPTS + 1):

        print("\n----------------------------")
        print(f"🔍 Intruder attempt {attempt} for section {intruder_index}")

        # Progressive relaxation factor (0 → 1)
        relax = (attempt - 1) / (MAX_ATTEMPTS - 1)

        # --- Length relaxation ---
        length_slack = int(avg_len * 0.40 * relax)   # up to ±40%
        min_len = max(5, min_len_base - length_slack)
        max_len = max_len_base + length_slack

        # --- Density relaxation ---
        min_density = max(0.35, min_density_base - (0.30 * relax))

        try:
            raw = llm_chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.85,   # slight creativity boost
                max_tokens=220
            )

            candidate = raw.strip("'\"")
            tokens = tokenize_words_lower(candidate)

            if not tokens:
                print("❌ Rejected — empty output.")
                continue

            wc = len(tokens)
            content_wc = sum(1 for t in tokens if t not in STOPWORDS)
            density = content_wc / wc
            norm = normalize_sentence(candidate)

            print(f"📏 Words: {wc} (target {min_len}-{max_len})")
            print(f"📊 Density: {density:.2f} (min {min_density:.2f})")

            # ------------------------------------
            # Similarity check (global threshold)
            # ------------------------------------
            too_similar = intruder_too_similar(candidate, existing_sentences)

            # ------------------------------------
            # Scoring (for fallback selection)
            # ------------------------------------
            length_score = 1 - abs(wc - avg_len) / max(1, avg_len)
            density_score = min(1, density / max(0.0001, avg_density))
            similarity_penalty = 0.5 if too_similar else 0

            score = length_score + density_score - similarity_penalty

            if score > best_score:
                best_score = score
                best_candidate = candidate

            # ------------------------------------
            # Acceptance condition (relaxed)
            # ------------------------------------
            if (
                min_len <= wc <= max_len and
                density >= min_density and
                norm not in existing_norms and
                not too_similar
            ):
                print(f"✅ Accepted intruder (relax level {relax:.2f})")
                return candidate

            print("❌ Rejected — constraints not met.")

        except Exception as e:
            print(f"⚠️ Intruder error: {e}")

    # --------------------------------------------------
    # Final fallback: return best observed candidate
    # --------------------------------------------------
    if best_candidate:
        print("⚠️ Returning best available candidate after relaxation.")
        return best_candidate

    raise RuntimeError(
        f"Failed to generate usable intruder sentence for section {intruder_index}"
    )

def get_sentence_embeddings(sentences):
    """
    Returns a list of embedding vectors for the given sentences.
    Uses OpenAI embeddings endpoint.
    """

    if not sentences:
        return []

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "text-embedding-3-small",  # fast + cheap
        "input": sentences
    }

    last_error = None

    for attempt in range(1, OPENAI_MAX_RETRIES + 1):

        print(f"\n⏱ Embedding call start (attempt {attempt}): {time.strftime('%H:%M:%S')}")
        api_start = time.perf_counter()

        try:
            r = requests.post(
                "https://api.openai.com/v1/embeddings",
                headers=headers,
                json=payload,
                timeout=60
            )

            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(
                    f"Transient embedding error {r.status_code}",
                    response=r
                )

            r.raise_for_status()

            elapsed = time.perf_counter() - api_start
            print(
                f"⏱ Embedding call end: {time.strftime('%H:%M:%S')} "
                f"(elapsed {elapsed:.2f}s)"
            )

            data = r.json()

            # Preserve order
            embeddings = [item["embedding"] for item in data["data"]]

            return embeddings

        except Exception as e:
            elapsed = time.perf_counter() - api_start
            print(f"⚠️ Embedding attempt {attempt} failed after {elapsed:.2f}s: {e}")

            last_error = e

            if attempt < OPENAI_MAX_RETRIES:
                wait = 2 ** attempt
                print(f"⏳ Retrying in {wait}s...")
                time.sleep(wait)

    raise RuntimeError(
        f"Embedding call failed after {OPENAI_MAX_RETRIES} attempts: {last_error}"
    )


def compute_quarters(sentences):

    n = len(sentences)

    if n < MIN_QUARTER_SIZE:
        return []

    max_possible = n // MIN_QUARTER_SIZE
    num_segments = min(NUM_INTRUDERS, max_possible)

    if num_segments <= 1:
        return [(0, n)]

    try:
        embeddings = np.array(get_sentence_embeddings(sentences))

        if len(embeddings) != n:
            raise RuntimeError("Embedding length mismatch.")

        similarities = []
        for i in range(n - 1):
            sim = cosine_similarity(
                embeddings[i].reshape(1, -1),
                embeddings[i + 1].reshape(1, -1)
            )[0][0]
            similarities.append(sim)

        split_points = sorted(
            range(len(similarities)),
            key=lambda i: similarities[i]
        )[: num_segments - 1]

        split_points = sorted(split_points)

    except Exception as e:
        print("⚠️ Embeddings failed — reverting to equal segmentation.")
        # ---- SAFE FALLBACK ----
        base_size = n // num_segments
        split_points = [base_size * (i + 1) - 1 for i in range(num_segments - 1)]

    # ---- Build segments ----
    segments = []
    start = 0
    for sp in split_points:
        end = sp + 1
        segments.append((start, end))
        start = end
    segments.append((start, n))

    return segments


def insert_intruders_into_sentences(sentences):

    if not sentences:
        return sentences, [], []

    original_sentences = list(sentences)
    
    quarters = compute_quarters(original_sentences)

    if not quarters:
        print("⚠️ Text too short for intruders — skipping insertion.")
        return original_sentences, [], []

    intruder_texts = []
    intruder_index = 1
    existing_for_generation = list(original_sentences)

    for (start, end) in quarters:
        section = original_sentences[start:end]

        intruder = generate_intruder_sentence(
            essay_section_sentences=section,
            existing_sentences=existing_for_generation,
            intruder_index=intruder_index
        )

        intruder_texts.append(intruder)
        existing_for_generation.append(intruder)
        intruder_index += 1

    if not intruder_texts:
        return original_sentences, [], []

    # -------------------------------------------------
    # SCORING TRACKERS
    # -------------------------------------------------
    best_candidate = None
    best_combined_score = -1
    best_detection_rate = 1.0
    best_dispersion_score = 0.0

    for attempt in range(1, MAX_PARAGRAPH_RETRIES + 1):

        print("\n======================================")
        print(f"🎲 Paragraph placement attempt {attempt}")
        print("======================================")

        try:
            augmented = api_place_intruders(
                original_sentences,
                intruder_texts
            )
        except Exception as e:
            print(f"⚠️ Placement API failed: {e}")
            continue

        pdf_intruder_numbers = get_pdf_intruder_numbers_from_augmented(
            augmented,
            intruder_texts
        )

        dispersion_score = spacing_irregularity_score(
            pdf_intruder_numbers,
            total_sentences=len(augmented)
        )

        hybrid_paragraph = build_unified_paragraph(augmented)

        flagged_numbers = api_detect_outlier_sentences(hybrid_paragraph)

        detection_rate = compute_detection_rate(
            flagged_numbers,
            pdf_intruder_numbers
        )

        combined_score = (
            (1 - detection_rate) * 0.70 +
            dispersion_score * 0.30
        )

        print(f"📊 Detection rate: {detection_rate:.2f}")
        print(f"📐 Dispersion score: {dispersion_score:.3f}")
        print(f"⭐ Combined score: {combined_score:.3f}")
        print(f"🔎 Flagged: {flagged_numbers}")
        print(f"🎯 True intruders: {pdf_intruder_numbers}")

        # ----------------------------
        # Track best candidate
        # ----------------------------
        if combined_score > best_combined_score:
            best_combined_score = combined_score
            best_candidate = (augmented, pdf_intruder_numbers)
            best_detection_rate = detection_rate
            best_dispersion_score = dispersion_score

        # ----------------------------
        # Early acceptance
        # ----------------------------
        if (
            detection_rate <= MAX_INTRUDER_DETECTION_RATE and
            dispersion_score >= MIN_DISPERSION_SCORE
        ):
            print("✅ Paragraph accepted (meets detection + dispersion threshold).")
            return augmented, pdf_intruder_numbers, intruder_texts

    # -------------------------------------------------
    # Fallback to best candidate
    # -------------------------------------------------
    if best_candidate:
        print(
            f"\n🏆 Selecting best candidate "
            f"(detect={best_detection_rate:.2f}, "
            f"dispersion={best_dispersion_score:.3f}, "
            f"combined={best_combined_score:.3f})"
        )
        return best_candidate[0], best_candidate[1], intruder_texts

    print("⚠️ All placement attempts failed — using random fallback.")

    augmented = list(original_sentences)
    available_positions = list(range(1, len(augmented)))

    if len(available_positions) >= len(intruder_texts):
        insertion_points = random.sample(available_positions, len(intruder_texts))
        insertion_points.sort(reverse=True)

        for idx, intr in zip(insertion_points, intruder_texts):
            augmented.insert(idx, intr)

        pdf_intruder_numbers = get_pdf_intruder_numbers_from_augmented(
            augmented,
            intruder_texts
        )

        return augmented, pdf_intruder_numbers, intruder_texts

    return original_sentences, [], []
    
# ====================================================
# PDF GENERATION (COMBINED TEST)
# ====================================================

def generate_pdf(student_id, name, sentences, replacements, pdf_dir=PDF_DIR):
    """
    Generate a single PDF that contains:
      - Instructions for both tests at the top
      - ONE unified paragraph with numbered sentences
      - A 3-row table:
            1) part of speech
            2) replacements  (blank cells)
            3) originals     (first-letter hints)
    """

    os.makedirs(pdf_dir, exist_ok=True)
    safe_name = sanitize_filename(name)
    pdf_path = os.path.join(pdf_dir, f"{safe_name}.pdf")

    esc_name = html.escape(name)
    esc_number = html.escape(student_id)

    # Build unified paragraph
    unified_paragraph = build_unified_paragraph(sentences)
    esc_paragraph = html.escape(unified_paragraph)

    html_parts = [
        "<html>",
        "<head>",
        "<meta charset='utf-8'>",
        "<style>",
        "@page { margin: 1.5cm; size: A4; }",
        "body { font-family: Arial, sans-serif; font-size: 13pt; line-height: 1.4; margin: 0; padding: 0; }",
        ".header { font-weight: bold; margin-bottom: 0.5em; }",
        ".instructions { white-space: normal; margin: 0.3em 0; text-indent: 0; }",
        ".paragraph { white-space: pre-wrap; margin: 0.6em 0; text-indent: 2em; }",
        "table.syn-table { border-collapse: collapse; margin-top: 1.2em; }",
        "table.syn-table td { border: 1px solid #000; padding: 0.2em 0.3em; min-width: 2.5cm; }",
        "table.syn-table td.label-cell { font-weight: bold; white-space: nowrap; }",
        "</style>",
        "</head>",
        "<body>",
        f"<div class='header'>Name: {esc_name}<br>Student Number: {esc_number}</div>",
        "<div class='header'>Sentence Intruders & Synonym Replacements</div>",
        "<div class='instructions'>",
        "<b>Extra sentences have been added. Circle the added sentence numbers. Lose one point for incorrect guesses. <br>",
        f"Five words have been replaced. Find the replacements and provide the originals. No penalty for guessing.</b>",
        "</div>",
        f"<div class='paragraph'>{esc_paragraph}</div>",
        "<table class='syn-table'>",
    ]

    # ======================================================
    # 🔵 ROW 1 — PART OF SPEECH HINTS
    # ======================================================
    html_parts.append("<tr>")
    html_parts.append("<td class='label-cell'>part of speech</td>")
    for idx in range(NUM_WORDS_TO_REPLACE):
        if idx < len(replacements):
            orig, syn, pos_label = replacements[idx]
            html_parts.append(f"<td>{html.escape(pos_label)}</td>")
        else:
            html_parts.append("<td>&nbsp;</td>")
    html_parts.append("</tr>")

    # ======================================================
    # 🔵 ROW 2 — REPLACEMENTS (blank for students)
    # ======================================================
    html_parts.append("<tr>")
    html_parts.append("<td class='label-cell'>replacements</td>")
    for _ in range(NUM_WORDS_TO_REPLACE):
        html_parts.append("<td>&nbsp;</td>")
    html_parts.append("</tr>")

    # ======================================================
    # 🔵 ROW 3 — ORIGINAL WORDS (first-letter hints)
    # ======================================================
    html_parts.append("<tr>")
    html_parts.append("<td class='label-cell'>originals</td>")
    for idx in range(NUM_WORDS_TO_REPLACE):
        if idx < len(replacements):
            orig, syn, pos_label = replacements[idx]
            first_letter = next((ch.lower() for ch in orig if ch.isalpha()), "")
            html_parts.append(f"<td>{html.escape(first_letter)}</td>")
        else:
            html_parts.append("<td>&nbsp;</td>")
    html_parts.append("</tr>")

    html_parts.append("</table>")
    html_parts.append("</body></html>")

    html_doc = "\n".join(html_parts)
    HTML(string=html_doc).write_pdf(pdf_path)
    print(f"📄 PDF created: {pdf_path}")

from rapidfuzz import fuzz

def extract_bracket_sentences(text):
    """
    Extract [ nn ] sentences from final hybrid paragraph.
    Returns list of (num_as_int, sentence_text)
    """
    text = re.sub(r"\s+", " ", text)
    pattern = r"\[\s*([0-9OIl]+)\s*\]\s*(.*?)(?=\[\s*[0-9OIl]+\s*\]|$)"

    out = []
    for m in re.finditer(pattern, text):
        raw = m.group(1)
        cleaned = (
            raw.replace("O","0")
               .replace("o","0")
               .replace("I","1")
               .replace("l","1")
        )
        try:
            num = int(cleaned)
        except ValueError:
            continue
        sent = m.group(2).strip()
        out.append((num, sent))
    return out


def get_pdf_intruder_numbers_from_augmented(augmented_sentences, intruder_texts):
    """
    Compute PDF intruder numbers from the FINAL sentence list.

    For each intruder_text, find its index in augmented_sentences and
    return 1-based indices (these correspond to [ nn ] labels).

    Handles possible duplicates by not reusing the same index twice.
    """
    pdf_nums = []
    used_indices = set()

    for intr in intruder_texts:
        found_idx = None
        for i, s in enumerate(augmented_sentences):
            if i in used_indices:
                continue
            if s.strip() == intr.strip():
                found_idx = i
                break
        if found_idx is not None:
            used_indices.add(found_idx)
            pdf_nums.append(found_idx + 1)  # 1-based for [ nn ]
    return pdf_nums


def detect_intruders_by_fuzz(original_text, hybrid_text, threshold=0.60):
    """
    Similarity-based intruder detection over ALL labeled sentences.

    Returns:
        fuzz_intruder_numbers (list[int])
        fuzz_intruder_sents   (list[str])
        fuzz_intruder_scores  (list[float])
    """
    orig_sents = split_into_sentences(original_text)
    hyb_sents = extract_bracket_sentences(hybrid_text)

    fuzz_nums = []
    fuzz_texts = []
    fuzz_scores = []

    for num, sent in hyb_sents:
        best = 0.0
        for o in orig_sents:
            score = fuzz.ratio(sent.lower(), o.lower()) / 100
            if score > best:
                best = score

        if best < threshold:
            fuzz_nums.append(num)
            fuzz_texts.append(sent)
            fuzz_scores.append(round(best, 3))

    return fuzz_nums, fuzz_texts, fuzz_scores


def process_tsv(input_tsv, output_tsv):
    """
    Final hybrid processor with CORRECT intruder numbering.

    Answer key columns:

        student_id
        name
        original_text
        hybrid_text
        pdf_intruder_numbers      (from actual positions of intruder_texts)
        fuzz_intruder_numbers     (from RapidFuzz < 0.60 on hybrid text)
        fuzz_intruder_scores      (best similarity for those fuzz intruders)
        intruder_sentences        (the inserted intruder texts)
        replacement_words         (synonyms)
        original_words            (pre-replacement)
        pos_labels                (LLM POS)
    """

    freq_ranks = load_frequency_ranks(FREQ_FILE)

    with open(output_tsv, "w", newline="", encoding="utf-8") as keyfile:
        writer = csv.writer(keyfile, delimiter="\t")

        # ---------- HEADER ----------
        writer.writerow([
            "student_id",
            "name",
            "original_text",
            "hybrid_text",
            "pdf_intruder_numbers",
            "fuzz_intruder_numbers",
            "fuzz_intruder_scores",
            "intruder_sentences",
            "replacement_words",
            "original_words",
            "pos_labels",
        ])

        with open(input_tsv, newline="", encoding="utf-8") as infile:
            reader = csv.reader(infile, delimiter="\t")

            for row in reader:
                if len(row) < 3:
                    continue

                student_id, name, text = row[0], row[1], row[2]
                print(f"\n=== Processing {student_id} / {name} ===")

                # 1. original sentences
                orig_sentences = split_into_sentences(text)
                if not orig_sentences:
                    print("⚠️ No sentences found. Skipping.")
                    continue

                # 2. synonym transformation
                synonym_modified_text, replacements = transform_text_with_synonyms(
                    text, freq_ranks
                )
                modified_original_sents = split_into_sentences(synonym_modified_text)

                # 3. insert intruders
                augmented_sents, intruder_positions, intruder_texts = \
                    insert_intruders_into_sentences(modified_original_sents)

                # ✅ CORRECT pdf intruder numbers: based on FINAL augmented_sents
                pdf_intruder_numbers = get_pdf_intruder_numbers_from_augmented(
                    augmented_sents,
                    intruder_texts
                )

                # 4. build unified hybrid paragraph with [ nn ]
                hybrid_paragraph = build_unified_paragraph(augmented_sents)

                # 5. RapidFuzz-based intruder detection (independent checksum)
                fuzz_nums, fuzz_sents, fuzz_scores = detect_intruders_by_fuzz(
                    text,
                    hybrid_paragraph,
                    threshold=0.60
                )

                # 6. generate PDF
                generate_pdf(
                    student_id=student_id,
                    name=name,
                    sentences=augmented_sents,
                    replacements=replacements,
                    pdf_dir=PDF_DIR
                )

                # 7. replacement metadata
                original_words = [orig for (orig, syn, pos) in replacements]
                replacement_words = [syn for (orig, syn, pos) in replacements]
                pos_labels = [pos for (orig, syn, pos) in replacements]

                # 8. write answer key row
                writer.writerow([
                    student_id,
                    name,
                    text,
                    hybrid_paragraph,
                    ",".join(str(n) for n in pdf_intruder_numbers),
                    ",".join(str(n) for n in fuzz_nums),
                    ",".join(str(s) for s in fuzz_scores),
                    " || ".join(intruder_texts),
                    ",".join(replacement_words),
                    ",".join(original_words),
                    ",".join(pos_labels),
                ])

    print(f"\n🎯 Done. Answer key saved to: {output_tsv}")


# ====================================================
# MAIN
# ====================================================

if __name__ == "__main__":
    random.seed()
    process_tsv(INPUT_TSV, ANSWER_KEY)