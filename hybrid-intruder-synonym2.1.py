#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import re
import html
from weasyprint import HTML
import nltk
import spacy
nlp = spacy.load("en_core_web_sm")
from sentence_transformers import SentenceTransformer, util
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
import requests
import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
from rapidfuzz import fuzz
import random
from rapidfuzz.distance import Levenshtein

# -----------------------------
# CONFIG
# -----------------------------

INPUT_TSV = "students.tsv"   # student_id, name, text
PDF_DIR = "PDFs-hybrid-synonym-intruders"
ANSWER_KEY = "answer_key_hybrid_synonym_intruders.tsv"

try:
    nltk.data.find("tokenizers/punkt")
except:
    nltk.download("punkt")

STOPWORDS = {
    "the","a","an","and","or","but","if","than","then","therefore","so","because",
    "of","to","in","on","for","at","by","from","with","as","about","into","through",
    "after","over","between","out","against","during","without","before","under","around",
    "among","is","am","are","was","were","be","been","being","have","has","had","do","does",
    "did","can","could","will","would","shall","should","may","might","must","i","you",
    "he","she","it","we","they","me","him","her","us","them","my","your","his","their",
    "our","its","this","that","these","those","there","here","up","down","very","also",
    "just","only","not","no","yes","such","many","much","few","several","some",
    "any","all","each","every","both","either","neither","one","two","three","four",
    "five","first","second","third"
}

MAX_PARAGRAPH_RETRIES = 4
MAX_INTRUDER_DETECTION_RATE = 0.34
MIN_DISPERSION_SCORE = 0.20

# -----------------------------
# TUNABLE THRESHOLDS
# -----------------------------

SEMANTIC_THRESHOLD = 0.80
PINC_O_MAX = 0.20
PINC_SLASH_MIN = 0.30
PINC_SLASH_MAX = 0.60
INTRUDER_PINC_MIN = 0.80

# -----------------------------
# UTILITIES
# -----------------------------

def tokenize_words_lower(text):
    return re.findall(r"[A-Za-z']+", str(text).lower())

def build_unified_paragraph(sentences):
    parts = []
    for i, sent in enumerate(sentences, start=1):
        parts.append(f"[ {i:02d} ] {sent}")
    return " ".join(parts)

def normalize_sentence(sent):
    return " ".join(tokenize_words_lower(sent)).strip()

def get_pdf_intruder_numbers_from_augmented(augmented_sentences, intruder_texts):
    nums = []
    used = set()

    for intr in intruder_texts:
        found = None
        for i, s in enumerate(augmented_sentences):
            if i in used:
                continue
            if s.strip() == intr.strip():
                found = i
                break
        if found is not None:
            used.add(found)
            nums.append(found + 1)

    return nums

def split_into_sentences(text):
    return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]

def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "", name)

def is_divisible(sentence):
    doc = nlp(sentence)

    clause_heads = {
        "ROOT",
        "conj",
        "ccomp",
        "xcomp",
        "advcl",
        "relcl",
    }
    
    count = 0
    for token in doc:
        if token.dep_ in clause_heads:
            count += 1

    # divisible = more than just main clause
    return count > 1

def get_max_sentences(input_tsv):
    max_len = 0
    with open(input_tsv, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 3:
                continue
            sentences = split_into_sentences(row[2])
            max_len = max(max_len, len(sentences))
    return max_len

def encode_sentence(sentence):
    return sbert_model.encode(sentence, convert_to_tensor=True)

def semantic_similarity_from_emb(orig_emb, candidate_sentence):
    cand_emb = sbert_model.encode(candidate_sentence, convert_to_tensor=True)
    score = util.cos_sim(orig_emb, cand_emb).item()
    return score

def build_rewrite_prompt():
    return (
        "You are a precise academic paraphrasing assistant.\n\n"

        "Rewrite the sentence while preserving meaning EXACTLY.\n\n"

        "STRICT REQUIREMENTS:\n"
        "- Preserve ALL details, examples, and meaning\n"
        "- Do NOT remove or add information\n"
        "- Do NOT replace specific nouns with pronouns (it, they, this, etc.)\n"
        "- Maintain full semantic equivalence\n\n"

        "You MUST produce THREE versions with CLEARLY DIFFERENT WORD OVERLAP.\n\n"

        "LOW (≈80–90% of words the same):\n"
        "- Keep most original words\n"
        "- Only minor synonym changes\n"
        "- Sentence should look almost identical to the original\n\n"

        "MEDIUM (≈40–60% of words the same):\n"
        "- Replace ABOUT HALF of the words\n"
        "- Use clear synonym substitution\n"
        "- Some reordering is allowed\n"
        "- Sentence should look noticeably different\n\n"

        "HIGH (≈0–20% of words the same):\n"
        "- Replace MOST OR ALL content words\n"
        "- Avoid reusing original wording wherever possible\n"
        "- Use different phrasing for nearly every part of the sentence\n"
        "- Sentence should look VERY different at the word level\n\n"

        "IMPORTANT:\n"
        "- The difference must be VISIBLE at the word level\n"
        "- HIGH must NOT reuse most of the original words\n"
        "- Do NOT produce similar sentences\n\n"

        "Output EXACTLY three lines:\n"
        "LOW: ...\n"
        "MEDIUM: ...\n"
        "HIGH: ..."
    )

def clean_line(text):
    return text.strip().strip('"').strip()

def parse_rewrite_output(raw):
    out = {"low": None, "medium": None, "high": None}

    lines = [l.strip() for l in raw.splitlines() if l.strip()]

    # strict parsing
    for line in lines:
        lower = line.lower()

        if lower.startswith("low:"):
            out["low"] = clean_line(line[4:])
        elif lower.startswith("medium:"):
            out["medium"] = clean_line(line[7:])
        elif lower.startswith("high:"):
            out["high"] = clean_line(line[5:])

    # fallback ONLY if clearly 3 standalone sentences
    if not all(out.values()) and len(lines) == 3:
        out["low"] = clean_line(lines[0])
        out["medium"] = clean_line(lines[1])
        out["high"] = clean_line(lines[2])

    return out

def generate_rewrite_set(sentence, max_attempts=3):

    system_prompt = build_rewrite_prompt()

    for attempt in range(max_attempts):

        try:
            raw = llm_chat(
                system_prompt,
                sentence,
                temperature=0.8,
                max_tokens=300
            )

            parsed = parse_rewrite_output(raw)

            # full success
            if all(parsed.values()):
                return parsed

            # partial success (acceptable)
            if any(parsed.values()):
                return parsed

        except Exception as e:
            print(f"⚠️ Rewrite set error (attempt {attempt+1}): {e}")

    # fallback
    return {
        "low": None,
        "medium": None,
        "high": None
    }

def llm_chat(system_prompt, user_prompt, temperature=0.6, max_tokens=100):
    url = "https://api.openai.com/v1/responses"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    prompt = f"{system_prompt}\n\n{user_prompt}"

    payload = {
        "model": "gpt-4.1-mini",
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt}
                ]
            }
        ],
        "temperature": temperature,
        "max_output_tokens": max(16, max_tokens),
    }

    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()

    text = ""
    for item in data.get("output", []):
        for block in item.get("content", []):
            if block.get("type") == "output_text":
                text += block.get("text", "")

    return text.strip()

def get_pos_sequence(sentence):
    doc = nlp(sentence)
    return " ".join(token.pos_ for token in doc)

def compute_edit_distance(original, candidate):
    return Levenshtein.normalized_similarity(original, candidate)

def compute_ngram_overlap(original_sentence, candidate_sentence, n=2):
    tokens1 = [t.text.lower() for t in nlp(original_sentence) if t.is_alpha]
    tokens2 = [t.text.lower() for t in nlp(candidate_sentence) if t.is_alpha]

    ngrams1 = get_ngrams(tokens1, n)
    ngrams2 = get_ngrams(tokens2, n)

    if not ngrams1:
        return 0.0

    overlap = len(ngrams1 & ngrams2)
    return overlap / len(ngrams1)

def compute_token_index(original_sentence, candidate_sentence):
    """
    Current token index:
        - RapidFuzz ratio only
    Later you can extend this into a composite.
    Returns a float in [0, 1].
    """
    return fuzz.ratio(original_sentence, candidate_sentence) / 100.0

def compute_sts_score(orig_emb, candidate_sentence):
    try:
        cand_emb = sbert_model.encode(candidate_sentence, convert_to_tensor=True)
        return float(util.cos_sim(orig_emb, cand_emb).item())
    except:
        return 0.0

def compute_dep_overlap(original_sentence, candidate_sentence):
    doc1 = nlp(original_sentence)
    doc2 = nlp(candidate_sentence)

    deps1 = [token.dep_ for token in doc1]
    deps2 = [token.dep_ for token in doc2]

    if not deps1:
        return 0.0

    overlap = len(set(deps1) & set(deps2))
    return overlap / len(set(deps1))

def compute_tree_edit_proxy(original_sentence, candidate_sentence):
    doc1 = nlp(original_sentence)
    doc2 = nlp(candidate_sentence)

    edges1 = {
        (token.dep_, token.head.pos_, token.pos_)
        for token in doc1
    }
    edges2 = {
        (token.dep_, token.head.pos_, token.pos_)
        for token in doc2
    }

    if not edges1:
        return 0.0

    overlap = len(edges1 & edges2)
    return overlap / len(edges1)

def compute_structural_index(original_sentence, candidate_sentence):
    """
    Current structural index:
        - POS-sequence similarity only
    Later you can extend this into a composite.
    Returns a float in [0, 1].
    """
    orig_pos = get_pos_sequence(original_sentence)
    cand_pos = get_pos_sequence(candidate_sentence)
    return fuzz.ratio(orig_pos, cand_pos) / 100.0

def get_ngrams(tokens, n):

    return set(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

def compute_pinc(original_sentence, candidate_sentence, max_n=4):
    tokens1 = [t.text.lower() for t in nlp(original_sentence) if t.is_alpha]
    tokens2 = [t.text.lower() for t in nlp(candidate_sentence) if t.is_alpha]
    if not tokens2:
        return 0.0

    scores = []
    for n in range(1, max_n + 1):
        ngrams1 = get_ngrams(tokens1, n)
        ngrams2 = get_ngrams(tokens2, n)
        if not ngrams2:
            continue

        overlap = ngrams1 & ngrams2
        non_overlap_ratio = 1 - (len(overlap) / len(ngrams2))
        scores.append(non_overlap_ratio)

    if not scores:
        return 0.0

    return sum(scores) / len(scores)

# -----------------------------
# SENTENCE GENERATION
# -----------------------------
def choose_intruder_positions_llm(base_sentences, num_intruders):
    """
    Use LLM to pick insertion points (1..n-1), returned as integers.
    """

    n = len(base_sentences)

    numbered = "\n".join(
        f"S{i+1}: {s}" for i, s in enumerate(base_sentences)
    )

    system_prompt = (
        "You are selecting insertion points for additional sentences in a paragraph.\n\n"
        "Choose positions where a new sentence could naturally fit.\n\n"
        "Return EXACTLY {k} positions as comma-separated numbers.\n"
        "Positions are BETWEEN sentences (1 means after S1, before S2).\n\n"
        "Rules:\n"
        "- spread them across the paragraph\n"
        "- avoid clustering\n"
        "- avoid always choosing the beginning or end\n"
        "- do not repeat positions\n"
        "- only return numbers\n"
    ).replace("{k}", str(num_intruders))

    raw = llm_chat(system_prompt, numbered, temperature=0.3, max_tokens=50)

    nums = [int(x) for x in re.findall(r"\d+", raw)]

    # sanitize
    nums = [n for n in nums if 1 <= n <= (len(base_sentences) - 1)]
    nums = list(dict.fromkeys(nums))  # dedupe

    # fallback if bad output
    if len(nums) != num_intruders:
        print("⚠️ LLM positions fallback → random spread")
        return choose_intruder_positions(base_sentences, num_intruders)

    return sorted(nums)


def generate_slash_modifications(original_sentences, get_emb):
    """
    Build candidate buckets for each sentence:
      - O (very low PINC)
      - SLASH (controlled paraphrase: PINC in target band)

    No X generation here anymore.
    """

    sentence_candidates = []

    for idx, s in enumerate(original_sentences):

        print(f"\n🔁 Original [{idx+1}]: {s}")

        candidate_dict = generate_rewrite_set(s)
        orig_emb = get_emb(s)

        buckets = {"O": [], "SLASH": []}

        for level, cand in candidate_dict.items():

            if not cand:
                continue

            semantic_index = semantic_similarity_from_emb(orig_emb, cand)
            pinc = compute_pinc(s, cand, max_n=3)

            print(f"\n➡️ {level.upper()}: {cand}")
            print(f"📊 🧠 {semantic_index:.3f} | 🧬 PINC: {pinc:.3f}")

            if semantic_index < SEMANTIC_THRESHOLD:
                continue

            # ---- STRICT O
            if pinc < PINC_O_MAX:
                buckets["O"].append((cand, pinc, semantic_index, level))

            # ---- STRICT SLASH BAND
            elif PINC_SLASH_MIN <= pinc <= PINC_SLASH_MAX:
                buckets["SLASH"].append((cand, pinc, semantic_index, level))

            # ---- RELAXED FALLBACK (prevents empty buckets)
            elif 0.275 <= pinc <= 0.70:
                buckets["SLASH"].append((cand, pinc, semantic_index, level))

        sentence_candidates.append(buckets)

    return sentence_candidates

def select_slash_sentences(original_sentences, sentence_candidates, target_slash):
    total = len(original_sentences)

    modified_sentences = list(original_sentences)
    selected = []
    used_indices = set()

    # -----------------------------
    # BUILD STRICT + RELAXED POOLS
    # -----------------------------
    strict_pool = []
    relaxed_pool = []

    for i, buckets in enumerate(sentence_candidates):
        for cand, pinc, sem, level in buckets["SLASH"]:

            if PINC_SLASH_MIN <= pinc <= PINC_SLASH_MAX:
                strict_pool.append({
                    "idx": i,
                    "sentence": cand,
                    "pinc": pinc
                })

            elif 0.25 <= pinc <= 0.70:
                relaxed_pool.append({
                    "idx": i,
                    "sentence": cand,
                    "pinc": pinc
                })

    print(f"📦 Strict SLASH candidates: {len(strict_pool)}")
    print(f"📦 Relaxed SLASH candidates: {len(relaxed_pool)}")

    # -----------------------------
    # PRIMARY: MAX–MIN DISTRIBUTION
    # -----------------------------
    pool = list(strict_pool)

    if not pool:
        print("⚠️ No strict SLASH candidates — using relaxed pool")
        pool = list(relaxed_pool)

    if pool:

        # seed with center-most candidate (≈0.45)
        seed = min(pool, key=lambda x: abs(x["pinc"] - 0.45))
        selected.append(seed)
        used_indices.add(seed["idx"])
        pool.remove(seed)

        # greedy max–min spread
        while len(selected) < target_slash and pool:

            best = None
            best_dist = -1

            for cand in pool:

                if cand["idx"] in used_indices:
                    continue

                dist = min(abs(cand["pinc"] - s["pinc"]) for s in selected)

                if dist > best_dist:
                    best = cand
                    best_dist = dist

            if not best:
                break

            selected.append(best)
            used_indices.add(best["idx"])
            pool.remove(best)

    # -----------------------------
    # FALLBACK STAGE 2:
    # per-sentence best (~0.45)
    # -----------------------------
    if len(selected) < target_slash:

        print("⚠️ Fallback Stage 2: per-sentence selection")

        per_sentence_best = []

        for i, buckets in enumerate(sentence_candidates):

            if i in used_indices:
                continue

            if not buckets["SLASH"]:
                continue

            best = min(
                buckets["SLASH"],
                key=lambda x: abs(x[1] - 0.45)
            )

            per_sentence_best.append({
                "idx": i,
                "sentence": best[0],
                "pinc": best[1]
            })

        for item in per_sentence_best:
            if len(selected) >= target_slash:
                break

            selected.append(item)
            used_indices.add(item["idx"])

    # -----------------------------
    # FALLBACK STAGE 3:
    # controlled PINC relaxation
    # -----------------------------
    if len(selected) < target_slash:

        print("⚠️ Fallback Stage 3: relaxed PINC band")

        relaxed = []

        for i, buckets in enumerate(sentence_candidates):

            if i in used_indices:
                continue

            for cand, pinc, sem, level in (buckets["SLASH"] + buckets["O"]):
                if 0.25 <= pinc <= 0.65:
                    relaxed.append({
                        "idx": i,
                        "sentence": cand,
                        "pinc": pinc
                    })

        relaxed.sort(key=lambda x: abs(x["pinc"] - 0.45))

        for item in relaxed:
            if len(selected) >= target_slash:
                break

            if item["idx"] in used_indices:
                continue

            selected.append(item)
            used_indices.add(item["idx"])

    # -----------------------------
    # APPLY SELECTION
    # -----------------------------
    selected_counts = {"O": 0, "SLASH": 0, "X": 0}

    for item in selected:
        idx = item["idx"]
        modified_sentences[idx] = item["sentence"]
        selected_counts["SLASH"] += 1

        print(
            f"🟡 SELECTED SLASH [{idx+1}] "
            f"(PINC={item['pinc']:.3f}) | {selected_counts}"
        )

    print("📊 Final SLASH PINCs:", [round(s["pinc"], 3) for s in selected])

    # -----------------------------
    # REMAINING → O
    # -----------------------------
    for i in range(total):
        if modified_sentences[i] == original_sentences[i]:
            selected_counts["O"] += 1
            print(f"🟢 SELECTED O [{i+1}] | {selected_counts}")

    return modified_sentences, selected_counts

def generate_intruders_step(original_sentences, slash_sentences, num_intruders):
    """
    Generate intruders using LOCAL context (position-aware),
    but final placement is handled separately.
    """

    if not slash_sentences:
        return [], []

    # -----------------------------
    # STEP 1: choose positions (LOCAL CONTEXT ONLY)
    # -----------------------------
    insertion_positions = choose_intruder_positions_llm(
        slash_sentences,
        num_intruders
    )

    intruders = []
    reference = list(original_sentences) + list(slash_sentences)

    # -----------------------------
    # STEP 2: generate per position
    # -----------------------------
    for i, pos in enumerate(insertion_positions):

        prev_sentence = slash_sentences[pos - 1]
        next_sentence = slash_sentences[pos]

        system_prompt = (
            "You are inserting ONE sentence into a paragraph.\n\n"

            "PREVIOUS SENTENCE:\n"
            f"{prev_sentence}\n\n"

            "NEXT SENTENCE:\n"
            f"{next_sentence}\n\n"

            "Write ONE sentence that:\n"
            "- connects naturally to the previous sentence\n"
            "- does not disrupt the flow into the next sentence\n"
            "- stays on topic\n"
            "- does NOT copy wording\n"
            "- introduces a slightly different or tangential idea\n\n"

            "Return ONE sentence only."
        )

        best_candidate = None

        for attempt in range(10):

            try:
                cand = llm_chat(system_prompt, "", temperature=0.9, max_tokens=80).strip()
            except Exception as e:
                print(f"⚠️ Intruder gen error: {e}")
                continue

            if not cand:
                continue

            if intruder_too_similar(cand, reference):
                continue

            for pinc_min, scope in [
                (0.80, "all"),
                (0.75, "all"),
                (0.70, "all"),
                (0.65, "recent"),
            ]:

                print(f"🔧 Intruder tier: PINC≥{pinc_min} scope={scope}")

                refs = reference if scope == "all" else reference[-3:]

                if intruder_has_required_pinc(cand, refs, min_pinc=pinc_min):
                    intruders.append(cand)
                    reference.append(cand)
                    best_candidate = cand
                    break

            if best_candidate:
                break

        if not best_candidate:
            raise RuntimeError(f"Failed to generate intruder {i+1}")

    return intruders, insertion_positions

def api_detect_outlier_sentences(hybrid_paragraph):
    system_prompt = (
        "You are reading a student paragraph.\n"
        "Some sentences may feel less connected to the writer's flow of ideas.\n"
        "Do NOT focus on grammar mistakes.\n"
        "Identify the sentence numbers that stand out as least connected.\n"
        "Return ONLY comma-separated sentence numbers, or NONE."
    )

    raw = llm_chat(system_prompt, hybrid_paragraph, temperature=0.2, max_tokens=80).strip()

    if raw.upper() == "NONE":
        return []

    nums = re.findall(r"\d+", raw)
    return [int(n) for n in nums]

def intruder_has_required_pinc(candidate_sentence, reference_sentences, min_pinc=INTRUDER_PINC_MIN):
    """
    Return True only if candidate_sentence has PINC >= min_pinc
    against EVERY sentence in reference_sentences.
    """
    for ref in reference_sentences:
        pinc = compute_pinc(ref, candidate_sentence, max_n=3)
        if pinc < min_pinc:
            return False
    return True

def compute_detection_rate(flagged_numbers, true_intruder_numbers):
    flagged = set(flagged_numbers)
    true_intruders = set(true_intruder_numbers)

    if not true_intruders:
        return 0.0

    return len(flagged.intersection(true_intruders)) / len(true_intruders)

def spacing_irregularity_score(intruder_positions, total_sentences):
    if len(intruder_positions) < 3:
        return 1.0  # <-- FIX: don't penalize small N

    positions = sorted(intruder_positions)
    gaps = [positions[i+1] - positions[i] for i in range(len(positions)-1)]

    mean_gap = sum(gaps) / len(gaps)
    variance = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
    std = variance ** 0.5

    return min(1.0, std / mean_gap) if mean_gap > 0 else 0.0

def place_and_validate_intruders_step(base_sentences, intruders):
    best_candidate = None
    best_score = -1

    for attempt in range(1, MAX_PARAGRAPH_RETRIES + 1):
        print("\n======================================")
        print(f"🎲 Intruder placement attempt {attempt}")
        print("======================================")

        try:
            augmented, source_map = hybrid_place_intruders(base_sentences, intruders)
        except Exception as e:
            print(f"⚠️ Placement failed: {e}")
            continue

        intruder_numbers = get_pdf_intruder_numbers_from_augmented(augmented, intruders)
        paragraph = build_unified_paragraph(augmented)

        flagged = api_detect_outlier_sentences(paragraph)
        detection_rate = compute_detection_rate(flagged, intruder_numbers)
        dispersion = spacing_irregularity_score(intruder_numbers, len(augmented))

        print(f"👁 Flagged: {flagged}")
        print(f"🎯 True intruders: {intruder_numbers}")
        print(f"📊 Detection rate: {detection_rate:.2f}")
        print(f"📐 Dispersion score: {dispersion:.3f}")

        score = (1 - detection_rate) * 0.7 + dispersion * 0.3

        if score > best_score:
            best_score = score
            best_candidate = (augmented, intruder_numbers, source_map)

        if (
            detection_rate <= MAX_INTRUDER_DETECTION_RATE and
            dispersion >= MIN_DISPERSION_SCORE
        ):
            return augmented, intruder_numbers, source_map

    if best_candidate:
        return best_candidate

    augmented = list(base_sentences)
    source_map = [("BASE", i) for i in range(len(base_sentences))]

    positions = random.sample(range(len(augmented) + 1), len(intruders))
    positions.sort(reverse=True)

    for intr_idx, pos in enumerate(positions):
        intr = intruders[intr_idx]
        augmented.insert(pos, intr)
        source_map.insert(pos, ("X", intr_idx))

    intruder_numbers = get_pdf_intruder_numbers_from_augmented(augmented, intruders)
    return augmented, intruder_numbers, source_map

def intruder_too_similar(candidate, existing_sentences, threshold=0.75):
    cand_norm = normalize_sentence(candidate)
    cand_tokens = set(tokenize_words_lower(candidate))

    for sent in existing_sentences:
        sent_norm = normalize_sentence(sent)
        sent_tokens = set(tokenize_words_lower(sent))

        if not sent_norm:
            continue

        # rapidfuzz whole-sentence similarity
        sim = fuzz.ratio(cand_norm, sent_norm) / 100.0
        if sim >= threshold:
            return True

        # content overlap check
        content_overlap = len(cand_tokens & sent_tokens)
        if content_overlap >= max(4, int(0.5 * max(1, len(cand_tokens)))):
            return True

    return False

def hybrid_place_intruders(base_sentences, intruders, trials=12):

    best_augmented = None
    best_source_map = None
    best_score = -1

    for _ in range(trials):

        # -----------------------------
        # 1. RANDOM PLACEMENT (exploration)
        # -----------------------------
        n = len(base_sentences)
        k = len(intruders)

        positions = sorted(random.sample(range(n + 1), k))

        augmented = list(base_sentences)
        source_map = [("BASE", i) for i in range(len(base_sentences))]

        for intr, pos, idx in sorted(
            zip(intruders, positions, range(len(intruders))),
            key=lambda x: x[1],
            reverse=True
        ):
            augmented.insert(pos, intr)
            source_map.insert(pos, ("X", idx))

        # -----------------------------
        # 2. LLM COHERENCE SCORING
        # -----------------------------
        paragraph = build_unified_paragraph(augmented)

        system_prompt = (
            "You are evaluating paragraph coherence.\n\n"
            "Rate how NATURALLY the paragraph flows overall.\n"
            "Do NOT identify intruders.\n\n"
            "Return ONLY a number from 0 to 1.\n"
            "1 = perfectly natural\n"
            "0 = very unnatural"
        )

        try:
            raw = llm_chat(system_prompt, paragraph, temperature=0.0, max_tokens=10)
            matches = re.findall(r"\d*\.?\d+", raw)
            score = float(matches[0]) if matches else 0.5
        except:
            score = 0.5

        # -----------------------------
        # 3. KEEP BEST
        # -----------------------------
        if score > best_score:
            best_score = score
            best_augmented = augmented
            best_source_map = source_map

    return best_augmented, best_source_map


# -----------------------------
# PDF GENERATION
# -----------------------------

def generate_pdf(student_id, name, sentences):
    import os, html, re
    from weasyprint import HTML

    os.makedirs(PDF_DIR, exist_ok=True)
    filename = sanitize_filename(name) + ".pdf"
    path = os.path.join(PDF_DIR, filename)

    esc_name = html.escape(name)
    esc_id = html.escape(student_id)

    # -----------------------------
    # BUILD PARAGRAPH (UPDATED)
    # -----------------------------
    paragraph_parts = []

    for i, s in enumerate(sentences, 1):
        esc = html.escape(s.strip())
        label = f"<span class='label'>[{i} ___ ]</span>"
        paragraph_parts.append(f"{label} {esc}")

    paragraph_text = " ".join(paragraph_parts)

    # -----------------------------
    # HTML
    # -----------------------------
    html_parts = [
        "<html><head><meta charset='utf-8'>",
        "<style>",
        "@page { margin: 1.5cm; size: A4; }",
        "body { font-family: Arial, sans-serif; font-size: 14pt; }",
        ".header { margin-bottom: 10px; }",
        ".paragraph { text-indent: 2em; line-height: 1.5; text-align: justify; }",

        # 🔥 KEY FIX
        ".label { white-space: nowrap; }",

        "</style></head><body>",

        f"<div class='header'><b>Name:</b> {esc_name}<br>",
        f"<b>Student Number:</b> {esc_id}</div>",

        "<div><b>Instructions:</b> For each sentence, mark O, /, or X.</div><br>",

        f"<div class='paragraph'>{paragraph_text}</div>",

        "</body></html>"
    ]

    HTML(string="\n".join(html_parts)).write_pdf(path)
    print(f"PDF created: {path}")

# -----------------------------
# TEST GENERATION
# -----------------------------

def process():

    max_sents = get_max_sentences(INPUT_TSV)

    with open(ANSWER_KEY, "w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out, delimiter="\t")

        # -----------------------------
        # HEADER
        # -----------------------------
        header = [
            "student_id",
            "name",
            "original_paragraph",
            "revised_paragraph",
            "O_sentences",
            "X_sentences",
            "/_sentences",
        ]

        for i in range(1, max_sents + 1):
            header += [
                f"orig_{i}",
                f"mod_{i}",
                f"rapidfuzz_{i}",
                f"edit_distance_{i}",
                f"ngram_overlap_{i}",
                f"sbert_cosine_{i}",
                f"sts_score_{i}",
                f"pos_overlap_{i}",
                f"dep_overlap_{i}",
                f"tree_edit_{i}",
                f"pinc_{i}",
            ]

        writer.writerow(header)

        with open(INPUT_TSV, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")

            for row in reader:
                if len(row) < 3:
                    continue

                student_id, name, text = row
                original_sentences = split_into_sentences(text)
                total = len(original_sentences)

                if total == 0:
                    continue

                print("\n🎯 TARGETS")
                print(f"Total: {total}")

                # -----------------------------
                # EMBEDDING CACHE
                # -----------------------------
                embedding_cache = {}

                def get_emb(sentence):
                    if sentence not in embedding_cache:
                        embedding_cache[sentence] = encode_sentence(sentence)
                    return embedding_cache[sentence]

                # -----------------------------
                # STEP 1: generate / candidates
                # -----------------------------
                sentence_candidates = generate_slash_modifications(
                    original_sentences,
                    get_emb
                )

                # -----------------------------
                # STEP 2: generate X intruders
                # (FIXED: pass BOTH original + slash)
                # -----------------------------
                target_o = total // 2
                target_slash = total - target_o
                target_x = target_o

                # -----------------------------
                # STEP 3: apply / selection
                # -----------------------------
                slash_sentences, selected_counts = select_slash_sentences(
                    original_sentences,
                    sentence_candidates,
                    target_slash
                )

                # -----------------------------
                # STEP 4: generate X intruders (position-first)
                # -----------------------------
                intruders, _ = generate_intruders_step(
                    original_sentences,
                    slash_sentences,
                    target_x
                )

                # -----------------------------
                # STEP 5: insert intruders (WITH validation loop restored)
                # -----------------------------
                final_sentences, intruder_numbers, source_map = \
                    place_and_validate_intruders_step(
                        slash_sentences,
                        intruders
                    )

                print("\n📊 FINAL DISTRIBUTION READY")

                # -----------------------------
                # STEP 6: LABELING (FIXED)
                # -----------------------------
                O, X, P = [], [], []

                for i, (kind, ref_idx) in enumerate(source_map, 1):

                    if kind == "X":
                        X.append(i)

                    else:
                        # BASE sentence → check if modified or not
                        if slash_sentences[ref_idx] == original_sentences[ref_idx]:
                            O.append(i)
                        else:
                            P.append(i)

                # -----------------------------
                # BASE ROW
                # -----------------------------
                row_out = [
                    student_id,
                    name,
                    text,
                    " ".join(final_sentences),
                    ",".join(map(str, O)),
                    ",".join(map(str, X)),
                    ",".join(map(str, P)),
                ]

                # -----------------------------
                # METRICS (FIXED ALIGNMENT)
                # -----------------------------
                for i in range(max_sents):

                    if i < len(final_sentences):

                        mod = final_sentences[i]
                        kind, ref_idx = source_map[i]

                        if kind == "X":
                            # intruder → compare to itself
                            orig = mod
                        else:
                            # correctly aligned original sentence
                            orig = original_sentences[ref_idx]

                        orig_emb = get_emb(orig)

                        row_out += [
                            orig,
                            mod,
                            f"{compute_token_index(orig, mod):.4f}",
                            f"{compute_edit_distance(orig, mod):.4f}",
                            f"{compute_ngram_overlap(orig, mod, 2):.4f}",
                            f"{semantic_similarity_from_emb(orig_emb, mod):.4f}",
                            f"{compute_sts_score(orig_emb, mod):.4f}",
                            f"{compute_structural_index(orig, mod):.4f}",
                            f"{compute_dep_overlap(orig, mod):.4f}",
                            f"{compute_tree_edit_proxy(orig, mod):.4f}",
                            f"{compute_pinc(orig, mod, 3):.4f}",
                        ]

                    else:
                        row_out += [""] * 11

                # -----------------------------
                # PDF
                # -----------------------------
                generate_pdf(student_id, name, final_sentences)

                writer.writerow(row_out)

    print("Done.")
    
# -----------------------------
# MAIN
# -----------------------------

if __name__ == "__main__":
    process()