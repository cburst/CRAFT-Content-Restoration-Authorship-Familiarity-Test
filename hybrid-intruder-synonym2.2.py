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
# Version 2.1: save metrics for every final item, including positions
# added by intruder insertion. Existing output filenames/schema are retained.
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
PINC_SLASH_MAX = 0.50

PINC_SLASH_FALLBACK_INCREMENT = 0.025
PINC_SLASH_MAX_FALLBACK_STEPS = 5

PINC_SLASH_ATTEMPTS_PER_STAGE = 2

INTRUDER_PINC_MIN = 0.90
INTRUDER_PINC_FALLBACK_INCREMENT = 0.025
INTRUDER_PINC_MAX_FALLBACK_STEPS = 5
INTRUDER_ATTEMPTS_PER_STAGE = 2

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
    text = str(text).strip()

    # Remove a leading title line, including Markdown formatting.
    # Examples:
    # Title: Learning from Korean Writing
    # **Title: Learning from Korean Writing**
    # **Title:** Learning from Korean Writing

    text = re.sub(
        r'^\s*(?:\*\*)?\s*Title\s*:\s*'
        r'[^\r\n]*(?:\r?\n|$)',
        '',
        text,
        count=1,
        flags=re.IGNORECASE
    ).strip()

    # Tokenize the remaining text into sentences.
    return [
        s.strip()
        for s in nltk.sent_tokenize(text)
        if s.strip()
    ]


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
def choose_intruder_positions(base_sentences, num_intruders):
    """
    Fallback: randomly select unique insertion positions
    between existing sentences.

    Return positions in ascending order.
    """

    available_positions = list(
        range(1, len(base_sentences))
    )

    if num_intruders > len(available_positions):
        raise ValueError(
            f"Cannot insert {num_intruders} intruders "
            f"into {len(base_sentences)} sentences "
            "using unique internal positions."
        )

    return sorted(
        random.sample(
            available_positions,
            num_intruders
        )
    )

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
    Initialize candidate storage.

    Actual LLM generation occurs during progressive
    selection in select_slash_sentences().

    Each source sentence retains all previously
    generated candidates across fallback stages.
    """

    sentence_candidates = []

    for idx, sentence in enumerate(original_sentences):

        sentence_candidates.append({
            "original": sentence,
            "embedding": get_emb(sentence),
            "O": [],
            "SLASH": [],
            "seen": set(),
            "attempts": 0,
        })

    print()
    print("======================================")
    print("📦 SLASH CANDIDATE POOLS INITIALIZED")
    print("======================================")

    print(
        f"Source sentences: "
        f"{len(sentence_candidates)}"
    )

    print(
        f"Strict PINC range: "
        f"{PINC_SLASH_MIN:.3f}–{PINC_SLASH_MAX:.3f}"
    )

    print(
        f"Attempts per stage: "
        f"{PINC_SLASH_ATTEMPTS_PER_STAGE}"
    )

    print(
        f"Maximum fallback stages: "
        f"{PINC_SLASH_MAX_FALLBACK_STEPS}"
    )

    return sentence_candidates

def select_slash_sentences(
    original_sentences,
    sentence_candidates,
    target_slash,
):
    """
    Progressive PINC selection with retained candidates
    and two generation attempts per fallback stage.

    Stage 0:
        PINC_SLASH_MIN to PINC_SLASH_MAX

    Each subsequent stage:
        Expand both boundaries by
        PINC_SLASH_FALLBACK_INCREMENT.

    Previously generated candidates remain available.

    Each LLM generation call produces LOW, MEDIUM,
    and HIGH paraphrases.

    Stop when enough distinct source sentences have
    eligible candidates.
    """

    total = len(original_sentences)

    if target_slash < 0 or target_slash > total:
        raise ValueError(
            "Invalid target number of modified sentences."
        )

    modified_sentences = list(original_sentences)

    target_pinc = (
        PINC_SLASH_MIN + PINC_SLASH_MAX
    ) / 2

    widest_min = max(
        0.0,
        PINC_SLASH_MIN
        - PINC_SLASH_MAX_FALLBACK_STEPS
        * PINC_SLASH_FALLBACK_INCREMENT
    )

    widest_max = min(
        1.0,
        PINC_SLASH_MAX
        + PINC_SLASH_MAX_FALLBACK_STEPS
        * PINC_SLASH_FALLBACK_INCREMENT
    )

    print()
    print("======================================")
    print("🔎 PROGRESSIVE SLASH GENERATION")
    print("======================================")

    print(f"Target modifications: {target_slash}")
    print(f"Preferred PINC: {target_pinc:.3f}")
    print(
        f"Maximum permitted range: "
        f"{widest_min:.3f}–{widest_max:.3f}"
    )

    # ------------------------------------------------
    # GENERATE AND RETAIN CANDIDATES
    # ------------------------------------------------

    def generate_candidates(bucket, idx, lower, upper):
        """
        Make one independent LLM generation call.

        Retain every unique candidate satisfying
        semantic similarity and the widest permitted
        PINC range.

        Return True if the source sentence has at
        least one candidate in the current band.
        """

        sentence = bucket["original"]
        orig_emb = bucket["embedding"]

        bucket["attempts"] += 1

        print(
            f"\n🔁 Source [{idx + 1}] "
            f"| Generation attempt "
            f"{bucket['attempts']}"
        )

        # One call normally generates three candidates.
        # Existing API-error retries remain available.

        candidate_dict = generate_rewrite_set(
            sentence
        )

        for level, candidate in candidate_dict.items():

            if not candidate:
                continue

            candidate = candidate.strip()

            if not candidate:
                continue

            # Avoid retaining duplicate paraphrases.

            normalized = normalize_sentence(candidate)

            if normalized in bucket["seen"]:
                continue

            bucket["seen"].add(normalized)

            semantic_index = semantic_similarity_from_emb(
                orig_emb,
                candidate
            )

            pinc = compute_pinc(
                sentence,
                candidate,
                max_n=3
            )

            print(
                f"  ➡️ {level.upper()}: {candidate}"
            )

            print(
                f"     🧠 Semantic={semantic_index:.3f} "
                f"| 🧬 PINC={pinc:.3f}"
            )

            if semantic_index < SEMANTIC_THRESHOLD:

                print("     ❌ Semantic threshold")

                continue

            entry = (
                candidate,
                pinc,
                semantic_index,
                level,
            )

            if pinc < PINC_O_MAX:
                bucket["O"].append(entry)

            if widest_min <= pinc <= widest_max:
                bucket["SLASH"].append(entry)

            else:
                print("     ❌ Outside maximum PINC range")

        return any(
            lower <= entry[1] <= upper
            for entry in bucket["SLASH"]
        )

    def eligible_candidates(bucket, lower, upper):
        return [
            entry
            for entry in bucket["SLASH"]
            if lower <= entry[1] <= upper
        ]

    def eligible_source_count(lower, upper):
        return sum(
            bool(eligible_candidates(bucket, lower, upper))
            for bucket in sentence_candidates
        )

    # ------------------------------------------------
    # PROGRESSIVE GENERATION
    # ------------------------------------------------

    final_stage = None

    for step in range(
        PINC_SLASH_MAX_FALLBACK_STEPS + 1
    ):

        lower = max(
            0.0,
            PINC_SLASH_MIN
            - step * PINC_SLASH_FALLBACK_INCREMENT
        )

        upper = min(
            1.0,
            PINC_SLASH_MAX
            + step * PINC_SLASH_FALLBACK_INCREMENT
        )

        print()
        print("======================================")

        if step == 0:
            print(
                f"🟢 STRICT STAGE "
                f"| PINC {lower:.3f}–{upper:.3f}"
            )
        else:
            print(
                f"🟡 FALLBACK {step}/"
                f"{PINC_SLASH_MAX_FALLBACK_STEPS} "
                f"| PINC {lower:.3f}–{upper:.3f}"
            )

        print("======================================")

        # First reconsider every retained candidate.
        # No additional LLM calls if the existing
        # candidate pool is already sufficient.

        current_count = eligible_source_count(
            lower,
            upper
        )

        print(
            f"📦 Eligible sources from retained "
            f"candidates: {current_count}/{target_slash}"
        )

        if current_count >= target_slash:
            final_stage = step
            break

        # Two rounds of generation per stage.
        # Each round gives each still-ineligible
        # source sentence one opportunity.

        for attempt_round in range(
            1,
            PINC_SLASH_ATTEMPTS_PER_STAGE + 1
        ):

            print(
                f"\n🔄 Generation round "
                f"{attempt_round}/"
                f"{PINC_SLASH_ATTEMPTS_PER_STAGE}"
            )

            for idx, bucket in enumerate(
                sentence_candidates
            ):

                # Stop immediately once enough
                # distinct source sentences qualify.

                current_count = eligible_source_count(
                    lower,
                    upper
                )

                if current_count >= target_slash:
                    break

                # Do not regenerate a sentence that
                # already has an eligible candidate
                # in the current PINC range.

                if eligible_candidates(
                    bucket,
                    lower,
                    upper
                ):
                    continue

                generate_candidates(
                    bucket,
                    idx,
                    lower,
                    upper
                )

            current_count = eligible_source_count(
                lower,
                upper
            )

            print(
                f"\n📊 Eligible sources after "
                f"generation round {attempt_round}: "
                f"{current_count}/{target_slash}"
            )

            if current_count >= target_slash:
                final_stage = step
                break

        if final_stage is not None:
            break

    # ------------------------------------------------
    # VALIDATE CANDIDATE AVAILABILITY
    # ------------------------------------------------

    if final_stage is None:

        raise RuntimeError(
            "Insufficient eligible modified sentences "
            "after all generation attempts and "
            "progressive PINC relaxation. "
            f"Required: {target_slash}; "
            f"available: {eligible_source_count(widest_min, widest_max)}. "
            f"Final PINC range: "
            f"{widest_min:.3f}–{widest_max:.3f}."
        )

    # ------------------------------------------------
    # FINAL SELECTION
    # ------------------------------------------------

    lower = max(
        0.0,
        PINC_SLASH_MIN
        - final_stage * PINC_SLASH_FALLBACK_INCREMENT
    )

    upper = min(
        1.0,
        PINC_SLASH_MAX
        + final_stage * PINC_SLASH_FALLBACK_INCREMENT
    )

    pool = []

    for idx, bucket in enumerate(sentence_candidates):

        for candidate, pinc, sem, level in eligible_candidates(
            bucket,
            lower,
            upper
        ):

            pool.append({
                "idx": idx,
                "sentence": candidate,
                "pinc": pinc,
                "semantic": sem,
                "level": level,
            })

    selected = []
    used_indices = set()

    while len(selected) < target_slash:

        available = [
            item
            for item in pool
            if item["idx"] not in used_indices
        ]

        if not available:
            raise RuntimeError(
                "Candidate pool exhausted during "
                "final SLASH selection."
            )

        if not selected:

            best = min(
                available,
                key=lambda item: (
                    abs(item["pinc"] - target_pinc),
                    -item["semantic"],
                ),
            )

        else:

            best = max(
                available,
                key=lambda item: (
                    min(
                        abs(
                            item["pinc"] - chosen["pinc"]
                        )
                        for chosen in selected
                    ),
                    -abs(item["pinc"] - target_pinc),
                    item["semantic"],
                ),
            )

        selected.append(best)
        used_indices.add(best["idx"])

    # ------------------------------------------------
    # APPLY SELECTION
    # ------------------------------------------------

    selected_counts = {
        "O": 0,
        "SLASH": 0,
        "X": 0,
    }

    for item in selected:

        idx = item["idx"]

        modified_sentences[idx] = item["sentence"]

        selected_counts["SLASH"] += 1

        print(
            f"🟡 SELECTED SLASH [{idx + 1}] "
            f"| PINC={item['pinc']:.3f} "
            f"| Semantic={item['semantic']:.3f}"
        )

    for idx in range(total):

        if idx not in used_indices:

            selected_counts["O"] += 1

            print(
                f"🟢 SELECTED O [{idx + 1}]"
            )

    print()
    print("======================================")
    print("📊 FINAL SLASH DISTRIBUTION")
    print("======================================")

    print(
        f"Fallback stage: {final_stage}"
    )

    print(
        f"Final permitted PINC range: "
        f"{lower:.3f}–{upper:.3f}"
    )

    print(
        "Selected PINCs:",
        [
            round(item["pinc"], 3)
            for item in selected
        ],
    )

    print(
        f"Original: {selected_counts['O']} "
        f"| Modified: {selected_counts['SLASH']}"
    )

    return modified_sentences, selected_counts


def generate_intruders_step(
    original_sentences,
    slash_sentences,
    num_intruders,
):
    """
    Generate fabricated sentences using progressive
    minimum-PINC relaxation and adaptive prompt
    diversification.

    Retain candidates across fallback stages.

    Each candidate is evaluated against:
        - All original sentences
        - All modified/base sentences
        - All previously accepted fabricated sentences

    After generation, recalculate each fabricated
    sentence's minimum PINC against the complete
    reference set, including all other fabricated
    sentences.

    Returns:
        intruders
        insertion_positions
        intruder_pinc_scores

    The three returned lists have matching indices.
    """

    if not slash_sentences or num_intruders <= 0:
        return [], [], []

    # ------------------------------------------------
    # STEP 1: CHOOSE LOCAL CONTEXT POSITIONS
    # ------------------------------------------------

    insertion_positions = choose_intruder_positions_llm(
        slash_sentences,
        num_intruders
    )

    intruders = []
    intruder_pinc_scores = []

    reference = (
        list(original_sentences)
        + list(slash_sentences)
    )

    final_minimum = max(
        0.0,
        INTRUDER_PINC_MIN
        - INTRUDER_PINC_MAX_FALLBACK_STEPS
        * INTRUDER_PINC_FALLBACK_INCREMENT
    )

    print()
    print("======================================")
    print("🟣 PROGRESSIVE INTRUDER GENERATION")
    print("======================================")

    print(
        f"Target fabricated sentences: "
        f"{num_intruders}"
    )

    print(
        f"Initial minimum PINC: "
        f"{INTRUDER_PINC_MIN:.3f}"
    )

    print(
        f"Final minimum PINC: "
        f"{final_minimum:.3f}"
    )

    print(
        f"Attempts per stage: "
        f"{INTRUDER_ATTEMPTS_PER_STAGE}"
    )

    print(
        f"Maximum fallback stages: "
        f"{INTRUDER_PINC_MAX_FALLBACK_STEPS}"
    )

    # ------------------------------------------------
    # STEP 2: GENERATE FABRICATED SENTENCES
    # ------------------------------------------------

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
            "- introduces a slightly different or tangential idea\n"
            "- does not contradict the surrounding text\n\n"

            "Return ONE sentence only."
        )

        candidates = []
        seen = set()

        selected = None
        selected_stage = None

        generation_attempts = 0

        print()
        print("======================================")
        print(
            f"🎯 FABRICATED SENTENCE "
            f"{i + 1}/{num_intruders}"
        )
        print("======================================")

        print(f"Context position: {pos}")
        print(f"Reference sentences: {len(reference)}")

        # --------------------------------------------
        # PROGRESSIVE PINC RELAXATION
        # --------------------------------------------

        for step in range(
            INTRUDER_PINC_MAX_FALLBACK_STEPS + 1
        ):

            minimum_pinc = max(
                0.0,
                INTRUDER_PINC_MIN
                - step * INTRUDER_PINC_FALLBACK_INCREMENT
            )

            print()
            print("--------------------------------------")

            if step == 0:

                print(
                    f"🟢 STRICT STAGE "
                    f"| Minimum PINC={minimum_pinc:.3f}"
                )

            else:

                print(
                    f"🟡 FALLBACK {step}/"
                    f"{INTRUDER_PINC_MAX_FALLBACK_STEPS} "
                    f"| Minimum PINC={minimum_pinc:.3f}"
                )

            print("--------------------------------------")

            # ----------------------------------------
            # RECONSIDER RETAINED CANDIDATES
            # ----------------------------------------

            print(
                f"📦 Retained candidates: "
                f"{len(candidates)}"
            )

            for candidate in candidates:

                if intruder_too_similar(
                    candidate,
                    reference
                ):
                    continue

                if intruder_has_required_pinc(
                    candidate,
                    reference,
                    min_pinc=minimum_pinc
                ):

                    selected = candidate
                    selected_stage = step

                    print(
                        "♻️ Accepted previously "
                        "generated candidate."
                    )

                    break

            if selected:
                break

            # ----------------------------------------
            # GENERATE NEW CANDIDATES
            # ----------------------------------------

            for attempt in range(
                1,
                INTRUDER_ATTEMPTS_PER_STAGE + 1
            ):

                generation_attempts += 1

                print()
                print(
                    f"🔄 Generation attempt "
                    f"{attempt}/"
                    f"{INTRUDER_ATTEMPTS_PER_STAGE} "
                    f"at stage {step}"
                )

                print(
                    f"Total generation attempts: "
                    f"{generation_attempts}"
                )

                # ------------------------------------
                # ADAPTIVE PROMPT DIVERSIFICATION
                # ------------------------------------

                if generation_attempts <= 2:

                    diversity_instruction = (
                        "Generate a natural, contextually "
                        "appropriate sentence introducing "
                        "a slightly tangential idea."
                    )

                elif generation_attempts <= 4:

                    diversity_instruction = (
                        "Introduce a NEW concrete example "
                        "or observation related to the "
                        "paragraph. Do not repeat the "
                        "previous sentence's main idea."
                    )

                elif generation_attempts <= 6:

                    diversity_instruction = (
                        "Introduce a different consequence, "
                        "implication, or practical application "
                        "of the surrounding ideas. Avoid "
                        "repeating earlier explanations."
                    )

                elif generation_attempts <= 8:

                    diversity_instruction = (
                        "Introduce a relevant contrast, "
                        "qualification, or limitation. "
                        "Use different vocabulary and "
                        "sentence structure."
                    )

                else:

                    diversity_instruction = (
                        "Introduce an entirely NEW but "
                        "contextually plausible observation. "
                        "Avoid repeating the main ideas "
                        "or expressions in nearby sentences."
                    )

                # Include recent rejected candidates
                # so the LLM can avoid repeating them.

                recent_candidates = candidates[-5:]

                avoid_text = "\n".join(
                    f"- {sentence}"
                    for sentence in recent_candidates
                )

                if not avoid_text:
                    avoid_text = "None"

                dynamic_prompt = (
                    system_prompt
                    + "\n\nADDITIONAL REQUIREMENT:\n"
                    + diversity_instruction
                    + "\n\nDO NOT REPEAT OR CLOSELY "
                      "PARAPHRASE THESE PREVIOUSLY "
                      "REJECTED CANDIDATES:\n"
                    + avoid_text
                    + "\n\nMaintain natural paragraph "
                      "coherence despite using "
                      "different wording."
                )

                print(
                    f"🧠 Prompt diversification "
                    f"level: {generation_attempts}"
                )

                # ------------------------------------
                # LLM GENERATION
                # ------------------------------------

                try:

                    candidate = llm_chat(
                        dynamic_prompt,
                        "",
                        temperature=0.9,
                        max_tokens=100
                    ).strip()

                except Exception as error:

                    print(
                        f"⚠️ Intruder generation error: "
                        f"{error}"
                    )

                    continue

                if not candidate:

                    print("❌ Empty candidate.")
                    continue

                normalized = normalize_sentence(
                    candidate
                )

                if not normalized:

                    print("❌ Candidate has no words.")
                    continue

                if normalized in seen:

                    print("⚠️ Duplicate candidate.")
                    continue

                seen.add(normalized)
                candidates.append(candidate)

                print(f"➡️ Candidate: {candidate}")

                # ------------------------------------
                # LEXICAL SIMILARITY SAFEGUARD
                # ------------------------------------

                if intruder_too_similar(
                    candidate,
                    reference
                ):

                    print(
                        "❌ Rejected: too similar "
                        "to an existing sentence."
                    )

                    continue

                # ------------------------------------
                # PINC VALIDATION
                # ------------------------------------

                if intruder_has_required_pinc(
                    candidate,
                    reference,
                    min_pinc=minimum_pinc
                ):

                    selected = candidate
                    selected_stage = step

                    print(
                        f"✅ Accepted candidate "
                        f"at PINC≥{minimum_pinc:.3f}"
                    )

                    break

                print(
                    f"❌ Candidate below minimum "
                    f"PINC={minimum_pinc:.3f}"
                )

            if selected:
                break

        # --------------------------------------------
        # VALIDATE FINAL RESULT
        # --------------------------------------------

        if not selected:

            raise RuntimeError(
                f"Failed to generate fabricated "
                f"sentence {i + 1}/{num_intruders}. "
                f"Attempts: {generation_attempts}. "
                f"Final minimum PINC: "
                f"{final_minimum:.3f}. "
                f"Retained candidates: "
                f"{len(candidates)}."
            )

        # --------------------------------------------
        # CALCULATE ACCEPTANCE PINC
        # --------------------------------------------

        acceptance_pinc = min(
            compute_pinc(
                ref,
                selected,
                max_n=3
            )
            for ref in reference
        )

        required_minimum = max(
            0.0,
            INTRUDER_PINC_MIN
            - selected_stage
            * INTRUDER_PINC_FALLBACK_INCREMENT
        )

        if acceptance_pinc < required_minimum - 1e-9:

            raise RuntimeError(
                "Selected fabricated sentence "
                "violates its PINC threshold. "
                f"Actual: {acceptance_pinc:.4f}; "
                f"required: {required_minimum:.4f}."
            )

        # --------------------------------------------
        # ACCEPT SENTENCE
        # --------------------------------------------

        intruders.append(selected)

        # Temporarily record acceptance PINC.
        # Final values are recalculated after
        # all fabricated sentences are generated.

        intruder_pinc_scores.append(
            acceptance_pinc
        )

        reference.append(selected)

        print()
        print(
            f"🟣 SELECTED FABRICATED "
            f"[{i + 1}]"
        )

        print(f"Sentence: {selected}")
        print(f"Fallback stage: {selected_stage}")

        print(
            f"Required minimum PINC: "
            f"{required_minimum:.3f}"
        )

        print(
            f"Acceptance PINC: "
            f"{acceptance_pinc:.4f}"
        )

        print(
            f"Generation attempts: "
            f"{generation_attempts}"
        )

    # ------------------------------------------------
    # STEP 3: RECALCULATE FINAL PINC SCORES
    # ------------------------------------------------

    print()
    print("======================================")
    print("🔬 FINAL FABRICATED PINC VALIDATION")
    print("======================================")

    final_pinc_scores = []

    for i, candidate in enumerate(intruders):

        # Include every original and modified
        # sentence, plus all OTHER fabricated
        # sentences. Exclude the candidate itself.

        complete_reference = (
            list(original_sentences)
            + list(slash_sentences)
            + [
                intruder
                for j, intruder in enumerate(intruders)
                if j != i
            ]
        )

        final_pinc = min(
            compute_pinc(
                ref,
                candidate,
                max_n=3
            )
            for ref in complete_reference
        )

        final_pinc_scores.append(final_pinc)

        print(
            f"🟣 Fabricated [{i + 1}] "
            f"| Acceptance PINC="
            f"{intruder_pinc_scores[i]:.4f} "
            f"| Final PINC={final_pinc:.4f}"
        )

        if final_pinc < final_minimum - 1e-9:

            print(
                f"⚠️ Final PINC below maximum "
                f"relaxation threshold "
                f"({final_minimum:.3f})."
            )

    # Replace acceptance scores with final scores.

    intruder_pinc_scores = final_pinc_scores

    # ------------------------------------------------
    # STEP 4: FINAL VALIDATION
    # ------------------------------------------------

    if not (
        len(intruders)
        == len(intruder_pinc_scores)
        == len(insertion_positions)
        == num_intruders
    ):

        raise RuntimeError(
            "Fabricated-sentence generation produced "
            "inconsistent output lengths."
        )

    print()
    print("======================================")
    print("📊 FINAL FABRICATED-SENTENCE SUMMARY")
    print("======================================")

    print(
        f"Generated: "
        f"{len(intruders)}/{num_intruders}"
    )

    print(
        f"Initial PINC threshold: "
        f"{INTRUDER_PINC_MIN:.3f}"
    )

    print(
        f"Final permitted minimum: "
        f"{final_minimum:.3f}"
    )

    print(
        "Final fabricated minimum PINCs:",
        [
            round(score, 4)
            for score in intruder_pinc_scores
        ],
    )

    return (
        intruders,
        insertion_positions,
        intruder_pinc_scores,
    )


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

    # Each source paragraph retains N base sentences
    # and adds floor(N / 2) fabricated sentences.
    max_final_items = max_sents + max_sents // 2

    with open(
        ANSWER_KEY,
        "w",
        newline="",
        encoding="utf-8"
    ) as out:

        writer = csv.writer(
            out,
            delimiter="\t"
        )

        # ------------------------------------
        # STEP 1: HEADER
        # ------------------------------------

        header = [
            "student_id",
            "name",
            "original_paragraph",
            "revised_paragraph",
            "O_sentences",
            "X_sentences",
            "/_sentences",
        ]

        for i in range(1, max_final_items + 1):

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

        # ------------------------------------
        # STEP 2: READ INPUT
        # ------------------------------------

        with open(
            INPUT_TSV,
            encoding="utf-8"
        ) as f:

            reader = csv.reader(
                f,
                delimiter="\t"
            )

            for row in reader:

                if len(row) < 3:
                    continue

                student_id, name, text = row[:3]

                original_sentences = split_into_sentences(
                    text
                )

                total = len(original_sentences)

                if total == 0:
                    continue

                print()
                print("======================================")
                print(f"🎯 PROCESSING: {name}")
                print(f"Student ID: {student_id}")
                print(f"Original sentences: {total}")
                print("======================================")

                # ------------------------------------
                # STEP 3: EMBEDDING CACHE
                # ------------------------------------

                embedding_cache = {}

                def get_emb(sentence):

                    if sentence not in embedding_cache:

                        embedding_cache[sentence] = (
                            encode_sentence(sentence)
                        )

                    return embedding_cache[sentence]

                # ------------------------------------
                # STEP 4: MODIFIED SENTENCES
                # ------------------------------------

                sentence_candidates = (
                    generate_slash_modifications(
                        original_sentences,
                        get_emb
                    )
                )

                target_o = total // 2
                target_slash = total - target_o
                target_x = target_o

                slash_sentences, selected_counts = (
                    select_slash_sentences(
                        original_sentences,
                        sentence_candidates,
                        target_slash
                    )
                )

                # ------------------------------------
                # STEP 5: FABRICATED SENTENCES
                # ------------------------------------

                (
                    intruders,
                    insertion_positions,
                    intruder_pinc_scores
                ) = generate_intruders_step(
                    original_sentences,
                    slash_sentences,
                    target_x
                )

                # Validate the returned PINC scores.

                if len(intruders) != len(intruder_pinc_scores):

                    raise RuntimeError(
                        f"{student_id}: "
                        "Fabricated sentence count "
                        "does not match PINC score count."
                    )

                if len(intruders) != target_x:

                    raise RuntimeError(
                        f"{student_id}: "
                        f"Expected {target_x} fabricated "
                        f"sentences, received {len(intruders)}."
                    )

                # ------------------------------------
                # STEP 6: INTRUDER PLACEMENT
                # ------------------------------------

                (
                    final_sentences,
                    intruder_numbers,
                    source_map
                ) = place_and_validate_intruders_step(
                    slash_sentences,
                    intruders
                )

                # ------------------------------------
                # VALIDATE FINAL STRUCTURE
                # ------------------------------------

                if len(final_sentences) != len(source_map):

                    raise RuntimeError(
                        "Final sentences and source map "
                        "have different lengths."
                    )

                if len(final_sentences) > max_final_items:

                    raise RuntimeError(
                        f"{student_id}: "
                        f"{len(final_sentences)} final items "
                        f"exceed {max_final_items} "
                        "reserved metric slots."
                    )

                if len(final_sentences) != (
                    len(slash_sentences) + len(intruders)
                ):

                    raise RuntimeError(
                        "Final item count does not match "
                        "base plus fabricated count."
                    )

                # ------------------------------------
                # STEP 7: LABELING
                # ------------------------------------

                O = []
                X = []
                P = []

                for i, (kind, ref_idx) in enumerate(
                    source_map,
                    start=1
                ):

                    if kind == "X":

                        if not 0 <= ref_idx < len(intruders):
                            raise RuntimeError(
                                f"{student_id}: Invalid "
                                "fabricated sentence index."
                            )

                        X.append(i)

                    elif kind == "BASE":

                        if not 0 <= ref_idx < len(original_sentences):
                            raise RuntimeError(
                                f"{student_id}: Invalid "
                                "base sentence index."
                            )

                        if (
                            slash_sentences[ref_idx]
                            == original_sentences[ref_idx]
                        ):

                            O.append(i)

                        else:

                            P.append(i)

                    else:

                        raise RuntimeError(
                            f"Unknown source-map kind: {kind}"
                        )

                print()
                print("📊 FINAL DISTRIBUTION READY")

                print(
                    f"Original: {len(O)} | "
                    f"Modified: {len(P)} | "
                    f"Fabricated: {len(X)}"
                )

                # ------------------------------------
                # STEP 8: BASE ANSWER-KEY ROW
                # ------------------------------------

                row_out = [
                    student_id,
                    name,
                    text,
                    " ".join(final_sentences),
                    ",".join(map(str, O)),
                    ",".join(map(str, X)),
                    ",".join(map(str, P)),
                ]

                # ------------------------------------
                # STEP 9: METRICS
                # ------------------------------------

                for i in range(max_final_items):

                    # Unused positions retain
                    # exactly 11 empty fields.

                    if i >= len(final_sentences):

                        row_out += [""] * 11

                        continue

                    mod = final_sentences[i]

                    kind, ref_idx = source_map[i]

                    # --------------------------------
                    # FABRICATED SENTENCE (X)
                    # --------------------------------

                    if kind == "X":

                        if not 0 <= ref_idx < len(intruders):

                            raise RuntimeError(
                                f"{student_id}: Invalid "
                                "fabricated source index."
                            )

                        # The source map preserves
                        # the original intruder index
                        # even after random placement.

                        pinc = intruder_pinc_scores[ref_idx]

                        # Preserve the existing
                        # 11-column metric schema.
                        #
                        # orig_i = fabricated text
                        # mod_i  = fabricated text
                        #
                        # Other comparison metrics
                        # remain blank.
                        #
                        # pinc_i receives the actual
                        # minimum PINC against all
                        # reference sentences.

                        row_out += [
                            mod,                  # orig_i
                            mod,                  # mod_i
                            "",                   # rapidfuzz_i
                            "",                   # edit_distance_i
                            "",                   # ngram_overlap_i
                            "",                   # sbert_cosine_i
                            "",                   # sts_score_i
                            "",                   # pos_overlap_i
                            "",                   # dep_overlap_i
                            "",                   # tree_edit_i
                            f"{pinc:.4f}",        # pinc_i
                        ]

                        print(
                            f"🟣 X [{i + 1}] "
                            f"| PINC={pinc:.4f}"
                        )

                        continue

                    # --------------------------------
                    # ORIGINAL / MODIFIED SENTENCE
                    # --------------------------------

                    elif kind == "BASE":

                        if not 0 <= ref_idx < len(original_sentences):

                            raise RuntimeError(
                                f"{student_id}: Invalid "
                                "base sentence source index."
                            )

                        orig = original_sentences[ref_idx]

                    else:

                        raise RuntimeError(
                            f"Unknown source-map kind: {kind}"
                        )

                    orig_emb = get_emb(orig)

                    # Preserve all existing metrics
                    # and their original positions.

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

                # ------------------------------------
                # STEP 10: VALIDATE TSV ROW
                # ------------------------------------

                if len(row_out) != len(header):

                    raise RuntimeError(
                        f"{student_id}: "
                        f"Answer-key row has "
                        f"{len(row_out)} fields; "
                        f"expected {len(header)}."
                    )

                # ------------------------------------
                # STEP 11: GENERATE PDF
                # ------------------------------------

                generate_pdf(
                    student_id,
                    name,
                    final_sentences
                )

                # ------------------------------------
                # STEP 12: WRITE ANSWER KEY
                # ------------------------------------

                writer.writerow(row_out)

                out.flush()

                print()
                print(
                    f"✅ Completed: {name} "
                    f"({student_id})"
                )

    print()
    print("======================================")
    print("✅ ALL TESTS GENERATED")
    print("======================================")
    
# -----------------------------
# MAIN
# -----------------------------

if __name__ == "__main__":
    process()