# 🧠 CRAFT Test Generator  
**Content Restoration Authorship Familiarity Tests (CRAFT)**

This tool generates authorship verification tasks by measuring a writer’s familiarity with their own text.

It produces:
- 📄 Student test PDFs  
- 📊 Answer key CSV files  

---

## 🚀 Quick Start (Recommended)

### 🍎 macOS (current workflow)
1. Place the repository in:
   ~/CRAFTtests
2. Launch the app:
   CRAFT.app
3. Select your TSV file and generate tests

> 📦 A full installer (macOS + Windows) is coming soon.

---

## 📄 Input Format

- Tab-separated file (.tsv)  
- ❗ No header row  
- Exactly 3 columns:

    student_number    name    text

### Example:
    N6MAA10816    Roy Batty    I've seen things you people wouldn't believe. Attack ships on fire off the shoulder of Orion. I watched C-beams glitter in the dark near the Tannhäuser Gate. All those moments will be lost in time, like tears in rain.

---

## 🎯 Research Motivation
As large language models become increasingly capable of generating fluent academic text, traditional authorship verification methods based on stylistic features are becoming less reliable. The CRAFT framework addresses this challenge by shifting the focus from static text features to dynamic author knowledge. Instead of asking whether a text "looks like" a writer’s work, CRAFT evaluates whether a claimed author demonstrates familiarity with the content and structure of the text itself. By requiring authors to identify inserted sentences and recover original lexical choices, the test captures reconstruction-based evidence of authorship that is difficult to replicate without genuine involvement in the writing process.

## 🧪 What the Test Does

Each generated test includes:

### ✍️ Sentence Intruders
- Additional sentences are inserted using an LLM  
- The author must identify sentences that do not belong  

### 🔤 Synonym Replacement
- Rare words are identified using a frequency list  
- 5 words are replaced with synonyms  
- The author must recover the original wording  

👉 Core assumption:  
True authors demonstrate familiarity that cannot be easily replicated.

---

## ⚙️ Requirements

- Python 3.11  
- OpenAI-compatible API key  

### Required packages:
- weasyprint  
- nltk  
- numpy  
- scikit-learn  

---

## 🧠 Core Script

### ✅ Final CRAFT Test
    app/hybrid_intruder_synonym.py

- Combines:
  - sentence intruders  
  - synonym replacement   

---

## 🪦 Legacy / Experimental Scripts

Earlier test variants are included for reference and experimentation.

### 🔀 Hybrid Tests
- hybrid-intruders.py  
- hybrid-assembler-replacer.py  

### 🧩 Standalone Tests
- sentence reconstruction  
- synonym tasks  
- authorship recognition tasks  

> ⚠️ These are not part of the main GUI workflow.

---

## 🔧 Pipelines

- real_pipeline.py → human-authored texts  
- hard_pipeline.py → LLM-generated texts  

---

## 📁 Output

- /output/ → final PDFs and answer keys  
- /archive/ → logs and saved runs  

---

## 👤 Author

Richard Rose  
Hankuk University of Foreign Studies (HUFS)  

---
