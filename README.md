![License](https://img.shields.io/badge/license-Source--Available--Non--Production-blue)
![Python](https://img.shields.io/badge/python-3.11+-blue)

# 🧠 CRAFT Test Generator  
**Content Restoration Authorship Familiarity Tests (CRAFT)**

Generates authorship verification tests that measure a writer’s familiarity with their own text.

Produces:
- 📄 Test PDFs  
- 📊 Answer keys  

---

## 🚀 Quick Start

### 🪟 Windows
Download and run [CRAFT-Installer.exe](https://github.com/cburst/CRAFT-Content-Restoration-Authorship-Familiarity-Test/releases/latest/download/CRAFT-Installer.exe).  
If prompted, click **More info → Run anyway**.

### 🍎 macOS
Download [CRAFTinstaller.zip](https://github.com/cburst/CRAFT-Content-Restoration-Authorship-Familiarity-Test/releases/latest/download/CRAFTinstaller.zip), unzip, and run the installer.  

---

## 📄 Input Format

- `.tsv` file  
- No header row  
- 3 columns: student_number, name, text  

---

## 🧪 What it does

Each test includes:

- ✍️ **Sentence intruders** (identify LLM-generated added sentences)  
- 🔤 **Synonym replacements** (identify replacements words recover original words)  

👉 Core idea: real authors can reconstruct their own text.

---

## 🔑 First Run

You will be prompted for an **OpenAI API key**.

---

## 📁 Output

- `output/` → PDFs and answer keys  
- `archive/` → logs and saved runs  

---

## 👤 Author

Richard Rose  
Hankuk University of Foreign Studies (HUFS)

**License:** Source-Available (Non-Production Use Only)  
Free for personal, educational, and research use; commercial (production) use requires a separate license.
