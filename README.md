# 🧠 CRAFT Test Generator  
**Content Restoration Authorship Familiarity Tests (CRAFT)**

Generates authorship verification tests that measure a writer’s familiarity with their own text.

Produces:
- 📄 Test PDFs  
- 📊 Answer keys  

---

## 🚀 Quick Start

### 🪟 Windows
Run **CRAFT-Installer.exe** and follow the setup.  
If prompted, click **More info → Run anyway**.

### 🍎 macOS
Download **CRAFTinstaller.zip**, unzip, and run the installer.  
Dependencies (Homebrew, Python, etc.) are installed automatically.

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