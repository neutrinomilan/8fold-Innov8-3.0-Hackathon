# Machau_Innov8_3 – Bonus Challenge

This folder contains the full submission for the **Machau Innov8 3.0 – Bonus Challenge**. It includes the source code, outputs, data folders, and documentation required for evaluation.

---

## 📂 Folder Contents

| File / Folder | Description |
|---------------|-------------|
| `Prelims_Source_code-main.py` | Main Python script for the challenge (Truth Weaver engine). |
| `transcribed.txt` | Auto-generated transcripts of all candidate audio files. |
| `PrelimsSubmission.json` | Structured output summarizing each candidate’s skills, experience, and contradictions. |
| `README.md` | This documentation file, explaining setup and usage. |
| `INNOV8 3.0/` | Folder containing all provided audio datasets: <br> ├─ `Evaluation set/audio/` (evaluation `.mp3` files) <br> └─ `INNOV8 3.0/` (extra / practice audio files). |

> Place any new candidate `.mp3` files inside the appropriate subdirectories of `INNOV8 3.0` before running the script.

---

## 🚀 Quick Start for Judges

1. **Install Requirements**
   ```bash
   pip install openai-whisper nltk
   ```
   Ensure [ffmpeg](https://ffmpeg.org/download.html) is installed on your system.

2. **Download NLTK Stopwords** (only once):
   ```python
   import nltk
   nltk.download('stopwords')
   ```

3. **Run the Code**
   ```bash
   python Prelims_Source_code-main.py
   ```

4. **Review the Outputs**
   - `transcribed.txt` – text transcripts of each audio file.
   - `PrelimsSubmission.json` – JSON with each candidate’s revealed truths and deception patterns.

---

## 🧭 How It Works

| Stage | Description |
|-------|-------------|
| **1. Transcription** | Uses [OpenAI Whisper](https://github.com/openai/whisper) to convert `.mp3` audio to text. |
| **2. Keyword Extraction** | Parses transcripts to identify skills, durations (e.g., “6 years”, “internship”), leadership terms, and self-corrections. |
| **3. Contradiction Detection** | Compares claims across recordings for inconsistencies (e.g., “lead architect” vs. “junior dev”). |
| **4. Truth Consolidation** | Produces a summary of real experience, skill level, and team role. |

---

## 📌 Notes for Judges

- The `INNOV8 3.0` folder holds all audio inputs for testing and evaluation.
- The script can handle any `.mp3` interviews placed inside these directories.
- If no audio files are found, the script prints an error and exits.
- `transcribed.txt` and `PrelimsSubmission.json` are overwritten each time the script runs.

---

## 🧰 Requirements

- Python 3.8+
- [OpenAI Whisper](https://github.com/openai/whisper)
- `nltk` (stopwords corpus)
- `ffmpeg` for audio decoding

> No other Python packages or tools are required.

---

## 🏁 Summary

This submission implements an end-to-end pipeline to:
1. Transcribe candidate interviews from the `INNOV8 3.0` folder.
2. Extract key claims and skills.
3. Identify exaggerations or contradictions.
4. Present clear, structured insights for evaluation.

All logic is contained in `Prelims_Source_code-main.py`; outputs (`transcribed.txt`, `PrelimsSubmission.json`) demonstrate its effectiveness.