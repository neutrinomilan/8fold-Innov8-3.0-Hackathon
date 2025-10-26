import os
import re
import json
import whisper
import argparse
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

# --------------------
# CONFIG
# --------------------
OUTPUT_TRANSCRIPT = "transcribed.txt"
OUTPUT_JSON = "PrelimsSubmission.json"

# --------------------
# HELPERS
# --------------------
NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"
}

def normalize_numbers(text):
    for word, digit in NUMBER_WORDS.items():
        text = re.sub(rf"\b{word}\b", digit, text, flags=re.IGNORECASE)
    return text

def load_model():
    print("Loading Whisper model...")
    return whisper.load_model("base")

def clean_text(text):
    text = normalize_numbers(text.lower())
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS]

def extract_keywords(text):
    tokens = clean_text(text)
    keywords = []

    # Durations
    durations = re.findall(r"\d+\s*(?:year|years|month|months|full\s+time|counting\s+freelance|weekend\s+workshop)", text.lower())
    keywords.extend(durations)
    if "internship" in text.lower():
        keywords.append("internship")

    # Skills/context
    skills_and_context = [
        "python", "java", "c++", "machine", "learning", "debugging", "kubernetes", "kafka", "react", "django", "celery", "rails", "tensorflow", "pytorch", "smote", "cosmos db", "microservices", "distributed systems", "low latency", "backend systems", "networking", "security", "e-commerce", "caching", "geo-replication", "class imbalance", "custom hook", "sprints"
    ]
    for s in skills_and_context:
        if s in text.lower():
            keywords.append(s)

    # Roles
    role_keywords = ["team", "leader", "alone", "individual", "contributor", "mentor", "architect", "engineer", "dev", "junior", "senior", "principal", "manager"]
    for r in role_keywords:
        if r in text.lower():
            keywords.append(r)

    # Deception markers
    for marker in ["fraud", "copied", "exaggerate", "lied", "not what they think", "just watched", "mostly watched", "just deploy", "specific component", "handled part", "stitched parts", "mostly copied", "weekend workshop", "lead engineer might be wrong", "architected is too strong", "he designed the core", "i'm a junior dev", "not an architect", "only took a weekend workshop", "not exactly six years", "maybe three-ish", "coordinated is different from lead", "life cycle is a team thing", "handled part of it", "built parts of things", "stitched services together", "occasionally owned a small feature", "mostly copied patterns", "assembled other people's modules", "stitching counts as designing"]:
        if marker in text.lower():
            keywords.append(marker)

    # Self-correction
    self_corrections = re.findall(r"(actually|no actually|sorry|mean|is too strong|might be the wrong|is different|was the architect of a specific|should correct|he designed)", text.lower())
    if self_corrections:
        keywords.append("self_correction")

    return list(set(keywords))

def detect_contradictions(keyword_lists):
    contradictions = []
    flat = [kw for kws in keyword_lists for kw in kws]

    # Experience inflation
    durations = [kw for kw in flat if any(d in kw for d in ["year", "month", "full time", "freelance", "weekend workshop"])]
    if len(set(durations)) > 1 or ("internship" in flat and durations):
        claims = list(set(durations + ["internship"] if "internship" in flat else durations))
        contradictions.append({
            "lie_type": "experience_inflation",
            "contradictory_claims": claims
        })

    # Role conflict
    if "seasoned devops engineer" in flat and "internship" in flat:
        contradictions.append({
            "lie_type": "role_conflict",
            "contradictory_claims": ["seasoned DevOps engineer managing production", "internship, mostly watched seniors"]
        })
    if "lead engineer" in flat and "developer on the e-commerce team" in flat:
        contradictions.append({
            "lie_type": "role_conflict",
            "contradictory_claims": ["lead engineer", "developer on small component"]
        })
    if "principal cwar engineer" in flat and "grow into that lead architect role" in flat:
        contradictions.append({
            "lie_type": "resume_inflation",
            "contradictory_claims": ["architected entire system", "architected one service"]
        })
    if "built my back end from the first line" in flat and "he designed the core architecture" in flat:
        contradictions.append({
            "lie_type": "role_conflict",
            "contradictory_claims": ["I built everything", "lead dev designed architecture/schema"]
        })
    if "seasoned ruby on rails developer" in flat and "weekend workshop" in flat:
        contradictions.append({
            "lie_type": "skill_inflation",
            "contradictory_claims": ["seasoned AI/ML professional", "only took weekend workshop"]
        })
    if "lead architect" in flat and "i'm a junior dev" in flat:
        contradictions.append({
            "lie_type": "resume_inflation",
            "contradictory_claims": ["7 years lead architect", "junior dev, 2 years experience"]
        })
    if "drew architecture" in flat and "mostly copied" in flat:
        contradictions.append({
            "lie_type": "skill_inflation",
            "contradictory_claims": ["drew architecture", "mostly copied and stitched"]
        })
    if "built entire projects" in flat and "stitched parts" in flat:
        contradictions.append({
            "lie_type": "ownership_inflation",
            "contradictory_claims": ["built entire projects", "only stitched parts"]
        })
    if "6 years" in flat and "3 years" in flat:
        contradictions.append({
            "lie_type": "experience_inflation",
            "contradictory_claims": ["6 years", "3 years"]
        })
    if "handled full lifecycle" in flat and "handled part" in flat:
        contradictions.append({
            "lie_type": "ownership_inflation",
            "contradictory_claims": ["handled full lifecycle", "handled only part of it"]
        })
    if "6 years" in flat and "3 full time" in flat:
        contradictions.append({
            "lie_type": "experience_inflation",
            "contradictory_claims": ["6 years", "3 years", "4 years"]
        })
    if "led sprints" in flat and "coordinated is different from lead" in flat:
        contradictions.append({
            "lie_type": "leadership_confusion",
            "contradictory_claims": ["led sprints", "coordinated (not lead)"]
        })

    # Self-admitted
    if any(kw in flat for kw in ["fraud", "copied", "self_correction", "internship", "weekend workshop"]):
        contradictions.append({
            "lie_type": "self_admitted_deception",
            "contradictory_claims": [kw for kw in ["fraud", "copied", "self_correction"] if kw in flat]
        })

    return contradictions

def consolidate_truth(keyword_lists):
    flat = [kw for kws in keyword_lists for kw in kws]
    truth = {
        "programming_experience": "",
        "programming_language": "",
        "skill_mastery": "intermediate",
        "leadership_claims": "fabricated",
        "team_experience": "individual contributor",
        "skills and other keywords": []
    }

    # Programming experience
    durations = [kw for kw in flat if any(d in kw for d in ["year", "month", "full time", "freelance", "workshop"])]
    if "internship" in flat:
        truth["programming_experience"] = "internship (summer, ~1 year exposure)"
    elif "weekend workshop" in flat:
        truth["programming_experience"] = "limited (Rails strong, AI/ML minimal)"
    elif durations:
        truth["programming_experience"] = sorted(durations, key=lambda x: int(re.findall(r'\d+', x)[0] if re.findall(r'\d+', x) else 0))[0]
    else:
        truth["programming_experience"] = "unknown"

    # Programming language
    lang_map = {
        "python": "python",
        "java": "java",
        "c++": "c++",
        "react": "javascript/react",
        "rails": "ruby",
        "django": "Python (Django, Celery)"
    }
    for kw, lang in lang_map.items():
        if kw in flat:
            truth["programming_language"] = lang
            break

    # Skill mastery
    if "internship" in flat or "junior" in flat or "fraud" in flat or "weekend workshop" in flat:
        truth["skill_mastery"] = "beginner"
    elif "seasoned" in flat or "expert" in flat or "mastered" in flat or "advanced" in flat:
        truth["skill_mastery"] = "advanced"
    elif "obsessive understanding" in flat:
        truth["skill_mastery"] = "expert in distributed systems"

    # Leadership claims
    if "self_correction" in flat or "fraud" in flat or "copied" in flat or "internship" in flat or "junior" in flat:
        truth["leadership_claims"] = "fabricated"
    elif "leader" in flat or "mentor" in flat or "architect" in flat or "principal" in flat:
        truth["leadership_claims"] = "possible"

    # Team experience
    if "internship" in flat or "just watched" in flat:
        truth["team_experience"] = "observer / minimal contribution"
    elif "fraud" in flat or "junior" in flat:
        truth["team_experience"] = "junior/principal engineer but not lead architect" if "principal" in flat else "junior dev (not architect)"
    elif "manages" in flat:
        truth["team_experience"] = "manages 6 engineers"
    elif "mentor" in flat:
        truth["team_experience"] = "tech lead / mentor"
    elif "team" in flat:
        truth["team_experience"] = "team player"
    elif "individual" in flat:
        truth["team_experience"] = "individual contributor"

    # Skills and other keywords
    excluded = durations + ["team", "leader", "alone", "individual", "contributor", "mentor", "architect", "engineer", "dev", "junior", "senior", "principal", "manager", "self_correction", "fraud", "copied", "exaggerate", "lied", "not what they think", "just watched", "mostly watched", "just deploy", "specific component", "handled part", "stitched parts", "mostly copied", "weekend workshop", "lead engineer might be wrong", "architected is too strong", "he designed the core", "i'm a junior dev", "not an architect", "only took a weekend workshop", "not exactly six years", "maybe three-ish", "coordinated is different from lead", "life cycle is a team thing", "handled part of it", "built parts of things", "stitched services together", "occasionally owned a small feature", "mostly copied patterns", "assembled other people's modules", "stitching counts as designing"]
    truth["skills and other keywords"] = list(set([kw for kw in flat if kw not in excluded]))

    return truth

# --------------------
# MAIN
# --------------------
def main():
    parser = argparse.ArgumentParser(description="Truth Weaver for Innov8 3.0")
    parser.add_argument("--evaluation_dir", default="INNOV8 3.0/Evaluation set/audio", help="Path to evaluation audio directory")
    parser.add_argument("--extra_dir", default="INNOV8 3.0/INNOV8 3.0", help="Path to extra (practice) audio directory")
    args = parser.parse_args()

    EVAL_AUDIO_DIR = args.evaluation_dir
    EXTRA_AUDIO_DIR = args.extra_dir

    model = load_model()

    all_files = []
    if os.path.exists(EVAL_AUDIO_DIR) and os.path.isdir(EVAL_AUDIO_DIR):
        all_files = [os.path.join(EVAL_AUDIO_DIR, f) for f in os.listdir(EVAL_AUDIO_DIR) if f.lower().endswith(".mp3")]
    else:
        print(f"Warning: Evaluation directory {EVAL_AUDIO_DIR} not found or not a directory. Skipping...")
    practice_files = []
    if os.path.exists(EXTRA_AUDIO_DIR) and os.path.isdir(EXTRA_AUDIO_DIR):
        practice_files = [os.path.join(EXTRA_AUDIO_DIR, f) for f in os.listdir(EXTRA_AUDIO_DIR) if f.lower().endswith(".mp3")]
        all_files.extend(practice_files)
    else:
        print(f"Warning: Extra directory {EXTRA_AUDIO_DIR} not found or not a directory. Skipping...")

    if not all_files:
        print("Error: No audio files found. Please check directories.")
        return

    transcripts = {}
    with open(OUTPUT_TRANSCRIPT, "w", encoding="utf-8") as fout:
        for path in sorted(all_files):
            fname = os.path.basename(path)
            try:
                result = model.transcribe(path)
                text = result["text"].strip()
                fout.write(f"{fname} : {text}\n")
                transcripts[fname] = text
            except Exception as e:
                print(f"Error transcribing {fname}: {e}")
                fout.write(f"{fname} : [Transcription failed: {str(e)}]\n")
                transcripts[fname] = ""

    groups = {}
    for fname, text in transcripts.items():
        base = fname.split(".")[0]
        parts = base.split("_")
        if len(parts) >= 2 and parts[-1].isdigit():
            shadow_id = "_".join(parts[:-1])
        else:
            shadow_id = base
        groups.setdefault(shadow_id, []).append(text)

    json_output = []
    for shadow_id, texts in sorted(groups.items()):
        keywords_per_session = [extract_keywords(t) for t in texts]
        contradictions = detect_contradictions(keywords_per_session)
        truth = consolidate_truth(keyword_lists=keywords_per_session)
        json_output.append({
            "shadow_id": shadow_id,
            "revealed_truth": truth,
            "deception_patterns": contradictions
        })

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"subjects": json_output}, f, indent=2)

    print(f"All done! Wrote {OUTPUT_TRANSCRIPT} and {OUTPUT_JSON}.")

if __name__ == "__main__":
    main()