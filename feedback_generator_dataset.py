"""
Project: Rubric Aligned Feedback Generation for Short Answers using Small Language Models
File: rubric_feedback_dataset_generator.py

What this file contains
- A ready-to-run Python script that programmatically builds a synthetic yet realistic dataset of
  rubric-aligned short-answer grading instances suitable for training/evaluating feedback
  generation systems.

Default dataset produced by running this script
- 4 short-answer questions (introductory CS / AI topics)
- 150 student answers per question (total 600 instances)
- Each question has a structured rubric with 3 criteria
- Each instance contains: student_answer, rubric (criteria + score per criterion),
  instructor_feedback (short + detailed), and metadata

Why synthetic?
- Ethical: avoids sharing real student data
- Flexible: you can control difficulty, number of examples, and error patterns
- Good starting point for model development and ablation studies

How the generator works (high level)
1. For each question we provide a handful of "seed" ideal answers and a list of common
   mistake templates.
2. For each required student instance the script samples whether the answer is
   Excellent / Good / Fair / Poor / Incorrect / Incomplete and applies templates to
   generate realistic short answers.
3. Scores per rubric criterion are produced using simple deterministic rules tied to the
   answer category; small random noise is added for realism.
4. Instructor feedback is generated from templates aligned to the rubric (so it is
   "rubric-aligned").

How to run
- Python 3.8+
- Run: python feedback_generator_dataset.py
- Outputs saved to ./output/records.json and ./output/records_sample.csv

Customize
- Adjust NUM_QUESTIONS, INSTANCES_PER_QUESTION at the top.
- Edit QUESTIONS list to add/remove question seeds and rubrics.

Ethics note
- If you replace synthetic answers with real student submissions, obtain consent and
  appropriate ethics approval. Remove any PII.

"""

import os
import json
import csv
import random
import uuid
from datetime import datetime

# ------------------------- CONFIG ---------------------------------
NUM_QUESTIONS = 4
INSTANCES_PER_QUESTION = 150  # set to 100-200 as suggested in your project
OUTPUT_DIR = "output"
SEED = 42
random.seed(SEED)

# ------------------------- HELPERS --------------------------------

def uid():
    return str(uuid.uuid4())


def clamp(x, a, b):
    return max(a, min(b, x))


def jitter_score(base, jitter=1, min_score=0, max_score=5):
    return clamp(int(round(base + random.uniform(-jitter, jitter))), min_score, max_score)


# ------------------------- RUBRIC + QUESTIONS ----------------------
# Each question entry: id, course, prompt, seeds, rubric (list of criteria dicts)

QUESTIONS = [
    {
        "id": "q1",
        "course": "Intro to Machine Learning",
        "prompt": "What is overfitting in machine learning, and how can cross-validation help reduce it? (2-3 sentences)",
        "seeds": [
            "Overfitting happens when a model learns noise or patterns specific to the training data and fails to generalize to new data. Cross-validation helps by splitting data and validating across folds so we can detect models that don't generalize and select hyperparameters accordingly.",
            "When a model performs well on training data but poorly on unseen data, it is overfitting. Cross-validation estimates performance on unseen data and helps choose models or regularization settings to avoid overfitting."
        ],
        "mistakes": [
            "Describes overfitting vaguely as 'model does bad on test' without saying it learns noise.",
            "Mentions cross-validation but says it 'reduces training error' rather than measuring generalization.",
            "Confuses cross-validation with data augmentation or early stopping.",
            "Gives only definition, no method to reduce it."
        ],
        "rubric": [
            {"code": "C1", "criterion": "Definition correctness", "max_score": 5, "description": "Clear definition describing learning noise/poor generalization"},
            {"code": "C2", "criterion": "Cross-validation explanation", "max_score": 5, "description": "Explains how cross-validation helps detect or reduce overfitting"},
            {"code": "C3", "criterion": "Conciseness and clarity", "max_score": 5, "description": "Answer is concise (2-3 sentences) and uses correct terminology"}
        ]
    },
    {
        "id": "q2",
        "course": "Data Structures",
        "prompt": "Briefly compare BFS and DFS. When would you use one over the other? (2-3 sentences)",
        "seeds": [
            "BFS explores neighbors level by level and is useful for finding shortest paths in unweighted graphs. DFS goes deep along a branch before backtracking and is useful for topological ordering or searching for any path when memory is limited.",
            "Use BFS when you need shortest path or level information; use DFS for tasks like cycle detection or when you want to explore deep structure with less memory overhead."
        ],
        "mistakes": [
            "Says BFS is always faster than DFS or vice versa.",
            "Mixes up use-cases (e.g., says DFS finds shortest paths).",
            "Only describes one algorithm but not the other."
        ],
        "rubric": [
            {"code": "C1", "criterion": "Algorithm characteristics", "max_score": 5, "description": "Mentions traversal order and core property (level-order vs depth)"},
            {"code": "C2", "criterion": "Use-case justification", "max_score": 5, "description": "Gives correct reasons for choosing BFS or DFS"},
            {"code": "C3", "criterion": "Concise comparison", "max_score": 5, "description": "Clear, short comparison in 2-3 sentences"}
        ]
    },
    {
        "id": "q3",
        "course": "Databases",
        "prompt": "What does ACID mean in databases? Provide a short explanation of each property. (List and 1-line explanation)",
        "seeds": [
            "ACID stands for Atomicity, Consistency, Isolation, Durability. Atomicity means transactions are all-or-nothing; Consistency means DB moves between valid states; Isolation ensures concurrent transactions don't interfere; Durability ensures committed changes persist.",
            "Atomicity: either all operations of a transaction happen or none. Consistency: DB constraints hold before and after transactions. Isolation: concurrent transactions appear serial. Durability: once committed, data survive crashes."
        ],
        "mistakes": [
            "Mixes up Isolation and Consistency, or gives vague descriptions.",
            "Forgets one of the properties.",
            "Gives overly technical answer beyond short scope."
        ],
        "rubric": [
            {"code": "C1", "criterion": "Coverage of properties", "max_score": 5, "description": "Lists all four ACID properties correctly"},
            {"code": "C2", "criterion": "Correctness of explanations", "max_score": 5, "description": "Each property is explained correctly in one line"},
            {"code": "C3", "criterion": "Brevity and clarity", "max_score": 5, "description": "Concise list-style explanations"}
        ]
    },
    {
        "id": "q4",
        "course": "Computer Networks",
        "prompt": "Explain the TCP three-way handshake in one or two sentences.",
        "seeds": [
            "TCP uses SYN, SYN-ACK, ACK messages: client sends SYN, server replies SYN-ACK, client replies ACK, establishing a reliable connection. This ensures both sides are ready and agree on initial sequence numbers.",
            "A three-way handshake is: client SYN, server SYN-ACK, client ACK; it's used to synchronize sequence numbers and establish a TCP connection."
        ],
        "mistakes": [
            "Says handshake uses SYN, ACK only (missing SYN-ACK).",
            "Confuses UDP with TCP.",
            "Mentions extra steps not part of the handshake."
        ],
        "rubric": [
            {"code": "C1", "criterion": "Sequence correctness", "max_score": 5, "description": "Mentions SYN, SYN-ACK, ACK in the right order"},
            {"code": "C2", "criterion": "Purpose explanation", "max_score": 5, "description": "Explains why it's done (sync seq numbers, ensure readiness)"},
            {"code": "C3", "criterion": "Conciseness", "max_score": 5, "description": "Explanation within 1-2 short sentences"}
        ]
    }
]

# ------------------------- TEXT TEMPLATES --------------------------
GRADE_CATEGORIES = [
    ("excellent", 0.7),
    ("good", 0.15),
    ("fair", 0.08),
    ("poor", 0.05),
    ("incorrect", 0.02)
]

FEEDBACK_TEMPLATES = {
    "definition_missing": [
        "You described the topic but missed the key definition — make sure to include that next time.",
        "Good attempt, but the core definition was missing or unclear. Add a clear definition for full marks."
    ],
    "partial_correct": [
        "You partially answered the question — correct on some points but missed: {missed}.",
        "Partly correct; to improve, expand on: {missed}."
    ],
    "minor_fix": [
        "Small fix needed: {fix}. Then your answer will be complete.",
        "Minor correction — {fix}. Good otherwise."
    ],
    "excellent_short": [
        "Clear and correct—well done.",
        "Excellent answer; concise and accurate."
    ],
    "incorrect_short": [
        "The answer is incorrect or misunderstands the concept. Review {topic} and try again.",
        "Incorrect: there is a misunderstanding about {topic}. Revise the core concept."
    ]
}

# ------------------------- GENERATION LOGIC ------------------------


def pick_grade_category():
    r = random.random()
    cum = 0
    for cat, prob in GRADE_CATEGORIES:
        cum += prob
        if r <= cum:
            return cat
    return GRADE_CATEGORIES[-1][0]


def mutate_answer(seed, mistakes, category):
    """Produce a mutated answer string given a seed and a category label."""
    if category == "excellent":
        # maybe paraphrase and sometimes add a short example
        s = seed
        if random.random() < 0.3:
            s = paraphrase(seed)
        return s

    if category == "good":
        s = paraphrase(seed)
        # remove one small detail sometimes
        if random.random() < 0.5 and mistakes:
            m = random.choice(mistakes)
            s = remove_detail(s)
            s += " " + random.choice([m])
        return s

    if category == "fair":
        # keep partial correctness, may mix up parts
        s = paraphrase(seed)
        s = shorten(s)
        if random.random() < 0.6 and mistakes:
            s += " " + random.choice(mistakes)
        return s

    if category == "poor":
        # short, vague, or partially wrong
        s = random.choice(mistakes) if mistakes else shorten(seed)
        if random.random() < 0.4:
            s = "I think " + s
        return s

    if category == "incorrect":
        # an incorrect statement
        s = random.choice(mistakes) if mistakes else "Incorrect description"
        # swap some keywords to make it look plausible
        s = swap_keywords(s)
        return s

    return seed


# Simple string transformation helpers

def paraphrase(s):
    # naive paraphrase: reorder clauses, swap words
    parts = s.split(',')
    if len(parts) > 1 and random.random() < 0.6:
        random.shuffle(parts)
        s = ','.join(p.strip() for p in parts)
    # replace some words
    s = s.replace('model', random.choice(['classifier', 'fit model']))
    s = s.replace('training data', random.choice(['the data used for training', 'training set']))
    return s


def shorten(s):
    # keep first sentence only
    if '.' in s:
        return s.split('.')[0] + '.'
    return s


def remove_detail(s):
    # remove phrases like "so we can detect" or clauses
    s = s.replace('so we can detect models that don\'t generalize and select hyperparameters accordingly', '')
    s = s.replace('and helps choose models or regularization settings to avoid overfitting', '')
    return s


def swap_keywords(s):
    s = s.replace('cross-validation', 'data augmentation')
    s = s.replace('SYN-ACK', 'ACK')
    return s


def score_from_category(category, rubric):
    # base mapping for category to mean score fraction of max
    mapping = {
        'excellent': 0.9,
        'good': 0.75,
        'fair': 0.5,
        'poor': 0.25,
        'incorrect': 0.05
    }
    base_frac = mapping.get(category, 0.5)
    scores = {}
    for crit in rubric:
        base = base_frac * crit['max_score']
        scores[crit['code']] = jitter_score(base, jitter=1.2, min_score=0, max_score=crit['max_score'])
    return scores


def generate_feedback_text(scores, rubric, question_prompt, student_answer, category):
    # Build a short feedback aligned to rubric by checking low scoring criteria
    low = []
    missing = []
    fixes = []
    for crit in rubric:
        code = crit['code']
        sc = scores.get(code, 0)
        if sc <= max(1, int(0.3 * crit['max_score'])):
            low.append(crit['criterion'])
            missing.append(crit['criterion'])
        elif sc < crit['max_score'] and sc < int(0.7 * crit['max_score']):
            fixes.append(crit['criterion'])

    # Compose short feedback
    if category == 'excellent' and not low:
        short = random.choice(FEEDBACK_TEMPLATES['excellent_short'])
        detail = "".join(["Great: "+c+". " for c in [r['criterion'] for r in rubric]])
        return short, detail

    if category == 'incorrect' or (len(low) == len(rubric)):
        short = random.choice(FEEDBACK_TEMPLATES['incorrect_short']).format(topic=question_prompt.split(':')[0])
        detail = "You should review the core concepts and definitions for this topic."
        return short, detail

    parts = []
    if low:
        miss = ', '.join(low[:2])
        parts.append(random.choice(FEEDBACK_TEMPLATES['partial_correct']).format(missed=miss))
    if fixes:
        fix = ', '.join(fixes[:2])
        parts.append(random.choice(FEEDBACK_TEMPLATES['minor_fix']).format(fix=fix))
    short = ' '.join([p for p in parts])
    if not short:
        short = random.choice(FEEDBACK_TEMPLATES['excellent_short'])
    detail = "Details: " + "; ".join([f"{c['code']}({c['criterion']}): {scores[c['code']]}/{c['max_score']}" for c in rubric])
    return short, detail


# ------------------------- MAIN GENERATION ------------------------


def build_dataset(questions, instances_per_q):
    records = []
    for q in questions[:NUM_QUESTIONS]:
        for i in range(instances_per_q):
            category = pick_grade_category()
            seed = random.choice(q['seeds'])
            ans = mutate_answer(seed, q.get('mistakes', []), category)
            scores = score_from_category(category, q['rubric'])
            short_fb, detailed_fb = generate_feedback_text(scores, q['rubric'], q['prompt'], ans, category)

            record = {
                'id': uid(),
                'question_id': q['id'],
                'course': q['course'],
                'prompt': q['prompt'],
                'student_answer': ans,
                'grade_category': category,
                'rubric_scores': scores,
                'rubric': q['rubric'],
                'instructor_feedback': {
                    'short_comment': short_fb,
                    'detailed_comment': detailed_fb,
                    'overall_score': sum(scores.values()),
                    'max_overall': sum([c['max_score'] for c in q['rubric']])
                },
                'created_at': datetime.utcnow().isoformat() + 'Z'
            }
            records.append(record)
    return records


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    records = build_dataset(QUESTIONS, INSTANCES_PER_QUESTION)

    out_json = os.path.join(OUTPUT_DIR, 'records.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    # also write a small CSV sample for quick inspection
    out_csv = os.path.join(OUTPUT_DIR, 'records_sample.csv')
    sample_fields = ['id', 'question_id', 'course', 'student_answer', 'grade_category', 'instructor_feedback_short', 'overall_score', 'max_overall']
    with open(out_csv, 'w', newline='', encoding='utf-8') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=sample_fields)
        writer.writeheader()
        for r in records[:min(200, len(records))]:
            writer.writerow({
                'id': r['id'],
                'question_id': r['question_id'],
                'course': r['course'],
                'student_answer': r['student_answer'][:200].replace('\n', ' '),
                'grade_category': r['grade_category'],
                'instructor_feedback_short': r['instructor_feedback']['short_comment'],
                'overall_score': r['instructor_feedback']['overall_score'],
                'max_overall': r['instructor_feedback']['max_overall']
            })

    print(f"Dataset generated: {out_json} ({len(records)} records)")
    print(f"CSV sample: {out_csv}")
