"""
prepare_data_for_training.py
Converts the generated dataset into a format suitable for fine-tuning the SLM.
"""

import json
import random
from datasets import Dataset
from sklearn.model_selection import train_test_split

def format_rubric_for_prompt(rubric):
    """Convert rubric list to readable text"""
    text = "**Marking Rubric:**\n"
    for criterion in rubric:
        text += f"-** [{criterion['code']}] {criterion['criterion']}** "
        text += f"({criterion['max_score']} marks): {criterion['description']}\n"
    return text
