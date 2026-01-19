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

def format_scores_context(rubric_scores,rubric):
    """Format the scores the student received (optional context)"""
    text = "\n**Student Scores:**\n"
    for criterion in rubric:
        code = criterion['code']
        score = rubric_scores.get(code,0)
        text += f"- {code}: {score}/{criterion['max_score']}\n"
    return text

def create_training_example(record, include_scores=False):
    """ 
    Convert a record from your generator into a training example

    Args:
        record: Dictionary from records.json
        include_scores: Whether to include the student's scores in the prompt (experiment with True/False)

    Returns:
        Dictionary with 'instruction' and 'response' fields
    """
    # Build the instruction prompt
    rubric_text = format_rubric_for_prompt(record['rubric'])

    instruction = f"""You are a university teaching assistant providing constructive feedback on student answers.

**Course:** {record['course']}
**Question:** {record['prompt']}

{rubric_text}
**Student Answer:** {record['student_answer']}"""

    # Optionally include scores (for experiments)
    if include_scores:
        instruction += format_scores_context(record['rubric_scores'],record['rubric'])
    
    instruction += "\n\n**Provide brief, constructive feedback (2-3 sentences):"

    # The response is the teacher's feedback
    response = record['instructor_feedback']['short_comment']

    return {
        'instruction': instruction,
        'response': response,
        'metadata': {
            'question_id': record['question_id'],
            'course': record['course'],
            'grade_category': record['grade_category'],
            'overall_score': record['instructor_feedback']['overall_score'],
            'max_score': record['instructor_feedback']['max_overall']
        }
    }

def prepare_dataset(json_path, test_size=0.15, val_size=0.15, include_scores=False):
    """
    Load generated data and split into train/val/test

    Args:
        json_path: Path to records.json
        test_size: Proportion for test set
        val_size: Proportion for validation set
        include_scores: Whether to include scores in prompts

    Returns:
        train_dataset, val_dataset, test_dataset (HuggingFace Dataset objects)
    """

    # Load data
    with open(json_path,'r') as f:
        records = json.load(f)

    print(f"Loaded {len(records)} records")

    # Convert to training examples
    examples = [create_training_example(r, include_scores) for r in records]

    # Shuffle
    random.shuffle(examples)

    # Split: train/temp -> temp: val/test
    train_examples, temp_examples = train_test_split(
        examples,
        test_size=(test_size + val_size),
        random_state=42
    )

    val_examples, test_examples = train_test_split(
        temp_examples,
        test_size=(test_size / (test_size + val_size)),
        random_state=42
    )

    print(f"Split: {len(train_examples)} train, {len(val_examples)} val, {len(test_examples)} test")

    # Convert to HuggingFace Datasets
    train_dataset = Dataset.from_list(train_examples)
    val_dataset = Dataset.from_list(val_examples)
    test_dataset = Dataset.from_list(test_examples)

    return train_dataset , val_dataset , test_dataset

def save_datasets(train_ds,val_ds, test_ds, output_dir='./data_processed'):
    """Save processed datasets"""
    import os
    os.makedirs(output_dir, exist_ok=True)

    train_ds.to_json(f'{output_dir}/train.json')
    val_ds.to_json(f'{output_dir}/val.json')
    test_ds.to_json(f'{output_dir}/test.json')

    print(f"Saved datasets to {output_dir}/")

if __name__ == '__main__':
    # Prepare datasets
    train_ds , val_ds , test_ds = prepare_dataset(
        'output/records.json',
        include_scores=False 
    )

    # Save
    save_datasets(train_ds, val_ds , test_ds)

    # Show example
    print("\n" + "="*80)
    print("SAMPLE TRAINING EXAMPLE")
    print("="*80)
    print("\nINSTRUCTION:")
    print(train_ds[0]['instruction'])
    print("\nRESPONSE:")
    print(train_ds[0]['response'])
    print("="*80)