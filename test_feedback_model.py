"""
test_feedback_model.py
Test your fine-tuned model and generate feedback
"""

import torch
import json
from unsloth import FastLanguageModel
from datasets import load_dataset

# ============================================================================
# LOAD MODEL
# ============================================================================

def load_finetuned_model(model_path="./feedback_model_phi3/lora_adapters"):
    """ Load your fine-tuned model"""

    model , tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    # Enable fast inference
    FastLanguageModel.for_inference(model)

    print(f"✓ Model loaded from {model_path}")
    return model, tokenizer    

# ============================================================================
# INFERENCE FUNCTIONS
# ============================================================================

def format_rubric_for_prompt(rubric):
    """Convert rubric to readable text"""
    text = "**Marking Rubric:**\n"
    for criterion in rubric:
        text += f" - **[{criterion['code']}] {criterion['criterion']}** "
        text += f"({criterion['max_score']} marks): {criterion['description']}\n"
    return text

def generate_feedback(model,tokenizer,question,rubric,student_answer,course=""):
    """
    Generate feedback for a student answer

    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer
        question: Question text
        rubric: List of rubric criteria dicts
        student_answer: Student's answer text
        course: Course name (optional)

    Returns:
        Generated feedback string
    """

    # Format rubric
    rubric_text = format_rubric_for_prompt(rubric)

    # Build instruction
    instruction = f"""You are a university teaching assistant providing constructive feedback on student answers."""

    if course:
        instruction += f"\n\n**Course:** {course}"
    
    instruction += f"""

**Question:** {question}

{rubric_text}
**Student Answer:** {student_answer}

Provide brief, constructive feedback (2-3 sentences):"""

    # Format as chat
    messages = [
        {"role": "user", "content": instruction}
    ]

    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.1,
        use_cache=True,
    )

    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract assistant's response (after the last </assistant/> tag)
    if "<|assistant|>" in response:
        feedback = response.split("<|assistant|>")[-1].strip()
    else:
        feedback = response.split(instruction)[-1].strip()

    return feedback

# ============================================================================
# EVALUATION ON TEST SET
# ============================================================================

def evaluate_on_test_set(model, tokenizer, test_path='./data_processed/test.json'):
    """
    Evaluate model on test set
    """

    print("\n" + "="*80)
    print("EVALUATING ON TEST SET")
    print("="*80)
    # Load test data
    test_dataset = load_dataset('json', data_files=test_path, split='train')
    print(f"✓ Loaded {len(test_dataset)} test examples")

    # Generate feedback for each
    results = []

    for i, example in enumerate(test_dataset):
        if i >=10:
            break

        print(f"\n--- Example {i+1} ---")

        # Parse instruction to get components
        instruction = example['instruction']

        # Extract question (simple parsing)
        question_start = instruction.find("**Question**") + len("**Question:**")
        question_end = instruction.find("**Marking Rubric**")
        question = instruction[question_start:question_end].strip()

        print(f"Question: {question[:100]}...")
        print(f"Expected: {example['response']}")
        
        # Generate (simplified - using parsed instruction directly)
        messages = [{"role": "user", "content": instruction}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs, max_new_tokens=256, temperature=0.7,
            do_sample=True, repetition_penalty=1.1
        )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "<|assistant|>" in generated:
            generated_feedback = generated.split("<|assistant|>")[-1].strip()
        else:
            generated_feedback = generated.split(instruction)[-1].strip()
        
        print(f"Generated: {generated_feedback}")
        
        results.append({
            'example_id': i,
            'expected': example['response'],
            'generated': generated_feedback
        })
    
    # Save results
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to evaluation_results.json")
    
    return results

# ============================================================================
# INTERACTIVE TESTING
# ============================================================================

def interactive_test(model, tokenizer):
    """
    Interactive mode to test custom inputs
    """
    
    print("\n" + "="*80)
    print("INTERACTIVE TESTING MODE")
    print("="*80)
    print("Enter question, rubric, and student answer to get feedback")
    print("Type 'quit' to exit\n")
    
    # Example rubric for testing
    example_rubric = [
        {
            "code": "C1",
            "criterion": "Definition correctness",
            "max_score": 5,
            "description": "Clear definition with key concepts"
        },
        {
            "code": "C2",
            "criterion": "Explanation completeness",
            "max_score": 5,
            "description": "Complete explanation of the concept"
        },
        {
            "code": "C3",
            "criterion": "Clarity and conciseness",
            "max_score": 5,
            "description": "Clear and concise answer"
        }
    ]
    
    while True:
        print("\n" + "-"*80)
        question = input("Question: ")
        if question.lower() == 'quit':
            break
        
        student_answer = input("Student answer: ")
        if student_answer.lower() == 'quit':
            break
        
        # Generate feedback
        feedback = generate_feedback(
            model, tokenizer,
            question=question,
            rubric=example_rubric,
            student_answer=student_answer
        )
        
        print(f"\n📝 Generated Feedback:\n{feedback}\n")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # Load model
    model, tokenizer = load_finetuned_model()
    
    # Evaluate on test set
    evaluate_on_test_set(model, tokenizer)
    
    # Interactive testing
    interactive_test(model, tokenizer)