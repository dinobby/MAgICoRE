c_postfix = """### Question:\n{question}\n\n### Current Solution:\n{cur_sol}\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nLet's review the steps with lower scores and identify the errors."""
r_postfix = """### Question:\n{question}\n\n### Current Solution:\n{cur_sol}\n\n### Teacher Feedback:\n{feedback}\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n### Revised Solution:\n\nLet's review the solution and the feedback, and then refine the current solution.\nAfter reviewing the solution and feedback, let's solve this math problem again."""

llama_reviewer_prompt = """<|start_header_id|>user<|end_header_id|>

Your task is to provide step-by-step feedback to the current solution.
You will be given a math problem and a current solution, along with the scores for each step based on its correctness.
- You will find (Score: n/10) at the end of each step. 
- The maximum (best) score is 10, which means that this step is 100% correct (and 0% incorrect).
- The minimum (worst) score is 0, which means that this step is 100% incorrect (and 0% correct).
- Pay attention to the steps having scores lower than 6, and carefully identify the errors in those steps.
- Provide your explanation of the error and how it can be fixed. DO NOT propose a new solution, just the explanation.

### Question:
{icl_question}

### Current Solution:
{icl_solution}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Let's review the steps with lower scores and identify the errors.

{icl_feedback}

<|start_header_id|>user<|end_header_id|>

"""

# ==========================================================================================================================

llama_refiner_prompt = """<|start_header_id|>user<|end_header_id|>

Your task is to fix the error in the given solution, based on the teacher's feedback.
- After reviewing the solution and feedback, provide a better solution to fix the mistakes you found.
- Your final answer must be in A, B, C, or D. Place your final answer in \\boxed{{}}.

### Question:
{icl_question}

### Current Solution:
{icl_solution}

### Teacher Feedback:
{icl_feedback}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

### Revised Solution:

Let's review the solution and the feedback, and then refine the current solution.
After reviewing the solution and feedback, let's solve this math problem again.

{icl_refined_solution}

<|start_header_id|>user<|end_header_id|>

"""

# ====================================================================================

gpt_reviewer_prompt = """Your task is to provide step-by-step feedback to the current solution.
You will be given a math problem and a current solution, along with the scores for each step based on its correctness.
- You will find (Score: n/10) at the end of each step. 
- The maximum (best) score is 10, which means that this step is 100% correct (and 0% incorrect).
- The minimum (worst) score is 0, which means that this step is 100% incorrect (and 0% correct).
- Pay attention to the steps having scores lower than 6, and carefully identify the errors in those steps.
- Provide your explanation of the error and how it can be fixed. DO NOT propose a new solution, just the explanation.

### Question:
{icl_question}

### Current Solution:
{icl_solution}

Let's review the steps with lower scores and identify the errors.

{icl_feedback}

"""

# ==========================================================================================================================

gpt_refiner_prompt = """Your task is to fix the error in the given solution, based on the teacher's feedback.
- After reviewing the solution and feedback, provide a better solution to fix the mistakes you found.
- Your final answer must be in A, B, C, or D. Place your final answer in \\boxed{{}}.

### Question:
{icl_question}

### Current Solution:
{icl_solution}

### Teacher Feedback:
{icl_feedback}

### Revised Solution:

Let's review the solution and the feedback, and then refine the current solution.
After reviewing the solution and feedback, let's solve this math problem again.

{icl_refined_solution}

"""
