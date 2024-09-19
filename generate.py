import os
import time
import json
import glob
import torch
import datetime
import argparse
import warnings
import numpy as np
from tqdm import tqdm
from utils import *
from math_utils import *
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='GSM8K', type=str)
    parser.add_argument('--model', default='Llama3', type=str)
    parser.add_argument('--k', default=40, type=int)
    args = parser.parse_args()
    task = args.task
    model = args.model

    with open(f"./test_data/{task}.json", "r") as f:
        test_samples = json.load(f)

    if model == "Llama3":
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1000, stop_token_ids=[tokenizer.eos_token_id])
        llm = LLM(model=model_id)
        prompts = [f"<|start_header_id|>user<|end_header_id|>\n\n{i['question']} Place your answer in \\boxed{{}}.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nStep 1:" for i in test_samples]

        for _ in range(args.k):
            outputs = llm.generate(prompts, sampling_params)
            results = []
            for output in outputs:
                results.append(output.outputs[0].text)

            results = []
            for i in range(len(test_samples)):
                tmp = {}
                tmp['question'] = test_samples[i]['question']
                tmp['gold_answer'] = test_samples[i]['gold_answer']
                tmp['reasoning'] = "Step 1:" + results[i]  
                if task in ["MATH"]:
                    tmp['pred'] = parse_math_boxed(results[i])     
                if task in ["GSM8K", "SVAMP"]:
                    tmp['pred'] = parse_boxed(results[i])
                elif task in ["MMLU", "SAT"]:
                    tmp['pred'] = get_choice(results[i])
                results.append(tmp)
                
            acc = evaluate_math(results)
            timestamp = datetime.datetime.now().strftime('%m%d_%H%M')

            filename = f"./pred/{task}/{model}/{acc}_{timestamp}.json"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "w") as f:
                json.dump(results, f)