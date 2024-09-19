import io
import re
import ast
import sys
import time
import json
import openai
import random
import backoff
import numpy as np
import pandas as pd
from math_utils import is_math_correct
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from collections import Counter
from openai.error import RateLimitError, APIError, ServiceUnavailableError, APIConnectionError, InvalidRequestError

# fill in yours
openai.api_type = ""
openai.api_base = ""
openai.api_version = ""
openai.api_key = ""
openai.azure_endpoint = ""

@backoff.on_exception(backoff.constant, (KeyError, RateLimitError, APIError, ServiceUnavailableError, APIConnectionError), interval=20, max_tries=5) 
def gpt_generate(model, prompt):
    assert model in ["gpt3", "gpt4"]
    if model == "gpt3":
        engine = "gpt-35-turbo-1106"
    elif model == "gpt4":
        engine = "gpt-4-1106"
    else:
        print("invalid model name.")
        return None
    contexts = [{"role": "user", "content": f"{prompt}"}]
    try:
        completion = openai.ChatCompletion.create(
                  engine=engine,
                  messages=contexts,
                  temperature=1)
    except InvalidRequestError:
        return None
    except Exception as e:
        print(e)
    output = completion['choices'][0]['message']['content']
    return output

def std_normalize(data, mean, std):
    if type(data) is list:
        data = np.array(data)
    output = (data - mean) / std
    return output

def get_choice(text):
    if text:
        match = re.findall(r'([A|B|C|D|E])\)', text)
    if not match: 
        match = re.findall(r'([A|B|C|D|E])', text)
    return match[-1] if match else "N/A"

def merge_samples(files, k):
    CoT_results = {idx: json.load(open(f)) for idx, f in files}
    samples = []
    for i in tqdm(range(len(CoT_results[0]))):
        tmp = {}
        tmp['pred'] = []
        tmp['reasoning'] = []
        for k, CoT_result in CoT_results.items():  
            tmp['question'] = CoT_result[i]['question']
            tmp['gold_answer'] = CoT_result[i]['gold_answer']
            if len(tmp['pred']) < k:
                tmp['pred'].append(CoT_result[i]['pred'])
                tmp['reasoning'].append(CoT_result[i]['reasoning'])

        majority = Counter(tmp['pred']).most_common(1)[0][0]
        tmp['majority'] = majority
        samples.append(tmp)
    return samples

def get_top_k_values(data, k):
    return dict(sorted(data.items(), key=lambda item: item[1], reverse=True)[:k])

def get_top_k(pred, reasoning, orm_scores, prm_scores, k):
    prm_scores = [np.prod(i) for i in prm_scores]
    combined = list(zip(pred, reasoning, orm_scores, prm_scores))
    sorted_list = sorted(combined, key=lambda x: x[2] + x[3], reverse=True)
    top_k_pred = [item[0] for item in sorted_list[:k]]
    top_k_reasoning = [item[1] for item in sorted_list[:k]]
    top_k_orm = [item[2] for item in sorted_list[:k]]
    top_k_prm = [item[3] for item in sorted_list[:k]]
    return top_k_pred, top_k_reasoning, top_k_orm, top_k_prm