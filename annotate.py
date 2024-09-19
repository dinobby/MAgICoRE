import time
import json
import glob
import random
import argparse
import warnings
import numpy as np
from utils import *
from math_utils import *
from models import ORM, PRM
from tqdm import tqdm
from scipy.stats import entropy
from collections import Counter 

import torch
import accelerator
from transformers import AutoTokenizer, pipeline
from transformers import AutoModelForCausalLM

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='GSM8K', type=str)
    parser.add_argument('--model', default='Llama3', type=str)
    parser.add_argument('--k', default=40, type=int)
    parser.add_argument('--batch_size', default=10, type=int)
    
    args = parser.parse_args()
    task = args.task
    model_name = args.model
    batch_size = args.batch_size
    files = glob.glob(f"./pred/{task}/{model_name}/0.*.json")
    samples = merge_samples(enumerate(files), args.k)

    orm = ORM()
    prm = PRM()

    num_correct = 0
    for sample in samples:
        if is_math_correct(sample['majority'], sample['gold_answer']):
            num_correct += 1
    print("majority acc: ", round((num_correct / len(samples)), 4))

    for sample in tqdm(samples):
        question = sample['question']
        reasoning = sample['reasoning']
        pred = sample['pred']
        gold_answer = sample['gold_answer']
        orm_scores, prm_scores = [], []
        steps_with_scores = []

        for b in range(0, len(reasoning), batch_size):
            orm_score = orm.get_outcome_rewards(question, reasoning[b:b+batch_size])
            orm_scores.extend(orm_score)

        for r, p in zip(reasoning, pred): 
            r = prm.add_step_lines(r)
            step_with_score = prm.add_pr_to_question(question, r)
            steps_with_scores.append(step_with_score)
            if p:
                prm_score = prm.get_process_reward(question, prm.reformat_for_prm_CoT(r)).tolist()
                prm_scores.append(prm_score)
            else:
                prm_scores.append([0])
                continue

        sample['step_with_score'] = steps_with_scores
        sample['orm_scores'] = orm_scores
        sample['prm_scores'] = prm_scores

    with open(f"./pred/{task}/{model_name}/annotated_{args.k}.json", "w") as f:
        json.dump(samples, f)