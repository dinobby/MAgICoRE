import time
import json
import glob
import random
import argparse
import warnings
import numpy as np
from tqdm.notebook import tqdm
from utils import *
from prompt import *
from math_utils import *
from models import ORM, PRM
from scipy.stats import entropy
from collections import Counter 
from icl_samples import icl_sample
import torch
import accelerator
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from vllm import LLM, SamplingParams

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='GSM8K', type=str)
    parser.add_argument('--model', default='Llama3', type=str)
    parser.add_argument('--k', default=40, type=int)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--iter', default=3, type=int)

    args = parser.parse_args()
    task = args.task
    model_name = args.model
    batch_size = args.batch_size
    iters = args.iter

    orm = ORM(device=1)
    prm = PRM(device=2)
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    vllm_tokenizer = AutoTokenizer.from_pretrained(model_id)
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, min_tokens=50, max_tokens=1000, stop_token_ids=[vllm_tokenizer.eos_token_id])
    llm = LLM(model=model_id, tensor_parallel_size=1)
    
    icl_question = icl_sample[task]['question']
    icl_solution = icl_sample[task]['solution']
    icl_feedback = icl_sample[task]['feedback']
    icl_refined_solution = icl_sample[task]['refined_solution']

    reviewer_icl = llama_reviewer_prompt.format(icl_question=icl_question, icl_solution=icl_solution, icl_feedback=icl_feedback)
    refinement_icl = llama_refiner_prompt.format(icl_question=icl_question, icl_solution=icl_solution, icl_feedback=icl_feedback, icl_refined_solution=icl_refined_solution)

    with open(f"./pred/{task}/{model_name}/annotated_{args.k}.json", "r") as f:
        samples = json.load(f)

    for sample in samples:
        sample['iter0_orm'] = sample['orm_scores']
        sample['iter0_prm'] = sample['prm_scores']
        sample['iter0_reasoning'] = sample['reasoning']
        sample['iter0_pred'] = sample['pred']

    all_scores = []
    for i in samples:
        all_scores.append(i['orm_scores'][0] + np.product(i['prm_scores'][0]))

    reward_avg = np.mean(all_scores)
    reward_std = np.std(all_scores)

    for n in range(iters):
        num_need_refinement = 0
        for sample in samples:
            assert len(sample[f'iter{n}_pred']) == len(sample[f'iter{n}_orm']) == len(sample[f'iter{n}_prm'])
            pred, orms, prms = sample[f'iter{n}_pred'], sample[f'iter{n}_orm'], sample[f'iter{n}_prm']

            rewards = []
            for p, o, s in zip(pred, orms, prms):
                if is_math_correct(p, sample['majority']):
                    s = np.product(s)
                    reward = (o+s)
                    reward = std_normalize(reward, reward_avg, reward_std)
                    rewards.append(reward)
            
            # condition 1
            if np.mean(rewards) < 0.0:
                sample['cond1'] = 'false'
            else:
                sample['cond1'] = 'true'

            # condition 2
            orm_votes, prm_votes = {}, {}
            for p, o, s in zip(pred, orms, prms):
                s = np.product(s)
                if p not in orm_votes:
                    orm_votes[p] = o
                else:
                    orm_votes[p] += o
                    
                if p not in prm_votes:
                    prm_votes[p] = s
                else:
                    prm_votes[p] += s
                    
            orm_pq = [count for item, count in orm_votes.items()]
            orm_uncertainty = torch.nn.functional.sigmoid(torch.tensor(entropy(orm_pq))).item() 
            orm_confidence = 2 * (1 - orm_uncertainty)

            prm_pq = [count for item, count in prm_votes.items()]
            prm_uncertainty = torch.nn.functional.sigmoid(torch.tensor(entropy(prm_pq))).item() 
            prm_confidence = 2 * (1 - prm_uncertainty)
            
            if sample['cond1'] == 'false' and (orm_confidence < 0.5 or prm_confidence < 0.5):
                sample['cond2'] = 'false'
                sample['need_refinement'] = 'yes'
                num_need_refinement += 1
            else:
                sample['cond2'] = 'true'
                sample['need_refinement'] = 'no'

        for sample in tqdm(samples):
            if sample['need_refinement'] == 'yes':
                tqdm.write(f"yes")
                question = sample['question']
                reasoning = sample[f'iter{n}_reasoning']
                pred = sample[f'iter{n}_pred']
                
                steps_with_scores = []
                for r, p in zip(reasoning, pred): 
                    r = prm.add_step_lines(r)
                    step_with_score = prm.add_pr_to_question(question, r)
                    steps_with_scores.append(step_with_score)

                reviewer_prompts = []
                for res in steps_with_scores:
                    reviewer_postfix = c_postfix.format(question=question, cur_sol=res)
                    reviewer_prompt = reviewer_icl + reviewer_postfix
                    reviewer_prompts.append(reviewer_prompt)

                outputs = llm.generate(reviewer_prompts, sampling_params, use_tqdm=False)
                feedback = [output.outputs[0].text for output in outputs]

                refine_prompts = []
                for res, fb in zip(steps_with_scores, feedback):
                    refine_postfix = r_postfix.format(question=question, cur_sol=res, feedback=fb)
                    refinement_prompt = refinement_icl + refine_postfix
                    refine_prompts.append(refinement_prompt)

                outputs = llm.generate(refine_prompts, sampling_params, use_tqdm=False)
                updated_sol = [output.outputs[0].text for output in outputs]
                if task in ["MATH"]:
                    updated_pred = [parse_math_boxed(i) for i in updated_sol]
                elif task in ["GSM8K", "SVAMP"]:
                    updated_pred = [parse_boxed(i) for i in updated_sol]
                elif task in ["MMLU", "SAT"]:
                    updated_pred = [get_choice(i) for i in updated_sol]

                prm_scores = []
                steps_with_scores = []
                for r, p in zip(updated_sol, updated_pred): 
                    r = prm.add_step_lines(r)
                    step_with_score = prm.add_pr_to_question(question, r)
                    steps_with_scores.append(step_with_score)
                    if p:
                        prm_score = prm.get_process_reward(question, prm.reformat_for_prm_CoT(r)).tolist()
                        prm_scores.append(prm_score)
                    else:
                        prm_scores.append([0])
                        continue
                        
                orm_scores = []
                for b in range(0, len(updated_sol), batch_size):
                    orm_score = orm.get_outcome_rewards(question, updated_sol[b:b+batch_size])
                    orm_scores.extend(orm_score)
                    
                sample[f'iter{n+1}_orm'] = orm_scores
                sample[f'iter{n+1}_prm'] = prm_scores
                sample[f'iter{n+1}_steps_with_scores'] = steps_with_scores
                sample[f'iter{n+1}_reasoning'] = updated_sol
                sample[f'iter{n+1}_pred'] = updated_pred   
            else:
                sample[f'iter{n+1}_orm'] = sample['iter0_orm']
                sample[f'iter{n+1}_prm'] = sample['iter0_prm']
                sample[f'iter{n+1}_reasoning'] = sample['iter0_reasoning']
                sample[f'iter{n+1}_pred'] = sample['iter0_pred']

        print(f"Evaluating for the {n} iteration")
        num_correct = 0
        for sample in samples:
            if sample['need_refinement'] == 'yes':
                reasoning = sample['iter0_reasoning'] + sample['iter1_reasoning']
                pred = sample[f'iter{n}_pred'] + sample[f'iter{n+1}_pred']
                prm_scores = sample[f'iter{n}_prm'] + sample[f'iter{n+1}_prm']
                orm_scores = sample[f'iter{n}_orm'] + sample[f'iter{n+1}_orm']
            else:
                reasoning = sample[f'iter{n}_reasoning']
                pred = sample[f'iter{n}_pred']
                prm_scores = sample[f'iter{n}_prm']
                orm_scores = sample[f'iter{n}_orm']
                
            pred, reasoning, orm_scores, prm_scores = get_top_k(pred, reasoning, orm_scores, prm_scores, args.k)
            sample[f'iter{n+1}_top_pred'] = pred
            sample[f'iter{n+1}_top_reasoning'] = reasoning
            sample[f'iter{n+1}_top_orm'] = orm_scores
            sample[f'iter{n+1}_top_prm'] = prm_scores
            
            votes = {}
            for p, o, s in zip(pred, orm_scores, prm_scores):
                if p:
                    s = np.product(s)
                    if p not in votes:
                        votes[p] = (o+s)
                    else:
                        votes[p] += (o+s)

            ans = max(votes, key=votes.get)    
            if is_math_correct(ans, sample['gold_answer']):
                num_correct += 1
        acc = round((num_correct / len(samples)), 3)    
        print(acc)

    with open(f"./pred/{task}/{model_name}/refined_{args.k}.json", "w") as f:
        json.dump(samples, f)