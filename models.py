import re
import torch
import accelerator
from transformers import AutoModel, AutoTokenizer, pipeline, AutoModelForCausalLM

class ORM:
    def __init__(self, model_name="internlm/internlm2-7b-reward", device=0):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
                        "internlm/internlm2-7b-reward", 
                        device_map=f"cuda:{device}", 
                        torch_dtype=torch.float16, 
                        trust_remote_code=True
                    )
        
    def format_chat(self, question, answer):
        chat = [
                {"role": "user", "content": f"{question}"},
                {"role": "assistant", "content": f"{answer}"}
                ]
        return chat
    
    def get_outcome_reward(self, question, answer):
        chat = self.format_chat(question, answer)
        outcome_reward = self.model.get_score(self.tokenizer, chat)
        return outcome_reward
    
    def get_outcome_rewards(self, question, answers):
        chats = []
        for a in answers:
            chat = self.format_chat(question, a)
            chats.append(chat)
        r = self.model.get_scores(self.tokenizer, chats)
        outcome_rewards = torch.nn.functional.sigmoid(torch.tensor(r)).tolist()
        return outcome_rewards

class PRM:
    def __init__(self, model_name="peiyi9979/math-shepherd-mistral-7b-prm", device=1):     
        good_token = '+'
        bad_token = '-'
        step_tag = 'ки'

        self.prm_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.prm_tokenizer.pad_token = self.prm_tokenizer.eos_token
        self.candidate_tokens = self.prm_tokenizer.encode(f"{good_token} {bad_token}")[1:] # [648, 387]
        self.step_tag_id = self.prm_tokenizer.encode(f"{step_tag}")[-1] # 12902
        self.prm = AutoModelForCausalLM.from_pretrained(model_name,
                                                    torch_dtype=torch.bfloat16,
                                                    device_map=f'cuda:{device}').eval()

    def get_process_reward(self, question, reasoning):
        input_for_prm = [f"{question} {reasoning}"]
        input_id = self.prm_tokenizer(input_for_prm, padding=True, return_tensors="pt").input_ids.to(self.prm.device)
        with torch.no_grad():
            logits = self.prm(input_id).logits[:,:,self.candidate_tokens]
            scores = logits.softmax(dim=-1)[:,:,0] 
            step_scores = scores[input_id == self.step_tag_id]
        return step_scores

    def add_pr_to_question(self, question, reasoning):
        reasoning = self.reformat_for_prm_CoT(reasoning)
        prs = self.get_process_reward(question, reasoning).tolist()
        whole_text = self.add_process_reward(reasoning, prs)
        return whole_text

    def add_step_lines(self, text):
        text = text.replace("\n\n", "\n")
        text = re.sub(r"\n(Step [2-9])", r"\1", text).strip()
        return re.sub(r"(Step [2-9])", r"\n\n\1", text)

    def get_step_score_pair(self, chunk):
        pairs = re.findall(r"(.+?)\(Score: (0.\d+)\)", chunk, re.DOTALL)
        results = []
        for p in pairs:
            results.append((p[0], float(p[1])))
        return results

    def reformat_for_prm_CoT(self, text):
        text = text.strip()
        text = text.replace("\n\n", " ки\n")
        text = text + " ки"
        return text

    def add_process_reward(self, text, scores):
        parts = text.split('ки')
        res = []
        for part, score in zip(parts, scores):
            res.append(f"{part} (Score: {int(round(score, 1)*10)}/10)\n")
        formatted_text = "".join(res)
        return formatted_text
        
    def get_answer_out_of_boxed(self, text):
        text = text.split("I hope it is correct")[0]
        res = re.findall(r"final answer is (.*). ", text)
        return res[-1] if res else None