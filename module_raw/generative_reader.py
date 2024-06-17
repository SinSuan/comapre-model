"""
generative_reader.py from the raw code
"""
import os
import configparser
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm import trange
from .subfolder.context_prefix_tree import ContextPrefixTree

# config 讀取資訊
path_to_config = os.getenv('path_to_config')
config = configparser.ConfigParser()
# config.read('config.ini')
config.read(path_to_config)

model_path = config['model']['generativeQA']
num_beams = int(config['constant']['num_beams'])

# tokenizer = AutoTokenizer.from_pretrained(model_path)
# tokenizer.padding_side = "left"

# model = AutoModelForCausalLM.from_pretrained(
#     model_path,
#     device_map='auto',
#     torch_dtype=torch.float16,
#     #load_in_8bit=True
# )
# model.eval()

generation_config = GenerationConfig(
    max_new_tokens=32,
    num_beams = num_beams,
    num_return_sequences = num_beams,
)

def get_prompt(question, context):
    print(f"enter get_prompt")
    # print(f"\tcontext = {context}")
    prompt = f"<s>[INST] <<SYS>>\n\n請根據提供的問題，從提供的內文中尋找答案並回答，回答時只需要輸出答案，不需輸出其他資訊，如果從提供的內文無法找到答案，請回答\"無法回答\"。\n<</SYS>>問題:\n{question}\n\n內文:\n"

    context_start = len(prompt)
    context_end = context_start + len(context)

    prompt = prompt + context + "\n [/INST]答案:\n"
    print(f"exit get_prompt")
    return prompt, context_start, context_end


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='auto',
        torch_dtype=torch.float16,
        #load_in_8bit=True
    )
    model.eval()
    return tokenizer, model

def generative_reader(question: str, 
                    contexts: list[str],
                    batch_size: int = 1, 
                    max_length: int = 768,
                    only_return_best: bool = True) -> (list[str], list[str]):
    print(f"enter generative_reader")
    tokenizer, model = load_model()
    
    prompts = []
    context_starts = []
    context_ends = []

    # convert question and context to llama2 prompt format
    for context in contexts:
        print(f"context = {context}")
        prompt, context_start, context_end = get_prompt(question, context)
        # print(f"prompt = {prompt}")
        
        input_len = tokenizer(prompt, add_special_tokens=False, return_tensors="pt",).input_ids.size(1)
        cut_len = input_len - max_length
        while input_len > max_length: # recursive remove word from context while the length of input_ids is longer than max_length
            prompt, context_start, context_end = get_prompt(question, context[:-cut_len])
            input_len = tokenizer(prompt, add_special_tokens=False, return_tensors="pt",).input_ids.size(1)
            
            cut_len += 10
            print(f"cut_len = {cut_len}")
            print(f"input_len = {input_len}")
            print(f"max_length = {max_length}")
            print(f"context[:-cut_len] = {context[:-cut_len]}")
            print()
        print("after while loop")

        prompts.append(prompt)
        context_starts.append(context_start)
        context_ends.append(context_end)
    print("after loop: get_prompt")

    eos_token_id = tokenizer.eos_token_id
    num_return_sequences = 1 if not hasattr(generation_config, "num_return_sequences") else generation_config.num_return_sequences
    context_prefix_tree =  ContextPrefixTree(eos_token_id = tokenizer.eos_token_id, 
                                            allow_token_ids = [30267]) # 29871代表空白
    print("after init ContextPrefixTree")
    
    answers = []
    answer_contexts = []
    unable_to_answer_string = ["無法回答", "無法回", "無法", "無", "無法回答。", "無法回。", "無法。", "無。"]
    for i in trange(0, len(contexts), batch_size):
        print(f"i = {i}")
        inputs = tokenizer(prompts[i:i+batch_size],
                    padding="max_length",
                    truncation=True, 
                    max_length = max_length, 
                    add_special_tokens=False,
                    return_tensors="pt",
                    return_offsets_mapping=True).to(model.device)
        context_prefix_tree.generate_prefix_tree(inputs, context_starts[i:i+batch_size], context_ends[i:i+batch_size])
        context_prefix_tree.add_string_to_prefix_tree([43684, 35616, 30267])  # "無法回答"的token

        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    generation_config=generation_config,
                                    repetition_penalty=1.0,
                                    prefix_allowed_tokens_fn=context_prefix_tree.find_allowed_tokens)
        generate_answers = tokenizer.batch_decode(outputs[:,inputs.input_ids.size(1):], skip_special_tokens=True)

        for j in range(0, len(generate_answers), num_return_sequences):
            print(generate_answers[j:j+num_return_sequences])
            if not any([answer.replace(" ", "") in unable_to_answer_string for answer in generate_answers[j:j+num_return_sequences]]): # remove all answer from the same context if one of the answer is "無法回答。"
                answers_len = [len(answer) for answer in generate_answers[j:j+num_return_sequences]]
                longest_answer_idx = answers_len.index(max(answers_len))

                answer = (generate_answers[j + longest_answer_idx].replace(" ", "")).replace("。", "")
                if only_return_best and answer not in answers:
                    answers.append(answer)
                    answer_contexts.append(contexts[i + j//num_return_sequences])
            else:
                print("no answer")
    print(f"exit generative_reader")
    return answers, answer_contexts
