"""
create parameters for various models and call the models
"""

import configparser
import json
import os
import requests

token = os.getenv('taide_api_key')
# print(f"token:{token}")
# path_to_config = "/user_data/DG/113_0426/compare_model/private/config.ini"
path_to_config = os.getenv('path_to_config')
# print(f"path_to_config:{path_to_config}")

# config 讀取資訊
config = configparser.ConfigParser()
config.read(path_to_config)
host_tsg = config['host']['nstc_taide']
host_txl = config['host']['nlplab_taide_llama3']
host_braw = config['host']['nlplab_breeze']
host_bcot = config['host']['nlplab_breeze_cot']

def call_model(model_type: str, full_prompt: str) -> str:
    """ general model caller for various models

    Var:
        model_type: str
            tsg   taide e.1.1.0-SG
            txl   taide Llama3-TAIDE-LX-8B-Chat-Alpha1
            braw  Breeze-7B-Instruct-v1_0
            bcot  Breeze-v1_0-CoT-v0_2-full
    """
    try:

        host, headers, data = get_param(model_type, full_prompt)
        if model_type == "tsg":
            r = requests.post(host+"/completions", headers=headers, json=data)
            result = r.json()["choices"][0]["text"]

        elif model_type in ["tsg", "braw", "bcot"]:
            r = requests.request("POST", host_braw, headers=headers, data=data)
            result = json.loads(r.text)["generated_text"]

        return str(result)

    except Exception as e:
        print(f"發生錯誤:{e}")
        print(r.text)
        return 'exception occurred'

def get_param(model_type: str, full_prompt: str):
    """ general parameter creator for various models
    
    Var:
        model_type: str
            tsg   taide e.1.1.0-SG
            txl   taide Llama3-TAIDE-LX-8B-Chat-Alpha1
            braw  Breeze-7B-Instruct-v1_0
            bcot  Breeze-v1_0-CoT-v0_2-full
    """

    if model_type == "tsg":
        host, headers, data = param_4_tsg(full_prompt)
    elif model_type == "tsg":
        host, headers, data = param_4_txl(full_prompt)
    elif model_type == "braw":
        host, headers, data = param_4_braw(full_prompt)
    elif model_type == "bcot":
        host, headers, data = param_4_bcot(full_prompt)

    return host, headers, data

def param_4_tsg(full_prompt):
    """ define parameters taide e.1.1.0-SG
    """
    host = host_tsg
    headers = {
        "Authorization": "Bearer "+token
    }
    data = {
        "model": "TAIDE/e.1.1.0-SG",
        "prompt": full_prompt,
        "temperature": 0.9,
        "top_p": 0.9,
        "presence_penalty": 1,
        "frequency_penalty": 1,
        "max_tokens": 600,
        "repetition_penalty":1.2
    }
    return host, headers, data

def param_4_txl(full_prompt):
    """ define parameters taide Llama3-TAIDE-LX-8B-Chat-Alpha1
    """
    host = host_txl
    headers = {
        'Content-Type': 'application/json',
        'accept': 'application/json'
    }
    parameters = {
        "temperature": 0.9,
        "top_p": 0.9,
        "max_new_tokens": 200,
    }
    data = json.dumps({
        "inputs": full_prompt,
        "parameters": parameters
    })
    return host, headers, data

def param_4_braw(full_prompt):
    """ define parameters for Breeze-7B-Instruct-v1_0
    """
    host = host_braw
    headers = {
        'Content-Type': 'application/json',
        'accept': 'application/json'
    }
    parameters = {
        "do_sample": True,
        "top_p": 0.95,
        "temperature": 0.01
    }
    data = json.dumps({
        "inputs": full_prompt,
        "parameters": parameters
    })
    return host, headers, data

def param_4_bcot(full_prompt):
    """ define parameters for Breeze-v1_0-CoT-v0_2-full
    """
    host = host_braw
    headers = {
        'Content-Type': 'application/json',
        'accept': 'application/json'
    }
    parameters = {
        "do_sample": True,
        "top_p": 0.95,
        "temperature": 0.01,
        "max_new_tokens":2000 # the number of output text
    }
    data = json.dumps({
        "inputs": full_prompt,
        "parameters": parameters
    })
    return host, headers, data
