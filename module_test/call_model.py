import requests
import re
import json
import configparser
import os

token = os.getenv('taide_api_key')

# config 讀取資訊
config = configparser.ConfigParser()
config.read('./config.ini')
host_tsg = config['host']['nstc_taide']
host_txl = config['host']['nlplab_taide_llama3']
host_braw = config['host']['nlplab_breeze']

# def call_tsg(full_prompt):
#     """call api taide e.1.1.0-SG
#     """
#     headers = {
#         "Authorization": "Bearer "+token
#     }
#     host = host_tsg
#     data = {
#         "model": "TAIDE/e.1.1.0-SG",
#         "prompt": full_prompt,
#         "temperature": 0.9,
#         "top_p": 0.9,
#         "presence_penalty": 1,
#         "frequency_penalty": 1,
#         "max_tokens": 200,
#         "repetition_penalty":1.2
#     }
#     try:
#         r = requests.post(host+"/completions", headers=headers, json=data)
#         result = r.json()["choices"][0]["text"]
#         return result

#     except Exception as e:
#         print(f"發生錯誤:{e}")
#         print(r.json())
#         return ""

# def call_txl(full_prompt):
#     """ call api taide Llama3-TAIDE-LX-8B-Chat-Alpha1
#     """
#     headers = {
#         'Content-Type': 'application/json',
#         'accept': 'application/json'
#     }
#     parameters = {
#         "temperature": 0.9,
#         "top_p": 0.9,
#         "max_new_tokens": 200,
#     }
#     payload = json.dumps({
#         "inputs": full_prompt,
#         "parameters": parameters,
#     })

#     try:

#         r = requests.request("POST", host_txl, headers=headers, data=payload)
#         result = json.loads(r.text)["generated_text"]

#         return result

#     except Exception as e:
#         print(f"發生錯誤:{e}")
#         print(r.json())
#         return ""

# def call_braw(full_prompt):
#     """ call api Breeze-7B-Instruct-v1_0
#     """
#     headers = {
#         'Content-Type': 'application/json',
#         'accept': 'application/json'
#     }
#     parameters = {
#         "do_sample": True,
#         "top_p": 0.95,
#         "temperature": 0.01
#     }
#     payload = json.dumps({
#         "inputs": full_prompt,
#         "parameters": parameters
#     })
#     try:
#         r = requests.request("POST", host_braw, headers=headers, data=payload)
#         result = json.loads(r.text)["generated_text"]
#         return result

#     except Exception as e:
#         print(f"發生錯誤:{e}")
#         print(r.text)
#         return 'exception occurred'

def call_model(model_type, full_prompt):
    """ general model caller for various models
    """
    try:
        
        host, headers, data = param_4_tsg(full_prompt)
        if model_type == "tsg":
            r = requests.post(host+"/completions", headers=headers, json=data)
            result = r.json()["choices"][0]["text"]
        elif model_type in ["tsg", "braw"]:
            r = requests.request("POST", host_braw, headers=headers, data=data)
            result = json.loads(r.text)["generated_text"]
        elif model_type == "bcot":
            pass

        return result

    except Exception as e:
        print(f"發生錯誤:{e}")
        print(r.text)
        return 'exception occurred'

def get_param(model_type, full_prompt):
    """ general parameter creator for various models
    """

    if model_type == "tsg":
        host, headers, data = param_4_tsg(full_prompt)
    elif model_type == "tsg":
        host, headers, data = param_4_txl(full_prompt)
    elif model_type == "braw":
        host, headers, data = param_4_braw(full_prompt)
    elif model_type == "bcot":
        pass

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
        "max_tokens": 200,
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
    pass