import requests
import json
import re
import configparser
import os

# config 讀取資訊
config = configparser.ConfigParser()
config.read('./config.ini') 
host_from_nlplab = config['host']['nlplab_breeze']


def QuestionTypeClassificationWithBREEZE(question):
    
    host = host_from_nlplab
    headers = {
        'Content-Type': 'application/json',
        'accept': 'application/json'
    }

    prompt = f'''你現在是一位農業專家，請對輸入問題進行三種分類。農業事實驗證問題(例子:水稻稻熱病是由水稻稻熱病病毒所引起的嗎？、農藥會造成植物根部萎縮？)、\
                        農業知識型問答(例子:水稻稻熱病的可能原因?、如何提高作物的產量？)、\
                        非農業問題(例子:什麼是歷史確定論，中興大學位於哪裡，國泰信用卡申辦規則有哪些。)，\
                        你的回答只能是一個數字，有三種: 0、1、2 ，如果是農業事實驗證問題請回答0 ，如果是農業一般問答請回答1 ，如果非農業問題請回答2。\
                        '''
    
    system = f"You are a helpful AI assistant built by MediaTek Research. The user you are helping speaks Traditional Chinese and comes from Taiwan.{prompt}"

    payload = json.dumps({
        "inputs": f"<s>{system} [INST] {question} [/INST]",
        "parameters": {
            "do_sample": True,
            "temperature": 0.01,
            "top_p": 0.95
        }
    })
    try:
        response = requests.request("POST", host, headers=headers, data=payload)
        # 使用正則表達式來找到所有的數字
        numbers = re.findall(r'\d+', response.json()['generated_text'])
        # re.findall 返回的是list，直接取出
        type_num = 0
        if len(numbers) > 0:
            num = int(numbers[0])
            if num >=0 and num <=2:
                type_num = num
        question_type_category = ["Fact Verification", "General QA", "Non-Agriculural Question"]
        return question_type_category[type_num]

    except Exception as e:
        print(f"發生錯誤:{e}")
        print(response.text)
        return 'exception occurred'

def QuestionExpansionHYDEWithBREEZE(question):
    
    host = host_from_nlplab
    headers = {
        'Content-Type': 'application/json',
        'accept': 'application/json'
    }

    system = "You are a helpful AI assistant built by MediaTek Research. The user you are helping speaks Traditional Chinese and comes from Taiwan. 你是一個農業專家，請幫我解答以下問題，請用中文回答。"
    payload = json.dumps({
        "inputs": f"<s>{system} [INST] {question} [/INST]",
        "parameters": {
            "do_sample": True,
            "temperature": 0.01,
            "top_p": 0.95
        }
    })
    try:
        response = requests.request("POST", host, headers=headers, data=payload)
        result = response.json()['generated_text']
        
        return result[0]

    except Exception as e:
        print(f"發生錯誤:{e}")
        print(response.text)
        return ''

def SummaryGeneratorWithBREEZE(question, answers):
    headers = {
        'Content-Type': 'application/json',
        'accept': 'application/json'
    }
    host = host_from_nlplab

    system = "You are a helpful AI assistant built by MediaTek Research. The user you are helping speaks Traditional Chinese and comes from Taiwan. 你是一個農業專家，請幫完成任務以下問題，請用中文回答。"
    task = f"我有以下片段關於{question}的答案：{str(answers)} 你可以試著幫我將結果進行統整摘要嗎？"
    
    payload = json.dumps({
        "inputs": f"<s>{system} [INST] {task} [/INST]",
        "parameters": {
            "do_sample": True,
            "temperature": 0.01,
            "top_p": 0.95
        }
    })
    try:
        response = requests.request("POST", host, headers=headers, data=payload)
        result = response.json()['generated_text']

        return result[0]

    except Exception as e:
        print(f"發生錯誤:{e}")
        print(response.text)
        return ''
