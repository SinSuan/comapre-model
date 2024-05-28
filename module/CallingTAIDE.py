import requests
import re
import configparser
import os
import time

token = os.getenv('taide_api_key')

# config 讀取資訊
config = configparser.ConfigParser()
config.read('./config.ini') 
host_from_nlplab = config['host']['nlplab_taide']
host_from_nstc = config['host']['nstc_taide']

# 國科會api
def QuestionTypeClassificationWithTAIDE(question):
    headers = {
        "Authorization": "Bearer "+token
    }
    host = host_from_nstc
    prompt_2 = f'''你現在是一位農業專家，請對輸入問題進行三種分類。農業事實驗證問題(例子:水稻稻熱病是由水稻稻熱病病毒所引起的嗎？、農藥會造成植物根部萎縮？)、\
                        農業知識型問答(例子:水稻稻熱病的可能原因?、如何提高作物的產量？)、\
                        非農業問題(例子:什麼是歷史確定論，中興大學位於哪裡，國泰信用卡申辦規則有哪些。)，\
                        你的回答只能是一個數字，有三種: 0、1、2 ，如果是農業事實驗證問題請回答0 ，如果是農業一般問答請回答1 ，如果非農業問題請回答2。\
                        USER: {question} ASSISTANT:'''

    data = {
        "model": "TAIDE/e.1.1.0-SG",
        "prompt": prompt_2,
        "temperature": 0.9,
        "top_p": 0.9,
        "presence_penalty": 1,
        "frequency_penalty": 1,
        "max_tokens": 200,
        "repetition_penalty":1.2
    }

    start = time.time()
    delta_time = 0
    sleep_time = 30
    delta_sleep = 10
    while(delta_time < 100):
        try:
            # 遞減睡眠時間
            if delta_time != 0 and sleep_time > 10:
                sleep_time -= delta_sleep

            r = requests.post(host+"/completions", json=data, headers=headers)
            result = r.json()["choices"][0]["text"]
            print(str(result))

            # 使用正則表達式來找到所有的數字
            numbers = re.findall(r'\d+', str(result))
            # re.findall 返回的是list，直接取出
            type_num = 0
            if len(numbers) > 0:
                num = int(numbers[0])
                if num >=0 and num <=2:
                    type_num = num
            question_type_category = ["Fact Verification", "General QA", "Non-Agriculural Question"]

            return question_type_category[type_num]

        except Exception as e:
            delta_time = time.time() - start
            time.sleep(sleep_time)
            print(f"發生錯誤:{e}")
            print(r.json())

    return 'time out'

def QuestionExpansionHYDE(question):
    headers = {
        "Authorization": "Bearer "+token
    }
    host = host_from_nstc
    # prompt_2 = f"[INST] <<SYS>>\n你是一個農業專家，請幫我解答以下問題，請用中文回答。\n<</SYS>>\n\n {question} [/INST]"
    prompt_2 = f'你是一個農業專家，請幫我解答以下問題，請用中文回答。USER: {question} ASSISTANT:'
    data = {
        "model": "TAIDE/e.1.1.0-SG",
        "prompt": prompt_2,
        "temperature": 0.9,
        "top_p": 0.9,
        "presence_penalty": 1,
        "frequency_penalty": 1,
        "max_tokens": 200,
        "repetition_penalty":1.2
    }

    start = time.time()
    delta_time = 0
    sleep_time = 30
    delta_sleep = 10
    while(delta_time < 100):
        try:
            # 遞減睡眠時間
            if delta_time != 0 and sleep_time > 10:
                sleep_time -= delta_sleep
            r = requests.post(host+"/completions", json=data, headers=headers)
            result = r.json()["choices"][0]["text"]
            return str(result)

        except Exception as e:
            delta_time = time.time() - start
            time.sleep(sleep_time)
            print(f"發生錯誤:{e}")
            print(r.json())

    return 'time out'

def SummaryGeneratorWithTAIDE(question, answers):
    headers = {
        "Authorization": "Bearer "+token
    }
    host = host_from_nstc
    task = f"我有以下片段關於{question}的答案：{str(answers)} 你可以試著幫我將結果進行統整摘要嗎？"

    # prompt = f"[INST] <<SYS>>\n你是一個農業專家，請幫完成任務以下問題，請用中文回答。\n<</SYS>>\n\n {task} [/INST]"
    prompt = f'你是一個農業專家，請幫完成任務以下問題，請用中文回答。USER: {task} ASSISTANT:'
    data = {
        "model": "TAIDE/e.1.1.0-SG",
        "prompt": prompt,
        "temperature": 0.9,
        "top_p": 0.9,
        "presence_penalty": 1,
        "frequency_penalty": 1,
        "max_tokens": 600,
        "repetition_penalty":1.2
    }

    start = time.time() 
    delta_time = 0
    sleep_time = 30
    delta_sleep = 10
    while(delta_time < 100):
        try:
            # 遞減睡眠時間
            if delta_time != 0 and sleep_time > 10:
                sleep_time -= delta_sleep
            r = requests.post(host+"/completions", json=data, headers=headers)
            result = r.json()["choices"][0]["text"]
            return str(result)

        except Exception as e:
            delta_time = time.time() - start
            time.sleep(sleep_time)
            print(f"發生錯誤:{e}")
            print(r.json())

    return 'time out'
