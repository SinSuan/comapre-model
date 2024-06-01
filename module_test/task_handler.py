""" handle the following task

        cls   question_classification
        hyde  question_expansion
        sum   summary_generator
"""

import re
from .about_model.call_model import call_model
from .about_model.create_full_prompt import prompt_creator

def task_handler(task_type: str, model_type: str, query: str, answers=None)->str:
    """ get task system prompt
    Var:
        task_type: str
            cls   question_classification
            hyde  question_expansion
            sum   summary_generator
        
        model_type: str
            tsg   taide e.1.1.0-SG
            txl   taide Llama3-TAIDE-LX-8B-Chat-Alpha1
            braw  Breeze-7B-Instruct-v1_0
            bcot  Breeze-v1_0-CoT-v0_2-full

        query: str
        
        answers: str

    Return:
        String
    """
    system_prompt = get_system_prompt(task_type)
    user_prompt = get_user_prompt(task_type, query, answers)
    full_prompt = prompt_creator(model_type, system_prompt, user_prompt)
    response = call_model(model_type, full_prompt)

    if task_type in ["hyde", "sum"]:
        result = response

    elif task_type=="cls":
        numbers = re.findall(r'\d+', response)
        # re.findall 返回的是list，直接取出
        type_num = 0
        if len(numbers) > 0:
            num = int(numbers[0])
            if 0 <= num <=2:
                type_num = num
        question_type_category = ["Fact Verification", "General QA", "Non-Agriculural Question"]

        result = question_type_category[type_num]

    return result

def get_system_prompt(task_type:str) -> str:
    """ get task system prompt
    Var:
        task_type: str
            cls   question_classification
            hyde   question_expansion
            sum   summary_generator
    
    Return:
        String
    """

    if task_type=="cls":
        system_prompt ="""你現在是一位農業專家，請對輸入問題進行三種分類。農業事實驗證問題(例子:水稻稻熱病是由水稻稻熱病病毒所引起的嗎？、農藥會造成植物根部萎縮？)、\
                        農業知識型問答(例子:水稻稻熱病的可能原因?、如何提高作物的產量？)、\
                        非農業問題(例子:什麼是歷史確定論，中興大學位於哪裡，國泰信用卡申辦規則有哪些。)，\
                        你的回答只能是一個數字，有三種: 0、1、2 ，如果是農業事實驗證問題請回答0 ，如果是農業一般問答請回答1 ，如果非農業問題請回答2。\
        """
    elif task_type=="hyde":
        system_prompt = "你是一個農業專家，請幫我解答以下問題，請用中文回答。"
    elif task_type=="sum":
        system_prompt = "你是一個農業專家，請幫我完成任務以下問題，請用中文回答。"

    return system_prompt

def get_user_prompt(task_type: str, query: str, answers=None) -> str:
    """ get task system prompt
    Var:
        task_type: str
            cls   question_classification
            hyde   question_expansion
            sum   summary_generator
    
    Return:
        String
    """

    if task_type in ["cls", "hyde"]:
        user_prompt = query
    elif task_type=="sum":
        user_prompt = f"我有以下片段關於{query}的答案：{str(answers)} 你可以試著幫我將結果進行統整摘要嗎？"

    return user_prompt
