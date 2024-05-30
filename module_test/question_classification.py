"""123
"""

from .question_classification import prompt_of_tsg, prompt_of_txl, prompt_of_breeze

def prompt_4_question_classification(model_type, query):
    """ create prompt for question classification
    
    Var:
        model_type: str
            taide
                tsg     e.1.1.0-SG
                txl     Llama3-TAIDE-LX-8B-Chat-Alpha1
            breeze
                b       Breeze-7B-Instruct-v1_0, Breeze-v1_0-CoT-v0_2-full
                
    Return:
        String

    """
    system_prompt ='''你現在是一位農業專家，請對輸入問題進行三種分類。農業事實驗證問題(例子:水稻稻熱病是由水稻稻熱病病毒所引起的嗎？、農藥會造成植物根部萎縮？)、\
                        農業知識型問答(例子:水稻稻熱病的可能原因?、如何提高作物的產量？)、\
                        非農業問題(例子:什麼是歷史確定論，中興大學位於哪裡，國泰信用卡申辦規則有哪些。)，\
                        你的回答只能是一個數字，有三種: 0、1、2 ，如果是農業事實驗證問題請回答0 ，如果是農業一般問答請回答1 ，如果非農業問題請回答2。\
                        '''
    full_prompt = prompt_creator(model_type, system_prompt, query)

    
    
    return ""
