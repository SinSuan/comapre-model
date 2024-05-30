
import json
import requests

def td8b(question, t, parameters):
    url = "http://140.120.13.248:10201/generate"

    if t==0:
        text="""你現在是一位農業專家，請對輸入問題進行三種分類。農業事實驗證問題(例子:水稻稻熱病是由水稻稻熱病病毒所引起的嗎？、農藥會造成植物根部萎縮？)、\
                        農業知識型問答(例子:水稻稻熱病的可能原因?、如何提高作物的產量？)、\
                        非農業問題(例子:什麼是歷史確定論，中興大學位於哪裡，國泰信用卡申辦規則有哪些。)，\
                        你的回答只能是一個數字，有三種: 0、1、2 ，如果是農業事實驗證問題請回答0 ，如果是農業一般問答請回答1 ，如果非農業問題請回答2。
        """
    elif t==1:
        text="""你是一個農業專家，請幫我解答以下問題，請用中文回答。
        """
    elif t==2:
        text="""你是一個農業專家，請幫完成任務以下問題，請用中文回答。
        """

    instruction = f"""
    {text}
    """

    SUGGEST_SYS_PROMPT = f"""你是一個來自台灣的AI助理，你的名字是 TAIDE，樂於以台灣人的立場幫助使用者，會用繁體中文回答問題。
    {instruction}
    """
    prompt = f"""{question}
    """

    payload = json.dumps({
    "inputs": f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

    {{ {SUGGEST_SYS_PROMPT} }}<|eot_id|><|start_header_id|>user<|end_header_id|>

    {{ {prompt} }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    "parameters": parameters,
    })
    
    headers = {
    'Content-Type': 'application/json',
    'accept': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    result = json.loads(response.text)["generated_text"]
    return result
