"""
prompt creators for various models
"""

SUGGEST_SYS_PROMPT_4_TXL = \
    "你是一個來自台灣的AI助理，你的名字是 TAIDE，樂於以台灣人的立場幫助使用者，會用繁體中文回答問題。"
SUGGEST_SYS_PROMPT_4_BREEZE = \
    "You are a helpful AI assistant built by MediaTek Research. The user you are helping speaks Traditional Chinese and comes from Taiwan."

def prompt_creator(model_type: str, system_prompt: str, user_prompt: str) -> str:
    """ general prompt creator for various models

    Var:
        model_type: str
            tsg   taide e.1.1.0-SG
            txl   taide Llama3-TAIDE-LX-8B-Chat-Alpha1
            braw  Breeze-7B-Instruct-v1_0
            bcot  Breeze-v1_0-CoT-v0_2-full
    """
    print(f"enter prompt_creator")

    if model_type == "tsg":
        full_prompt = prompt_of_tsg(system_prompt, user_prompt)
    elif model_type == "tsg":
        full_prompt = prompt_of_txl(system_prompt, user_prompt)
    elif model_type[0] == "b":
        full_prompt = prompt_of_breeze(system_prompt, user_prompt)

    print(f"exit prompt_creator")
    return full_prompt

def prompt_of_tsg(system_prompt: str, user_prompt: str) -> str:
    """ create prompt for taide e.1.1.0-SG
    """
    print(f"enter prompt_of_tsg")
    
    full_prompt = f"{system_prompt} \n USER: {user_prompt} ASSISTANT:"
    
    print(f"exit prompt_of_tsg")
    return full_prompt

def prompt_of_txl(system_prompt: str, user_prompt: str) -> str:
    """ create prompt for taide Llama3-TAIDE-LX-8B-Chat-Alpha1
    """
    print(f"enter prompt_of_txl")

    full_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

    {{ {SUGGEST_SYS_PROMPT_4_TXL}\n{system_prompt} }}<|eot_id|><|start_header_id|>user<|end_header_id|>

    {{ {user_prompt} }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    
    print(f"exit prompt_of_txl")
    return full_prompt

def prompt_of_breeze(system_prompt: str, user_prompt: str) -> str:
    """ create prompt for Breeze-7B-Instruct-v1_0, Breeze-v1_0-CoT-v0_2-full
    """
    print("enter prompt_of_breeze")

    full_prompt = f"<s> {SUGGEST_SYS_PROMPT_4_BREEZE} {system_prompt} [INST] {user_prompt} [/INST]"

    print("exit prompt_of_breeze")
    return full_prompt
