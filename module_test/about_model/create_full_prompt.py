"""
prompt creators for various models
"""

def prompt_creator(model_type, system_prompt, user_prompt):
    """ general prompt creator for various models
    """

    if model_type == "tsg":
        full_prompt = prompt_of_tsg(system_prompt, user_prompt)
    elif model_type == "tsg":
        full_prompt = prompt_of_txl(system_prompt, user_prompt)
    elif model_type[0] == "b":
        full_prompt = prompt_of_breeze(system_prompt, user_prompt)

    return full_prompt


def prompt_of_tsg(system_prompt, user_prompt):
    """ create prompt for taide e.1.1.0-SG
    """
    full_prompt = f"{system_prompt} \n USER: {user_prompt} ASSISTANT:"
    return full_prompt

SUGGEST_SYS_PROMPT_4_TAIDE = "你是一個來自台灣的AI助理，你的名字是 TAIDE，樂於以台灣人的立場幫助使用者，會用繁體中文回答問題。"
def prompt_of_txl(system_prompt, user_prompt):
    """ create prompt for taide Llama3-TAIDE-LX-8B-Chat-Alpha1
    """

    full_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

    {{ {SUGGEST_SYS_PROMPT_4_TAIDE}\n{system_prompt} }}<|eot_id|><|start_header_id|>user<|end_header_id|>

    {{ {user_prompt} }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    return full_prompt

SUGGEST_SYS_PROMPT_4_BREEZE = \
    "You are a helpful AI assistant built by MediaTek Research. The user you are helping speaks Traditional Chinese and comes from Taiwan."
def prompt_of_breeze(system_prompt, user_prompt):
    """ create prompt for Breeze-7B-Instruct-v1_0, Breeze-v1_0-CoT-v0_2-full
    """
    full_prompt = f"<s> {SUGGEST_SYS_PROMPT_4_BREEZE} {system_prompt} [INST] {user_prompt} [/INST]"
    return full_prompt
