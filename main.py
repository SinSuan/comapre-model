""" for experiementing the pipeline of the model
"""

from dotenv import load_dotenv
path_2_env = "/user_data/DG/113_0426/compare_model/private/.env"
# print(path_2_env)
load_dotenv(path_2_env)

from time import time
from module_test.task_handler import task_handler
from module_test.get_hints import get_hints
from module_test.recording import load_record, save_record
from module_raw.sparse_retriever import sparse_retriever
from module_raw.generative_reader import generative_reader

def process_task(model_type, query, retrieve_K, batch_size):
    """ process given query with the pipeline
    Var:        
        model_type: str
            tsg   taide e.1.1.0-SG
            txl   taide Llama3-TAIDE-LX-8B-Chat-Alpha1
            braw  Breeze-7B-Instruct-v1_0
            bcot  Breeze-v1_0-CoT-v0_2-full
    """
    print(f"enter process_task")
    
    inference_time = {}
    # result = {}

    ttl_time_start = time()
    # step 0: question classification
    print("step 0: question classification")
    time_start = time()
    question_classification_result = task_handler("cls", model_type, query)
    inference_time["step0_cls"] = str(time() - time_start)
    # result["step0_cls"] = question_classification_result
    
    if question_classification_result == "Non-Agriculural Question":
        content={}
        result =["很抱歉，這不是個農業問題，神農taide身為一個針對農業知識問答強化的語言模型，無法處理這樣的問題。"]
        print(f"early return\t{result}")
        return result, [content], inference_time
    
    elif question_classification_result == "exception occurred":
        content={}
        result =["exception occurred"]
        print(f"early return\t{result}")
        return result, [content], inference_time

    else:

        # step 1: question expansion with hyde
        print("step 1: question expansion with hyde")
        time_start = time()
        hyDE_query = task_handler("hyde", model_type, query)
        inference_time["step1_hyde"] = str(time() - time_start)

        # step 2: sparse retriever
        print("step 2: sparse retriever")
        time_start = time()
        hints = get_hints(query)
        query_4_retrieval = f"{hints} {hyDE_query}"
        relevant_contexts, relevant_info = sparse_retriever(query_4_retrieval,retrieve_K)
        inference_time["step2_retr"] = str(time() - time_start)

        # step 3: generative reader
        print("step 3: generative reader")
        # possible_answers for summarization
        # answer_contexts for summarization
        time_start = time()
        possible_answers, answer_contexts = \
            generative_reader(
                query, 
                relevant_contexts, 
                batch_size = batch_size, 
                max_length = 768,
                only_return_best = True
            )
        inference_time["step3_read"] = str(time() - time_start)
            
        answers_by_ExtractiveLM, contexts_by_ExtractiveLM = possible_answers, answer_contexts
        if len(answers_by_ExtractiveLM) == 0:
            content=['']
            result =['很抱歉，模型無法根據現有資料集回答您的問題。']
            print(f"early return\t{result}")
            return result, content, inference_time

        # step 4: summarization
        print("step 4: summarization")
        time_start = time()
        summarization_reply = task_handler("sum", model_type, query, answers_by_ExtractiveLM)
        inference_time["step4_sum"] = str(time() - time_start)
        
        # 整理 answer_context_pair
        context_info = \
            [
                item for context in contexts_by_ExtractiveLM \
                    for item in relevant_info \
                    if item["contents"] == context
            ]

        answer_context_pair = []
        for answer, info in zip(answers_by_ExtractiveLM, context_info):
            answer_context_pair.append({
                'answer':answer,
                'context_info':info,
            })
        
        inference_time["ttl_time"] = str(time() - ttl_time_start)
        
        print(f"exit process_task")
        return  summarization_reply, answer_context_pair, inference_time
    
def test_task(model_type, task_type, query, *args):
    """
        # step 0: question classification
        # step 1: question expansion with hyde
        # step 2: sparse retriever
        # step 3: generative reader
        # step 4: summarization
    
    Hyper Var:
    
        model_type: str
            tsg   taide e.1.1.0-SG
            txl   taide Llama3-TAIDE-LX-8B-Chat-Alpha1
            braw  Breeze-7B-Instruct-v1_0
            bcot  Breeze-v1_0-CoT-v0_2-full

        query: str
            from user input
        
        retrieve_K: int
            for step 2: sparse retriever

        batch_size: int
            for step 3: generative reader

    """
    print(f"enter test_task")
    
    print(f"args = {args}")
    if task_type=="ttl":
        retrieve_K, batch_size = args
        summarization_reply, answer_context_pair, time_spent = process_task(model_type, query, retrieve_K, batch_size)
        # model_record = {
        #     "query": query,
        #     "model_type": model_type,
        #     "time_spent": time_spent,
        #     "result": summarization_reply,
        #     "reference": answer_context_pair,
        # }
        result = {
            "result": summarization_reply,
            "reference": answer_context_pair,
        }

    elif task_type in ["cls", "hyde", "sum"]:
        answers = None if args==() else args[0]

        time_start = time()
        result = task_handler(task_type, model_type, query, answers)
        time_spent = time() - time_start
    
    elif task_type == "retr":
        hyDE_query, retrieve_K = args
        time_spent = {}
        
        time_start = time()
        hints = get_hints(query)
        time_spent['get_hints'] = str(time() - time_start)

        # query_4_retrieval = f"{hints} {hyDE_query}"
        query = f"{hints} {hyDE_query}"
        
        time_start = time()
        # relevant_contexts, relevant_info = sparse_retriever(query_4_retrieval, retrieve_K)
        relevant_contexts, relevant_info = sparse_retriever(query, retrieve_K)
        time_spent['sparse_retriever'] = str(time() - time_start)

        result = {
            'relevant_contexts': relevant_contexts,
            'relevant_info': relevant_info,
        }
    
    elif task_type == "read":
        relevant_contexts, relevant_info, batch_size = args
        
        time_start = time()
        possible_answers, answer_contexts = \
        generative_reader(
            query, 
            relevant_contexts, 
            batch_size = batch_size, 
            max_length = 768,
            only_return_best = True
        )
        time_spent = time() - time_start
        
        answers_by_ExtractiveLM, contexts_by_ExtractiveLM = possible_answers, answer_contexts
        context_info = \
            [
                item for context in contexts_by_ExtractiveLM \
                    for item in relevant_info \
                    if item["contents"] == context
            ]

        result = []
        for answer, info in zip(answers_by_ExtractiveLM, context_info):
            result.append({
                'answer':answer,
                'context_info':info,
            })
    
    model_record = {
        "query": query,
        "task_type": task_type,
        "model_type": model_type,
        "time_spent": time_spent,
        "result": result,
    }
    print(f"exit test_task")
    return model_record

def main(model_type, idx_step, query, path_folder):
    """ main function
    """
    ttl_task = ["cls", "hyde", "retr", "read", "sum", "ttl"]
    task_type = ttl_task[idx_step]

    if idx_step in [2,3,4]:
        name_file_result = f"{query[:10]}_step{idx_step-1}_{ttl_task[idx_step-1]}_{model_type}"
        text_input = load_record(path_folder, name_file_result)
    
    # print(f"text_input.keys() = {text_input.keys()}")
    
    name_file_result = f"{query[:10]}_step{idx_step}_{task_type}_{model_type}"
    if idx_step in [0,1]:
        model_record = test_task(model_type, task_type, query)
    elif idx_step==2:
        hyde_query = text_input
        print(f"hyDE_query =\n{hyde_query}")
        retrieve_k = 20
        model_record = test_task(model_type, task_type, query, hyde_query, retrieve_k)
    elif idx_step==3:
        relevant_contexts = text_input['relevant_contexts']
        relevant_info = text_input['relevant_info']
        batch_size = 1
        model_record = test_task(model_type, task_type, query, relevant_contexts, relevant_info, batch_size)
    elif idx_step==4:
        model_record = test_task(model_type, task_type, query, text_input)
    elif idx_step==5:
        retrieve_k = 20
        batch_size = 1
        # model_record = test_task(model_type, task_type, text_input, retrieve_k, batch_size)
        model_record = test_task(model_type, task_type, query, retrieve_k, batch_size)

    path_file = save_record(model_record, path_folder, name_file_result)
    print(f"path_file =\n{path_file}")

if __name__ == "__main__":
    path_2_env = "/user_data/DG/113_0426/compare_model/private/.env"
    print(path_2_env)
    load_dotenv(path_2_env)
    
    PATH_FOLDER = "model_record/test/2024_0617_1405"
    QUERY = "水稻稻熱病是由水稻稻熱病病毒所引起的嗎？"
    
    # parameters
    TTL_MODEL = ["tsg", "txl", "braw", "bcot"]
    MODEL_TYPE = TTL_MODEL[0]

    ## ttl_step
    # step 0: question classification       cls
    # step 1: question expansion with hyde  hyde
    # step 2: sparse retriever
    # step 3: generative reader
    # step 4: summarization                 sum
    IDX_STEP = 0
    # TTL_TASK = ["cls", "hyde", "retr", "read", "sum", "ttl"]
    # TASK_TYPE = TTL_TASK[IDX_STEP]

    # if IDX_STEP in [2,3,4,5]:
    #     name_file_result = f"{QUERY[:10]}_step{IDX_STEP-1}_{TTL_TASK[IDX_STEP-1]}_{MODEL_TYPE}"
    #     text_input = load_record(PATH_FOLDER, name_file_result)
    
    # # print(f"text_input.keys() = {text_input.keys()}")
    
    # name_file_result = f"{QUERY[:10]}_step{IDX_STEP}_{TASK_TYPE}_{MODEL_TYPE}"
    # if IDX_STEP in [0,1]:
    #     model_record = test_task(MODEL_TYPE, TASK_TYPE, QUERY)
    # elif IDX_STEP in [2]:
    #     hyDE_query = text_input
    #     print(f"hyDE_query =\n{hyDE_query}")
    #     retrieve_K = 20
    #     model_record = test_task(MODEL_TYPE, TASK_TYPE, QUERY, hyDE_query, retrieve_K)
    # elif IDX_STEP in [3]:
    #     relevant_contexts = text_input['relevant_contexts']
    #     relevant_info = text_input['relevant_info']
    #     batch_size = 1
    #     model_record = test_task(MODEL_TYPE, TASK_TYPE, QUERY, relevant_contexts, relevant_info, batch_size)
    # elif IDX_STEP in [4]:
    #     model_record = test_task(MODEL_TYPE, TASK_TYPE, QUERY, text_input)
    # elif IDX_STEP in [5]:
    #     retrieve_K = 20
    #     batch_size = 1
    #     model_record = test_task(MODEL_TYPE, TASK_TYPE, text_input, retrieve_K, batch_size)

    # save_record(model_record, PATH_FOLDER, name_file_result)

    main(MODEL_TYPE, IDX_STEP, QUERY, PATH_FOLDER)
