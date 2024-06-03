""" for experiementing the pipeline of the model
"""

from dotenv import load_dotenv
from time import time
from module_test.task_handler import task_handler
from module_test.get_hints import get_hints
from module_test.save_record import save_record
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
    inference_time = {}
    ttl_time_start = time()
    # step 0: question classification
    time_start = time()
    question_classification_result = task_handler("cls", model_type, query)
    inference_time["question classification"] = time() - time_start
    
    if question_classification_result == "Non-Agriculural Question":
        content={}
        result =["很抱歉，這不是個農業問題，神農taide身為一個針對農業知識問答強化的語言模型，無法處理這樣的問題。"]
        return result, [content]
    
    elif question_classification_result == "exception occurred":
        content={}
        result =["exception occurred"]
        return result, [content]

    else:

        # step 1: question expansion with hyde
        time_start = time()
        hyDE_query = task_handler("hyde", model_type, query)
        inference_time["question expansion with hyde"] = time() - time_start

        # step 2: sparse retriever
        time_start = time()
        hints = get_hints(query)
        query_4_retrieval = f"{hints} {hyDE_query}"
        relevant_contexts, relevant_info = sparse_retriever(query_4_retrieval,retrieve_K)
        inference_time["sparse retriever"] = time() - time_start

        # step 3: generative reader
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
        inference_time["generative reader"] = time() - time_start
            
        answers_by_ExtractiveLM, contexts_by_ExtractiveLM = possible_answers, answer_contexts
        if len(answers_by_ExtractiveLM) == 0:
            return ['很抱歉，模型無法根據現有資料集回答您的問題。'], ['']

        # step 4: summarization
        time_start = time()
        summarization_reply = task_handler("sum", model_type, query, answers_by_ExtractiveLM)
        inference_time["summarization"] = time() - time_start
        
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
        
        inference_time["ttl_time"] = time() - ttl_time_start
        return  summarization_reply, answer_context_pair, inference_time
    
def main():
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
    
    start_time = time()
    
    # parameters
    all_model = ["tsg", "txl", "braw", "bcot"]

    model_type = all_model[0]
    query = "水稻稻熱病是由水稻稻熱病病毒所引起的嗎？"
    retrieve_K = 20
    batch_size = 1
    
    # inference
    summarization_reply, answer_context_pair, inference_time = process_task(model_type, query, retrieve_K, batch_size)
    
    # save model record
    model_record = {
        "query": query,
        "model_type": model_type,
        "query_time": start_time,
        "inference_time": inference_time,
        "result": summarization_reply,
        "reference": answer_context_pair,
    }
    
    path_folder = "summarization_reply"
    path_file = f"{path_folder}/{query[:10]}_{model_type}"
    save_record(model_record, path_file)
    
    

if __name__ == "__main__":
    load_dotenv(".env")
    main()
