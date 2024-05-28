import configparser
import time
from .CallingTAIDE import QuestionExpansionHYDE, SummaryGeneratorWithTAIDE, QuestionTypeClassificationWithTAIDE
from .sparse_retriever import sparse_retriever
from .generative_reader import generative_reader


# config 讀取資訊
config = configparser.ConfigParser()
config.read('config.ini')
retrieve_K = int(config['constant']['retrieve_K'])
batch_size = int(config['constant']['batch_size'])

def agri_QA(retrieval_hint, question): 
    start_time = time.time()

    answers_by_ExtractiveLM, answer_context_pair = agriQA_with_TAIDE(retrieval_hint, question)

    end_time = time.time()
    duration = end_time - start_time
    print("Execution Time: {:.2f} seconds".format(duration))

    result = {
        "query": question,
        "result": answers_by_ExtractiveLM,
        "reference": answer_context_pair,
        "query_time":start_time,
        "duration":duration
    }

    return result


def agriQA_with_TAIDE(retrieval_hint, question, confidence = 0.8):
    question_classification_result = QuestionTypeClassificationWithTAIDE(question)

    if question_classification_result == "Non-Agriculural Question":
        content={}
        result =["很抱歉，這不是個農業問題，神農taide身為一個針對農業知識問答強化的語言模型，無法處理這樣的問題。"]
        return result, [content]

    elif question_classification_result == "exception occurred":
        content={}
        result =["exception occurred"]
        return result, [content]

    else:
        print("TAIDE Model")
        hyDE_query = QuestionExpansionHYDE(question)
        # print(hyDE_query)
        relevant_contexts, relevant_info = sparse_retriever(retrieval_hint+" "+hyDE_query,retrieve_K)   #使用sparse retrievier
        print("Enable TAIDE HYDE Query Expansion")
        # relevant_contexts, relevant_info = sparse_retriever(retrieval_hint,retrieve_K)   #使用sparse retrievier

        # possible_answers, answer_contexts = extractive_reader(
        #                                 relevant_contexts,
        #                                 question,
        #                                 confidence)
        possible_answers, answer_contexts = generative_reader(question,
                                        relevant_contexts,
                                        batch_size = batch_size,
                                        max_length = 768,
                                        only_return_best = True)
        print(f"possible answers : \n{possible_answers}")

        # select a subset of answers
        answers_by_ExtractiveLM, contexts_by_ExtractiveLM = possible_answers, answer_contexts
        print(f"chosen answers : \n{answers_by_ExtractiveLM}")
        if len(answers_by_ExtractiveLM) == 0:
            return ['很抱歉，模型無法根據現有資料集回答您的問題。'], ['']

        # 處理參考文章資訊
        context_info = [item for context in contexts_by_ExtractiveLM for item in relevant_info if item["contents"] == context]

        # find answer-context pair
        answer_context_pair = []
        for answer, info in zip(answers_by_ExtractiveLM, context_info):
            answer_context_pair.append({
                'answer':answer,
                'context_info':info,
            })

        # summarize by TAIDE  ****************************************************************************
        summarization_reply = SummaryGeneratorWithTAIDE(question, answers_by_ExtractiveLM)
        # summarization_reply = SummaryGeneratorWithBREEZE(question, answers_by_ExtractiveLM)
        print(f"summarization_reply : \n{summarization_reply}")
        # answers_by_ExtractiveLM.insert(0,summarization_reply.replace('\n','<br> '))
        summarization_reply = [summarization_reply.replace('\n','<br> ')]

        return  summarization_reply, answer_context_pair
