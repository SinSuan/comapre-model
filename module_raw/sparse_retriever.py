import os
import json
from pyserini.search.lucene import LuceneSearcher
from pyserini.index import IndexReader
import sys
import configparser

# sys.path.append("..")
path_to_config = os.getenv('path_to_config')

# config 讀取資訊
config = configparser.ConfigParser()
config.read(path_to_config)
inverted_index_filepath = config['filepath']['inverted_index']

print('Loading Inverted Index for 神農gpt...')

searcher = LuceneSearcher(inverted_index_filepath)
index_reader = IndexReader(inverted_index_filepath)



def sparse_retriever(question, k=40) -> list[str]:
    searcher.set_language('zh')
    hits = searcher.search(question, k+10)
    candicate_context = []
    candicate_info = []

    for i in hits:
        _content = json.loads(i.raw)["contents"]
        _author = json.loads(i.raw)["author"]
        _url = json.loads(i.raw)["url"]
        _type = json.loads(i.raw)["type"]
        _date = json.loads(i.raw)["date"]
        _source = json.loads(i.raw)["source"]
        
        info_dic = {
            'id':i.docid,
            'type':_type,
            'source':_source,
            'date':_date,
            'url':_url,
            'author':_author,
            'contents':_content,
        }


        if _content not in candicate_context:
            candicate_context.append(_content)
            candicate_info.append(info_dic)

    return candicate_context,  candicate_info[:k]
