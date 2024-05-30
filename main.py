from udicOpenData.stopwords import rmsw

#model1
def get_hints(query: str):
    hints = ""
    for i in rmsw(query, flag=True):
        print(i)
        if (i[1] not in ['ng']) and (i[0] not in ["原因", "為何"]):
            hints = hints+" "+i[0]
    return hints

#step 0: question classification
#step 1: question expansion with hyde
#step 2: sparse retriever
#step 3: generative reader
#step 4: summarization


if __name__ == "__main__":
    hints = get_hints("為何徒長病發生")
    print(hints)
