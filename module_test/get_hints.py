""" 取得關鍵字
"""

from udicOpenData.stopwords import rmsw

#model1
def get_hints(query: str) -> str:
    """ 斷詞後取得關鍵字
    """
    print(f"enter get_hints")

    hints = ""
    for i in rmsw(query, flag=True):
        print(i)
        if (i[1] not in ['ng']) and (i[0] not in ["原因", "為何"]):
            hints = hints+" "+i[0]

    print(f"exit get_hints")
    return hints
