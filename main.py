""" for experiementing the pipeline of the model
"""

from compare_model.module_test.task_handler import task_handler
from compare_model.module_test.get_hints import get_hints

def main(query):
    """ main function for the pipeline
    """
    hints = get_hints(query)

#step 0: question classification
#step 1: question expansion with hyde
#step 2: sparse retriever
#step 3: generative reader
#step 4: summarization


if __name__ == "__main__":
    QUERY = "水稻稻熱病是由水稻稻熱病病毒所引起的嗎？"
    main(QUERY)
