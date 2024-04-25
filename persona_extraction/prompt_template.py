from langchain import PromptTemplate

template = """
Given that you have seen a given persons writing style meta data on several research papers provided to you as a json key value pair as "context", 
write a paragraph about a given paper name passed as "Question" keeping the given writing style in mind and trying to addhere to it.

Context: {context}
Question: {question}
Answer: 

"""

prompt = PromptTemplate(
  template=template, 
  input_variables=["context", "question"]
)