from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

from persona_extraction.llm import chatbot_llm_chain
from persona_extraction.prompt_template import prompt
from persona_extraction.persona import person_persona


class ChatBot():


  llm = chatbot_llm_chain(prompt)

  rag_chain = (
    # {"context": docsearch.as_retriever(),  "question": RunnablePassthrough()} 
    {"context":  lambda x: str(person_persona)[:512],  "question": RunnablePassthrough()} 
    | prompt 
    | llm
    | StrOutputParser() 
  )


# Outside ChatBot() class
bot = ChatBot()
# input = input("Ask me anything: ")
input = "Can Computers Create Art?"
result = bot.rag_chain.invoke(input)
# print(result)
