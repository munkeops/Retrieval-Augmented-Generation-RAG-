from langchain_community.llms import Ollama
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from bs4 import BeautifulSoup as Soup
from langchain.docstore.document import Document
from langchain.utils.html import (PREFIXES_TO_IGNORE_REGEX,
                                  SUFFIXES_TO_IGNORE_REGEX)

from rag_server.config import *
import logging
import sys
import chromadb


logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


global conversation
conversation = None
write_ups = {
    "Artifical General Intelligence": """/
                                    The authors aim to create a framework for classifying the capabilities and behavior of Artificial General Intelligence (AGI) models and their precursors. The paper claims that the proposed framework will help track progress toward AGI, identify potential risks, and guide research efforts. As a comparison the authors analyzed nine case studies of prior prominent formulations of the concept of AGI, including the Loebner Prize, the Turing Test, and the Winograd Schema Challenge. /
                                    Through this paper we see the authors propose a matrixed ontology, the "Levels of AGI," which considers a combination of performance and generality. The six levels of Performance (No AI, Emerging, Competent, Expert, Virtuoso, and Superhuman) and the six levels of Autonomy (No AI, AI as a Tool, AI as a Consultant, AI as a Collaborator, AI as an Expert, and AI as an Agent) provide a nuanced metric for evaluating AGI systems. /
                                    The authors demonstrate the applicability of the Levels of AGI framework by analyzing several AI systems, including AlphaGo, Watson, and Siri. The framework provides a clear and operationalizable definition of AGI, enabling researchers and policymakers to communicate more effectively about progress toward AGI and potential risks./
                                    Apart from its nuanced approach and comprehensive framework, the framework also draws my attention to several of its issues. For beginners the framework may be too complicated for some stake holders requiring significant experitise. Additionally the assignment of AGI levels may involve subjective judgements, potentially leading to disagreements and inconsistencies. The framework also has a limited generalizibility (to all models / AI and contexts) potentially limiting its applicability. It would also need to evolve as AGI capabilities and autononomy continue to adavnce hence potentially requiring revision. In my personal opinion I also believe that there was a lack of clear boundaries between AGI levels, they seems to be confusing at some points. Furthermore some may argue that the framework overemphasizes autonomy and could be overlooking other aspects of AGI. Last but not the least i feel like the framework is a bit reliant on benchmark and metrics which could be incomplete/ biased/ inaccurate which may impact the validity./
                                    Overall the framework proposed in this paper has done a great job in the domain of AGI. The framework serves as a foundation for tracking progress toward AGI and guiding research efforts to achieve responsible and beneficial AGI. Shared terminology and frameworks can help researches, policymakers and other stakeholders communicate more clearly about progress toward powerful AI systems. I think this paves the path for better understanding of the concept.
                                    """,
    "The Moral Machine experiment" : """The authors of this paper aim to understand how humans make moral decisions in situations where machines (e.g. autonomous cars) must make decisions that involve trade-offs between different values such as safety or lives. They set the building blocks to understand and study how cultural and personal factors influence moral decisions. The paper uses classical thought experiments in ethics such as the infamous Trolley Problem to establish a foundation in understanding human moral decision making and argue that they are limited and need to be expanded to include more diverse and realistic perspectives./ 
                                        The proposed moral machine is used as a metric to gather human decisions on a large scale (impressive 40 million decisions from millions of people, providing a vast and diverse dataset). The metric included moral preferences, values and cultural + personal factors that went into making these decisions. Furthermore The experiment's scenarios were inspired by real-world dilemmas, making the findings more applicable to actual ethical challenges and  uncovered subtle differences in moral values and priorities, going beyond simple right-or-wrong answers./
                                        The effort put into this paper was impressive and sets the baseline for AI development, as the results could be used to align algorithms to match real human values and ethical principles. But on the flip side the experiment’s scenarios did feel like they were potentially oversimplified. Additionally participants lacked detailed context about the scenarios which may have influenced their decisions (reminds me of the Tom Hanks starrer Sully). The study may have inadvertently also collected more data from certain cultures tipping the scale in the favor of their biases and given that only certain participants chose to participate, could have also affected the stats. This in general is a hard problem to solve as generalizing findings across vast populations could be very hard. Philosophical arguments and considerations of alternate perspectives could have potentially been overlooked as well./ 
                                        Overall I think the study demonstrates the importance of considering human moral values and provides a great baseline (or a starting point to generate the perfect one). It gives us great insights in cultural preferences, gender preferences and age preferences on the moral scale. The results truly show us the need and potential for considering such human moral values but in my opinion fails to provide a concrete decision due to data generalization issues./ 
                                        Discussion question: While it seems like there might be no correct answer, how do you make a decision for the collective masses when implementing an algorithm's moral scale as a whole and who takes accountability for this decision? [the movie “ I Robot” movie also kind of touches on this topic, great watch!]
                                    """,

    "The Disagreement Deconvolution: Bringing Machine Learning Performance Metrics In Line With Reality" : """The authors of the paper aim to bridge the gap between the performance provided by highly scored models -  on current machine learning performance metrics (eg like ROC AUC) -  and the actual user-facing experience. They attempt to reduce the disconnect by providing a more accurate assessment of model effectiveness in social computing tasks. The proposed metric Disagreement Deconvolution measures the degree to which multiple annotators agree on the models predictions. It is based on the idea that the model performance should be evaluated in terms of how well it aligns with human judgment.\
                                                                                                            I think the efforts put in by the author show that there is definitely an increase in accuracy of the model being designed due to more correct assessments being evaluated and developed with the new metric.  This helps improve the reflection on the design as the metric forces to align the model predictions with human judgment making it a more relevant metric. The metric is also robust to variations in the annotator disagreements and dataset sizes making it reliable. The works of this paper highlight the limitation of the current metrics and urge us to  not solely rely on them but to improve our approach with a more sophisticated metric like the one proposed in the paper. \
                                                                                                            The authors through DD were successfully able to demonstrate how significantly the current models have been overstating the results through the existing metrics. But the way I see it, I think the approach provided in this paper to bridge the gap (DD) has a very limited applicability and may not apply to all ML problems (due to its reliance on human annotation and disagreement). DD also brings about an increased complexity in calculation of the metric and also has a high dependency on annotator quality, both of which could significantly increase the time and cost of the project. Poor annotations may be prone to biases, and in many cases relatively decent results may tend to incorporate biases too, depending on the sampling group of annotators. \
                                                                                                            Overall I still thing the paper presents a valuable contribution to the evaluation metric field by pointing out the significant flaws of the current metrics. But its applicability and limitations (atleast for the metric - DD - provided in this paper) should be considered well before applying it to actual models in practice.\
                                                                                                            Discussion question : How might the use of human-centered metrics impact the development of explainable AI and interpretability techniques?""",
    "Datasheets for Datasets" :"""The authors of the paper note that machine learning datasets are often poorly documented, making it difficult to understand their contents limitations and potential biases. They strongly believe that this lack of transparency leads to issues such as misuse or misinterpretation of datasets, difficulty in reproduction of results and lack of accountability of dataset creation and curation. I like the proposal of the authors to have a standardized framework for documenting such datasets. This would streamline the process for dataset discovery and evaluation saving time and resources and furthermore improve collaboration through the common documentation.
The paper demonstrates how in the survey of top 100 popular machine learning datasets only 15% provide any form of documentation beyond a brief description and only 5 % provide detailed information about the datasets contents, limitations and biases. Clearly lack of such information makes it hard for researchers to pick datasets and understand them well to use them responsibly. The authors propose a simple but effective technique by evaluating - Coverage, Accuracy, Completeness and Clarity. The proposed framework included sections like Dataset descriptions, data collection and preprocessing, data characteristics, potential biases and limitations and ethical considerations. This information improves the understanding of the dataset and furthermore makes the creators more accountable for their data. This approach reduces misinterpretations and errors and furthermore promotes data literacy as the importance of documenting information about datasets is now a priority.
That being said I do understand that this would bring about additional overhead over dataset curators, but I believe its for the greater good. Furthermore it would be hard to standardize the required information, and in many cases maintenance and updates would be hard. The amount of information could also overload certain audiences that might find it hard to understand the data. Last but not the least due to a lack of incentive and this being a significant technical challenge for some, it would be hard to monitor quality and also train creators to balance between detail and brevity. The standard would require a presiding body to apply and enforce the standard in order to bring fruitful results.
Overall I do think this is a great approach, and if we could come up with a presiding body (could be open source for that matter) that could enforce this, I think this would benefit the whole machine learning community a great ton. 
Discussion Question: How could we develop a body that could help implement this standard. Could we have a central presiding body that accepts and rejects proposed datasets? Or could we have open sourced methods where the community contributes to improving and maintaining these standards.
"""

}
db_persistent_client = chromadb.PersistentClient()


def create_vector_store(author_name):
    collection = db_persistent_client.get_or_create_collection(author_name)

    # documents =  []

    documents = []

    for topic in write_ups:
        documents.append(Document(page_content=write_ups[topic], metadata={"source": "local"}))


    logging.info("index creating with `%d` documents", len(documents))

    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


    # tell LangChain to use our client and collection name
    # author_chroma = Chroma(
    #     client=db_persistent_client,
    #     collection_name= author_name,
    #     embedding_function=embedding_function,
    # )

    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embedding_function,
        collection_name= author_name,
        persist_directory=INDEX_PERSIST_DIRECTORY
    )
    vectordb.persist()
    # return vectordb


def init_conversation(author_name):
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
   
    # vectordb = create_vector_store(author_name)
    vectordb_runtime = Chroma(
        # client=db_persistent_client,
        collection_name= author_name,
        embedding_function=embedding_function,
        persist_directory=INDEX_PERSIST_DIRECTORY
    )

    # llama2 llm which runs with ollama
    # ollama expose an api for the llam in `localhost:11434`
    llm = Ollama(
        model="llama3",
        base_url="http://localhost:11434",
        verbose=True,
    )

    # create conversation
    conversation = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vectordb_runtime.as_retriever(),
        return_source_documents=True,
        verbose=True,
    )
    return conversation


def chat(question, user_id):
    # global conversation

    # create_vector_store("sanil")
    conversation = init_conversation(user_id)
    

    chat_history = []
    response = conversation({"question": question, "chat_history": chat_history})
    answer = response['answer']

    logging.info("got response from llm - %s", answer)

    # TODO save history

    return answer