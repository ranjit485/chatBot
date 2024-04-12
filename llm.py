#
# __import__('pysqlite3')
#
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os

import nest_asyncio
from langchain.schema import retriever
from langchain_community.llms.huggingface_hub import HuggingFaceHub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from db.web import vectorstore

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

nest_asyncio.apply()

llm = HuggingFaceHub(
    repo_id="huggingfaceh4/zephyr-7b-alpha",
    model_kwargs={"temperature": 0.5, "max_length": 64, "max_new_tokens": 512}
)


#  Generate response with llm
def llm_ans_question(input):
    # search in db
    search_db = vectorstore.similarity_search(input)
    context = search_db[0].page_content

    template = f"""

    User: You are Snehal an AI Assistant of Adarsh Institute of Technology and Research Centre Vita,Maharashtra,
     in that follows instructions extremely well.
    Please be truthful and give direct answers. Please tell 'I don't know' if user query is not in CONTEXT
    Do not give long answer.
    Keep in mind, you will lose the job, if you answer out of CONTEXT questions.
    Dont expose anyone's profile those who not related with CONTEXT.
    You can use list , bullets , emojis, for better conversation.
    CONTEXT: {context}
    Query: {input}

    Remember only return answer from CONTEXT
    Don’t justify your answers. Don’t give information not mentioned in the CONTEXT INFORMATION
    I don't know' if {input} is not in {context}  
    Assistant:"""

    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
            {"context": retriever, "query": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    response = rag_chain.invoke(input)
    print("RAG RESPONSE :", response)
    # print(template)
    onlyAns = response.split("Assistant:")
    print("Index 0 ", onlyAns[0])
    print("Index 1 ", onlyAns[1])
    return onlyAns[1]
