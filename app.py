from flask_cors import CORS
from langchain_community.document_loaders import TextLoader, WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFaceHub
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
import nest_asyncio
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import Chroma
import os
from flask import Flask, request, render_template
import sys

__import__('pysqlite3')

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

nest_asyncio.apply()
#
# loader = TextLoader("Data/d1.txt")
# docs = loader.load()
#
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=109, chunk_overlap=20)
# chunks = text_splitter.split_documents(docs)

URLS = [
    'http://aitrcvita.edu.in/index.php',
    'http://aitrcvita.edu.in/',
    'http://aitrcvita.edu.in/about-us.php',
    'http://aitrcvita.edu.in/about-institute.php',
    'http://aitrcvita.edu.in/fpmessage.php',
    'http://aitrcvita.edu.in/pmessage.php',
    'http://aitrcvita.edu.in/edmessage.php',
    'http://aitrcvita.edu.in/cdmessage.php',
    'http://aitrcvita.edu.in/tpomessage.php',
    'http://aitrcvita.edu.in/prmessage.php',
    'http://aitrcvita.edu.in/ddmessage.php',
    'http://aitrcvita.edu.in/btech_civil.php',
    'http://aitrcvita.edu.in/btech_cse.php',
    'http://aitrcvita.edu.in/btech_e&tc.php',
    'http://aitrcvita.edu.in/btech_mech.php',
    'http://aitrcvita.edu.in/department.php',
    'http://aitrcvita.edu.in/btech_ai_ml.php',
    'http://aitrcvita.edu.in/btech_ece.php',
    'http://aitrcvita.edu.in/diploma_ce.php',
    'http://aitrcvita.edu.in/diploma_cm.php',
    'http://aitrcvita.edu.in/diploma_ej.php',
    'http://aitrcvita.edu.in/diploma_ee.php',
    'http://aitrcvita.edu.in/diploma_me.php',
    'http://aitrcvita.edu.in/diploma_gsc.php',
    'http://aitrcvita.edu.in/diploma_ai.php',
    'http://aitrcvita.edu.in/library.php',
    'http://aitrcvita.edu.in/infrastructure.php',
    'http://aitrcvita.edu.in/admission-btech.php',
    'http://aitrcvita.edu.in/admission-poly.php',
    'http://aitrcvita.edu.in/academics_btech.php',
    'http://aitrcvita.edu.in/academics_poly.php',
    'http://aitrcvita.edu.in/tnp.php',
    'http://aitrcvita.edu.in/governance.php',
    'http://aitrcvita.edu.in/rti.php',
    'http://aitrcvita.edu.in/contact-us.php',
    'http://aitrcvita.edu.in/naac.php',
    'http://aitrcvita.edu.in/shop-details.html',
    'http://aitrcvita.edu.in/placement.php',
    'https://www.aitrcvita.edu.in/rti.php',
]


loader = WebBaseLoader(URLS)
data = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
chunks = text_splitter.split_documents(data)

# for i in data:
#     page_con = i.page_content
#     print(len(page_con))
#     print(page_con)

print("this is demo chunk :", chunks[1])

tokn = "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"
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=tokn, model_name="BAAI/bge-base-en-v1.5"
)

vectorstore = Chroma.from_documents(chunks, embeddings)

retriever = vectorstore.as_retriever(
    search_type="mmr",  # similarity
    search_kwargs={'k': 4}
)

llm = HuggingFaceHub(
    repo_id="huggingfaceh4/zephyr-7b-alpha",
    model_kwargs={"temperature": 0.5, "max_length": 64, "max_new_tokens": 512}
)


# Search in database
def search_in_db(userinput):
    search_db = vectorstore.similarity_search(userinput)
    semetric_result_from_db = search_db[0].page_content
    return semetric_result_from_db


#  Generate response with llm
def ans_question(input):
    search_db = vectorstore.similarity_search(input)
    context = search_db[0].page_content

    inp = "{query}"

    template = f"""
    
    User: You are an AI Assistant of Adarsh Institute of Technology and Research Centre, in that follows instructions extremely well.
    Please be truthful and give direct answers. Please tell 'I don't know' if user query is not in CONTEXT
    
    Keep in mind, you will lose the job, if you answer out of CONTEXT questions
    
    CONTEXT: {context}
    Query: {input}
    
    Remember only return AI answer
    Assistant:
    """

    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
            {"context": retriever, "query": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    response = rag_chain.invoke(input)
    print("RAG RESPONSE :", response)
    print(template)
    onlyAns = response.split("Assistant:")
    print("Index 0 ", onlyAns[0])
    print("Index 1 ", onlyAns[1])
    return onlyAns[1]


app = Flask(__name__, template_folder='template')

CORS(app)  # Enable CORS for all routes in the app


@app.route('/')
def start():
    return render_template("index.html")


@app.route('/chat')
def chat():
    return render_template("index.html")


@app.route('/api/v0/ask', methods=['GET'])
def generate_ans():
    user_query = request.args.get('question')

    if user_query:
        return ans_question(user_query)
    else:
        return "No question provided in the request."


# if __name__ == '__main__':
#     app.run()
