from flask_cors import CORS
from langchain.document_loaders import TextLoader
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

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

nest_asyncio.apply()

loader = TextLoader("Data/d1.txt")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=109, chunk_overlap=20)
chunks = text_splitter.split_documents(docs)

for i in docs:
    page_con = i.page_content
    print(len(page_con))
    print(page_con)

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
    semetric_result = search_db[0].page_content

    inp = "{query}"

    template = f"""
    <|system|>
    You are an AI assistant of Adarsh Institute of Technology and Research Centre Vita that follows instruction extremely well.
      Please be truthful and give direct answers from India Maharashtra,Sangli,
      Consider Only Given information for response : {semetric_result},
      Instructions 1: Use bullets,list,symbols,emojis for better conversation.
      </s>
      <|user|>{inp}</s><|assistant|>"""

    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
            {"context": retriever, "query": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    response = rag_chain.invoke(input)
    print(response)
    print(template)
    onlyAns = response.split("<|assistant|>")
    print(onlyAns)
    return onlyAns[1]


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes in the app


@app.route('/')
def start():
    return "Chat Bot is running"


@app.route('/chat')
def chat():
    return render_template("Templates/index.html")


@app.route('/api/v0/ask', methods=['GET'])
def generate_ans():
    user_query = request.args.get('question')

    if user_query:
        return ans_question(user_query)
    else:
        return "No question provided in the request."


if __name__ == '__main__':
    app.run()
