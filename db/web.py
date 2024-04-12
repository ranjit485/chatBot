import nest_asyncio
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_community.embeddings.huggingface import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter

nest_asyncio.apply()

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

text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=10)
chunks = text_splitter.split_documents(data)

for i in data:
    page_con = i.page_content
    print(len(page_con))
    # print(page_con)

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