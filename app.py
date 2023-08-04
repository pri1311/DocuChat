import streamlit as st
from langchain.document_loaders import OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import pinecone

st.set_page_config(page_title="DocuChat", page_icon=":bookmark_tabs", layout="wide")

OPENAI_API_KEY = st.sidebar.text_input('OpenAI API Key')
PINECONE_API_KEY = st.sidebar.text_input('Pinecone API Key')
PINECONE_API_ENV = st.sidebar.text_input('Pinecone Enviroment')
PINECONE_INDEX_NAME = st.sidebar.text_input('Pinecone Index Name')
PDF_LINK = st.sidebar.text_input('Document Link')

def generate_response(query):
    loader = OnlinePDFLoader(PDF_LINK)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
    texts = text_splitter.split_documents(data)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    pinecone.init(
        api_key=PINECONE_API_KEY,  # find at app.pinecone.io
        environment=PINECONE_API_ENV  # next to api key in console
    )
    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    chain = load_qa_chain(llm, chain_type="stuff")
    docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=PINECONE_INDEX_NAME)
    docs = docsearch.similarity_search(query)
    chain.run(input_documents=docs, question=query)

with st.form('my_form'):
    text = st.text_area('Enter your question:', 'What are the three key pieces of advice for learning how to code?')
    submitted = st.form_submit_button('Submit')
    if not OPENAI_API_KEY.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
    if not PINECONE_API_KEY:
        st.warning('Please enter your Pinecone API key!', icon='⚠')
    if submitted and OPENAI_API_KEY.startswith('sk-'):
        generate_response(text)