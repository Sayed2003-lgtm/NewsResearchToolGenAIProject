import os
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import (
GoogleGenerativeAI,
GoogleGenerativeAIEmbeddings
)

from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain

#Load API Key

load_dotenv()

#Streamlit UI

st.title("News Research Tool ")

#Sidebar

st.sidebar.title("News Article URLs")

url1 = st.sidebar.text_input("URL 1")
url2 = st.sidebar.text_input("URL 2")
url3 = st.sidebar.text_input("URL 3")

process_url_clicked = st.sidebar.button("Process URLs")

file_path = "faiss_index"

#Process URLs

if process_url_clicked:
    urls = [url1, url2, url3]
    urls = [url for url in urls if url.strip() != ""]

    if len(urls) == 0:
        st.error("Please enter at least one URL")
    else:
        with st.spinner("Loading URLs and creating embeddings..."):

            # Load Articles
            loader = UnstructuredURLLoader(urls=urls)
            data = loader.load()

            # Split Text
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            docs = text_splitter.split_documents(data)

            # Embeddings
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001"
            )

        # Create FAISS Index
        vectorstore = FAISS.from_documents(
            docs,
            embeddings
        )

        # Save Index
        vectorstore.save_local(file_path)

        st.success("URLs processed successfully!")
#Question Input

query = st.text_input("Ask a Question")

if query:

    try:

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001"
        )

        vectorstore = FAISS.load_local(
            file_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

        llm = GoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.7
        )

        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever()
        )

        result = chain.invoke(
            {"question": query}
        )

        st.header("Answer")
        st.write(result["answer"])

    except Exception as e:
        st.error(f"Error: {str(e)}")