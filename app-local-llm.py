
import os
from dotenv import load_dotenv

import streamlit as st

from langchain.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores.qdrant import Qdrant
from qdrant_client import QdrantClient
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter
)
from langchain.prompts import PromptTemplate
from tempfile import NamedTemporaryFile

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

model_name = "mistral"
# model_name = "llama2"

collection_name="ui_local_collection",

def main():

    index_placeholder = None
    st.set_page_config(page_title = "Chat with your PDF Using Ollama Local Model", page_icon="📔")
    st.header('📔 Chat with your PDF using Ollama Local Model & Qdrant')

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "activate_chat" not in st.session_state:
        st.session_state.activate_chat = False

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar = message['avatar']):
            st.markdown(message["content"])


    with st.sidebar:
        st.subheader('Upload Your PDF File')
        docs = st.file_uploader('⬆️ Upload your PDF & Click to process',
                                accept_multiple_files = True, 
                                type=['pdf'])
        if st.button('Process'):
            documents = []
            with NamedTemporaryFile(dir='.', suffix='.pdf') as f:
                with st.spinner('Processing'):
                    for file in docs:
                        f.write(file.getbuffer())
                        file_name = f.name
                        loader = PyPDFLoader(file_name)
                        documents.extend(loader.load())

                    text_splitter = RecursiveCharacterTextSplitter(
                        # Set a really small chunk size, just to show.
                        chunk_size=1000,
                        chunk_overlap=200,
                        length_function=len,
                    )

                    metaData = []
                    page_content= []
                    for d in documents:
                        page_content.append(d.page_content)
                        metaData.append(d.metadata)

                    print("metaData ", metaData)

                    pdfData = text_splitter.create_documents(texts=page_content, metadatas=metaData)

                    llm = Ollama(model=model_name)
                    embeddings = OllamaEmbeddings(model=model_name)

                    pdf_index = Qdrant.from_documents(
                        pdfData,
                        embeddings,
                        url="http://localhost:6333",
                        collection_name=collection_name,
                    )

                    if "pdf_index" not in st.session_state:
                        st.session_state.pdf_index = pdf_index
                    st.session_state.activate_chat = True

    if st.session_state.activate_chat == True:
        if prompt := st.chat_input("Ask your question from the PDF?"):
            with st.chat_message("user", avatar = '👨🏻'):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", 
                                              "avatar" :'👨🏻',
                                              "content": prompt})
            print("============ prompt =============", prompt)
            index_placeholder = st.session_state.pdf_index

            found_docs = index_placeholder.similarity_search(prompt, k=4)

            template = """Given the following extracted parts of a long document and a question, create a final answer. 
            If you don't know the answer, just say that you don't know. Don't try to make up an answer.
            QUESTION: 
            {question}
            =========
            CONTEXT:
            {context}
            =========
            FINAL ANSWER:"""
            PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

            chain = load_qa_chain(Ollama(temperature=0), chain_type="stuff", prompt=PROMPT)
            answer = chain.run(input_documents=found_docs, question=prompt)
            with st.chat_message("assistant", avatar='🤖'):
                st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", 
                                              "avatar" :'🤖',
                                              "content": answer})
        
        else:
            st.markdown(
                'Upload your PDFs to chat'
                )


if __name__ == '__main__':
    main()