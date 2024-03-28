from langchain.vectorstores import qdrant
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct
from openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

import streamlit as st


from pathlib import Path
from pprint import pprint

import os

from langchain.schema import Document


def main():
    from dotenv import load_dotenv
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=OPENAI_API_KEY)

    st.set_page_config(page_title = "Chat with your PDF Using OpenAI", page_icon="📔")
    st.header('📔 Chat with your PDF using OpenAI & Qdrant')

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "activate_chat" not in st.session_state:
        st.session_state.activate_chat = False

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar = message['avatar']):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask your question from the PDF?"):
        with st.chat_message("user", avatar = '👨🏻'):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", 
                                            "avatar" :'👨🏻',
                                            "content": prompt})
        print("============ prompt =============", prompt)

        collection_name = "MI_SRH_collection"
        qclient = QdrantClient(url="http://localhost:6333")

        query_vector = client.embeddings.create(input = [prompt], model='text-embedding-ada-002').data[0].embedding

        found_docs = qclient.search(
            collection_name=collection_name,
            query_vector=query_vector, 
            limit=3
        )

        docs= []
        for data in found_docs:
            payload = data.payload
            doc = Document(page_content = payload['page_content'])
            docs.append(doc)
        
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

        print("docs ===  ", docs)

        chain = load_qa_chain(ChatOpenAI(temperature=0), chain_type="stuff", prompt=PROMPT)
        answer = chain.run(input_documents=docs, question=prompt)
        with st.chat_message("assistant", avatar='🤖'):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", 
                                            "avatar" :'🤖',
                                            "content": answer})

    else:
        st.markdown(
            'Feel free to inquire about PDFs via chat.'
            )
    

if __name__ == '__main__':
    main()