#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  7 20:22:21 2023

@author: michaelmulder
"""

from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from streamlit_autorefresh import st_autorefresh
import openai
import pinecone
import streamlit as st


#Initialize OpenAIEmbeddings and Pinecone
embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
pinecone.init(api_key=st.secrets["PINECONE_API_KEY"],environment=st.secrets["PINECONE_API_ENV"])
index_name = "langchain2" 

#Define Pinecone indices for semantic search
docsearch1 = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)

llm = OpenAI(temperature=0, openai_api_key=st.secrets["OPENAI_API_KEY"])
chain = load_qa_chain(llm, chain_type="stuff")


#Configure frontend
st.set_page_config(page_title="CounselorGPT", page_icon=":robot:")
st.header("CounselorGPT")
st.markdown("### Ask me a question. Imagine you are having a conversation with the UCLA Course Catalog.")

#Create function to get user's question
def get_text():
    input_text = st.text_area(label=" ", placeholder= "Your Question. . . ", key="question")
    return input_text

#Get user's question
query = get_text();

#Create area for chatbot's response
st.markdown("### CounselorGPT's Response:")

#Write chatbot's response
if query:
    docs = docsearch1.similarity_search(query)
    response = chain.run(input_documents=docs, question=query)
    st.write(response)  

##Initialize refresh counter component
count = st_autorefresh(interval=172800000, limit=None, key="refresh_counter")





            
            
            
            
            
            