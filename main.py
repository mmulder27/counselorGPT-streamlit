#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  7 20:22:21 2023

@author: michaelmulde
"""

from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

import pinecone
import streamlit as st
import requests
import json


#Initialize OpenAIEmbeddings and Pinecone
embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
pinecone.init(api_key=st.secrets["PINECONE_API_KEY"],environment=st.secrets["PINECONE_API_ENV"])
index_name = "langchain2" 

#Define Pinecone indices for semantic search
docsearch1 = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)

def getqaChain(query):
    llm = OpenAI(temperature=0, openai_api_key=st.secrets["OPENAI_API_KEY"])
    chain = load_qa_chain(llm, chain_type="stuff")
    query=query
    docs = docsearch1.similarity_search(query, include_metadata=True)
    return chain.run(input_documents=docs, question=query)
    



st.set_page_config(page_title="CounselorGPT", page_icon=":robot:")
st.header("CounselorGPT")

st.markdown("## Ask me a question. Imagine you are having a conversation with the UCLA Course Catalog.")

def get_text():
    input_text = st.text_area(label="", placeholder="Your Email. . .", keys='email_input')
    return input_text

email_input = get_text();

st.markdown("##CounselorGPT's Response:")
st.write(getqaChain(email_input))
            
            
            
            
            
            
            
            
            