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
    
def handle_post_request(request):
    # Get the request data from the request object
    request_data = request.json
    # Extract the data from the request data
    # For example, to extract a 'name' parameter:
    responseDic = json.loads(request_data)
    query = responseDic['data']['query']
    responseDic['data']['ai_response'] = getqaChain(query)
    responseJson = json.dumps(responseDic)
    return responseJson


st.set_page_config(page_title="CounselorGPT", page_icon=":robot:")
st.header("CounselorGPT")

# Get the incoming requests
requests = st.server.server_request_queue

# Loop through the incoming requests
for request in requests:
        # Check if the request is a POST request
        if request.method == 'POST':
            # Handle the POST request
            response_data = handle_post_request(request)



            
            
            
            
            
            
            
            
            