from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_openai import OpenAIEmbeddings

from pinecone import Pinecone
from dotenv import load_dotenv
import streamlit as st
from streamlit_chat import message
import os
import time

from kelly_bot_test import *

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPEN_API_KEY")

# Initialize Pinecone client and index
pc = Pinecone(api_key=os.getenv("PINE_API_KEY"))
index = pc.Index("energy-chunk")

# Initialize embedding model
model = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=os.getenv("OPEN_API_KEY"))

# Session states
if 'responses' not in st.session_state:
    st.session_state['responses'] = ["Hi, hope you're doing well. Please upload a renewable energy document to get started."]
if 'requests' not in st.session_state:
    st.session_state['requests'] = []
if 'buffer_memory' not in st.session_state:
    st.session_state['buffer_memory'] = ConversationBufferWindowMemory(k=5, return_messages=True)

# Chat setup
llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=os.getenv("OPEN_API_KEY"))

system_msg_template = SystemMessagePromptTemplate.from_template(
    template="Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say 'I don't know'"
)
human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")
prompt_template = ChatPromptTemplate.from_messages([
    system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template
])
conversation = ConversationChain(memory=st.session_state['buffer_memory'], prompt=prompt_template, llm=llm, verbose=True)

# UI
st.subheader("🌱 Renewable Energy PDF Reader")
if 'energy_namespaces' not in st.session_state:
    st.session_state['energy_namespaces'] = get_project_names(index)

response_container = st.container()
textcontainer = st.container()
bucket_name = os.getenv("S3_BUCKET_NAME")

if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = False
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'upload'

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ['Upload Document', 'Ask Questions'])

# Reset responses if changing tab
if st.session_state['current_page'] == 'Ask Questions' and page == 'Upload Document':
    st.session_state['responses'] = ["Hi, hope you're doing well. Please upload a renewable energy document to get started."]
    st.session_state['requests'] = []

st.session_state['current_page'] = page

# Upload Tab
if st.session_state['current_page'] == 'Upload Document':
    st.subheader("Upload Renewable Energy Document")
    energy_doc_name_input = st.text_input("Energy Document Namespace:", key="energy_namespace_input")
    energy_namespace = energy_doc_name_input.strip().replace(" ", "-")
    uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf", "docx"], key="file_uploader")

    if uploaded_file and st.button('Upload'):
        with st.spinner("Processing and embedding the document..."):
            try:
                file_content = uploaded_file.read()
                _, file_extension = os.path.splitext(uploaded_file.name)
                file_extension = file_extension.lower()

                text = pdf_to_text(file_content, file_extension)
                cleaned_text = remove_unwanted_spaces(text)
                data_upload = text_splitter(cleaned_text)
                texts = [doc.page_content for doc in data_upload]

                new_embeddings = model.embed_documents(texts)
                upserts = [{
                    "id": f"{uploaded_file.name}_{i}",
                    "values": new_embeddings[i],
                    "metadata": {"text": texts[i]}
                } for i in range(len(texts))]

                index.upsert(vectors=upserts, namespace=energy_namespace)

                st.session_state['energy_namespaces'] = get_project_names(index)
                st.success(" Document uploaded successfully! You can now ask questions.")
                st.session_state.file_uploaded = True
                st.session_state.uploaded_file = uploaded_file
                time.sleep(2)

            except Exception as e:
                st.error(f"Error handling file upload: {e}")

# Query Tab
elif st.session_state['current_page'] == 'Ask Questions':
    st.subheader("Ask a Question About Energy Documents")
    selected_namespace = st.selectbox("Select Energy Document Namespace:", st.session_state['energy_namespaces'], key="energy_namespace")
    query = st.text_input("Enter your question:", key="input")
    submit_button = st.button("Submit")

    if query and selected_namespace and submit_button:
        with st.spinner("Fetching answers from your documents..."):
            try:
                conversation_string = get_conversation_string()
                refined_query = query_refiner(conversation_string, query)
                context = find_match(refined_query, selected_namespace)
                response = conversation.predict(input=f"Context:\n{context}\n\nQuery:\n{query}")
                st.session_state.requests.append(query)
                st.session_state.responses.append(response)
            except Exception as e:
                st.error(f"Error processing your question: {e}")

# Chat History
with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i], key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state['requests'][i], is_user=True, key=str(i) + '_user')
