from pinecone import Pinecone
import openai
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from docx import Document
import os
import io
import fitz
from dotenv import load_dotenv




# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPEN_API_KEY")
model = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=os.getenv("OPEN_API_KEY")
)

# Initialize Pinecone v3 client
pc = Pinecone(api_key=os.getenv("PINE_API_KEY"))
index = pc.Index("energy-chunk")

# --- Pinecone Vector Search ---
def find_match(input, namespace):
    input_em = model.embed_query(input)

    result = index.query(
        vector=input_em,
        top_k=3,
        namespace=namespace,
        include_metadata=True
    )

    matches = result.matches

    if not matches:
        return "Hi, how may I help you? Please upload a document."
    elif len(matches) == 1:
        return matches[0].metadata['text']
    else:
        return "\n".join([match.metadata['text'] for match in matches[:3]])

# --- OpenAI Query Refiner ---
def query_refiner(conversation, query):
    client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content":
             f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base. "
             f"And if there is no conversation log please don't change the question and leave it as it is.\n\n"
             f"CONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:"}
        ],
    )
    return response.choices[0].message.content

# --- Session History as String ---
def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses']) - 1):
        conversation_string += "Human: " + st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: " + st.session_state['responses'][i + 1] + "\n"
    return conversation_string

# --- Document Parsing ---
def pdf_to_text(content, file_extension):
    if file_extension == ".txt":
        return content.decode('utf-8')
    elif file_extension == ".pdf":
        pdf_document = fitz.open("pdf", content)
        text = "".join([page.get_text() for page in pdf_document])
        pdf_document.close()
        return text
    elif file_extension == ".docx":
        doc = Document(io.BytesIO(content))
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

# --- Text Cleaning ---
def remove_unwanted_spaces(input_text):
    lines = input_text.split('\n')
    cleaned_lines = [remove_extra_spaces(line) for line in lines if line.strip()]
    return '\n'.join(cleaned_lines)

def remove_extra_spaces(line):
    return ' '.join(line.split()).strip()

# --- Text Splitter ---
def text_splitter(cleaned_text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=4500,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = splitter.split_text(cleaned_text)
    return [doc for text in chunks for doc in splitter.create_documents([text])]

# --- List Pinecone Namespaces (Projects) ---
def get_project_names(index):
    try:
        stats = index.describe_index_stats()
        return list(stats.namespaces.keys()) if stats.namespaces else []
    except Exception as e:
        print(f"Error retrieving project names: {e}")
        return []
