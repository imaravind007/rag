import tempfile
from typing import List
from pinecone import Pinecone
import openai
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from docx import Document
import pandas as pd
import os
import io
import fitz
from dotenv import load_dotenv
from langchain_docling import DoclingLoader
from docling.chunking import HierarchicalChunker
from langchain_core.documents import Document 



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
    sources = set([match.metadata['title'] for match in matches])
    if not matches:
        return "Sorry, I couldn't find any relevant information.", []
    elif len(matches) == 1:
        return matches[0].metadata['text'], sources
    else:
        return "\n".join([match.metadata['text'] for match in matches]), sources

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
def load_docs(content: bytes, ext: str) -> List[Document]:
    """
    Return a list[Document] ready for embedding.
      * For PDFs (and any other format Docling supports) we call DoclingLoader
        with HierarchicalChunker.
      * For everything else we fall back to the previous
        text‑cleanup  ➜  RecursiveCharacterTextSplitter path.
    """
    ext = ext.lower()
    if ext in {".pdf", ".docx", ".pptx", ".html"}:   
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(content)
            tmp.flush()                 
            loader = DoclingLoader(
                file_path=tmp.name,
                chunker=HierarchicalChunker(),           
            )
            docs = loader.load()           
        return docs

    # ----- For txt / xlsx (or anything Docling can’t parse yet)
    text      = pdf_to_text(content, ext)          
    cleaned   = remove_unwanted_spaces(text)             
    return text_splitter(cleaned)    

def pdf_to_text(content, file_extension):
    if file_extension == ".txt":
        return content.decode('utf-8')
    elif file_extension == ".pdf":
        pdf_document = fitz.open("pdf", content)
        text = "".join([f"{page.get_text()}\nPage:{str(page.number+1)}" for page in pdf_document])
        pdf_document.close()
        return text
    elif file_extension == ".docx":
        doc = Document(io.BytesIO(content))
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    elif file_extension in [".xlsx", ".xlsm", ".xlsb", ".xls"]:
        excel_io = io.BytesIO(content)
        dfs = pd.read_excel(excel_io, sheet_name=None)  # All sheets
        combined_text = ""
        for sheet_name, df in dfs.items():
            combined_text += f"\n--- Sheet: {sheet_name} ---\n"
            combined_text += df.fillna("").astype(str).to_csv(index=False, header=True)
        return combined_text
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
