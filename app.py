import os
import streamlit as st
from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf
from pydantic import BaseModel
from typing import Any
import requests
import uuid
from pinecone import Pinecone
from langchain_text_splitters import CharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Define Hugging Face API
model_id = "sentence-transformers/all-MiniLM-L6-v2"
api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}


def generate_embeddings(texts):
    response = requests.post(api_url, headers=headers, json={"inputs": texts, "options": {"wait_for_model": True}})
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error generating embeddings: {response.status_code}, {response.text}")
        return []


# Set up Pinecone
# index_name is your_vector_store_name and dimension should be equal to the length of your embeddings. 
# To check length of your embeddings run    len(generate_embeddings("How are you"))
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("your_vector_store_name")


# Define Pydantic Document model
class Document(BaseModel):
    page_content: Any
    metadata: dict


def process_pdfs(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        doc_path = f"/tmp/{uploaded_file.name}"
        with open(doc_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        raw_pdf_elements = partition_pdf(
            filename=doc_path,
            extract_images_in_pdf=False,
            infer_table_structure=True,
            chunking_strategy="by_title",
            max_characters=4000,
            new_after_n_chars=3800,
            combine_text_under_n_chars=2000,
        )

        count = 0
        for element in raw_pdf_elements:
            element_type = str(type(element))
            if "Table" in element_type:
                documents.append(Document(page_content=str(element),
                                          metadata={"source": uploaded_file.name, "page": count, "type": "table"}))
            elif "CompositeElement" in element_type:
                documents.append(Document(page_content=str(element),
                                          metadata={"source": uploaded_file.name, "page": count, "type": "text"}))
            count += 1

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    embedded_doc = []
    for doc in docs:
        id = str(uuid.uuid1())
        embedding = generate_embeddings(doc.page_content)
        if embedding:
            meta = doc.metadata
            meta["page_content"] = doc.page_content
            embedded_doc.append({"id": id, "values": embedding, "metadata": meta})

    index.upsert(vectors=embedded_doc)
    st.success("PDFs processed and uploaded to the vector store!")


# Initialize Groq model
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="Llama3-8b-8192")


def retreating_answer(query):
    embedded_query = generate_embeddings(query)
    query_response = index.query(top_k=3, vector=embedded_query, include_values=False, include_metadata=True)

    description = ""
    for match in query_response['matches']:
        description += match['metadata']['page_content'] + "\n\n"

    return description + query


# Streamlit UI
st.title("PDF Document Query System")

# File upload
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    if st.button("Process PDFs"):
        with st.spinner("Processing..."):
            process_pdfs(uploaded_files)

# Query input
query = st.text_input("Enter your query")

if query:
    if st.button("Search"):
        with st.spinner("Searching..."):
            llm_input = retreating_answer(query)
            prompt = ChatPromptTemplate.from_messages([("human", "Answer the question{topic}")])
            chain = prompt | llm

            answer = chain.invoke({"topic": llm_input})
            st.write(answer.content)
