# academic_deep_search.py

import streamlit as st
import os
import base64
from typing import List, Optional, Dict, Any, TypedDict, Annotated
import operator
import uuid
import PyPDF2
from fpdf import FPDF
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.chat_models import ChatOllama
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

# --- CONFIGURATION ---

# Set this to your Google API key if you want to use Gemini
os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY"

# Default academic domains to search
DEFAULT_DOMAINS = ["ieeexplore.ieee.org", "dl.acm.org", "arxiv.org", "scholar.google.com", "scopus.com"]

# --- STATE DEFINITION FOR LANGGRAPH ---

class ResearchState(TypedDict):
    query: str
    domains: List[str]
    num_references_per_domain: int
    paper_urls: Annotated[List[str], operator.add]
    downloaded_pdf_paths: Annotated[List[str], operator.add]
    extracted_text: str
    raptor_index: Any # Will hold the RAPTOR retriever
    conversation_history: Annotated[List[BaseMessage], operator.add]
    generation: str

# --- RAPTOR IMPLEMENTATION (Simplified for this example) ---
# A full RAPTOR implementation is complex, so we'll simulate its core idea:
# multi-level clustering and summarization. For a real-world scenario,
# you might use a more robust implementation.

def get_embeddings(llm_provider: str):
    if llm_provider == "gemini":
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    else:
        return OllamaEmbeddings(model="llama3")

def raptor_indexing(text: str, llm_provider: str):
    """
    A simplified RAPTOR-like indexing process.
    1. Splits text into chunks.
    2. Creates a vector store from the chunks.
    (A full implementation would add clustering and summarization layers)
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.create_documents([text])
    
    embeddings = get_embeddings(llm_provider)
    vector_store = FAISS.from_documents(docs, embedding=embeddings)
    return vector_store.as_retriever()

# --- LANGGRAPH NODES ---

def start_search_node(state: ResearchState) -> ResearchState:
    st.write("Starting research process...")
    return {
        "paper_urls": [],
        "downloaded_pdf_paths": [],
        "extracted_text": "",
    }

def web_search_node(state: ResearchState) -> ResearchState:
    st.write("Searching for academic papers...")
    all_urls = []
    for domain in state["domains"]:
        try:
            from duckduckgo_search import DDGS
            query = f'site:{domain} {state["query"]} filetype:pdf'
            with DDGS() as ddgs:
                results = [r['href'] for r in ddgs.text(query, max_results=state["num_references_per_domain"])]
                st.write(f"Found {len(results)} papers on {domain}.")
                all_urls.extend(results)
        except Exception as e:
            st.warning(f"Could not search {domain}: {e}")
    
    unique_urls = list(set(all_urls))
    st.write(f"Found a total of {len(unique_urls)} unique papers.")
    return {"paper_urls": unique_urls}

def download_pdfs_node(state: ResearchState) -> ResearchState:
    st.write("Downloading PDFs...")
    pdf_paths = []
    if not os.path.exists("temp_pdfs"):
        os.makedirs("temp_pdfs")

    for url in state["paper_urls"]:
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            
            # Generate a unique filename
            filename = f"temp_pdfs/{uuid.uuid4()}.pdf"
            with open(filename, "wb") as f:
                f.write(response.content)
            pdf_paths.append(filename)
            st.write(f"Successfully downloaded {url}")
        except Exception as e:
            st.warning(f"Failed to download {url}: {e}")
    
    return {"downloaded_pdf_paths": pdf_paths}

def extract_text_node(state: ResearchState) -> ResearchState:
    st.write("Extracting text from PDFs...")
    full_text = ""
    for path in state["downloaded_pdf_paths"]:
        try:
            loader = PyPDFLoader(path)
            pages = loader.load_and_split()
            for page in pages:
                full_text += page.page_content + "\n\n"
        except Exception as e:
            st.warning(f"Could not read {path}: {e}")
    
    return {"extracted_text": full_text}

def build_raptor_index_node(state: ResearchState) -> ResearchState:
    st.write("Building RAPTOR index...")
    llm_provider = st.session_state.get("llm_provider", "ollama")
    index = raptor_indexing(state["extracted_text"], llm_provider)
    st.success("Research and indexing complete! You can now ask questions.")
    return {"raptor_index": index}

# --- LANGGRAPH DEFINITION ---

builder = StateGraph(ResearchState)
builder.add_node("start_search", start_search_node)
builder.add_node("web_search", web_search_node)
builder.add_node("download_pdfs", download_pdfs_node)
builder.add_node("extract_text", extract_text_node)
builder.add_node("build_raptor_index", build_raptor_index_node)

builder.add_edge(START, "start_search")
builder.add_edge("start_search", "web_search")
builder.add_edge("web_search", "download_pdfs")
builder.add_edge("download_pdfs", "extract_text")
builder.add_edge("extract_text", "build_raptor_index")
builder.add_edge("build_raptor_index", END)

graph = builder.compile()

# --- HELPER FUNCTIONS FOR EXPORT ---

def generate_pdf_report(chat_history: List[Dict[str, str]]) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.cell(200, 10, txt="Academic QA Chat History", ln=True, align='C')
    
    for message in chat_history:
        if message['role'] == 'user':
            pdf.set_text_color(0, 0, 255) # Blue for user
            pdf.multi_cell(0, 10, f"Question: {message['content']}")
        else:
            pdf.set_text_color(0, 0, 0) # Black for AI
            pdf.multi_cell(0, 10, f"Answer: {message['content']}")
        pdf.ln(5)
        
    return pdf.output(dest='S').encode('latin-1')

def generate_mermaid_diagram(query: str, domains: List[str], num_refs: int, found_urls: int) -> str:
    diagram = f"""
graph TD
    A[Start: User Input] --> B(LangGraph Pipeline);
    B --> C[Web Search];
    C --> D[Download PDFs];
    D --> E[Extract Text];
    E --> F[Build RAPTOR Index];
    F --> G[Conversational QA];
    
    subgraph Parameters
        P1("Query: {query}");
        P2("Domains: {', '.join(domains)}");
        P3("References per Domain: {num_refs}");
    end
    
    subgraph Results
        R1("Found URLs: {found_urls}");
    end
    
    A -- Parameters --> P1;
    A -- Parameters --> P2;
    A -- Parameters --> P3;
    C -- Results --> R1;
"""
    return diagram

# --- STREAMLIT UI ---

def main():
    st.set_page_config(layout="wide", page_title="Academic Deep Search")
    st.title("📚 Academic Deep Search & QA")

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.research_done = False
        st.session_state.final_state = None

    with st.sidebar:
        st.header("Research Parameters")
        
        query = st.text_input("Academic Topic of Interest", "indoor quality monitoring using machine learning")
        
        domains_str = st.text_area("Webpage Domains (one per line)", "\n".join(DEFAULT_DOMAINS))
        
        num_references = st.slider("References per Domain", 1, 10, 3)
        
        llm_provider = st.selectbox("LLM Provider", ["ollama", "gemini"], key="llm_provider")
        
        if llm_provider == "ollama":
            ollama_model = st.text_input("Ollama Model", "llama3")
        else:
            gemini_model = "gemini-pro"
            st.write("Using Gemini Pro")
            if os.environ.get("GOOGLE_API_KEY") == "YOUR_GOOGLE_API_KEY":
                st.warning("Please set your Google API Key.")

        if st.button("Start Research"):
            with st.spinner("Running deep research pipeline... This may take a while."):
                domains = [d.strip() for d in domains_str.split("\n") if d.strip()]
                initial_state = {
                    "query": query,
                    "domains": domains,
                    "num_references_per_domain": num_references,
                }
                
                final_state = graph.invoke(initial_state)
                st.session_state.research_done = True
                st.session_state.final_state = final_state
                st.session_state.messages = [] # Reset chat
    
    if st.session_state.research_done:
        st.header("Conversational QA")
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask a question about the papers..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    retriever = st.session_state.final_state['raptor_index']
                    
                    if llm_provider == "gemini":
                        llm = ChatGoogleGenerativeAI(model=gemini_model, temperature=0.3)
                    else:
                        llm = ChatOllama(model=ollama_model, temperature=0.3)

                    prompt_template = ChatPromptTemplate.from_messages([
                        ("system", "You are an AI research assistant. Answer the user's question based on the following context:\n\n{context}\n\nIf the context doesn't contain the answer, say so."),
                        ("human", "{question}")
                    ])
                    
                    retrieved_docs = retriever.get_relevant_documents(prompt)
                    context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
                    
                    chain = prompt_template | llm
                    
                    response = chain.invoke({"context": context, "question": prompt})
                    response_content = response.content
                    st.markdown(response_content)
            
            st.session_state.messages.append({"role": "assistant", "content": response_content})
        
        # Export options
        st.header("Export Options")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Export Chat as PDF"):
                pdf_bytes = generate_pdf_report(st.session_state.messages)
                st.download_button(
                    label="Download PDF",
                    data=pdf_bytes,
                    file_name="chat_history.pdf",
                    mime="application/pdf"
                )

        with col2:
            if st.button("Generate Pipeline Diagram"):
                mermaid_code = generate_mermaid_diagram(
                    query=st.session_state.final_state['query'],
                    domains=st.session_state.final_state['domains'],
                    num_refs=st.session_state.final_state['num_references_per_domain'],
                    found_urls=len(st.session_state.final_state['paper_urls'])
                )
                st.code(mermaid_code, language="mermaid")
                st.info("You can copy this code and render it in a Mermaid.js compatible viewer.")

    else:
        st.info("Please configure your research parameters in the sidebar and click 'Start Research'.")


if __name__ == "__main__":
    main()