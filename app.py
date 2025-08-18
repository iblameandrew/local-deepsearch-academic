import streamlit as st
import os
import numpy as np
import json
from sklearn.cluster import KMeans
from typing import List, Optional, Dict, Any, TypedDict, Annotated, Tuple
import operator
import uuid
from fpdf import FPDF
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import arxiv
from collections import Counter, defaultdict
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---

DEFAULT_PUBLISHERS = ["IEEE", "ACM", "Springer", "Elsevier", "IEEE Explorer", "IEEE Transactions"]

# --- RAPTOR IMPLEMENTATION (No changes here) ---

class RAPTORRetriever(BaseRetriever):
    raptor_index: Any
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        return self.raptor_index.retrieve(query)

class RAPTOR:
    def __init__(self, llm, embeddings_model, session_id, chunk_size=1000, chunk_overlap=200):
        self.llm = llm
        self.embeddings_model = embeddings_model
        self.session_id = session_id
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.tree = {}
        self.all_nodes: Dict[str, Document] = {}
        self.vector_store = None
        self.checkpoint_path = f"checkpoint_{self.session_id}.json"

    def _save_checkpoint(self, level):
        state = {
            "level": level,
            "tree": {str(k): [node_id for node_id in v] for k, v in self.tree.items()},
            "all_nodes": {node_id: doc.to_json() for node_id, doc in self.all_nodes.items()},
        }
        with open(self.checkpoint_path, 'w') as f:
            json.dump(state, f)
        st.write(f"Checkpoint saved for level {level}.")

    def _load_checkpoint(self) -> int:
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, 'r') as f:
                    state = json.load(f)
                from langchain_core.load import load
                self.all_nodes = {node_id: load(doc) for node_id, doc in state["all_nodes"].items()}
                self.tree = state["tree"]
                start_level = state["level"]
                st.info(f"Resuming from checkpoint at level {start_level}.")
                return start_level
            except Exception as e:
                st.warning(f"Could not load checkpoint file due to error: {e}. Starting from scratch.")
                return 0
        return 0

    def add_documents(self, documents: List[Document]):
        start_level = self._load_checkpoint()
        if start_level == 0:
            st.write("Step 1: Assigning IDs to initial chunks (Level 0)...")
            level_0_node_ids = []
            for i, doc in enumerate(documents):
                node_id = f"0_{i}"
                self.all_nodes[node_id] = doc
                level_0_node_ids.append(node_id)
            self.tree[str(0)] = level_0_node_ids
            self._save_checkpoint(0)
        
        current_level = start_level
        while len(self.tree[str(current_level)]) > 1:
            next_level = current_level + 1
            st.write(f"Step 2: Building Level {next_level} of the tree...")
            current_level_node_ids = self.tree[str(current_level)]
            current_level_docs = [self.all_nodes[nid] for nid in current_level_node_ids]
            clustered_indices = self._cluster_nodes(current_level_docs)
            
            next_level_node_ids = []
            num_clusters = len(clustered_indices)
            summary_progress = st.progress(0, text=f"Summarizing Level {next_level}...")
            for i, indices in enumerate(clustered_indices):
                cluster_docs = [current_level_docs[j] for j in indices]
                summary, combined_metadata = self._summarize_cluster(cluster_docs)
                summary_doc = Document(page_content=summary, metadata=combined_metadata)
                node_id = f"{next_level}_{i}"
                self.all_nodes[node_id] = summary_doc
                next_level_node_ids.append(node_id)
                summary_progress.progress((i + 1) / num_clusters, text=f"Summarizing cluster {i+1}/{num_clusters} for Level {next_level}...")
            
            self.tree[str(next_level)] = next_level_node_ids
            self._save_checkpoint(next_level)
            current_level = next_level

        st.write("Step 3: Creating final vector store from all nodes...")
        final_docs = list(self.all_nodes.values())
        self.vector_store = FAISS.from_documents(documents=final_docs, embedding=self.embeddings_model)
        st.write("RAPTOR index built successfully!")
        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)

    def _cluster_nodes(self, docs: List[Document]) -> List[List[int]]:
        st.write(f"Embedding {len(docs)} nodes for clustering...")
        embeddings = self.embeddings_model.embed_documents([doc.page_content for doc in docs])
        n_clusters = max(2, len(docs) // 10)
        st.write(f"Clustering into {n_clusters} groups...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto').fit(embeddings)
        clusters = [[] for _ in range(n_clusters)]
        for i, label in enumerate(kmeans.labels_):
            clusters[label].append(i)
        return clusters

    def _summarize_cluster(self, cluster_docs: List[Document]) -> Tuple[str, dict]:
        context = "\n\n---\n\n".join([doc.page_content for doc in cluster_docs])
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are an AI assistant that summarizes academic texts. Create a concise, abstractive summary of the following content, synthesizing the key information."),
            HumanMessage(content="Please summarize the following content:\n\n{context}")
        ])
        chain = prompt | self.llm
        response = chain.invoke({"context": context})
        summary = response.content
        aggregated_sources = list(set(doc.metadata.get("url", "Unknown Source") for doc in cluster_docs))
        combined_metadata = {"sources": aggregated_sources}
        return summary, combined_metadata
    
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        return self.vector_store.similarity_search(query, k=k) if self.vector_store else []
    
    def as_retriever(self) -> BaseRetriever:
        return RAPTORRetriever(raptor_index=self)

# --- STATE DEFINITION FOR LANGGRAPH ---
class ResearchState(TypedDict):
    query: str
    publishers: List[str]
    num_references: int

    # Final papers for bibliography
    final_kept_papers: Dict[str, List[str]]
    path_to_metadata_map: Dict[str, Dict[str, str]]
    
    # Intermediate state for pipeline
    all_arxiv_results: List[Dict[str, Any]]
    papers_to_download: List[Dict[str, Any]]
    extracted_docs: List[Document]
    relevant_docs: List[Document]
    
    # Logging and RAPTOR index
    discard_log: List[Dict[str, str]]
    raptor_index: Any
    conversation_history: Annotated[List[BaseMessage], operator.add]

# --- LANGGRAPH NODES AND GRAPH DEFINITION ---
def start_search_node(state: ResearchState) -> ResearchState:
    st.write("Starting research process...")
    return {"discard_log": [], "final_kept_papers": {}}

def arxiv_search_node(state: ResearchState) -> ResearchState:
    st.write("Stage 1: Searching arXiv for candidate papers...")
    query_terms = state["query"]
    st.write(f"Query terms: {query_terms}")
    
    client = arxiv.Client()
    
    search = arxiv.Search(
        query = query_terms,
        max_results = 2000,
        sort_by = arxiv.SortCriterion.SubmittedDate
    )

    results =client.results(
        search = search
    )

    results = len(results)

    
    all_results = []
    search_progress = st.progress(0, text="Fetching results from arXiv...")
    
    try:
        total_results = len(results)

        for i, result in enumerate(results):
            search_progress.progress((i + 1) / total_results, text=f"Fetching result {i+1}/{total_results}")
            if result.pdf_url:
                all_results.append({
                    "url": result.pdf_url,
                    "title": result.title,
                    "result_obj": result
                })
    except arxiv.UnexpectedEmptyPageError:
        st.info("Processed all available results from arXiv for this query.")
    except Exception as e:
        st.error(f"An error occurred during arXiv search: {e}")

    st.success(f"Found {len(all_results)} candidate papers on arXiv.")
    return {"all_arxiv_results": all_results}

def filter_by_publisher_node(state: ResearchState) -> ResearchState:
    st.write("Stage 2: Filtering by Publisher...")
    
    filter_criteria = {
        "IEEE": lambda r: (r.journal_ref and "IEEE" in r.journal_ref) or (r.doi and "10.1109" in r.doi),
        "IEEE Explorer": lambda r: (r.journal_ref and "IEEE Explorer" in r.journal_ref),      
        "IEEE Transactions": lambda r: (r.journal_ref and "IEEE Transactions" in r.journal_ref),      
        "ACM": lambda r: (r.journal_ref and "ACM" in r.journal_ref) or (r.doi and "10.1145" in r.doi),
        "Springer": lambda r: (r.journal_ref and "Springer" in r.journal_ref) or (r.doi and "10.1007" in r.doi),
        "Elsevier": lambda r: (r.journal_ref and "Elsevier" in r.journal_ref) or (r.doi and "10.1016" in r.doi)
    }

    papers_to_download = []
    discard_log = state.get("discard_log", [])
    
    for paper in state["all_arxiv_results"]:
        result = paper["result_obj"]
        matched_publisher = "Unknown"
        is_kept = False
        
        for pub, checker in filter_criteria.items():
            if checker(result):
                matched_publisher = pub
                break
        
        if matched_publisher in state["publishers"]:
            paper_meta = paper.copy()
            paper_meta["publisher"] = matched_publisher
            papers_to_download.append(paper_meta)
            is_kept = True
        
        if not is_kept:
            discard_log.append({
                "url": paper["url"],
                "title": paper["title"],
                "reason": "Publisher not in selected list",
                "publisher": matched_publisher
            })
            
    discarded_count = len(state["all_arxiv_results"]) - len(papers_to_download)
    st.info(f"Discarded {discarded_count} papers due to publisher mismatch.")
    st.success(f"{len(papers_to_download)} papers passed the publisher filter.")
    
    return {"papers_to_download": papers_to_download, "discard_log": discard_log}

def download_pdfs_node(state: ResearchState) -> ResearchState:
    st.write("Stage 3: Downloading and validating PDFs...")
    path_to_metadata_map = {}
    discard_log = state.get("discard_log", [])
    
    if not os.path.exists("temp_pdfs"):
        os.makedirs("temp_pdfs")
    
    papers_to_download = state["papers_to_download"]
    total_urls = len(papers_to_download)
    
    if total_urls == 0:
        st.warning("No papers to download after filtering.")
        return {"path_to_metadata_map": {}, "discard_log": discard_log}
        
    download_progress = st.progress(0, text="Starting download...")
    
    for i, paper in enumerate(papers_to_download):
        url = paper["url"]
        download_progress.progress((i + 1) / total_urls, text=f"Downloading paper {i+1}/{total_urls}...")
        try:
            response = requests.get(url, timeout=20, headers={'User-Agent': 'Mozilla/5.0'}, stream=True)
            response.raise_for_status()
            content_type = response.headers.get('Content-Type', '')
            
            if 'application/pdf' in content_type.lower():
                filename = f"temp_pdfs/{uuid.uuid4()}.pdf"
                with open(filename, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                path_to_metadata_map[filename] = {
                    "url": url, 
                    "publisher": paper["publisher"], 
                    "title": paper["title"]
                }
            else:
                discard_log.append({
                    "url": url, "title": paper["title"], "reason": "File is not a PDF", "publisher": paper["publisher"]
                })
        except requests.exceptions.RequestException as e:
            discard_log.append({
                "url": url, "title": paper["title"], "reason": f"Download failed: {e}", "publisher": paper["publisher"]
            })
            
    discarded_count = total_urls - len(path_to_metadata_map)
    st.info(f"Discarded {discarded_count} papers due to download or format issues.")
    st.success(f"Successfully downloaded {len(path_to_metadata_map)} papers.")
            
    return {"path_to_metadata_map": path_to_metadata_map, "discard_log": discard_log}

def extract_text_node(state: ResearchState) -> ResearchState:
    st.write("Stage 4: Extracting text from PDFs...")
    all_docs = []
    discard_log = state.get("discard_log", [])
    path_to_metadata_map = state["path_to_metadata_map"]
    
    successful_urls = set()

    for path, metadata in path_to_metadata_map.items():
        try:
            loader = PyPDFLoader(path)
            pages = loader.load_and_split()
            for page in pages:
                page.metadata.update(metadata)
            all_docs.extend(pages)
            successful_urls.add(metadata["url"])
        except Exception as e:
            discard_log.append({
                "url": metadata["url"],
                "title": metadata["title"],
                "reason": "Corrupted or unreadable PDF",
                "publisher": metadata["publisher"]
            })
    
    discarded_count = len(path_to_metadata_map) - len(successful_urls)
    st.info(f"Discarded {discarded_count} papers that were corrupted or unreadable.")
    st.success(f"Successfully extracted text from {len(successful_urls)} papers.")
            
    return {"extracted_docs": all_docs, "discard_log": discard_log}

def get_llm_and_embeddings(model_name: str, embeddings_model_name: Optional[str] = None):
    llm = ChatOllama(model=model_name, temperature=0.3)
    embed_model = embeddings_model_name if embeddings_model_name else model_name
    embeddings = OllamaEmbeddings(model=embed_model)
    return llm, embeddings

def filter_by_relevance_node(state: ResearchState) -> ResearchState:
    st.write("Stage 5: Filtering by content relevance using LLM...")
    if not state["extracted_docs"]:
        st.warning("No documents to check for relevance.")
        return {"relevant_docs": [], "discard_log": state["discard_log"]}

    model_config = st.session_state.get("model_config", {})
    llm, _ = get_llm_and_embeddings(model_name=model_config.get("model_name"))
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "Analyze the abstract or the first page of the provided academic paper. Based SOLELY on this text, determine if the paper's primary focus is relevant to the query: '{query}'. Respond with only 'Yes' or 'No'."),
        ("human", "Paper content:\n\n{content}")
    ])
    chain = prompt_template | llm

    docs_by_url = defaultdict(list)
    for doc in state["extracted_docs"]:
        docs_by_url[doc.metadata["url"]].append(doc)
        
    relevant_docs = []
    discard_log = state.get("discard_log", [])
    total_papers = len(docs_by_url)
    relevance_progress = st.progress(0, text="Checking relevance...")

    for i, (url, docs) in enumerate(docs_by_url.items()):
        relevance_progress.progress((i + 1) / total_papers, f"Checking paper {i+1}/{total_papers} for relevance...")
        
        # Use first ~500 words for efficiency
        first_page_content = docs[0].page_content[:2500] 
        
        try:
            response = chain.invoke({"query": state["query"], "content": first_page_content})
            answer = response.content.strip().lower()
            
            if "yes" in answer:
                relevant_docs.extend(docs)
            else:
                metadata = docs[0].metadata
                discard_log.append({
                    "url": url,
                    "title": metadata.get("title", "N/A"),
                    "reason": "Not relevant to query",
                    "publisher": metadata.get("publisher", "N/A")
                })
        except Exception as e:
            st.warning(f"LLM relevance check failed for {url}: {e}")
            metadata = docs[0].metadata
            discard_log.append({
                "url": url,
                "title": metadata.get("title", "N/A"),
                "reason": "LLM relevance check failed",
                "publisher": metadata.get("publisher", "N/A")
            })

    discarded_count = total_papers - len(set(doc.metadata["url"] for doc in relevant_docs))
    st.info(f"Discarded {discarded_count} papers based on LLM relevance check.")
    st.success(f"{total_papers - discarded_count} papers deemed relevant and kept for indexing.")
    
    return {"relevant_docs": relevant_docs, "discard_log": discard_log}

def build_raptor_index_node(state: ResearchState) -> ResearchState:
    st.write("Stage 6: Building RAPTOR index... This may take some time.")
    
    relevant_docs = state["relevant_docs"]
    if not relevant_docs:
        st.error("No relevant documents found to build the index.")
        return {"raptor_index": None, "final_kept_papers": {}}

    model_config = st.session_state.get("model_config", {})
    chat_model_name = model_config.get("model_name")
    summary_model_name = model_config.get("summary_model_name")
    embeddings_model_name = model_config.get("embeddings_model_name")
    
    summarizer_llm = ChatOllama(model=summary_model_name if summary_model_name else chat_model_name, temperature=0.3)
    embeddings = OllamaEmbeddings(model=embeddings_model_name if embeddings_model_name else chat_model_name)
    
    raptor_index = RAPTOR(
        llm=summarizer_llm, 
        embeddings_model=embeddings, 
        session_id=st.session_state.session_id
    )
    raptor_index.add_documents(relevant_docs)
    
    # Populate final kept papers for bibliography
    final_kept_papers = defaultdict(list)
    kept_urls = set(doc.metadata["url"] for doc in relevant_docs)
    for url in kept_urls:
        publisher = next((doc.metadata["publisher"] for doc in relevant_docs if doc.metadata["url"] == url), "Unknown")
        final_kept_papers[publisher].append(url)

    st.success("Research and indexing complete! You can now ask questions.")
    return {"raptor_index": raptor_index, "final_kept_papers": dict(final_kept_papers)}

# --- GRAPH DEFINITION ---
builder = StateGraph(ResearchState)
builder.add_node("start_search", start_search_node)
builder.add_node("arxiv_search", arxiv_search_node)
builder.add_node("filter_by_publisher", filter_by_publisher_node)
builder.add_node("download_pdfs", download_pdfs_node)
builder.add_node("extract_text", extract_text_node)
builder.add_node("filter_by_relevance", filter_by_relevance_node)
builder.add_node("build_raptor_index", build_raptor_index_node)

builder.add_edge(START, "start_search")
builder.add_edge("start_search", "arxiv_search")
builder.add_edge("arxiv_search", "filter_by_publisher")
builder.add_edge("filter_by_publisher", "download_pdfs")
builder.add_edge("download_pdfs", "extract_text")
builder.add_edge("extract_text", "filter_by_relevance")
builder.add_edge("filter_by_relevance", "build_raptor_index")
builder.add_edge("build_raptor_index", END)
graph = builder.compile()

# --- HELPER FUNCTIONS & UI ---
def generate_pdf_report(chat_history: List[Dict[str, str]], used_sources: List[str]) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', size=16)
    pdf.cell(0, 10, txt="Academic Q&A Chat History", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    for message in chat_history:
        role, content = message.get('role', ''), message.get('content', '')
        if role == 'user':
            pdf.set_font('Arial', 'B', 12)
            pdf.set_text_color(0, 0, 128)
            pdf.multi_cell(0, 10, f"Question: {content}")
        else:
            pdf.set_font('Arial', '', 12)
            pdf.set_text_color(0, 0, 0)
            pdf.multi_cell(0, 10, f"Answer: {content}")
        pdf.ln(5)
    if used_sources:
        pdf.add_page()
        pdf.set_font("Arial", 'B', size=16)
        pdf.cell(0, 10, txt="References", ln=True, align='L')
        pdf.ln(5)
        pdf.set_font("Arial", size=10)
        for i, source in enumerate(used_sources):
            pdf.multi_cell(0, 8, f"{i+1}. {source}")
    return pdf.output(dest='S').encode('latin-1')

def generate_bibliography_pdf(papers_by_publisher: Dict[str, List[str]]) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', size=16)
    pdf.cell(0, 10, txt="Full Bibliography of Indexed Articles", ln=True, align='C')
    pdf.ln(10)
    
    for publisher, urls in papers_by_publisher.items():
        if urls:
            pdf.set_font("Arial", 'B', size=14)
            pdf.cell(0, 10, txt=f"--- {publisher} ---", ln=True, align='L')
            pdf.ln(5)
            pdf.set_font("Arial", size=10)
            for i, url in enumerate(urls):
                pdf.multi_cell(0, 8, f"{i+1}. {url}")
            pdf.ln(5)
            
    return pdf.output(dest='S').encode('latin-1')

def generate_mermaid_diagram(final_state: ResearchState) -> str:
    total_found = len(final_state.get('all_arxiv_results', []))
    discard_log = final_state.get('discard_log', [])
    
    discard_counts = Counter(item['reason'] for item in discard_log)
    publisher_discards = discard_counts.get("Publisher not in selected list", 0)
    download_discards = sum(count for reason, count in discard_counts.items() if "Download failed" in reason or "not a PDF" in reason)
    corrupt_discards = discard_counts.get("Corrupted or unreadable PDF", 0)
    relevance_discards = discard_counts.get("Not relevant to query", 0) + discard_counts.get("LLM relevance check failed", 0)

    after_search = total_found
    after_publisher = after_search - publisher_discards
    after_download = after_publisher - download_discards
    after_extract = after_download - corrupt_discards
    after_relevance = after_extract - relevance_discards
    
    def sanitize(text):
        return re.sub(r'[^a-zA-Z0-9_]', '', text.replace(" ", ""))

    diagram = f"""graph TD;
    A[Start: User Input] --> B(arXiv Search);
    B --> C{{Found {after_search} papers}};
    C --> D[Filter by Publisher];
    D -- Pass: {after_publisher} --> E[Download & Validate PDFs];
    D -- Discard: {publisher_discards} --> F_{sanitize('Publisher not in selected list')}[Discarded by Publisher];
    
    E -- Pass: {after_download} --> G[Extract Text];
    E -- Discard: {download_discards} --> F_{sanitize('Download failed')}[Discarded by Download/Format Error];
    
    G -- Pass: {after_extract} --> H[Filter by Relevance];
    G -- Discard: {corrupt_discards} --> F_{sanitize('Corrupted PDF')}[Discarded as Corrupted PDF];

    H -- Pass: {after_relevance} --> I[Build RAPTOR Index];
    H -- Discard: {relevance_discards} --> F_{sanitize('Not relevant')}[Discarded as Not Relevant];

    I --> J[Analysis];

    subgraph Parameters;
        P1("Query: {final_state['query']}");
        P2("Publishers: {', '.join(final_state['publishers'])}");
        P3("Years: {final_state['start_year']}-{final_state['end_year']}");
    end;
    """
    return diagram

@st.cache_data(show_spinner=False)
def get_ollama_models():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        response.raise_for_status()
        return [model['name'] for model in response.json().get('models', [])]
    except (requests.exceptions.RequestException, KeyError):
        return []

def main():
    st.set_page_config(layout="wide", page_title="Academic Deep Search")
    st.title("📚 Academic Deep Search & QA with RAPTOR")
    st.markdown("Powered by Ollama 🦙 and arXiv 📄")

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.research_done = False
        st.session_state.final_state = None
        st.session_state.model_config = {}
        st.session_state.used_sources = set()

    with st.sidebar:
        st.header("1. Research Parameters")
        query = st.text_input("Academic Topic", "indoor air quality monitoring using machine learning")
        
        publishers = st.multiselect(
            "Filter by Publisher (via arXiv metadata)",
            options=DEFAULT_PUBLISHERS,
            default=DEFAULT_PUBLISHERS
        )
        st.info("The pipeline will search arXiv and only keep papers matching these publishers.")

      
        st.header("2. AI Model Configuration")
        
        ollama_models = get_ollama_models()
        if ollama_models:
            default_chat_index = ollama_models.index("llama3:8b") if "llama3:8b" in ollama_models else 0
            model_name = st.selectbox("Select a Chat/Relevance Model", ollama_models, index=default_chat_index)
            
            default_summary_index = ollama_models.index("llama3:8b") if "llama3:8b" in ollama_models else 0
            summary_model_name = st.selectbox("Select a Summary Model", ollama_models, index=default_summary_index)
            
            default_embed_index = ollama_models.index("mxbai-embed-large") if "mxbai-embed-large" in ollama_models else 0
            embeddings_model_name = st.selectbox("Select an Embeddings Model", ollama_models, index=default_embed_index)
        else:
            st.warning("Ollama not detected. Please enter model names manually.")
            model_name = st.text_input("Ollama Chat/Relevance Model", "llama3")
            summary_model_name = st.text_input("Ollama Summary Model Name", "llama3")
            embeddings_model_name = st.text_input("Ollama Embeddings Model Name", "mxbai-embed-large")
        
        if model_name:
            st.session_state.model_config = {
                "model_name": model_name,
                "summary_model_name": summary_model_name,
                "embeddings_model_name": embeddings_model_name
            }
        
        st.header("3. Start Research")
        if st.button("Start Research Pipeline"):
            if not st.session_state.model_config.get("model_name"):
                st.error("Please configure the AI model before starting.")
            else:
                st.session_state.research_done = False
                st.session_state.messages = []
                st.session_state.used_sources = set()
                checkpoint_file = f"checkpoint_{st.session_state.session_id}.json"
                if os.path.exists(checkpoint_file):
                    os.remove(checkpoint_file)
                with st.spinner("Running deep research pipeline..."):
                    initial_state = {
                        "query": query, 
                        "publishers": publishers,  
                        "conversation_history": []
                    }
                    final_state = graph.invoke(initial_state)
                    if final_state.get("raptor_index"):
                        st.session_state.research_done = True
                        st.session_state.final_state = final_state
                    else:
                        st.error("Research pipeline failed to build an index. Check logs for errors.")

    if st.session_state.research_done:
        st.header("Conversational QA")
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask a question about the papers..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    retriever = st.session_state.final_state['raptor_index'].as_retriever()
                    model_config = st.session_state.model_config
                    llm, _ = get_llm_and_embeddings(
                        model_name=model_config["model_name"],
                        embeddings_model_name=model_config.get("embeddings_model_name")
                    )
                    retrieved_docs = retriever.get_relevant_documents(prompt)
                    
                    for doc in retrieved_docs:
                        if "url" in doc.metadata:
                            st.session_state.used_sources.add(doc.metadata["url"])
                        elif "sources" in doc.metadata:
                            for source_url in doc.metadata["sources"]:
                                st.session_state.used_sources.add(source_url)

                    context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
                    
                    prompt_template = ChatPromptTemplate.from_messages([
                        ("system", "You are an AI research assistant. Answer based on the following context from academic papers:\n\n{context}\n\nIf the answer isn't in the context, say so."),
                        ("human", "{question}")
                    ])
                    chain = prompt_template | llm
                    response = chain.invoke({"context": context, "question": prompt})
                    response_content = response.content
                    st.markdown(response_content)
            
            st.session_state.messages.append({"role": "assistant", "content": response_content})
        
        with st.expander("Export Options & Pipeline Summary"):
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Export Chat & Used References"):
                    pdf_bytes = generate_pdf_report(
                        chat_history=st.session_state.messages, 
                        used_sources=sorted(list(st.session_state.used_sources))
                    )
                    st.download_button(label="Download Q&A PDF", data=pdf_bytes, file_name="chat_history.pdf", mime="application/pdf")
            with col2:
                if st.button("Export Full Bibliography"):
                    bib_pdf_bytes = generate_bibliography_pdf(
                        papers_by_publisher=st.session_state.final_state.get('final_kept_papers', {})
                    )
                    st.download_button(label="Download Bibliography PDF", data=bib_pdf_bytes, file_name="full_bibliography.pdf", mime="application/pdf")

            st.subheader("Pipeline Execution Diagram")
            if st.button("Generate Pipeline Diagram"):
                final_state = st.session_state.final_state
                if final_state:
                    mermaid_code = generate_mermaid_diagram(final_state)
                    st.code(mermaid_code, language="mermaid")
                else:
                    st.warning("Final state not available to generate diagram.")

    else:
        st.info("Configure your research and AI model in the sidebar, then click 'Start Research Pipeline'.")

if __name__ == "__main__":
    main()