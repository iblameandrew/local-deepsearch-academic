import streamlit as st
import os
import json
from sklearn.cluster import KMeans
from typing import List, Optional, Dict, Any, TypedDict, Annotated, Tuple
import operator
import uuid
import requests
from datetime import datetime
from collections import Counter, defaultdict
import re
import base64
import time
import tempfile

from semanticscholar import SemanticScholar

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---

DEFAULT_PUBLISHERS = ["IEEE", "ACM", "Springer", "Elsevier", "Arxiv", "Scholar"]


# --- MOCKING UTILITIES FOR DEBUG MODE ---
def create_mock_data() -> List[Dict[str, Any]]:
    """Generates a list of fake paper data for debugging with publisher and abstract info."""
    mock_papers = [
        {
            "url": "https://www.semanticscholar.org/paper/mock-paper-1",
            "title": "Advanced LSTM Networks",
            "snippet": "This paper introduces a novel LSTM architecture for time-series prediction.",
            "pdf_url": "http://example.com/paper1.pdf",
            "publisher": "IEEE Transactions on Neural Networks",
        },
        {
            "url": "https://www.semanticscholar.org/paper/mock-paper-2",
            "title": "A Transformer-based Model",
            "snippet": "We present a new Transformer model for natural language understanding tasks.",
            "pdf_url": "http://example.com/paper2.pdf",
            "publisher": "ACM Transactions on Intelligent Systems",
        },
        {
            "url": "https://www.semanticscholar.org/paper/mock-paper-3",
            "title": "ML for Sensor Calibration",
            "snippet": "This work explores machine learning techniques for calibrating environmental sensors.",
            "pdf_url": "http://example.com/paper3.pdf",
            "publisher": "Springer Nature",
        },
        {
            "url": "https://www.semanticscholar.org/paper/mock-paper-4",
            "title": "Review of ML in Air Quality",
            "snippet": "A comprehensive review of machine learning applications in air quality monitoring.",
            "pdf_url": "http://example.com/paper4.pdf",
            "publisher": "Elsevier Science",
        },
        {
            "url": "https://www.semanticscholar.org/paper/mock-paper-5",
            "title": "Federated Learning for Privacy",
            "snippet": "We propose a federated learning framework that preserves user privacy.",
            "pdf_url": "http://example.com/paper5.pdf",
            "publisher": "IEEE Explorer Conference",
        },
        {
            "url": "https://www.semanticscholar.org/paper/mock-paper-6",
            "title": "Unsupervised Anomaly Detection",
            "snippet": "This paper presents an unsupervised method for detecting anomalies in industrial data.",
            "pdf_url": "http://example.com/paper6.pdf",
            "publisher": "Proceedings of the ACM",
        },
        {
            "url": "https://www.semanticscholar.org/paper/mock-paper-7",
            "title": "Novel Deep Learning Architectures",
            "snippet": None,
            "pdf_url": "http://example.com/paper7.pdf",
            "publisher": "Journal of Mock Science",
        },  # This should be filtered out
    ]
    return mock_papers


# --- RAPTOR IMPLEMENTATION ---


class RAPTORRetriever(BaseRetriever):
    raptor_index: Any

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return self.raptor_index.retrieve(query)


class RAPTOR:
    def __init__(
        self, llm, embeddings_model, session_id, chunk_size=1000, chunk_overlap=200
    ):
        self.llm = llm
        self.embeddings_model = embeddings_model
        self.session_id = session_id
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.tree = {}
        self.all_nodes: Dict[str, Document] = {}
        self.vector_store = None
        self.checkpoint_path = f"checkpoint_{self.session_id}.json"

    def _save_checkpoint(self, level):
        state = {
            "level": level,
            "tree": {str(k): [node_id for node_id in v] for k, v in self.tree.items()},
            "all_nodes": {
                node_id: doc.to_json() for node_id, doc in self.all_nodes.items()
            },
        }
        with open(self.checkpoint_path, "w") as f:
            json.dump(state, f)
        st.write(f"Checkpoint saved for level {level}.")

    def _load_checkpoint(self) -> int:
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, "r") as f:
                    state = json.load(f)
                from langchain_core.load import load

                self.all_nodes = {
                    node_id: load(doc) for node_id, doc in state["all_nodes"].items()
                }
                self.tree = state["tree"]
                start_level = state["level"]
                st.info(f"Resuming from checkpoint at level {start_level}.")
                return start_level
            except Exception as e:
                st.warning(
                    f"Could not load checkpoint file due to error: {e}. Starting from scratch."
                )
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
                summary_progress.progress(
                    (i + 1) / num_clusters,
                    text=f"Summarizing cluster {i + 1}/{num_clusters} for Level {next_level}...",
                )

            self.tree[str(next_level)] = next_level_node_ids
            self._save_checkpoint(next_level)
            current_level = next_level

        st.write("Step 3: Creating final vector store from all nodes...")
        final_docs = list(self.all_nodes.values())
        self.vector_store = FAISS.from_documents(
            documents=final_docs, embedding=self.embeddings_model
        )
        st.write("RAPTOR index built successfully!")
        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)

    def _cluster_nodes(self, docs: List[Document]) -> List[List[int]]:
        num_docs = len(docs)

        if num_docs <= 5:
            st.write(
                f"Grouping {num_docs} remaining nodes into a single summary to finalize the tree."
            )
            return [list(range(num_docs))]

        st.write(f"Embedding {num_docs} nodes for clustering...")
        embeddings = self.embeddings_model.embed_documents(
            [doc.page_content for doc in docs]
        )
        n_clusters = max(2, num_docs // 5)

        if n_clusters >= num_docs:
            n_clusters = num_docs - 1

        st.write(f"Clustering {num_docs} nodes into {n_clusters} groups...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto").fit(
            embeddings
        )

        clusters = [[] for _ in range(n_clusters)]
        for i, label in enumerate(kmeans.labels_):
            clusters[label].append(i)

        return clusters

    def _summarize_cluster(self, cluster_docs: List[Document]) -> Tuple[str, dict]:
        context = "\n\n---\n\n".join([doc.page_content for doc in cluster_docs])
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content="You are an AI assistant that summarizes academic texts. Create a concise, abstractive summary of the following content, synthesizing the key information."
                ),
                HumanMessage(
                    content="Please summarize the following content:\n\n{context}"
                ),
            ]
        )
        chain = prompt | self.llm
        response = chain.invoke({"context": context})
        summary = response.content
        aggregated_sources = list(
            set(doc.metadata.get("url", "Unknown Source") for doc in cluster_docs)
        )
        combined_metadata = {"sources": aggregated_sources}
        return summary, combined_metadata

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        return (
            self.vector_store.similarity_search(query, k=k) if self.vector_store else []
        )

    def as_retriever(self) -> BaseRetriever:
        return RAPTORRetriever(raptor_index=self)


# --- STATE DEFINITION FOR LANGGRAPH ---
class ResearchState(TypedDict):
    query: str
    publishers: List[str]
    max_results: int
    final_papers_by_publisher: Dict[str, List[str]]
    journal_counts: Dict[str, int]
    all_search_results: List[Dict[str, Any]]
    relevant_docs: List[
        Document
    ]  # This is the "final" list of papers that pass all criteria

    # Specific discard logs for each filtering process
    publisher_discard_log: List[Dict[str, str]]
    relevance_discard_log: List[Dict[str, str]]
    unreadable_discard_log: List[Dict[str, str]]

    raptor_index: Any
    conversation_history: Annotated[List[BaseMessage], operator.add]


# --- LANGGRAPH NODES AND GRAPH DEFINITION ---
def start_search_node(state: ResearchState) -> ResearchState:
    st.write("Starting research process...")
    return {
        "publisher_discard_log": [],
        "relevance_discard_log": [],
        "unreadable_discard_log": [],
        "final_papers_by_publisher": {},
        "journal_counts": {},
    }


def semantic_scholar_search_node(state: ResearchState) -> ResearchState:
    """
    Performs a search on Semantic Scholar, filters by publisher, and retrieves abstracts.
    Papers without abstracts are discarded into a specific log.
    Papers not matching publisher criteria are discarded into another specific log.
    """
    if st.session_state.get("debug_mode", False):
        st.warning("DEBUG MODE: Simulating search and API-based filtering.")

        all_mock_papers = create_mock_data()
        publishers = state["publishers"]

        filtered_papers = []
        journal_counts = Counter()
        publisher_discard_log = state.get("publisher_discard_log", [])
        unreadable_discard_log = state.get("unreadable_discard_log", [])

        for paper in all_mock_papers:
            # Check for abstract first
            if not paper.get("snippet"):
                unreadable_discard_log.append(
                    {
                        "url": paper["url"],
                        "title": paper["title"],
                        "reason": "Corrupted or unreadable PDF",
                        "publisher": paper.get("publisher", "Unknown"),
                    }
                )
                continue

            # Filter by publisher
            if not publishers:
                journal_counts["Uncategorized"] += 1
            else:
                paper_publisher = paper.get("publisher", "")
                found_match = False
                if paper_publisher:
                    for pub_filter in publishers:
                        if pub_filter.lower() in paper_publisher.lower():
                            paper["publisher"] = pub_filter
                            filtered_papers.append(paper)
                            journal_counts[pub_filter] += 1
                            found_match = True
                            break
                if not found_match:
                    publisher_discard_log.append(
                        {
                            "url": paper["url"],
                            "title": paper["title"],
                            "reason": "Publisher does not match filter",
                            "publisher": paper_publisher or "Unknown",
                        }
                    )

        st.success(
            f"DEBUG MODE: API search simulation finished. Found {len(filtered_papers)} candidate papers with abstracts."
        )
        return {
            "all_search_results": filtered_papers,
            "journal_counts": dict(journal_counts),
            "publisher_discard_log": publisher_discard_log,
            "unreadable_discard_log": unreadable_discard_log,
        }

    # --- Live API Logic ---
    query = state["query"]
    max_results = state["max_results"]
    publishers = state["publishers"]

    sch = SemanticScholar()
    all_found_papers = []
    seen_urls = set()
    journal_counts = Counter()
    publisher_discard_log = state.get("publisher_discard_log", [])
    unreadable_discard_log = state.get("unreadable_discard_log", [])

    st.write(
        "Stage 1: Searching Semantic Scholar API, filtering by publisher, and fetching abstracts..."
    )

    total_progress = st.progress(0, text="Starting API search...")
    papers_inspected = 0
    inspection_limit = max_results

    try:
        results = sch.search_paper(query, bulk=True)
        st.info(f"Found {results.total} results from the API.")
    except Exception as e:
        st.error(f"Failed to start Semantic Scholar search: {e}")
        return {
            "all_search_results": [],
            "journal_counts": {},
            "publisher_discard_log": publisher_discard_log,
            "unreadable_discard_log": unreadable_discard_log,
        }

    for paper in results:
        papers_inspected += 1
        paper_url = paper.url

        if paper_url and paper_url not in seen_urls:
            seen_urls.add(paper_url)

            # Discard if there is no abstract
            if not paper.abstract:
                unreadable_discard_log.append(
                    {
                        "url": paper_url,
                        "title": paper.title,
                        "reason": "Corrupted or unreadable PDF",  # Re-using this category for no abstract
                        "publisher": paper.journal.name if paper.journal else "Unknown",
                    }
                )
            else:
                passes_filter = False
                found_publisher = "Uncategorized"

                if not publishers:
                    passes_filter = True
                elif paper.journal and paper.journal.name:
                    journal_name = paper.journal.name
                    for pub_filter in publishers:
                        if pub_filter.lower() in journal_name.lower():
                            passes_filter = True
                            found_publisher = pub_filter
                            break
                    if (
                        not passes_filter
                        and len(journal_name.split()) == 1
                        and "Scholar" in publishers
                    ):
                        passes_filter = True
                        found_publisher = "Scholar"

                if passes_filter:
                    all_found_papers.append(
                        {
                            "url": paper_url,
                            "title": paper.title,
                            "snippet": paper.abstract,
                            "publisher": found_publisher,
                        }
                    )
                    journal_counts[found_publisher] += 1
                else:
                    publisher_discard_log.append(
                        {
                            "url": paper_url,
                            "title": paper.title,
                            "reason": "Publisher does not match filter",
                            "publisher": paper.journal.name
                            if paper.journal
                            else "Unknown",
                        }
                    )

        if papers_inspected % 25 == 0:
            progress = min(
                1.0, papers_inspected / inspection_limit if inspection_limit > 0 else 0
            )
            total_progress.progress(
                progress,
                text=f"Inspected {papers_inspected} papers from API...",
            )

        if papers_inspected >= inspection_limit:
            break

    st.success(
        f"API search finished. Found {len(all_found_papers)} candidate papers. Discarded {len(publisher_discard_log)} for publisher mismatch and {len(unreadable_discard_log)} for missing abstracts."
    )
    return {
        "all_search_results": all_found_papers,
        "journal_counts": dict(journal_counts),
        "publisher_discard_log": publisher_discard_log,
        "unreadable_discard_log": unreadable_discard_log,
    }


def get_llm_and_embeddings(
    model_name: str, embeddings_model_name: Optional[str] = None
):
    llm = ChatOllama(model=model_name, temperature=0.3)
    embed_model = embeddings_model_name if embeddings_model_name else model_name
    embeddings = OllamaEmbeddings(model=embed_model)
    return llm, embeddings


def filter_by_relevance_node(state: ResearchState) -> ResearchState:
    st.write("Stage 2: Filtering by content relevance using LLM...")
    search_results = state["all_search_results"]
    if not search_results:
        st.warning("No documents to check for relevance.")
        return {
            "relevant_docs": [],
            "relevance_discard_log": state["relevance_discard_log"],
        }

    model_config = st.session_state.get("model_config", {})
    llm, _ = get_llm_and_embeddings(model_name=model_config.get("model_name"))

    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Analyze the abstract of the provided academic paper. Based SOLELY on this text, determine if the paper's primary focus is relevant to the query: '{query}'. Respond with only 'Yes' or 'No'.",
            ),
            ("human", "Paper abstract:\n\n{content}"),
        ]
    )
    chain = prompt_template | llm

    relevant_docs = []
    relevance_discard_log = state.get("relevance_discard_log", [])
    total_papers = len(search_results)
    relevance_progress = st.progress(0, text="Checking relevance...")

    for i, paper in enumerate(search_results):
        relevance_progress.progress(
            (i + 1) / total_papers,
            f"Checking paper {i + 1}/{total_papers} for relevance...",
        )

        abstract_content = paper.get("snippet", "")

        if not abstract_content:
            continue

        try:
            answer = chain.invoke(
                {"query": state["query"], "content": abstract_content}
            )
            answer_text = answer.content.strip().lower()
            is_relevant = answer_text.startswith("yes")

            if is_relevant:
                doc = Document(
                    page_content=abstract_content,
                    metadata={
                        "url": paper["url"],
                        "title": paper["title"],
                        "publisher": paper.get("publisher", "N/A"),
                    },
                )
                relevant_docs.append(doc)
            else:
                relevance_discard_log.append(
                    {
                        "url": paper["url"],
                        "title": paper.get("title", "N/A"),
                        "reason": "Not relevant to query",
                        "publisher": paper.get("publisher", "N/A"),
                    }
                )
        except Exception as e:
            st.warning(f"LLM relevance check failed for {paper['url']}: {e}")
            relevance_discard_log.append(
                {
                    "url": paper["url"],
                    "title": paper.get("title", "N/A"),
                    "reason": "LLM relevance check failed",
                    "publisher": paper.get("publisher", "N/A"),
                }
            )

    discarded_count = len(relevance_discard_log)
    st.info(f"Discarded {discarded_count} papers based on LLM relevance check.")
    st.success(f"{len(relevant_docs)} papers deemed relevant and kept for indexing.")

    return {
        "relevant_docs": relevant_docs,
        "relevance_discard_log": relevance_discard_log,
    }


def build_raptor_index_node(state: ResearchState) -> ResearchState:
    st.write("Stage 3: Building RAPTOR index... This may take some time.")

    relevant_docs = state["relevant_docs"]
    if not relevant_docs:
        st.error("No relevant documents found to build the index.")
        return {"raptor_index": None, "final_papers_by_publisher": {}}

    model_config = st.session_state.get("model_config", {})
    chat_model_name = model_config.get("model_name")
    summary_model_name = model_config.get("summary_model_name")
    embeddings_model_name = model_config.get("embeddings_model_name")

    summarizer_llm = ChatOllama(
        model=summary_model_name if summary_model_name else chat_model_name,
        temperature=0.3,
    )
    embeddings = OllamaEmbeddings(
        model=embeddings_model_name if embeddings_model_name else chat_model_name
    )

    raptor_index = RAPTOR(
        llm=summarizer_llm,
        embeddings_model=embeddings,
        session_id=st.session_state.session_id,
    )
    raptor_index.add_documents(relevant_docs)

    final_papers_by_publisher = defaultdict(list)
    kept_urls = set(doc.metadata["url"] for doc in relevant_docs)
    for url in kept_urls:
        publisher = next(
            (
                doc.metadata["publisher"]
                for doc in relevant_docs
                if doc.metadata["url"] == url
            ),
            "Unknown",
        )
        final_papers_by_publisher[publisher].append(url)

    st.success("Research and indexing complete! You can now ask questions.")
    return {
        "raptor_index": raptor_index,
        "final_papers_by_publisher": dict(final_papers_by_publisher),
    }


# --- GRAPH DEFINITION ---
builder = StateGraph(ResearchState)
builder.add_node("start_search", start_search_node)
builder.add_node("semantic_scholar_search", semantic_scholar_search_node)
builder.add_node("filter_by_relevance", filter_by_relevance_node)
builder.add_node("build_raptor_index", build_raptor_index_node)

builder.add_edge(START, "start_search")
builder.add_edge("start_search", "semantic_scholar_search")
builder.add_edge("semantic_scholar_search", "filter_by_relevance")
builder.add_edge("filter_by_relevance", "build_raptor_index")
builder.add_edge("build_raptor_index", END)
graph = builder.compile()


# --- HELPER FUNCTIONS & UI ---
def render_mermaid_to_image(mermaid_code: str) -> Optional[bytes]:
    try:
        graphbytes = mermaid_code.encode("utf8")
        base64_bytes = base64.b64encode(graphbytes)
        base64_string = base64_bytes.decode("ascii")
        url = f"https://mermaid.ink/img/{base64_string}"

        response = requests.get(url)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to render Mermaid diagram: {e}")
        return None


def format_chat_as_report(chat_history: List[Dict[str, str]], llm: Any) -> str:
    """Uses an LLM to format a chat history into a formal academic-style Q&A report."""
    if not chat_history:
        return "No conversation to report."

    chat_log_string = ""
    for message in chat_history:
        role = "Question" if message.get("role") == "user" else "Answer"
        chat_log_string += f"{role}: {message.get('content')}\n\n---\n\n"

    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert academic writer. Your task is to transform a raw question-and-answer chat log into a formal academic report.
Follow these instructions precisely:
1.  For each 'Question' from the user, rephrase it into a formal, academic-style section heading or a research question. For example, 'what is lstm?' could become 'An Inquiry into Long Short-Term Memory Networks'.
2.  For each 'Answer' from the assistant, rewrite it in a formal, objective, and academic tone, suitable for a research paper. Remove any conversational filler.
3.  Structure the final output as a series of these formal questions and answers. Each question should serve as a subtitle for the answer that follows.
4.  Do not invent any information. Your response must be based *only* on the provided chat log content.
5.  The final output should read like a structured academic document, not a casual conversation.""",
            ),
            (
                "human",
                "Please convert the following chat log into a formal, academic-style Q&A report:\n\n---\n\n{chat_log}",
            ),
        ]
    )

    chain = prompt_template | llm

    try:
        response = chain.invoke({"chat_log": chat_log_string})
        report_content = response.content

        # Post-processing to remove <think> tags and their content
        cleaned_content = re.sub(
            r"<think>.*?</think>", "", report_content, flags=re.DOTALL
        ).strip()
        return cleaned_content
    except Exception as e:
        st.error(f"Failed to format report: {e}")
        return "Error: Could not format the chat log into a report."


def generate_pdf_report(
    report_content: str,
    used_sources: List[str],
    mermaid_image_bytes: Optional[bytes] = None,
) -> bytes:
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_font("DejaVu", "", "fonts/DejaVuSans.ttf", uni=True)
    pdf.add_font("DejaVu", "B", "fonts/DejaVuSans-Bold.ttf", uni=True)
    pdf.add_page()
    pdf.set_font("DejaVu", "B", size=16)
    pdf.cell(0, 10, txt="Research Findings Report", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("DejaVu", "", 12)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(
        0, 8, report_content
    )  # Reduced line height for better paragraph spacing
    pdf.ln(5)

    if mermaid_image_bytes:
        pdf.add_page()
        pdf.set_font("DejaVu", "B", size=16)
        pdf.cell(0, 10, txt="Pipeline Execution Diagram", ln=True, align="C")
        pdf.ln(5)
        tmpfile_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                tmpfile.write(mermaid_image_bytes)
                tmpfile_path = tmpfile.name

            pdf.image(tmpfile_path, x=10, y=30, w=190)
        except Exception as e:
            pdf.set_font("DejaVu", size=12)
            pdf.set_text_color(255, 0, 0)
            pdf.multi_cell(0, 10, f"Error rendering diagram: {e}")

    if used_sources:
        pdf.add_page()
        pdf.set_font("DejaVu", "B", size=16)
        pdf.cell(0, 10, txt="References", ln=True, align="L")
        pdf.ln(5)
        pdf.set_font("DejaVu", size=10)
        for i, source in enumerate(sorted(list(used_sources))):
            pdf.write(h=8, text=f"{i + 1}. {source}")
            pdf.ln(8)

    pdf_bytes = bytes(pdf.output(dest="S"))

    if tmpfile_path:
        try:
            os.remove(tmpfile_path)
        except OSError:
            pass

    return pdf_bytes


def generate_bibliography_pdf(
    final_papers: List[Document],
    publisher_discards: List[Dict[str, str]],
    relevance_discards: List[Dict[str, str]],
    unreadable_discards: List[Dict[str, str]],
) -> bytes:
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_font("DejaVu", "", "fonts/DejaVuSans.ttf", uni=True)
    pdf.add_font("DejaVu", "B", "fonts/DejaVuSans-Bold.ttf", uni=True)

    # --- Helper function for writing sections ---
    def write_section(title, papers_list):
        pdf.add_page()
        pdf.set_font("DejaVu", "B", size=16)
        pdf.cell(0, 10, txt=title, ln=True, align="C")
        pdf.ln(10)

        if not papers_list:
            pdf.set_font("DejaVu", "", size=12)
            pdf.cell(0, 10, txt="No articles in this category.", ln=True)
            return

        for i, paper in enumerate(papers_list):
            # Extract info. For Document objects, it's in metadata. For dicts, it's direct.
            if isinstance(paper, Document):
                title_text = paper.metadata.get("title", "N/A")
                url_text = paper.metadata.get("url", "N/A")
                publisher_text = paper.metadata.get("publisher", "N/A")
            else:  # It's a dict from a discard log
                title_text = paper.get("title", "N/A")
                url_text = paper.get("url", "N/A")
                publisher_text = paper.get("publisher", "N/A")

            # Using pdf.write() for more robust text wrapping
            pdf.set_font("DejaVu", "B", size=12)
            pdf.write(8, f"{i + 1}. {title_text}\n")

            pdf.set_font("DejaVu", "", size=10)
            pdf.write(8, f"   Publisher: {publisher_text}\n")
            pdf.write(8, f"   URL: {url_text}\n")
            pdf.ln(5)  # Add extra vertical space between entries

    # --- Generate PDF content with separate sections ---
    write_section("Final Indexed Articles", final_papers)
    write_section("Discarded by Publisher Filter", publisher_discards)
    write_section("Discarded by Relevance Filter", relevance_discards)
    write_section("Discarded for Missing Abstract", unreadable_discards)

    return bytes(pdf.output(dest="S"))


def generate_mermaid_diagram(final_state: ResearchState) -> str:
    # Get counts from the new specific logs
    publisher_discards = len(final_state.get("publisher_discard_log", []))
    no_abstract_discards = len(final_state.get("unreadable_discard_log", []))
    relevance_discards = len(final_state.get("relevance_discard_log", []))

    # Calculate the flow of numbers accurately
    after_relevance = len(final_state.get("relevant_docs", []))  # Final "kept" count
    after_search_and_filter = (
        after_relevance + relevance_discards
    )  # Input to relevance node
    initial_items_inspected = (
        after_search_and_filter + publisher_discards + no_abstract_discards
    )

    query = final_state.get("query", "N/A").replace('"', "'")
    journal_counts = final_state.get("journal_counts", {})
    publishers_str = (
        ", ".join(final_state.get("publishers", []))
        if final_state.get("publishers")
        else "All"
    )
    journal_counts_str = (
        "\\n".join([f"{journal}: {count}" for journal, count in journal_counts.items()])
        if journal_counts
        else "N/A"
    )

    diagram = f"""graph TD;
    A[Start: User Input] --> B(1. Search & Pre-Filter: Inspected {initial_items_inspected} papers);
    B -- Discard (No Abstract): {no_abstract_discards} --> F_abs(Discarded);
    B -- Discard (Publisher Mismatch): {publisher_discards} --> F_pub(Discarded);
    B -- Pass: {after_search_and_filter} --> H[2. Filter by Relevance];

    H -- Pass: {after_relevance} --> I[3. Ready for Analysis];
    H -- Discard (Not Relevant): {relevance_discards} --> F_rel(Discarded);

    subgraph Parameters;
        P1("Query: {query}");
        P2("Publishers Filter: {publishers_str}");
        P3("Final Article Counts by Publisher:\\n{journal_counts_str}");
    end;
    """

    return diagram


@st.cache_data(show_spinner=False)
def get_ollama_models():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        response.raise_for_status()
        models = response.json().get("models", [])
        return [model["name"] for model in models] if models else []
    except (requests.exceptions.RequestException, KeyError):
        return []


def main():
    st.set_page_config(layout="wide", page_title="Academic Deep Search")
    st.title("📚 Academic Deep Search & QA with RAPTOR")
    st.markdown("Powered by Ollama 🦙 and the Semantic Scholar API 🧑‍💻")

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.research_done = False
        st.session_state.final_state = None
        st.session_state.model_config = {}
        st.session_state.used_sources = None

    with st.sidebar:
        st.header("1. Research Parameters")

        debug_mode = st.checkbox("Enable Debug Mode (use mock data)", value=False)
        st.session_state.debug_mode = debug_mode
        if debug_mode:
            st.warning("Debug mode is ON. Live data will not be fetched.")

        query = st.text_input(
            "Academic Topic (searches Semantic Scholar)",
            "indoor air quality monitoring using machine learning",
        )

        publishers = st.multiselect(
            "Filter by Publisher",
            options=DEFAULT_PUBLISHERS,
            default=DEFAULT_PUBLISHERS,  # Default to all known publishers
        )
        st.info(
            "Filtering is based on journal metadata from the API. Papers without abstracts are discarded."
        )

        max_results = st.slider(
            "Max Papers to Find", min_value=10, max_value=5000, value=100, step=10
        )

        st.header("2. AI Model Configuration")

        ollama_models = get_ollama_models()
        if ollama_models:
            default_chat_index = (
                ollama_models.index("llama3:8b") if "llama3:8b" in ollama_models else 0
            )
            model_name = st.selectbox(
                "Select a Chat/Relevance/Report Model",
                ollama_models,
                index=default_chat_index,
            )

            default_summary_index = (
                ollama_models.index("llama3:8b") if "llama3:8b" in ollama_models else 0
            )
            summary_model_name = st.selectbox(
                "Select a Summary Model", ollama_models, index=default_summary_index
            )

            default_embed_index = (
                ollama_models.index("mxbai-embed-large")
                if "mxbai-embed-large" in ollama_models
                else 0
            )
            embeddings_model_name = st.selectbox(
                "Select an Embeddings Model", ollama_models, index=default_embed_index
            )
        else:
            st.warning("Ollama not detected. Please enter model names manually.")
            model_name = st.text_input("Ollama Chat/Relevance/Report Model", "llama3")
            summary_model_name = st.text_input("Ollama Summary Model Name", "llama3")
            embeddings_model_name = st.text_input(
                "Ollama Embeddings Model Name", "mxbai-embed-large"
            )

        if model_name:
            st.session_state.model_config = {
                "model_name": model_name,
                "summary_model_name": summary_model_name,
                "embeddings_model_name": embeddings_model_name,
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

                research_container = st.container()
                with research_container:
                    with st.spinner("Running deep research pipeline..."):
                        initial_state = {
                            "query": query,
                            "publishers": publishers,
                            "max_results": max_results,
                            "conversation_history": [],
                        }
                        final_state = graph.invoke(initial_state)

                        st.session_state.final_state = final_state
                        if final_state and final_state.get("raptor_index"):
                            st.session_state.research_done = True
                            st.rerun()
                        else:
                            st.error(
                                "Research pipeline failed to build an index. Check logs for errors."
                            )

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
                    retriever = st.session_state.final_state[
                        "raptor_index"
                    ].as_retriever()
                    model_config = st.session_state.model_config
                    llm, _ = get_llm_and_embeddings(
                        model_name=model_config["model_name"],
                        embeddings_model_name=model_config.get("embeddings_model_name"),
                    )
                    retrieved_docs = retriever.get_relevant_documents(prompt)

                    for doc in retrieved_docs:
                        if "url" in doc.metadata:
                            st.session_state.used_sources.add(doc.metadata["url"])
                        elif "sources" in doc.metadata:
                            for source_url in doc.metadata["sources"]:
                                st.session_state.used_sources.add(source_url)

                    context = "\n\n---\n\n".join(
                        [doc.page_content for doc in retrieved_docs]
                    )

                    prompt_template = ChatPromptTemplate.from_messages(
                        [
                            (
                                "system",
                                "You are an AI research assistant. Answer the user's question based *only* on the following context from academic papers:\n\n{context}\n\nIf the answer is not found in the context, clearly state that. Do not use any outside knowledge.",
                            ),
                            ("human", "{question}"),
                        ]
                    )
                    chain = prompt_template | llm
                    try:
                        response = chain.invoke(
                            {"context": context, "question": prompt}
                        )
                        response_content = re.sub(
                            r"<think>.*?</think>",
                            "",
                            response.content,
                            flags=re.DOTALL,
                        ).strip()
                    except Exception as e:
                        response_content = f"Error generating answer: {e}"
                    st.markdown(response_content)

            st.session_state.messages.append(
                {"role": "assistant", "content": response_content}
            )
            st.rerun()

        with st.expander("Export Options & Summary"):
            mermaid_code = generate_mermaid_diagram(st.session_state.final_state)
            st.code(mermaid_code, language="mermaid")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Export Full Report (Chat, Diagram, References)"):
                    mermaid_bytes = None
                    if st.session_state.final_state:
                        with st.spinner("Generating pipeline diagram..."):
                            mermaid_bytes = render_mermaid_to_image(mermaid_code)

                    with st.spinner("Formatting chat into an academic report..."):
                        model_config = st.session_state.model_config
                        llm, _ = get_llm_and_embeddings(
                            model_name=model_config["model_name"]
                        )
                        report_content = format_chat_as_report(
                            chat_history=st.session_state.messages, llm=llm
                        )

                    with st.spinner("Generating PDF report..."):
                        pdf_bytes = generate_pdf_report(
                            report_content=report_content,
                            used_sources=list(st.session_state.used_sources),
                            mermaid_image_bytes=mermaid_bytes,
                        )
                        st.download_button(
                            label="Download Report PDF",
                            data=pdf_bytes,
                            file_name=f"full_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                            mime="application/pdf",
                        )
            with col2:
                if st.button("Export Full Bibliography"):
                    final_state = st.session_state.final_state
                    bib_pdf_bytes = generate_bibliography_pdf(
                        final_papers=final_state.get("relevant_docs", []),
                        publisher_discards=final_state.get("publisher_discard_log", []),
                        relevance_discards=final_state.get("relevance_discard_log", []),
                        unreadable_discards=final_state.get(
                            "unreadable_discard_log", []
                        ),
                    )
                    st.download_button(
                        label="Download Bibliography PDF",
                        data=bib_pdf_bytes,
                        file_name=f"full_bibliography_{datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf",
                    )

    elif "final_state" in st.session_state and st.session_state.final_state is not None:
        st.error(
            "Research complete, but no valid papers were indexed. Please check the logs above and try adjusting your query or publisher filters."
        )
        with st.expander("View Execution Flow & Discard Log"):
            mermaid_code = generate_mermaid_diagram(st.session_state.final_state)
            st.code(mermaid_code, language="mermaid")
            st.write("Publisher Discard Log:")
            st.json(st.session_state.final_state.get("publisher_discard_log", []))
            st.write("Relevance Discard Log:")
            st.json(st.session_state.final_state.get("relevance_discard_log", []))
            st.write("Unreadable/Missing Abstract Log:")
            st.json(st.session_state.final_state.get("unreadable_discard_log", []))

    else:
        st.info(
            "Configure your research and AI model in the sidebar, then click 'Start Research Pipeline'."
        )


if __name__ == "__main__":
    main()
