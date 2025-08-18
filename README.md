# ![deep-research](https://github.com/user-attachments/assets/b98d01cb-0d7d-4cd9-a117-8af29d0b47e8)

## Academic Deep Search & QA 📚🧠

I built this tool to solve a problem I know well: you need to dive into a new research topic, but you're faced with a mountain of papers to find, filter, and read. This app is designed to be a personal research assistant that automates the entire pipeline, from discovery to insight.

You give it a topic, and it builds a custom, chat-ready knowledge base from thousands of real academic articles, all powered by local LLMs through Ollama. It turns days of manual work into a focused, interactive Q&A session.

## So, What Does It Actually Do? 🤔

This app is a multi-stage, intelligent pipeline that transforms a simple query into a fully interactive knowledge base.

*   🔍 **Automated Paper Discovery:** Instead of simple scraping, the app leverages the powerful **Semantic Scholar API** to search through millions of academic papers, finding thousands of relevant candidates for your topic in moments.
*   🎯 **Intelligent Filtering:** It runs a two-stage filtering process. First, it can narrow the field by publisher names found in the abstracts. Then, a local LLM reads the abstract of each remaining paper to verify its relevance to your query, ensuring only the most on-topic articles proceed.
*   📥 **Robust Downloading:** It intelligently downloads only the available Open Access PDFs, checking file headers to avoid saving invalid files or login pages. With built-in retries and backoff, it's designed to handle thousands of downloads without failing.
*   🦖 **Advanced Indexing with RAPTOR:** This is where the magic happens. Instead of just chunking text, it uses the **RAPTOR** (Recursive Abstractive Processing for Tree-Organized Retrieval) method. This builds a multi-level tree of summaries, allowing the AI to understand the research from fine-grained details to high-level concepts, resulting in far more insightful answers.
*   💬 **Conversational QA:** Once the index is built, a Streamlit-powered chat interface lets you ask complex questions. You get answers synthesized *exclusively* from the knowledge within the downloaded papers.
*   📤 **Professional Report Generation:** This is more than just a chat log. You can instantly transform your Q&A session into a polished, **academic-style report**. The app uses an LLM to rephrase your questions into formal section headers and rewrite the answers in an objective, academic tone. The final export includes this report, a visual diagram of the research pipeline, and a full bibliography of all cited sources.

## The Tech Stack 🛠️

*   **Backend Logic:** [LangGraph](https://langchain-ai.github.io/langgraph/) 🦜🔗 for creating a resilient, stateful research pipeline.
*   **Data Sourcing:** [Semantic Scholar API](https://www.semanticscholar.org/product/api) for comprehensive, structured access to academic literature.
*   **Indexing:** A custom **RAPTOR** implementation for intelligent, multi-level retrieval.
*   **AI Models:** Plugs into your local models via [Ollama](https://ollama.ai/). You can configure separate models for chat, summarization, and embeddings.
*   **Frontend:** [Streamlit](https://streamlit.io/) 🎈 for a fast, interactive web UI.
*   **The Glue:** The entire system is orchestrated with [LangChain](https://www.langchain.com/).

## Getting Started 🚀

Ready to build your own personal research assistant? Here’s how to get it running.

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/academic-deep-search.git
cd academic-deep-search
```

2. Set Up Your Environment
This project uses a requirements.txt file, so setup is straightforward. Using a virtual environment is highly recommended.
code
Bash
### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`


# Install the dependencies
pip install -r requirements.txt

```

### 3. Get Ollama Running

This app is designed to run entirely on your local machine using Ollama.
Install Ollama if you haven't already.
Pull the recommended models. You'll want a capable chat model and a specialized embedding model for the best results.


```bash
ollama pull qwen3:1.7b
```

```bash
ollama pull mxbai-embed-large
Ensure the Ollama server is running in the background.
```

### 4. Fire It Up! Run the Streamlit app from your terminal:

```bash
streamlit run app.py
```

Your browser should open with the app ready to go. Configure your research topic and select your desired Ollama models in the sidebar, then kick off the research pipeline.

## Use Cases & Who This Is For

I built this with a few people in mind:

- Students & Academics: Writing a literature review? Point this at a topic and get a massive head start. Quickly identify key themes, synthesize arguments from dozens of papers, and generate a structured Q&A report to guide your writing.
- Data Scientists & Engineers: Exploring a new machine learning architecture or a novel algorithm? Let the app grab the foundational papers and get you up to speed in minutes, not hours. Ask targeted questions about methodologies, datasets, and results.
- Curious Minds: Want to learn about quantum computing, cellular biology, or ancient history? This is a powerful, interactive way to dive deep into any topic by engaging directly with the primary literature.

## Contributing

Got an idea? Found a bug? Feel free to open an issue or submit a pull request! I'd love to see what the community can build on top of this. Let's make research less of a chore.
