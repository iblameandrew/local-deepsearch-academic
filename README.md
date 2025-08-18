# ![deep-research](https://github.com/user-attachments/assets/b98d01cb-0d7d-4cd9-a117-8af29d0b47e8)

## Academic Deep Search & QA 📚🧠

I built this tool because I wanted a way to quickly dive into a new research topic, pull down the relevant papers, and start asking questions right away—without spending days on manual searching and reading.

This app is a personal research assistant that automates the grunt work. You give it a topic, and it builds a custom, chat-ready knowledge base from real academic articles.


## So, What Does It Actually Do? 🤔



*   🔍 **Smart Search:** You provide a topic and optionally specify academic domains (like `arxiv.org`, `ieeexplore.ieee.org`). The app then scours the web for relevant PDF papers.
*   📥 **Automatic Downloading:** It finds the direct PDF links and downloads the papers for you, saving them locally.
*   📄 **Text Extraction:** It cracks open those PDFs and pulls out all the text content, getting it ready for the AI.
*   🦖 **Advanced Indexing with RAPTOR:** This is the cool part. Instead of just chunking the text, it uses the **RAPTOR** method to create a multi-level tree of summaries. This means it understands the papers from tiny details all the way up to high-level concepts, leading to much better answers.
*   💬 **Conversational QA:** Once the index is built, you get a chat interface (thanks to Streamlit!) where you can ask complex questions and get answers synthesized from the papers it just read.
*   📤 **Export Your Findings:** You can export your entire Q&A session as a clean, formatted PDF or generate a Mermaid diagram that visualizes the exact pipeline run, including your parameters and the resources it found.

## The Tech Stack 🛠️


*   **Backend Logic:** [LangGraph](https://langchain-ai.github.io/langgraph/) 🦜🔗 for creating a resilient, stateful pipeline.
*   **Indexing:** A custom **RAPTOR** implementation for intelligent, multi-level retrieval.
*   **AI Models:** Plugs into local models via [Ollama](https://ollama.ai/) (like Llama 3) and cloud models via [Google Gemini](https://ai.google.dev/).
*   **Frontend:** [Streamlit](https://streamlit.io/) 🎈 for a fast, interactive web UI.


## Getting Started 🚀

Ready to give it a spin? Here’s how to get it running on your machine.

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/academic-deep-search.git
cd academic-deep-search
```
2. Set Up Your Environment
This project uses a requirements.txt file, so setting up is a breeze. I recommend using a virtual environment.
# Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate # Install all the goodies
pip install -r requirements.txt
```


### 2. API Keys (Optional, for Gemini)

If you want to use Google's Gemini models, you'll need an API key.

*   Create a file named `.env` in the root of the project.
*   Add your key to it like this:

    ```env
    GOOGLE_API_KEY="your-super-secret-key-here"
    ```
The app will automatically load this. If you don't create this file, the app will ask for the key in the sidebar when you select Gemini.

### 4. Get Ollama Running

If you're using a local model, you need to have Ollama installed and running.

1.  [Install Ollama](https://ollama.ai/).
2.  Pull a model to chat with. Llama 3 is a great choice.

    ```bash
        ollama run dengcao/Qwen3-30B-A3B-Instruct-2507
    ```
3.  Make sure the Ollama server is running in the background.

### 5. Fire It Up!

Run the Streamlit app from your terminal:

```bash
streamlit run app.py
```

Your browser should open with the app ready to go! 

Configure your topic and model in the sidebar and kick off the research.
## Use Cases & Who This Is For

I built this with a few people in mind:

- Students & Academics: Need to write a literature review? Point this at a topic and get a massive head start. Quickly find key themes and ask targeted questions to build your arguments.
- Data Scientists & Engineers: Exploring a new machine learning architecture or a novel algorithm? Let the app grab the foundational papers and get you up to speed in minutes, not hours.

- Curious Minds: Just want to learn about something cool like quantum computing or cellular biology? This is a fun, interactive way to dive deep into a topic.

### Contributing

Got an idea? Found a bug? Feel free to open an issue or submit a pull request! I'd love to see what the community can build on top of this. Let's make research less of a chore.
