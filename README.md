# Simple RAG System with Hybrid Retrieval

This project is a simple yet powerful implementation of a Retrieval-Augmented Generation (RAG) system using Python, LangChain, and ChromaDB. It's designed to demonstrate how to effectively combine keyword-based and semantic search to answer questions based on a specific PDF document.

The system performs the following steps:
1.  **Document Ingestion**: Loads a PDF file from the `example_data` directory.
2.  **Chunking**: Splits the document into smaller, manageable text chunks.
3.  **Embedding**: Creates vector embeddings for each text chunk using a Hugging Face model.
4.  **Vector Store**: Stores the embeddings in a persistent ChromaDB vector store.
5.  **Hybrid Retrieval**: When a question is asked, it uses a hybrid approach to find the most relevant document chunks:
    * **Dense Retrieval**: A semantic search using the embeddings.
    * **BM25 Retrieval**: A keyword-based search.
    * The results are combined using an `EnsembleRetriever` to maximize relevance.
6.  **Generation**: The retrieved context is passed to the OpenAI `gpt-3.5-turbo` model to generate a final, informed answer.

## 🚀 Getting Started

This guide will walk you through the setup and usage of the project.

### 📋 Prerequisites

To run this project, you need the following:

* **Python 3.9+**
* **An OpenAI API Key** for the language model.

It is highly recommended to set your API key as an environment variable to prevent it from being exposed in your code or committed to your repository.

#### Setting Your OpenAI API Key

**For macOS/Linux:**
Open your terminal and run the following command. For a permanent solution, add this line to your `~/.bashrc` or `~/.zshrc` file.
```bash
export OPENAI_API_KEY="your_api_key_here"

**For Windows:**
1.  Open the Start Menu and search for "Environment Variables."
2.  Click on "Edit the system environment variables."
3.  In the System Properties window, click the **Environment Variables...** button.
4.  Under "User variables for [Your Username]," click **New...**.
5.  Set the variable name to `OPENAI_API_KEY`.
6.  Set the variable value to your API key.
7.  Click OK. Close and reopen your terminal or command prompt for the changes to take effect.

---

## 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Create and activate a Python virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On macOS/Linux
    source venv/bin/activate
    # On Windows
    .\venv\Scripts\activate
    ```

3.  **Install the required packages:**
    The project dependencies are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```
