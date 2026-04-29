# Academic Paper Summarizer

A Python utility that automates the process of reading, extracting, and summarizing academic research papers. It leverages **Ollama** to run Large Language Models (LLMs) locally, ensuring your data remains private and eliminating API costs.

The script extracts text from PDFs, identifies metadata (title/year), generates structured markdown summaries, and maintains a local cache to avoid re-processing files you've already summarized.

## Features

* **Local LLM Integration:** Uses Ollama to generate summaries (defaults to `gemma4:26b` but works with any model like `llama3.2`).
* **Smart Text Extraction:** Uses `PyMuPDF` to extract text and automatically strips references, bibliographies, and acknowledgments to save context window space.
* **Persistent Caching:** Stores processed paper metadata in `.paper_cache.json` so only new or updated files are sent to the LLM.
* **Structured Output:** Generates a clean `summaries.md` file with Problem, Method, Findings, and Significance sections.
* **Customizable:** Supports custom PDF directories, LLM models, and output paths via CLI arguments.

## Prerequisites

1.  **Ollama:** Install and run [Ollama](https://ollama.ai/).
2.  **Pull a Model:** Ensure you have a model downloaded (e.g., `ollama pull gemma4:26b` or `ollama pull llama3.2`).
3.  **Python Dependencies:**
    ```bash
    pip install pymupdf requests
    ```

## Installation

1.  Clone this repository or save the script as `summarize_papers.py`.
2.  Create a folder (e.g., `papers/`) and drop your academic PDFs into it.

## Usage

### Basic Usage
Run the script using default settings (looks for PDFs in `./papers` using `gemma4:26b`):
```bash
python summarize_papers.py
```

### Custom Directory and Model
Specify a specific folder and a lighter model like `llama3.2`:
```bash
python summarize_papers.py /path/to/my/research llama3.2
```

### Full Configuration
Specify the input directory, model, and a custom output location:
```bash
python summarize_papers.py ./pdfs llama3.2 -o ./literature_review
```

## How it Works

1.  **PDF Parsing:** The script reads the first 5 pages of each PDF (where the core introduction and methodology usually reside).
2.  **Cleaning:** It uses regex to truncate the text at the "References" or "Bibliography" section to keep the prompt focused.
3.  **Metadata Extraction:** It attempts to parse the publication year and a clean title from the document text and filename.
4.  **LLM Summarization:** The cleaned text is sent to your local Ollama instance with a system prompt designed for academic review.
5.  **Output:** A combined `summaries.md` is generated in the output folder.

## Configuration

You can change the Ollama connection URL via environment variables:
```bash
export OLLAMA_BASE="http://192.168.1.10:11434"
python summarize_papers.py
```

## License
MIT