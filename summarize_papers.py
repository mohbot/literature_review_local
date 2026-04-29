#!/usr/bin/env python3
"""
Summarize academic papers from a directory of PDFs using local LLMs via Ollama.
Extracts titles and years from each PDF, generates structured summaries,
and outputs a combined markdown file.

Only new/updated papers are sent to the LLM -- previously summarized papers
are loaded from a cache file and reused.

Usage:
    python summarize_papers.py
    python summarize_papers.py /path/to/pdfs
    python summarize_papers.py /path/to/pdfs llama3.2
    python summarize_papers.py /path/to/pdfs llama3.2 -o output_dir
"""

import os
import re
import sys
import json
import glob
import argparse
from pathlib import Path
from datetime import datetime

import requests
import fitz          # PyMuPDF

# --- Config -------------------------------------------------------------------
OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434")
CHAT_URL = f"{OLLAMA_BASE}/api/chat"
CACHE_FILE = ".paper_cache.json"

SUMMARY_PROMPT = """You are an expert academic reviewer. Read the following text from a research paper
and provide a concise, structured summary in markdown covering:
- **Problem**: What problem does the paper address?
- **Method**: What approach or methodology does it use?
- **Key Findings**: What are the main results or contributions?
- **Significance**: Why is this work important to the field?

Keep the summary to 200-300 words. Use clear, precise language.

--- PAPER TEXT ---
{paper_text}
--- END ---
"""

# --- PDF helpers --------------------------------------------------------------

def strip_references_and_acknowledgements(text):
    """Remove 'References' and 'Acknowledgement' sections from extracted text."""
    cutoff_patterns = [
        r'\n\s*#?\s*[Aa]cknowledgements?\s*(?:of\s*the\s*author)?\s*(?:and\s*References)?\s*\n',
        r'\n\s*#?\s*[Aa]cknowledgment\s*(?:of\s*the\s*author)?\s*(?:and\s*References)?\s*\n',
        r'\n\s*#?\s*[Rr]eferences\s*(?:and\s*[Aa]cknowledgements?)?\s*\n',
        r'\n\s*#?\s*[Rr]eferences\s*\n',
        r'\n\s*#?\s*[Ww]ork[Cc]ited\s*\n',
        r'\n\s*#?\s*[Bb]ibliography\s*\n',
    ]
    for pattern in cutoff_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            text = text[:match.start()]
            break
    text = re.sub(r'\n\s*#?\s*[Rr]eferences?\s*\n.*$', '', text, flags=re.IGNORECASE | re.DOTALL)
    return text.strip()


def extract_text(pdf_path, max_pages=5):
    """Extract text from the first few pages of a PDF, excluding references/acknowledgements."""
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for i in range(min(max_pages, len(doc))):
            text += doc[i].get_text()
        doc.close()
    except Exception as e:
        print(f"    [WARN] Text extraction failed: {e}")
    return strip_references_and_acknowledgements(text)


def extract_title_and_year(pdf_path, text):
    """Extract title and year from filename and PDF text."""
    year_match = re.search(r'\b(20\d{2})\b', text)
    year = year_match.group(1) if year_match else "Unknown"
    filename = Path(pdf_path).stem
    title = re.sub(r'[_\-\s]+', ' ', filename)
    title = re.sub(r'\s+', ' ', title).strip()
    title = title.replace(".pdf", "").strip()
    return title, year


# --- Cache management ---------------------------------------------------------

def load_cache(cache_path):
    """Load the cache of already-summarized papers."""
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache_path, cache):
    """Save the cache of summarized papers."""
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


# --- Ollama API ---------------------------------------------------------------

def summarize_with_ollama(paper_text, model):
    """Summarize paper text using the specified Ollama model."""
    prompt = SUMMARY_PROMPT.format(paper_text=paper_text[:80000])
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": 0.2,
            "top_p": 0.95,
            "top_k": 40,
            "num_predict": 2048,
        },
    }
    try:
        resp = requests.post(CHAT_URL, json=payload, timeout=300)
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except requests.exceptions.ConnectionError:
        return (
            "[ERROR] Could not connect to Ollama. "
            "Make sure 'ollama serve' is running on port 11434. "
            f"Try: {OLLAMA_BASE}/api/tags"
        )
    except requests.exceptions.HTTPError as e:
        detail = e.response.text if e.response is not None else str(e)
        return f"[ERROR] Ollama HTTP {e.response.status_code if e.response is not None else '?'}: {detail[:300]}"
    except Exception as e:
        return f"[ERROR] Failed to generate summary: {e}"


# --- Main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Summarize academic papers using local LLMs via Ollama.",
    )
    parser.add_argument(
        "pdf_dir", nargs="?", default="papers",
        help="Path to the folder containing PDF files (default: papers)",
    )
    parser.add_argument(
        "llm_model", nargs="?", default="gemma4:26b",
        help="Ollama LLM model to use (default: gemma4:26b)",
    )
    parser.add_argument(
        "--output", "-o", default="summaries",
        help="Output directory for the markdown file (default: summaries)",
    )
    args = parser.parse_args()

    PDF_DIR = args.pdf_dir
    LLM_MODEL = args.llm_model
    MD_DIR = args.output
    cache_path = os.path.join(PDF_DIR, CACHE_FILE)

    if not os.path.isdir(PDF_DIR):
        print(f"Error: '{PDF_DIR}' is not a valid directory.")
        sys.exit(1)

    pdf_paths = sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf")))
    if not pdf_paths:
        print(f"No PDF files found in '{PDF_DIR}/'")
        sys.exit(1)

    # Load cache of previously summarized papers
    cache = load_cache(cache_path)
    cached_count = len(cache)
    new_count = len(pdf_paths) - cached_count

    print(f"Using LLM model: {LLM_MODEL}\n")
    print(f"Found {len(pdf_paths)} PDF(s) in '{PDF_DIR}/'")
    if cached_count:
        print(f"Already cached: {cached_count} paper(s)")
        print(f"New papers to summarize: {new_count}\n")
    else:
        print()

    results = []
    processed_count = 0

    for i, pdf_path in enumerate(pdf_paths, 1):
        filename = Path(pdf_path).name
        already_cached = filename in cache

        if already_cached:
            print(f"[{i}/{len(pdf_paths)}] Skipping (cached): {filename}")
            continue

        print(f"[{i}/{len(pdf_paths)}] Processing: {filename}")

        # Extract text
        text = extract_text(pdf_path)
        if not text:
            results.append((filename, "Unknown", "Could not extract text from PDF."))
            print("  Skipped (empty text)\n")
            continue

        # Extract title and year
        title, year = extract_title_and_year(pdf_path, text)
        print(f"  Title: {title}")
        print(f"  Year:       {year}")

        # Generate summary with LLM
        summary = summarize_with_ollama(text, LLM_MODEL)
        print(f"  Summary: {summary[:120]}...\n")

        # Update cache
        cache[filename] = title
        results.append((title, year, summary))
        processed_count += 1

        # Save cache after each paper to avoid losing progress
        save_cache(cache_path, cache)

    # Save final cache
    save_cache(cache_path, cache)

    # Write markdown output
    os.makedirs(MD_DIR, exist_ok=True)
    output_file = os.path.join(MD_DIR, "summaries.md")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Paper Summaries\n\n")
        f.write(f"Total papers: {len(results)}\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("---\n\n")
        for title, year, summary in results:
            f.write(f"## {title}\n\n")
            f.write(f"**Year:** {year}\n\n")
            f.write(f"{summary}\n\n")
            f.write("---\n\n")

    print(f"Done! Written summaries to '{output_file}'")
    if processed_count:
        print(f"   {processed_count} new paper(s) summarized.")
    else:
        print("  No new papers to summarize.")


if __name__ == "__main__":
    main()
