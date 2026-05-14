#!/usr/bin/env python3
"""
Summarize academic papers from a directory of PDFs using local LLMs via Ollama.
Extracts titles and years from each PDF, generates structured summaries,
and outputs combined markdown and PDF files.

Only new/updated papers are sent to the LLM -- previously summarized papers
are loaded from a cache file and reused.

Usage:
    python summarize_papers.py
    python summarize_papers.py /path/to/pdfs
    python summarize_papers.py /path/to/pdfs llama3.2
    python summarize_papers.py /path/to/pdfs -o output_dir
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
import fitz                   # PyMuPDF
from markdown_pdf import MarkdownPdf, Section

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
        print(f"        [WARN] Text extraction failed: {e}")
    return strip_references_and_acknowledgements(text)


def extract_title_and_year(pdf_path, text):
    """Extract title and year from PDF text content (not filename)."""
    filename = Path(pdf_path).stem
    fallback_title = re.sub(r'[_\-\s]+', ' ', filename)
    fallback_title = re.sub(r'\s+', ' ', fallback_title).strip()
    fallback_title = fallback_title.replace(".pdf", "").strip()

    # Extract title from first substantial line in PDF text
    text_lines = text.split('\n')
    title = fallback_title
    for line in text_lines:
        stripped = line.strip()
        if len(stripped) < 10 or len(stripped.split()) < 4:
            continue
        if re.match(r'^[#\*\d\-\[\]]+', stripped):
            continue
        title = stripped
        break

    year_match = re.search(r'\b(20\d{2})\b', text)
    year = year_match.group(1) if year_match else "Unknown"
    return title, year


# --- Cache management ---------------------------------------------------------

_TITLE_INDEX_KEY = "_title_index"
_TEXT_HASH_KEY = "_text_hashes"

def _text_hash(text):
    """Compute a short hash for deduplication."""
    import hashlib
    return hashlib.sha256(text.strip().encode()).hexdigest()[:16]


def _normalize_title(title):
    """Normalize a title for deduplication: lowercase, strip punctuation."""
    title = title.lower().strip()
    title = re.sub(r'[^a-z0-9\s]', '', title)
    title = re.sub(r'\s+', ' ', title).strip()
    return title


def _title_overlap(norm_a, norm_b):
    """Check if two normalized titles share enough tokens to be the same paper."""
    tokens_a = set(norm_a.split())
    tokens_b = set(norm_b.split())
    if not tokens_a or not tokens_b:
        return False
    return len(tokens_a & tokens_b) >= 3


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


def build_title_index(cache):
    """Build a lookup from normalized title -> filename for all cached papers."""
    index = {}
    for fname, entry in cache.items():
        if fname == _TITLE_INDEX_KEY:
            continue
        if isinstance(entry, dict):
            title = entry.get("title", fname)
        else:
            title = entry
        norm = _normalize_title(title)
        index[norm] = fname
    return index


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
        help="Output directory for summaries (default: summaries)",
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
    title_index = build_title_index(cache)
    cached_count = len([k for k in cache if k != _TITLE_INDEX_KEY])

    print(f"Using LLM model: {LLM_MODEL}\n")
    print(f"Found {len(pdf_paths)} PDF(s) in '{PDF_DIR}/'")
    if cached_count:
        print(f"Already cached: {cached_count} paper(s)\n")
    else:
        print()

    new_summaries = []
    processed_count = 0

    for i, pdf_path in enumerate(pdf_paths, 1):
        filename = Path(pdf_path).name

           # Compute text hash to check against cache
        text = extract_text(pdf_path)
        if not text:
            print(f"[{i}/{len(pdf_paths)}] Skipped (empty text): {filename}\n")
            continue

        text_hash = _text_hash(text)
        text_hashes = cache.get(_TEXT_HASH_KEY, {})
        if text_hash in text_hashes:
            cached_fname = text_hashes[text_hash]
            entry = cache[cached_fname]
            if isinstance(entry, dict):
                title = entry.get("title", filename)
                summary = entry.get("summary", "No summary available.")
            else:
                title = entry
                summary = "Summary unavailable (legacy cache). Re-processing..."
                print(f"[{i}/{len(pdf_paths)}] [WARN] Legacy cache for {filename}, re-processing...")
            print(f"[{i}/{len(pdf_paths)}] Loaded from cache (hash match): {title}\n")
            continue

           # Extract title and year
        title, year = extract_title_and_year(pdf_path, text)
        print(f"[{i}/{len(pdf_paths)}] Processing: {filename}")
        print(f"  Title: {title}")
        print(f"  Year:   {year}")

           # Generate summary with LLM
        summary = summarize_with_ollama(text, LLM_MODEL)
        print(f"  Summary: {summary[:120]}...\n")

           # Update cache with hash and title index
        cache[filename] = {"title": title, "summary": summary}
        norm_title = _normalize_title(title)
        title_index[norm_title] = filename
        text_hashes[text_hash] = filename
        cache[_TEXT_HASH_KEY] = text_hashes
        new_summaries.append((title, year, summary))
        processed_count += 1

           # Save cache after each paper to avoid losing progress
        save_cache(cache_path, cache)

     # Save final cache
    save_cache(cache_path, cache)

       # Append new summaries to the latest summary.md
    os.makedirs(MD_DIR, exist_ok=True)
    existing_files = sorted(glob.glob(os.path.join(MD_DIR, "summaries_*.md")))
    latest_md = existing_files[-1] if existing_files else None

    if new_summaries:
        target_file = latest_md
        if target_file is None:
            timestamp = datetime.now().strftime('%Y-%m-%d_%H%M')
            target_file = os.path.join(MD_DIR, f"summaries_{timestamp}.md")

        with open(target_file, "a", encoding="utf-8") as f:
            if latest_md is None:
                # First run: write header
                f.write("# Paper Summaries\n\n")
                f.write(f"Total papers: 0\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
                f.write("---\n\n")
            for title, year, summary in new_summaries:
                f.write(f"## {title}\n\n")
                f.write(f"**Year:** {year}\n\n")
                f.write(f"{summary}\n\n")
                f.write("---\n\n")
            print(f"Appended {len(new_summaries)} summary(s) to '{os.path.basename(target_file)}'")
    else:
        print("  No new papers to summarize. Existing summaries preserved.")

     # Update total paper count in the latest summary file
    existing_files = sorted(glob.glob(os.path.join(MD_DIR, "summaries_*.md")))
    if existing_files:
        latest = existing_files[-1]
        try:
            with open(latest, "r", encoding="utf-8") as f:
                content = f.read()
            # Count all ## headers (paper entries)
            papers = re.findall(r"^## (.+)$", content, re.MULTILINE)
            count_line = f"Total papers: {len(papers)}\n"
            content = re.sub(r"Total papers: \d+", count_line, content)
            with open(latest, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Updated total paper count in '{os.path.basename(latest)}'")
        except Exception:
            pass

        # Also regenerate summaries.pdf from the latest .md
        if new_summaries:
            try:
                 # Read the full .md content for PDF generation
                target_md = latest_md
                if target_md is None:
                    target_md = target_file
                with open(target_md, "r", encoding="utf-8") as f:
                    md_content = f.read()
                timestamp = datetime.now().strftime('%Y-%m-%d_%H%M')
                pdf_file = os.path.join(MD_DIR, f"summaries_{timestamp}.pdf")
                pdf = MarkdownPdf()
                pdf.add_section(Section(md_content))
                pdf.save(pdf_file)
                print(f"Done! Written PDF summaries to '{pdf_file}'")
            except ImportError:
                pass
            except Exception as e:
                print(f"[WARN] PDF generation failed: {e}")


if __name__ == "__main__":
    main()

