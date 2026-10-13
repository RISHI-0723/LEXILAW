# LexiLaw — Legal Document Analysis and Information Retrieval

LexiLaw is a small, working college prototype that lets you upload a legal
document (PDF or DOCX), ask a question about it in plain English, and get an
answer grounded in passages retrieved from *your* document.

> **Legal disclaimer:** LexiLaw is an educational legal-information prototype.
> It may make mistakes and does **not** replace advice from a qualified legal
> professional.

---

## 1. Problem statement

Legal documents (contracts, agreements, policies) are long and hard to read.
People need to find specific clauses and understand them in plain language,
but keyword search misses meaning and general chatbots hallucinate statutes
and cases. LexiLaw answers questions using only the uploaded document's own
text, and shows the exact passages it used.

## 2. Objectives

- Extract text from legal PDFs and DOCX files while preserving source info.
- Retrieve the most relevant passages for a natural-language question.
- Generate a plain-language answer grounded strictly in those passages.
- Always show the supporting evidence (filename, page, excerpt, score).
- Abstain honestly when the document does not contain the answer.

## 3. Features

- Upload multiple PDF / DOCX files.
- Page-by-page PDF extraction with real page numbers.
- Paragraph-based chunking with long-paragraph splitting.
- Near-duplicate removal using Jaccard similarity.
- Local semantic embeddings (Sentence Transformers).
- Cosine-similarity semantic search (top-k).
- Groq-powered, evidence-grounded answers.
- Expandable evidence cards with similarity scores.
- Optional document summary and important-clauses review.
- Graceful handling of unsupported files, scanned PDFs, missing API key,
  and API errors.

## 4. Technology stack

| Purpose              | Library                                             |
|----------------------|-----------------------------------------------------|
| UI                   | Streamlit                                           |
| PDF extraction       | PyMuPDF (`fitz`)                                    |
| DOCX extraction      | python-docx                                         |
| Embeddings           | sentence-transformers (`all-MiniLM-L6-v2`)          |
| Vector math          | NumPy                                               |
| Answer generation    | Groq Python SDK                                     |
| Config / secrets     | python-dotenv                                       |

## 5. System architecture

```
PDF / DOCX ──▶ document_utils.py ──▶ retrieval.py ──▶ llm_utils.py ──▶ app.py
              (extract + chunk)      (dedupe +          (Groq answer)    (UI)
                                      embed + search)
```

1. **document_utils.py** — extracts text (keeping filename, source id, and PDF
   page number) and splits it into chunks.
2. **retrieval.py** — removes near-duplicate chunks (Jaccard), embeds the rest,
   and ranks them against the question by cosine similarity.
3. **llm_utils.py** — sends the question plus only the retrieved evidence to
   Groq with a strict, evidence-only system prompt.
4. **app.py** — the Streamlit interface tying it together with session state.

## 6. How Jaccard similarity works here

Jaccard is used **only to remove near-duplicate chunks**, not to rank
relevance. For two chunks A and B, each is normalized to a set of lowercase
word tokens and:

```
J(A, B) = |A ∩ B| / |A ∪ B|
```

If a new chunk's Jaccard similarity to an already-kept chunk is at least
`JACCARD_THRESHOLD` (default **0.85**, in `retrieval.py`), it is treated as a
duplicate and skipped. The first chunk's text and metadata are preserved.

## 7. How Sentence Transformers and cosine similarity work here

Each kept chunk is converted into a 384-dimensional vector using
`sentence-transformers/all-MiniLM-L6-v2`. Vectors are L2-normalized, so the
**cosine similarity** between two vectors equals their dot product. At query
time the question is embedded the same way, and chunks are ranked by cosine
similarity (highest first). The top-k chunks are returned with their scores.
The score is a *retrieval* measure, not a probability that the answer is
legally correct.

## 8. How Groq is used

Only the question and the retrieved evidence chunks are sent to Groq — never
the whole document. The system prompt instructs the model to answer only from
the evidence, cite supporting filenames/pages, avoid inventing statutes or
cases, abstain when evidence is missing, and treat uploaded text as untrusted
data (so instructions hidden in a document cannot hijack the assistant).

## 9. Project folder structure

```
LexiLaw/
├── app.py              # Streamlit interface and workflow
├── document_utils.py   # PDF/DOCX extraction and chunking
├── retrieval.py        # Jaccard dedupe, embeddings, cosine retrieval
├── llm_utils.py        # Groq client and evidence-grounded prompts
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── CLAUDE.md           # project instructions
```

## 10. Windows installation steps (PowerShell)

```powershell
# From the project folder:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> If PowerShell blocks the activation script, run this once in the same window:
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

## 11. Configure the API key

```powershell
Copy-Item .env.example .env
notepad .env
```

Then set:

```
GROQ_API_KEY=your_real_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

Get a key from <https://console.groq.com/keys>. Confirm the model name against
the models enabled in your account at
<https://console.groq.com/docs/models> and change `GROQ_MODEL` if needed.
Never commit `.env`.

## 12. Run the app

```powershell
streamlit run app.py
```

Streamlit opens the app in your browser (usually <http://localhost:8501>).

> **First run:** the embedding model (~90 MB) downloads automatically the
> first time you process a document. This needs an internet connection and may
> take a minute. Later runs use the cached model.

## 13. Example questions

- "What are the termination conditions in this agreement?"
- "What is the notice period mentioned?"
- "Who are the parties to this contract?"
- "What obligations does the service provider have?"
- "Is there a confidentiality clause, and what does it cover?"

## 14. Limitations

- Searches **only the uploaded documents** — no external legal database,
  statutes, or case law.
- No OCR: scanned / image-only PDFs cannot be read.
- DOCX files have no reliable page numbers (shown as "N/A").
- Cosine scores are retrieval measures, not legal-correctness guarantees.
- The LLM can still make mistakes; always verify with a professional.

## 15. Future enhancements

- OCR support for scanned PDFs.
- Persisted vector index for larger collections.
- Highlighting the exact sentence used within an excerpt.
- Multi-turn conversation / follow-up questions.

## 16. Legal disclaimer

LexiLaw provides informational summaries for educational purposes only. It may
make mistakes and is **not** a substitute for advice from a qualified legal
professional. Do not upload confidential documents without appropriate
authorization; retrieved excerpts are sent to the Groq API for answering, so
review Groq's privacy policy before using sensitive material.
