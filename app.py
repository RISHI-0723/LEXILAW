"""
app.py
------
LexiLaw - Legal Document Assistant (premium Streamlit UI).

The backend is unchanged: this module only redesigns the interface and wires
it to the existing functions in document_utils / retrieval / llm_utils.

Workflow:
  1. Upload PDF / DOCX files.
  2. Process: extract text -> chunk -> Jaccard de-duplicate -> embed.
  3. Ask a question: embed -> cosine search -> Groq answer with evidence.

Processed chunks and embeddings live in st.session_state so they are not
rebuilt on every rerun.
"""

from __future__ import annotations

import html
import uuid
from pathlib import Path

import streamlit as st

import document_utils as du
import retrieval as rt
import llm_utils as llm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DISCLAIMER = (
    "LexiLaw is an educational legal-information prototype. It may make "
    "mistakes and does not replace advice from a qualified legal professional."
)

SUGGESTED_QUESTIONS = [
    "What are the termination conditions?",
    "What are the main obligations of each party?",
    "Are there any penalties or late fees?",
    "What deadlines are mentioned?",
    "Which clauses should I review carefully?",
]

ICON_SCALES = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="#0F766E" stroke-width="1.8" '
    'stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v18"/>'
    '<path d="M7 7h10"/><path d="M5 21h14"/>'
    '<path d="M7 7l-3 6a3 3 0 0 0 6 0z"/><path d="M17 7l-3 6a3 3 0 0 0 6 0z"/>'
    '</svg>'
)
ICON_ASK = ('<svg viewBox="0 0 24 24" fill="none" stroke="#0F766E" '
            'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
            '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>'
            '</svg>')
ICON_CLAUSE = ('<svg viewBox="0 0 24 24" fill="none" stroke="#0F766E" '
               'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
               '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>'
               '<path d="M14 2v6h6"/><path d="M8 13h8"/><path d="M8 17h5"/></svg>')
ICON_EVIDENCE = ('<svg viewBox="0 0 24 24" fill="none" stroke="#0F766E" '
                 'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
                 '<circle cx="11" cy="11" r="7"/><path d="m21 21-4.3-4.3"/></svg>')
ICON_FILE = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
             'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
             '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>'
             '<path d="M14 2v6h6"/></svg>')
ICON_INFO = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
             'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
             '<circle cx="12" cy="12" r="9"/><path d="M12 11v5"/>'
             '<path d="M12 8h.01"/></svg>')
ICON_UPLOAD = ('<svg viewBox="0 0 24 24" fill="none" stroke="#0F766E" '
               'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
               '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>'
               '<path d="M17 8l-5-5-5 5"/><path d="M12 3v12"/></svg>')
ICON_STEP_UP = ('<svg viewBox="0 0 24 24" fill="none" stroke="#0F766E" '
                'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
                '<path d="M12 3v12"/><path d="M8 7l4-4 4 4"/>'
                '<path d="M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2"/></svg>')
ICON_STEP_ASK = ICON_ASK
ICON_STEP_VERIFY = ('<svg viewBox="0 0 24 24" fill="none" stroke="#0F766E" '
                    'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
                    '<path d="M9 12l2 2 4-4"/>'
                    '<path d="M12 3l7 3v5c0 4.5-3 7.5-7 9-4-1.5-7-4.5-7-9V6z"/></svg>')

st.set_page_config(
    page_title="LexiLaw - Legal Document Intelligence",
    page_icon="⚖️",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

def load_css():
    css_path = Path(__file__).parent / "styles.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>",
                    unsafe_allow_html=True)


def disclaimer_banner():
    """A visually distinct, readable legal disclaimer callout."""
    st.markdown(
        f'<div class="lx-disclaimer">{ICON_INFO}<span>{html.escape(DISCLAIMER)}'
        f'</span></div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def init_state():
    st.session_state.setdefault("chunks", [])           # list[Chunk], embedded
    st.session_state.setdefault("processed_files", [])  # list[str] filenames
    st.session_state.setdefault("answer", None)
    st.session_state.setdefault("asked_question", None)
    st.session_state.setdefault("evidence", [])         # list[SearchResult]
    st.session_state.setdefault("summary", None)
    st.session_state.setdefault("clauses", None)
    st.session_state.setdefault("confirm_clear", False)
    st.session_state.setdefault("q_input", "")
    st.session_state.setdefault("search_results", [])
    st.session_state.setdefault("full_text", {})        # {filename: extracted text}


def clear_session():
    for key in ("chunks", "processed_files", "answer", "asked_question",
                "evidence", "summary", "clauses", "confirm_clear",
                "q_input", "search_results", "full_text"):
        st.session_state.pop(key, None)
    init_state()


# ---------------------------------------------------------------------------
# Document processing (unchanged pipeline, nicer status)
# ---------------------------------------------------------------------------

def process_uploads(uploaded_files):
    """Extract, chunk, de-duplicate, and embed all uploaded files."""
    all_chunks = []
    processed = []
    errors = []
    full_text = {}

    progress = st.progress(0.0, text="Starting...")
    total = len(uploaded_files)

    for i, uploaded in enumerate(uploaded_files):
        progress.progress(i / total, text=f"Reading {uploaded.name}...")
        try:
            source_id = f"{uploaded.name}-{uuid.uuid4().hex[:8]}"
            file_bytes = uploaded.getvalue()
            passages = du.extract_document(file_bytes, uploaded.name, source_id)
            # Keep the full extracted text available (read-only) for inspection.
            full_text[uploaded.name] = "\n\n".join(p.text for p in passages)
            chunks = du.chunk_document(passages)
            all_chunks.extend(chunks)
            processed.append(uploaded.name)
        except ValueError as exc:
            errors.append(str(exc))
        except Exception as exc:  # unexpected - keep message concise
            errors.append(f"Unexpected error processing '{uploaded.name}': {exc}")

    if not all_chunks:
        progress.empty()
        return [], processed, errors, full_text

    progress.progress(0.7, text="Removing near-duplicate passages...")
    deduped = rt.deduplicate_chunks(all_chunks)

    progress.progress(0.8, text="Generating embeddings (first run downloads the model)...")
    try:
        rt.embed_chunks(deduped)
    except Exception as exc:
        progress.empty()
        errors.append(f"Could not generate embeddings: {exc}")
        return [], processed, errors, full_text

    progress.progress(1.0, text="Done.")
    progress.empty()
    return deduped, processed, errors, full_text


def run_processing(uploaded_files):
    """Shared handler for the Process action from any location."""
    if not uploaded_files:
        st.warning("Please choose at least one PDF or DOCX file first.")
        return
    chunks, processed, errors, full_text = process_uploads(uploaded_files)
    if chunks:
        st.session_state["chunks"] = chunks
        st.session_state["processed_files"] = processed
        st.session_state["full_text"] = full_text
        # New documents invalidate previous results.
        for k in ("answer", "asked_question", "evidence", "summary",
                  "clauses", "search_results"):
            st.session_state[k] = [] if k in ("evidence", "search_results") else None
    for err in errors:
        st.error(err)
    if chunks:
        st.rerun()


# ---------------------------------------------------------------------------
# UI pieces
# ---------------------------------------------------------------------------

def render_header():
    key_ok = bool(llm.get_api_key())
    key_chip = (
        '<span class="lx-chip"><span class="lx-dot"></span>AI-assisted analysis</span>'
        if key_ok else
        '<span class="lx-chip" style="color:#B54708;background:#FFF7E8;'
        'border-color:#F2D399;">API key not configured</span>'
    )
    st.markdown(
        f"""
        <div class="lx-header">
          <div class="lx-brand">
            <div class="lx-logo">{ICON_SCALES}</div>
            <div>
              <div class="lx-title">LexiLaw</div>
              <div class="lx-sub">Legal Document Intelligence</div>
            </div>
          </div>
          <div class="lx-status">
            <span class="lx-chip"><span class="lx-dot"></span>Local session</span>
            {key_chip}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_upload_controls(key_suffix: str):
    """The real Streamlit uploader + process button. Rendered in one place."""
    uploaded = st.file_uploader(
        "Upload PDF or DOCX files",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        key="doc_uploader",
        help="Text-based PDFs and DOCX files are supported. Scanned "
             "(image-only) PDFs cannot be read in this version.",
    )
    if st.button("Process documents", type="primary", use_container_width=True,
                 key=f"process_{key_suffix}"):
        run_processing(uploaded)
    return uploaded


def render_upload_card():
    """The upload card used in the hero: icon, titles, native uploader, note."""
    with st.container(border=True):
        st.markdown(
            f'<div class="lx-upload-head"><span class="lx-upload-ic">{ICON_UPLOAD}'
            f'</span><div><div class="lx-upload-t">Upload your legal documents</div>'
            f'<div class="lx-upload-s">PDF and DOCX supported</div></div></div>',
            unsafe_allow_html=True,
        )
        render_upload_controls("welcome")
        st.markdown(
            '<div class="lx-upload-note">Text extraction and semantic search run '
            'locally in this session. Only the short passages needed to answer a '
            'question are sent to the Groq API.</div>',
            unsafe_allow_html=True,
        )


def render_preview_panel():
    """Illustrative, clearly-labelled visual preview (not real analysis)."""
    st.markdown(
        f"""
        <div class="lx-preview">
          <div class="lx-preview-top">
            <span class="lx-preview-ey">Preview</span>
            <span class="lx-preview-ill">Illustrative only</span>
          </div>
          <div class="lx-doc-card">
            <div class="lx-doc-head">
              <span class="lx-doc-ic">{ICON_FILE}</span>
              <span class="lx-doc-name">service-agreement.pdf</span>
            </div>
            <div class="lx-doc-sections">
              <span class="lx-sec">Termination</span>
              <span class="lx-sec">Payment terms</span>
              <span class="lx-sec">Obligations</span>
            </div>
          </div>
          <div class="lx-evi-card">
            <div class="lx-evi-label">Evidence preview</div>
            <div class="lx-evi-text">"Either party may terminate this agreement
              with thirty (30) days' written notice to the other party."</div>
            <div class="lx-evi-foot">
              <span>service-agreement.pdf</span><span>·</span><span>Page 4</span>
              <span class="lx-score">score 0.82</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_welcome():
    if not llm.get_api_key():
        st.warning(
            "No Groq API key detected. You can still upload and process "
            "documents, but answer generation needs `GROQ_API_KEY` set in your "
            "`.env` file."
        )

    # ----- Hero: two columns -----
    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        st.markdown('<div class="lx-eyebrow">Legal Document Intelligence</div>',
                    unsafe_allow_html=True)
        st.markdown('<h1 class="lx-hero-h">Understand every clause.<br>'
                    'Find every answer.</h1>', unsafe_allow_html=True)
        st.markdown(
            '<p class="lx-hero-p">Upload an agreement to explore its clauses, '
            'surface key obligations, and ask questions in plain English — with '
            'every answer grounded in passages retrieved from your own '
            'document.</p>',
            unsafe_allow_html=True,
        )
        render_upload_card()
    with right:
        render_preview_panel()

    # ----- Feature section -----
    st.markdown('<div class="lx-section-h">Everything you need to review a '
                'document</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="lx-features">
          <div class="lx-feat">
            <div class="lx-feat-ic">{ICON_ASK}</div>
            <div class="lx-feat-t">Ask questions</div>
            <div class="lx-feat-d">Find answers in your documents using natural language.</div>
          </div>
          <div class="lx-feat">
            <div class="lx-feat-ic">{ICON_CLAUSE}</div>
            <div class="lx-feat-t">Analyze clauses</div>
            <div class="lx-feat-d">Explore obligations, deadlines, penalties, and termination terms.</div>
          </div>
          <div class="lx-feat">
            <div class="lx-feat-ic">{ICON_EVIDENCE}</div>
            <div class="lx-feat-t">Review evidence</div>
            <div class="lx-feat-d">Inspect the source passages supporting each answer.</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ----- Workflow strip -----
    st.markdown(
        f"""
        <div class="lx-steps">
          <div class="lx-step">
            <span class="lx-step-ic">{ICON_STEP_UP}</span>
            <span class="lx-step-n">01</span>
            <span class="lx-step-t">Upload document</span>
          </div>
          <span class="lx-step-sep"></span>
          <div class="lx-step">
            <span class="lx-step-ic">{ICON_STEP_ASK}</span>
            <span class="lx-step-n">02</span>
            <span class="lx-step-t">Ask a question</span>
          </div>
          <span class="lx-step-sep"></span>
          <div class="lx-step">
            <span class="lx-step-ic">{ICON_STEP_VERIFY}</span>
            <span class="lx-step-n">03</span>
            <span class="lx-step-t">Verify the evidence</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_documents():
    with st.sidebar:
        st.markdown("### Documents")
        st.caption(f"{len(st.session_state['processed_files'])} indexed · "
                   f"{len(st.session_state['chunks'])} chunks")

        for name in st.session_state["processed_files"]:
            st.markdown(
                f'<div class="lx-filechip">{ICON_FILE}'
                f'<span>{html.escape(name)}</span></div>',
                unsafe_allow_html=True,
            )

        st.divider()
        with st.expander("Add more documents"):
            render_upload_controls("sidebar")

        st.divider()
        # Clear workspace with a confirmation safeguard.
        if not st.session_state["confirm_clear"]:
            if st.button("Clear workspace", use_container_width=True):
                st.session_state["confirm_clear"] = True
                st.rerun()
        else:
            st.warning("Remove all documents and results?")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Yes, clear", type="primary", use_container_width=True):
                    clear_session()
                    st.rerun()
            with c2:
                if st.button("Cancel", use_container_width=True):
                    st.session_state["confirm_clear"] = False
                    st.rerun()

        st.divider()
        if llm.get_api_key():
            st.caption(f"Groq model: `{llm.get_model_name()}`")
        else:
            st.caption("Groq key not set — answers unavailable.")
        st.caption("Do not upload confidential documents without authorization; "
                   "evidence excerpts are sent to the Groq API for answering.")


def _sample_results(n: int):
    """Wrap the first n indexed chunks as SearchResults (score 0) for summary/clauses."""
    sample = st.session_state["chunks"][:n]
    return [rt.SearchResult(c, 0.0) for c in sample]


def render_evidence(results, intro=True):
    if intro:
        st.markdown(
            '<div class="lx-note">These passages were retrieved from your '
            'uploaded documents. The retrieval score is a cosine-similarity '
            'measure, not a probability that the answer is legally correct.'
            '</div>',
            unsafe_allow_html=True,
        )
        st.write("")
    for i, r in enumerate(results, start=1):
        page = f"Page {r.page}" if r.page else "Page unavailable"
        fname = html.escape(r.filename)
        cid = html.escape(str(r.chunk_id))
        # A plain bordered card in normal document flow (no expander header,
        # which previously rendered a dark bar that overflowed the card).
        with st.container(border=True):
            st.markdown(
                f'<div class="lx-evi-head"><span class="lx-evi-n">Evidence {i}'
                f'</span><span class="lx-score">score {r.score:.3f}</span></div>'
                '<div class="lx-evi-rows">'
                f'<div class="lx-mrow"><span class="lx-mk">Source</span>'
                f'<span class="lx-mv">{fname}</span></div>'
                f'<div class="lx-mrow"><span class="lx-mk">Location</span>'
                f'<span class="lx-mv">{page} · Chunk {cid}</span></div>'
                '</div>'
                '<div class="lx-ex-label">Excerpt</div>',
                unsafe_allow_html=True,
            )
            # Excerpt via st.write so the raw document text is never parsed as HTML.
            st.write(r.text)


# ----- Tab bodies ----------------------------------------------------------

def tab_ask():
    st.markdown("### Ask your documents")
    st.markdown('<div class="lx-note">Answers are generated only from passages '
                'retrieved from your uploaded documents.</div>',
                unsafe_allow_html=True)
    st.write("")

    # Apply a pending suggested question BEFORE the input is instantiated.
    if st.session_state.get("pending_question") is not None:
        st.session_state["q_input"] = st.session_state.pop("pending_question")

    question = st.text_input(
        "Your question",
        key="q_input",
        placeholder="e.g. What are the termination conditions in this contract?",
        label_visibility="collapsed",
    )

    ask = st.button("Ask question", type="primary")

    st.caption("Suggested questions")
    # Lay out in rows of up to 3 so each button has room for its full label
    # (labels wrap over lines via CSS; Streamlit stacks columns on mobile).
    per_row = 3
    for start in range(0, len(SUGGESTED_QUESTIONS), per_row):
        row = SUGGESTED_QUESTIONS[start:start + per_row]
        cols = st.columns(per_row, gap="small")
        for col, sug in zip(cols, row):
            with col:
                if st.button(sug, key=f"sug_{sug}", use_container_width=True):
                    st.session_state["pending_question"] = sug
                    st.rerun()

    if ask:
        if not question.strip():
            st.warning("Please type a question.")
        else:
            with st.spinner("Searching your documents..."):
                results = rt.semantic_search(
                    question, st.session_state["chunks"], top_k=4
                )
            st.session_state["evidence"] = results
            st.session_state["asked_question"] = question
            if not results:
                st.session_state["answer"] = (
                    "I couldn't find enough supporting information in the "
                    "indexed documents to answer this reliably. Try rephrasing "
                    "your question, or check whether the relevant document has "
                    "been uploaded."
                )
            else:
                try:
                    with st.spinner("Generating an evidence-grounded answer..."):
                        st.session_state["answer"] = llm.generate_answer(
                            question, results
                        )
                except llm.LLMError as exc:
                    st.session_state["answer"] = None
                    st.error(str(exc))

    # ----- Answer card -----
    if st.session_state["answer"]:
        st.write("")
        with st.container(border=True):
            st.markdown('<div class="lx-label">AI-generated analysis</div>',
                        unsafe_allow_html=True)
            if st.session_state["asked_question"]:
                st.caption(f"Question: {st.session_state['asked_question']}")
            st.write(st.session_state["answer"])
            st.write("")
            # Copy action in its own sized column so it never overlaps the
            # disclaimer; the revealed st.code block has a native copy button.
            act_col, _spacer = st.columns([1, 2], gap="small")
            with act_col:
                show_copy = st.toggle("Show copyable text", key="copy_toggle")
            if show_copy:
                st.code(st.session_state["answer"], language="text")
            st.write("")
            disclaimer_banner()

    # ----- Evidence under the answer -----
    if st.session_state["evidence"]:
        st.write("")
        st.markdown("### Supporting evidence")
        render_evidence(st.session_state["evidence"])


def tab_summary():
    st.markdown("### Document summary")
    st.caption("Source documents: " + ", ".join(st.session_state["processed_files"]))
    if st.button("Generate summary", type="primary", key="gen_summary"):
        try:
            with st.spinner("Summarizing..."):
                st.session_state["summary"] = llm.generate_summary(_sample_results(6))
        except llm.LLMError as exc:
            st.error(str(exc))

    if st.session_state["summary"]:
        st.write("")
        with st.container(border=True):
            st.markdown('<div class="lx-label">Overview</div>',
                        unsafe_allow_html=True)
            st.write(st.session_state["summary"])
            st.caption("Generated from a representative sample of the indexed "
                       "passages. " + DISCLAIMER)
    else:
        st.info("Generate a plain-language overview of the uploaded documents. "
                "This uses a representative sample of the indexed passages.")


def tab_clauses():
    st.markdown("### Key clauses & obligations")
    if st.button("Find key clauses", type="primary", key="gen_clauses"):
        try:
            with st.spinner("Reviewing clauses..."):
                st.session_state["clauses"] = llm.extract_clauses(_sample_results(8))
        except llm.LLMError as exc:
            st.error(str(exc))

    if st.session_state["clauses"]:
        st.write("")
        with st.container(border=True):
            st.markdown('<div class="lx-label">Extracted clauses</div>',
                        unsafe_allow_html=True)
            st.write(st.session_state["clauses"])
            st.caption("Potentially significant wording may deserve review by a "
                       "qualified legal professional. " + DISCLAIMER)
    else:
        st.info("Identify important clauses, obligations, deadlines, and "
                "penalties explicitly present in the documents.")


def tab_search():
    st.markdown("### Search evidence")
    st.markdown('<div class="lx-note">Inspect which passages the retriever '
                'returns for a query — no answer is generated here.</div>',
                unsafe_allow_html=True)
    st.write("")

    col_q, col_k = st.columns([3, 1])
    with col_q:
        sq = st.text_input("Search query", key="search_q",
                           placeholder="e.g. confidentiality",
                           label_visibility="collapsed")
    with col_k:
        k = st.slider("Top-k", 3, 8, 4, key="search_k", label_visibility="collapsed")

    if st.button("Search", key="run_search"):
        if not sq.strip():
            st.warning("Please type a search query.")
        else:
            with st.spinner("Searching..."):
                st.session_state["search_results"] = rt.semantic_search(
                    sq, st.session_state["chunks"], top_k=k
                )

    results = st.session_state["search_results"]
    if results:
        st.write("")
        render_evidence(results, intro=True)
    elif st.session_state.get("search_q"):
        st.info("No sufficiently relevant passages were found for that query.")


def render_workspace():
    st.markdown("### Document workspace")
    docs = len(st.session_state["processed_files"])
    chunks = len(st.session_state["chunks"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Documents indexed", docs)
    c2.metric("Chunks available", chunks)
    c3.metric("Retrieval method", "Semantic + Jaccard")

    st.write("")
    tabs = st.tabs(["Ask AI", "Document summary", "Key clauses", "Search evidence"])
    with tabs[0]:
        tab_ask()
    with tabs[1]:
        tab_summary()
    with tabs[2]:
        tab_clauses()
    with tabs[3]:
        tab_search()

    # Full extracted text kept accessible but out of the way, so evidence
    # excerpts stay focused on the actual retrieved passages.
    full_text = st.session_state.get("full_text") or {}
    if full_text:
        st.write("")
        with st.expander("View full extracted document"):
            names = list(full_text.keys())
            pick = st.selectbox("Document", names, key="fulltext_pick") \
                if len(names) > 1 else names[0]
            st.caption("Exact text extracted from the source (read-only).")
            st.text_area("Extracted text", value=full_text[pick], height=320,
                         disabled=True, label_visibility="collapsed",
                         key="fulltext_view")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    load_css()
    init_state()
    render_header()

    ready = bool(st.session_state["chunks"])
    if ready:
        render_sidebar_documents()
        render_workspace()
    else:
        render_welcome()

    st.write("")
    st.divider()
    disclaimer_banner()


main()
