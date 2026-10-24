"""Streamlit UI only. All logic lives in the rag/ package."""
import os

import streamlit as st

# Streamlit Cloud: copy the key from st.secrets into the environment BEFORE importing config.
try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ.setdefault("OPENAI_API_KEY", st.secrets["OPENAI_API_KEY"])
except Exception:
    pass

import config
from rag import pipeline

st.set_page_config(page_title="RealEstate Research Tool", page_icon="🏙️", layout="wide")
st.title("🏙️ RealEstate Research Tool")

if not config.api_key():
    st.error("OPENAI_API_KEY is missing. Copy `.env.example` to `.env`, add your key, and restart.")
    st.stop()

ss = st.session_state
ss.setdefault("report", None)
ss.setdefault("store_version", 0)
ss.setdefault("last", None)
if "store" not in ss:
    ss.store = pipeline.load_existing_store()


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("News Article URLs")
    pasted = st.text_area(
        "One URL per line",
        height=180,
        placeholder="https://www.cnbc.com/2026/10/20/example-article.html",
    )

    if st.button("Process URLs", type="primary"):
        lines = [u.strip() for u in pasted.splitlines() if u.strip()]
        bad = [u for u in lines if not u.startswith(("http://", "https://"))]
        items = [{"url": u} for u in lines if u not in bad]
        if bad:
            st.warning("Ignored (must start with http:// or https://): " + ", ".join(bad))
        if not items:
            st.warning("Paste at least one valid article URL.")
        else:
            with st.status("Processing articles...", expanded=True) as status:
                report = pipeline.ingest(items, progress=status.write)
                if report.store is None:
                    status.update(label="No articles could be processed", state="error")
                else:
                    status.update(label=f"Ready: {len(report.articles)} articles, "
                                        f"{report.n_chunks} chunks", state="complete")
            ss.report = report
            if report.store is not None:
                ss.store = report.store
                ss.store_version += 1
                ss.last = None

# ------------------------------------------------------------------ main area
report = ss.report
if report is not None:
    if report.articles:
        with st.expander(f"Knowledge base: {len(report.articles)} articles loaded", expanded=False):
            for a in report.articles:
                st.markdown(f"- [{a['title']}]({a['url']}) — {a['source']}, {a['published'] or 'date n/a'}")
    if report.failures:
        with st.expander(f"⚠️ {len(report.failures)} URL(s) skipped"):
            for url, reason in report.failures:
                st.markdown(f"- {url}\n  - {reason}")

question = st.text_input("Question:", placeholder="What is the latest Fed interest rate decision?")

if question:
    if ss.store is None:
        st.info("Fetch headlines or paste URLs, then click **Process URLs** first.")
    else:
        key = (question.strip(), ss.store_version)
        if ss.last is None or ss.last[0] != key:
            with st.spinner("Searching articles..."):
                try:
                    ss.last = (key, pipeline.answer(ss.store, question))
                except Exception as exc:
                    ss.last = None
                    st.error(f"Something went wrong: {exc}")
        if ss.last is not None:
            result = ss.last[1]
            st.header("Answer")
            st.markdown(result.answer.replace("$", "\\$"))

            st.subheader("Sources:")
            for s in result.sources:
                st.markdown(f"- [{s['title'] or s['url']}]({s['url']}) — {s['source']}, {s['published'] or 'date n/a'}")

            with st.expander("Retrieved passages (for verification)"):
                for i, p in enumerate(result.passages, start=1):
                    st.markdown(f"**[{i}]**")
                    st.text(p.page_content)