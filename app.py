import io
import os
import streamlit as st
from google import genai

st.set_page_config(
    page_title="AI Summarizer",
    page_icon="📝",
    layout="wide",
)

# Fetch API key silently from Secrets (never exposed in UI)
def get_secret_api_key() -> str:
    try:
        key = st.secrets.get("GEMINI_API_KEY", "") or st.secrets.get("gemini_api_key", "")
        if key:
            return key.strip()
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY", "").strip()


def extract_text(uploaded_file) -> str:
    ext = uploaded_file.name.split(".")[-1].lower()
    if ext in ["txt", "md"]:
        return uploaded_file.read().decode("utf-8", errors="replace")
    elif ext == "pdf":
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(uploaded_file.read()))
        return "\n\n".join([page.extract_text() or "" for page in reader.pages])
    return ""


# Sidebar Settings
with st.sidebar:
    st.header("⚙️ Settings")
    model_choice = st.selectbox(
        "Model",
        options=["gemini-2.5-flash", "gemini-3.5-flash-lite", "gemini-3.8-flash"],
        index=0,
        help="Fast, low-latency models for instant text generation.",
    )
    summary_style = st.selectbox(
        "Summary Style",
        options=["Key Bullet Points", "Executive Summary", "Concise Paragraph", "Detailed & Structured", "TL;DR"],
        index=0,
    )
    summary_length = st.select_slider("Target Length", options=["Short", "Medium", "Long"], value="Medium")
    custom_notes = st.text_area("Custom Instructions (Optional)", placeholder="e.g. Focus on financial metrics...")

# Main UI
st.title("📝 AI Document & Text Summarizer")

tab_text, tab_file = st.tabs(["✍️ Paste Text", "📂 Upload File (PDF / TXT / MD)"])
source_text = ""

with tab_text:
    pasted_text = st.text_area("Source Text", height=240, placeholder="Paste your text here...")
    if pasted_text:
        source_text = pasted_text

with tab_file:
    uploaded = st.file_uploader("Upload Document", type=["txt", "md", "pdf"])
    if uploaded:
        source_text = extract_text(uploaded)
        st.success(f"Loaded '{uploaded.name}' ({len(source_text.split()):,} words).")

col1, col2, _ = st.columns([1.5, 1.5, 7])
submit = col1.button("✨ Summarize", type="primary", use_container_width=True)
clear = col2.button("Clear", use_container_width=True)

if clear:
    st.session_state["summary"] = None
    st.rerun()

if submit:
    api_key = get_secret_api_key()

    if not api_key:
        st.error("Missing `GEMINI_API_KEY` in Streamlit Secrets. Please add it to App Settings > Secrets.")
    elif not source_text.strip():
        st.warning("Please paste some text or upload a document first.")
    else:
        style_prompts = {
            "Key Bullet Points": "Summarize with concise, high-impact bullet points focusing on key takeaways.",
            "Executive Summary": "Write a formal executive summary with an overview, core findings, and action items.",
            "Concise Paragraph": "Summarize into 1 to 2 clear, punchy paragraphs.",
            "Detailed & Structured": "Provide a comprehensive breakdown with structured markdown headings.",
            "TL;DR": "Provide a 2 to 3 sentence TL;DR capturing the essential message.",
        }

        prompt = (
            f"You are an expert analyst.\n"
            f"Style: {style_prompts.get(summary_style, '')}\n"
            f"Target Length: {summary_length}\n"
            f"{f'Additional instruction: {custom_notes}' if custom_notes else ''}\n\n"
            f"Text to summarize:\n{source_text}"
        )

        st.markdown("### 📋 Generated Summary")
        output_placeholder = st.empty()

        try:
            client = genai.Client(api_key=api_key)

            # Generator for real-time text streaming
            def generate_stream():
                response = client.models.generate_content_stream(
                    model=model_choice,
                    contents=prompt,
                )
                for chunk in response:
                    if chunk.text:
                        yield chunk.text

            # Stream words directly onto the page
            with output_placeholder.container():
                full_summary = st.write_stream(generate_stream())
                st.session_state["summary"] = full_summary

        except Exception as exc:
            st.error(f"Error generating summary: {exc}")

if st.session_state.get("summary"):
    st.download_button(
        label="📥 Download Summary (.md)",
        data=st.session_state["summary"],
        file_name="summary.md",
        mime="text/markdown",
    )
