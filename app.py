import os
import streamlit as st
from google import genai
from google.genai import types

st.set_page_config(
    page_title="AI Summarizer",
    page_icon="📝",
    layout="wide",
)

def get_secret_api_key() -> str:
    """Retrieve API key silently from Streamlit Secrets or Environment Variables."""
    try:
        key = st.secrets.get("GEMINI_API_KEY", "") or st.secrets.get("gemini_api_key", "")
        if key:
            return key.strip()
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY", "").strip()


# Sidebar Settings
with st.sidebar:
    st.header("⚙️ Settings")
    model_choice = st.selectbox(
        "Model",
        options=["gemini-2.5-flash", "gemini-3.5-flash-lite", "gemini-3.8-flash"],
        index=0,
    )
    summary_style = st.selectbox(
        "Summary Style",
        options=[
            "Key Bullet Points",
            "Executive Summary",
            "Concise Paragraph",
            "Detailed & Structured",
            "TL;DR",
        ],
        index=0,
    )
    summary_length = st.select_slider("Target Length", options=["Short", "Medium", "Long"], value="Medium")
    custom_notes = st.text_area("Custom Instructions (Optional)", placeholder="e.g., Highlight total pricing, client info, or deadlines...")

# Main UI
st.title("📝 AI Document & Text Summarizer")

tab_text, tab_file = st.tabs(["✍️ Paste Text", "📂 Upload File (PDF / TXT / MD)"])

text_input = ""
pdf_bytes = None
file_name = ""

with tab_text:
    text_input = st.text_area(
        "Source Text",
        height=240,
        placeholder="Paste your article, meeting notes, essay, or text here...",
    )

with tab_file:
    uploaded = st.file_uploader("Upload Document", type=["pdf", "txt", "md"])
    if uploaded is not None:
        file_name = uploaded.name
        ext = file_name.split(".")[-1].lower()

        if ext == "pdf":
            # Read full raw binary for native Gemini processing
            pdf_bytes = uploaded.getvalue()
            st.success(f"Loaded '{file_name}' ({len(pdf_bytes) / 1024:.1f} KB).")
        elif ext in ["txt", "md"]:
            text_input = uploaded.getvalue().decode("utf-8", errors="replace")
            st.success(f"Loaded '{file_name}' ({len(text_input.split()):,} words).")

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
    elif not text_input.strip() and not pdf_bytes:
        st.warning("Please paste some text or upload a document to summarize.")
    else:
        style_prompts = {
            "Key Bullet Points": "Summarize with concise, high-impact bullet points focusing on key takeaways.",
            "Executive Summary": "Write a formal executive summary with an overview, core findings, and action items.",
            "Concise Paragraph": "Summarize into 1 to 2 clear, punchy paragraphs.",
            "Detailed & Structured": "Provide a comprehensive breakdown with structured markdown headings.",
            "TL;DR": "Provide a 2 to 3 sentence TL;DR capturing the essential message.",
        }

        instruction_text = (
            f"You are an expert analyst.\n"
            f"Style: {style_prompts.get(summary_style, '')}\n"
            f"Target Length: {summary_length}\n"
            f"{f'Additional instruction: {custom_notes}' if custom_notes else ''}\n\n"
            "Please analyze and summarize the provided content accurately."
        )

        # Prepare payload: Native PDF part vs raw text
        if pdf_bytes:
            pdf_part = types.Part.from_bytes(
                data=pdf_bytes,
                mime_type="application/pdf",
            )
            contents_payload = [pdf_part, instruction_text]
        else:
            contents_payload = [f"{instruction_text}\n\n--- CONTENT ---\n{text_input}"]

        st.markdown("### 📋 Generated Summary")
        output_placeholder = st.empty()

        try:
            client = genai.Client(api_key=api_key)

            # Stream response chunks directly
            def generate_stream():
                response = client.models.generate_content_stream(
                    model=model_choice,
                    contents=contents_payload,
                )
                for chunk in response:
                    if chunk.text:
                        yield chunk.text

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
