import io
import os
import streamlit as st
from google import genai

st.set_page_config(
    page_title="AI Summarizer",
    page_icon="📝",
    layout="wide",
)

# 1. API key loaded securely behind the scenes (never displayed in UI)
def get_secret_api_key() -> str:
    if "GEMINI_API_KEY" in st.secrets:
        return st.secrets["GEMINI_API_KEY"]
    return os.getenv("GEMINI_API_KEY", "")


def extract_text(uploaded_file) -> str:
    ext = uploaded_file.name.split(".")[-1].lower()
    if ext in ["txt", "md"]:
        return uploaded_file.read().decode("utf-8", errors="replace")
    elif ext == "pdf":
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(uploaded_file.read()))
        return "\n\n".join([page.extract_text() or "" for page in reader.pages])
    return ""


def generate_summary(text: str, style: str, length: str, model_name: str, custom_instruction: str = "") -> str:
    api_key = get_secret_api_key()
    if not api_key:
        return "Error: `GEMINI_API_KEY` was not found in Streamlit Secrets. Please add it to your App Settings > Secrets."

    style_prompts = {
        "Key Bullet Points": "Summarize with concise, high-impact bullet points focusing on key takeaways.",
        "Executive Summary": "Write a formal executive summary with an overview, core findings, and action items.",
        "Concise Paragraph": "Summarize into 1 to 2 clear, punchy paragraphs.",
        "Detailed & Structured": "Provide a comprehensive breakdown with structured markdown headings.",
        "TL;DR": "Provide a 2 to 3 sentence TL;DR capturing the essential message.",
    }

    prompt = (
        f"You are an expert analyst.\n"
        f"Style: {style_prompts.get(style, '')}\n"
        f"Target Length: {length}\n"
        f"{f'Additional instruction: {custom_instruction}' if custom_instruction else ''}\n\n"
        f"Text to summarize:\n{text}"
    )

    # Automatically try chosen model, then fallback if endpoint reports not found
    candidate_models = [model_name]
    fallback_models = ["gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-4-argon"]
    for m in fallback_models:
        if m not in candidate_models:
            candidate_models.append(m)

    client = genai.Client(api_key=api_key)
    last_error = ""

    for target_model in candidate_models:
        try:
            response = client.models.generate_content(
                model=target_model,
                contents=prompt,
            )
            if response.text:
                return response.text
        except Exception as exc:
            last_error = str(exc)
            continue

    return f"Unable to generate summary: {last_error}"


# Sidebar Settings (API Key input completely removed)
with st.sidebar:
    st.header("⚙️ Settings")
    model_choice = st.selectbox(
        "Model",
        options=["gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-4-argon"],
        index=0,
    )
    summary_style = st.selectbox(
        "Summary Style",
        options=["Key Bullet Points", "Executive Summary", "Concise Paragraph", "Detailed & Structured", "TL;DR"],
        index=0,
    )
    summary_length = st.select_slider("Target Length", options=["Short", "Medium", "Long"], value="Medium")
    custom_notes = st.text_area("Custom Instructions (Optional)", placeholder="e.g. Focus on financial figures...")

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
    if not source_text.strip():
        st.warning("Please paste text or upload a document first.")
    else:
        with st.spinner("Generating summary..."):
            result = generate_summary(
                text=source_text,
                style=summary_style,
                length=summary_length,
                model_name=model_choice,
                custom_instruction=custom_notes,
            )
            st.session_state["summary"] = result

if st.session_state.get("summary"):
    st.markdown("### 📋 Generated Summary")
    st.markdown(st.session_state["summary"])
    st.download_button(
        label="📥 Download Summary (.md)",
        data=st.session_state["summary"],
        file_name="summary.md",
        mime="text/markdown",
    )
