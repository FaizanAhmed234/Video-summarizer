import io
import os
import streamlit as st
from google import genai

# Page configuration
st.set_page_config(
    page_title="AI Summarizer Pro",
    page_icon="📝",
    layout="wide",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #6c757d;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 0.8rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_api_key() -> str:
    """Retrieve API key from Streamlit secrets, environment variables, or sidebar input."""
    api_key = ""
    # Check st.secrets safely
    try:
        if "GEMINI_API_KEY" in st.secrets:
            api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

    # Check environment variable
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY", "")

    return api_key


def extract_text_from_file(uploaded_file) -> str:
    """Extract plain text from uploaded files (TXT, MD, PDF)."""
    file_type = uploaded_file.name.split(".")[-1].lower()

    if file_type in ["txt", "md"]:
        return uploaded_file.read().decode("utf-8", errors="replace")

    elif file_type == "pdf":
        try:
            import pypdf

            reader = pypdf.PdfReader(io.BytesIO(uploaded_file.read()))
            pages_text = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    pages_text.append(text)
            return "\n\n".join(pages_text)
        except ImportError:
            st.error("Missing dependency: please add `pypdf` to your requirements.txt.")
            return ""
        except Exception as e:
            st.error(f"Failed to read PDF file: {e}")
            return ""

    return ""


def generate_summary(
    api_key: str,
    text: str,
    style: str,
    length: str,
    custom_instruction: str = "",
    model_name: str = "gemini-2.5-flash",
) -> str:
    """Send text to Google Gemini and return a formatted summary."""
    try:
        client = genai.Client(api_key=api_key)

        style_prompts = {
            "Executive Summary": (
                "Provide a formal executive summary. Start with a high-level overview, "
                "follow with key findings/insights, and finish with conclusions or action items."
            ),
            "Key Bullet Points": (
                "Summarize the content as a crisp, well-organized bulleted list highlighting "
                "the most critical takeaways and metrics."
            ),
            "Concise Paragraph": (
                "Distill the core essence of the content into 1 to 2 clear, cohesive paragraphs. "
                "Keep it punchy, eliminating all fluff."
            ),
            "Detailed & Structured": (
                "Create a comprehensive summary with structured headings, sub-sections, and detailed explanations "
                "covering all major arguments and supporting details."
            ),
            "TL;DR": (
                "Provide an ultra-short TL;DR in 2 to 3 sentences capturing the absolute essentials."
            ),
        }

        length_prompts = {
            "Short": "Keep the summary brief and dense (approx. 100-200 words).",
            "Medium": "Provide balanced coverage of moderate length (approx. 250-450 words).",
            "Long": "Provide an in-depth summary covering all nuances (approx. 500-800 words).",
        }

        selected_style = style_prompts.get(style, style_prompts["Key Bullet Points"])
        selected_length = length_prompts.get(length, length_prompts["Medium"])

        prompt_parts = [
            "You are an expert executive analyst and text summarizer.",
            f"Format Instruction: {selected_style}",
            f"Length Instruction: {selected_length}",
        ]

        if custom_instruction.strip():
            prompt_parts.append(f"Additional User Instruction: {custom_instruction.strip()}")

        prompt_parts.append(
            "\nAnalyze and summarize the following source text accurately. "
            "Do not fabricate facts outside the text.\n\n"
            f"--- SOURCE TEXT ---\n{text}\n--- END SOURCE TEXT ---"
        )

        full_prompt = "\n".join(prompt_parts)

        response = client.models.generate_content(
            model=model_name,
            contents=full_prompt,
        )

        if response.text:
            return response.text

        return "Gemini returned an empty response. Please check your source text and try again."

    except Exception as exc:
        return (
            f"Error generating summary ({type(exc).__name__}): {exc}\n\n"
            "Please check that your Gemini API key is valid and has active quotas."
        )


# ==========================================
# UI Layout
# ==========================================

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Settings")

    detected_key = get_api_key()
    if detected_key:
        api_key_input = st.text_input(
            "Gemini API Key",
            value=detected_key,
            type="password",
            help="Loaded automatically from secrets. You can override it here.",
        )
    else:
        api_key_input = st.text_input(
            "Gemini API Key",
            type="password",
            placeholder="Enter your Gemini API Key",
            help="Get your API key at https://aistudio.google.com/app/apikey",
        )

    model_choice = st.selectbox(
        "Model",
        options=["gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
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

    summary_length = st.select_slider(
        "Target Length",
        options=["Short", "Medium", "Long"],
        value="Medium",
    )

    custom_notes = st.text_area(
        "Custom Instructions (Optional)",
        placeholder="e.g., Focus specifically on financial data or action items...",
        help="Add any custom guidance for the model.",
    )

# Main Page Header
st.markdown('<div class="main-title">📝 AI Document & Text Summarizer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Extract clear, structured insights from articles, notes, reports, and PDFs using Google Gemini.</div>',
    unsafe_allow_html=True,
)

# Input Section (Tabs)
tab_text, tab_file = st.tabs(["✍️ Paste Text", "📂 Upload File (PDF / TXT / MD)"])

source_text = ""

with tab_text:
    pasted_text = st.text_area(
        "Source Text",
        height=260,
        placeholder="Paste your article, meeting notes, essay, or raw document content here...",
    )
    if pasted_text:
        source_text = pasted_text

with tab_file:
    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["txt", "md", "pdf"],
        help="Supports .txt, .md, and .pdf documents",
    )
    if uploaded_file is not None:
        extracted = extract_text_from_file(uploaded_file)
        if extracted:
            source_text = extracted
            st.success(f"Loaded '{uploaded_file.name}' ({len(source_text.split()):,} words).")
            with st.expander("Preview Extracted Content", expanded=False):
                st.write(source_text[:2000] + ("..." if len(source_text) > 2000 else ""))

# Metrics bar for source text
if source_text.strip():
    word_count = len(source_text.split())
    char_count = len(source_text)
    col1, col2 = st.columns([1, 5])
    col1.caption(f"**Words:** {word_count:,}")
    col2.caption(f"**Characters:** {char_count:,}")

# Action Buttons
col_btn1, col_btn2, _ = st.columns([1.5, 1.5, 7])
generate_clicked = col_btn1.button("✨ Summarize", type="primary", use_container_width=True)
clear_clicked = col_btn2.button("Clear", use_container_width=True)

if clear_clicked:
    st.session_state["summary_result"] = None
    st.rerun()

# Execution
if generate_clicked:
    if not api_key_input.strip():
        st.error("Please provide a Gemini API Key in the sidebar or via Streamlit Secrets.")
    elif not source_text.strip():
        st.warning("Please paste some text or upload a document to summarize.")
    else:
        with st.spinner("Analyzing document and generating summary..."):
            summary_output = generate_summary(
                api_key=api_key_input.strip(),
                text=source_text,
                style=summary_style,
                length=summary_length,
                custom_instruction=custom_notes,
                model_name=model_choice,
            )
            st.session_state["summary_result"] = summary_output

# Output Display
if st.session_state.get("summary_result"):
    st.markdown("### 📋 Generated Summary")
    st.markdown(st.session_state["summary_result"])

    summary_words = len(st.session_state["summary_result"].split())
    original_words = len(source_text.split()) if source_text else 0

    if original_words > 0:
        ratio = round((1 - (summary_words / original_words)) * 100)
        st.caption(f"Summary contains **{summary_words:,} words** ({ratio}% compression reduction).")

    st.download_button(
        label="📥 Download Summary (.md)",
        data=st.session_state["summary_result"],
        file_name="summary.md",
        mime="text/markdown",
    )
