"""
Marine Species AI — Premium Chat Interface
Inspired by ChatGPT / Claude / Gemini
Dark black theme · Chat bubbles · Fixed bottom input
"""

import io
import base64
import requests
from PIL import Image

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Marine AI Assistant",
    page_icon="🐠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

BACKEND_URL = "http://127.0.0.1:8000/analyze"

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"], .stApp {
    background-color: #0a0a0a !important;
    color: #e8e8e8 !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header, .stDeployButton { visibility: hidden !important; }
.block-container {
    padding-top: 0 !important;
    padding-bottom: 180px !important;
    max-width: 760px !important;
}

/* ── Top bar ── */
.topbar {
    position: fixed;
    top: 0; left: 0; right: 0;
    z-index: 999;
    background: rgba(10, 10, 10, 0.95);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-bottom: 1px solid #1e1e1e;
    padding: 14px 0;
    text-align: center;
}
.topbar-title {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 1rem;
    color: #ffffff;
    letter-spacing: -0.01em;
}
.topbar-sub {
    font-size: 0.72rem;
    color: #555;
    margin-top: 1px;
}

/* ── Spacer below topbar ── */
.topbar-spacer { height: 72px; }

/* ── Welcome screen ── */
.welcome-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 60px 20px 20px;
    text-align: center;
}
.welcome-icon {
    font-size: 3.5rem;
    margin-bottom: 20px;
    filter: drop-shadow(0 0 30px rgba(0, 200, 255, 0.3));
}
.welcome-title {
    font-family: 'Inter', sans-serif;
    font-size: 1.9rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.03em;
    margin-bottom: 10px;
}
.welcome-desc {
    font-size: 0.9rem;
    color: #555;
    line-height: 1.7;
    max-width: 400px;
}
.welcome-pill {
    display: inline-block;
    background: #141414;
    border: 1px solid #222;
    border-radius: 999px;
    padding: 6px 16px;
    font-size: 0.78rem;
    color: #666;
    margin: 4px;
}

/* ── Chat messages ── */
.chat-area { padding: 10px 0 20px; }

.msg-row {
    display: flex;
    margin-bottom: 20px;
    align-items: flex-start;
    gap: 10px;
    animation: fadeSlideUp 0.3s ease forwards;
}
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* User row — right aligned */
.msg-row.user {
    flex-direction: row-reverse;
}

/* Avatars */
.avatar {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.85rem;
    flex-shrink: 0;
    margin-top: 2px;
}
.avatar.ai {
    background: linear-gradient(135deg, #00c8ff, #0066cc);
    color: white;
}
.avatar.user {
    background: #1e1e1e;
    border: 1px solid #2a2a2a;
    color: #888;
}

/* Bubbles */
.bubble {
    max-width: 78%;
    padding: 13px 16px;
    border-radius: 18px;
    font-size: 0.9rem;
    line-height: 1.65;
}
.bubble.user {
    background: #1a1a1a;
    border: 1px solid #252525;
    color: #d8d8d8;
    border-top-right-radius: 4px;
}
.bubble.ai {
    background: #111;
    border: 1px solid #1e1e1e;
    color: #e0e0e0;
    border-top-left-radius: 4px;
}

/* Image inside bubble */
.bubble img {
    width: 100%;
    border-radius: 10px;
    margin-bottom: 12px;
    display: block;
}

/* Species badges */
.species-row { margin-bottom: 10px; }
.species-label {
    font-size: 0.72rem;
    color: #444;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 6px;
    font-family: 'JetBrains Mono', monospace;
}
.badge {
    display: inline-block;
    background: #0a1f2e;
    border: 1px solid #0e4060;
    color: #00c8ff;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 3px 12px;
    border-radius: 999px;
    margin: 2px 3px 2px 0;
    font-family: 'Inter', sans-serif;
}
.badge.none { color: #3a3a3a; border-color: #1e1e1e; background: #0d0d0d; }

/* Answer section */
.answer-label {
    font-size: 0.72rem;
    color: #444;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 10px 0 6px;
    font-family: 'JetBrains Mono', monospace;
}
.answer-text {
    font-size: 0.88rem;
    line-height: 1.7;
    color: #c8c8c8;
}
.divider {
    border: none;
    border-top: 1px solid #1a1a1a;
    margin: 10px 0;
}

/* ── FIXED BOTTOM INPUT BAR ── */
.input-bar-outer {
    position: fixed;
    bottom: 0; left: 0; right: 0;
    z-index: 1000;
    background: rgba(10, 10, 10, 0.97);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-top: 1px solid #1a1a1a;
    padding: 14px 16px 20px;
}
.input-bar-inner {
    max-width: 760px;
    margin: 0 auto;
    display: flex;
    align-items: center;
    gap: 10px;
    background: #111;
    border: 1px solid #222;
    border-radius: 16px;
    padding: 6px 10px;
}
.input-disclaimer {
    text-align: center;
    font-size: 0.68rem;
    color: #333;
    margin-top: 8px;
    max-width: 760px;
    margin-left: auto;
    margin-right: auto;
}

/* ── Widget overrides ── */

/* Text area — borderless inside input bar */
.stTextArea textarea {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    color: #e0e0e0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.92rem !important;
    resize: none !important;
    padding: 6px 4px !important;
    caret-color: #00c8ff;
}
.stTextArea textarea:focus {
    box-shadow: none !important;
    border: none !important;
}
.stTextArea > div { background: transparent !important; border: none !important; }
.stTextArea > div > div { background: transparent !important; border: none !important; }

/* Buttons — flat icon style */
.stButton > button {
    background: transparent !important;
    border: 1px solid #252525 !important;
    color: #888 !important;
    border-radius: 10px !important;
    font-size: 1.1rem !important;
    padding: 6px 12px !important;
    line-height: 1 !important;
    transition: all 0.15s ease !important;
    min-height: 36px !important;
    white-space: nowrap !important;
}
.stButton > button:hover {
    background: #1a1a1a !important;
    color: #fff !important;
    border-color: #333 !important;
}

/* Send button special */
.send-btn > button {
    background: linear-gradient(135deg, #00c8ff22, #0066cc22) !important;
    border: 1px solid #0e4060 !important;
    color: #00c8ff !important;
    border-radius: 10px !important;
}
.send-btn > button:hover {
    background: linear-gradient(135deg, #00c8ff33, #0066cc33) !important;
    border-color: #00c8ff88 !important;
    color: #fff !important;
}

/* File uploader — hidden but functional */
div[data-testid="stFileUploader"] {
    background: #0d1117 !important;
    border: 1px dashed #1e2d3d !important;
    border-radius: 12px !important;
}
div[data-testid="stFileUploader"] label { color: #555 !important; }

/* Camera input */
div[data-testid="stCameraInput"] {
    background: #0d0d0d !important;
    border: 1px solid #1a1a1a !important;
    border-radius: 12px !important;
}

/* Spinner */
.stSpinner > div > div { border-top-color: #00c8ff !important; }

/* Selectbox / radio hidden */
div[data-testid="stRadio"] > div {
    flex-direction: row !important;
    gap: 8px;
}
div[data-testid="stRadio"] label {
    background: #111 !important;
    border: 1px solid #222 !important;
    border-radius: 8px !important;
    padding: 4px 12px !important;
    font-size: 0.8rem !important;
    color: #666 !important;
}

/* Error / info boxes */
.stAlert { border-radius: 10px !important; font-size: 0.85rem !important; }

/* Remove extra gaps */
.element-container { margin-bottom: 0 !important; }
.stMarkdown { margin-bottom: 0 !important; }

/* Expander for attachment menu */
.streamlit-expanderHeader {
    background: #111 !important;
    border: 1px solid #222 !important;
    border-radius: 12px !important;
    color: #888 !important;
    font-size: 0.85rem !important;
}
.streamlit-expanderContent {
    background: #0d0d0d !important;
    border: 1px solid #1a1a1a !important;
    border-top: none !important;
    border-radius: 0 0 12px 12px !important;
}

hr { border-color: #1a1a1a !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "show_attach" not in st.session_state:
    st.session_state.show_attach = False
if "pending_image" not in st.session_state:
    st.session_state.pending_image = None   # PIL Image
if "pending_image_name" not in st.session_state:
    st.session_state.pending_image_name = "image.jpg"
if "pending_image_bytes" not in st.session_state:
    st.session_state.pending_image_bytes = None
if "input_key" not in st.session_state:
    st.session_state.input_key = 0

# ── Helper: image → base64 data-URL ──────────────────────────────────────────
def pil_to_b64(img: Image.Image, fmt="JPEG") -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format=fmt)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

# ── Top bar ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <div class="topbar-title">🐠 Marine AI Assistant</div>
  <div class="topbar-sub">YOLOv8 · FAISS RAG · Phi-3</div>
</div>
<div class="topbar-spacer"></div>
""", unsafe_allow_html=True)

# ── Chat area ─────────────────────────────────────────────────────────────────
chat_container = st.container()

with chat_container:
    if not st.session_state.messages:
        st.markdown("""
        <div class="welcome-wrap">
          <div class="welcome-icon">🐋</div>
          <div class="welcome-title">Marine AI Assistant</div>
          <div class="welcome-desc">
            Upload a photo of any marine species and ask questions.<br>
            Powered by YOLOv8 detection and local AI.
          </div>
          <br>
          <div>
            <span class="welcome-pill">📷 Camera capture</span>
            <span class="welcome-pill">📁 Upload image</span>
            <span class="welcome-pill">🔍 Species detection</span>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="chat-area">', unsafe_allow_html=True)
        for msg in st.session_state.messages:
            role = msg["role"]
            if role == "user":
                img_html = ""
                if msg.get("image_b64"):
                    img_html = f'<img src="{msg["image_b64"]}" alt="uploaded"/>'
                st.markdown(f"""
                <div class="msg-row user">
                  <div class="avatar user">You</div>
                  <div class="bubble user">
                    {img_html}
                    {msg["text"]}
                  </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # AI message
                detected = msg.get("detected", [])
                answer = msg.get("text", "")
                img_html = ""
                if msg.get("image_b64"):
                    img_html = f'<img src="{msg["image_b64"]}" alt="analyzed"/>'

                if detected:
                    badges = "".join(f'<span class="badge">{s}</span>' for s in detected)
                else:
                    badges = '<span class="badge none">No species detected</span>'

                st.markdown(f"""
                <div class="msg-row ai">
                  <div class="avatar ai">🐠</div>
                  <div class="bubble ai">
                    {img_html}
                    <div class="species-row">
                      <div class="species-label">Detected Species</div>
                      {badges}
                    </div>
                    <hr class="divider"/>
                    <div class="answer-label">AI Analysis</div>
                    <div class="answer-text">{answer}</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ── Attachment popup (above input bar) ───────────────────────────────────────
attach_area = st.container()

with attach_area:
    if st.session_state.show_attach:
        st.markdown("---")
        st.markdown("<p style='font-size:0.78rem;color:#555;margin-bottom:8px;'>Choose attachment source</p>", unsafe_allow_html=True)
        att_col1, att_col2 = st.columns(2, gap="small")

        with att_col1:
            st.markdown("<p style='font-size:0.8rem;color:#888;margin-bottom:4px;'>📷 Camera</p>", unsafe_allow_html=True)
            cam_img = st.camera_input("", label_visibility="collapsed", key="camera_widget")
            if cam_img:
                img = Image.open(cam_img)
                buf = io.BytesIO()
                img.convert("RGB").save(buf, format="JPEG")
                st.session_state.pending_image = img
                st.session_state.pending_image_bytes = buf.getvalue()
                st.session_state.pending_image_name = "capture.jpg"
                st.session_state.show_attach = False
                st.rerun()

        with att_col2:
            st.markdown("<p style='font-size:0.8rem;color:#888;margin-bottom:4px;'>📁 Upload</p>", unsafe_allow_html=True)
            up_file = st.file_uploader("", type=["jpg", "jpeg", "png", "webp"],
                                       label_visibility="collapsed", key="file_widget")
            if up_file:
                img = Image.open(up_file)
                buf = io.BytesIO()
                img.convert("RGB").save(buf, format="JPEG")
                st.session_state.pending_image = img
                st.session_state.pending_image_bytes = buf.getvalue()
                st.session_state.pending_image_name = up_file.name
                st.session_state.show_attach = False
                st.rerun()

    # Preview pending image
    if st.session_state.pending_image is not None:
        prev_col1, prev_col2 = st.columns([5, 1])
        with prev_col1:
            st.image(st.session_state.pending_image, width=80, caption="")
            st.markdown("<p style='font-size:0.72rem;color:#555;margin-top:-10px;'>Image ready to send</p>", unsafe_allow_html=True)
        with prev_col2:
            if st.button("✕", key="remove_img"):
                st.session_state.pending_image = None
                st.session_state.pending_image_bytes = None
                st.rerun()

# ── Fixed bottom input bar ────────────────────────────────────────────────────
st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

bar_col1, bar_col2, bar_col3 = st.columns([1, 10, 1.5], gap="small")

with bar_col1:
    attach_label = "✕" if st.session_state.show_attach else "➕"
    if st.button(attach_label, key="attach_btn", help="Attach image"):
        st.session_state.show_attach = not st.session_state.show_attach
        st.rerun()

with bar_col2:
    user_text = st.text_area(
        "msg_input",
        placeholder="Ask about the detected marine species...",
        label_visibility="collapsed",
        height=50,
        key=f"user_input_{st.session_state.input_key}",
    )

with bar_col3:
    st.markdown('<div class="send-btn">', unsafe_allow_html=True)
    send_btn = st.button("Send ↑", key="send_btn")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div class='input-disclaimer'>Marine AI · 100% local · No cloud APIs</div>", unsafe_allow_html=True)

# ── Handle send ───────────────────────────────────────────────────────────────
if send_btn:
    query = (user_text or "").strip()

    if not query and st.session_state.pending_image is None:
        st.error("Please type a question or attach an image first.")
    elif st.session_state.pending_image is None:
        st.error("Please attach an image using the ➕ button before sending.")
    elif not query:
        query = "Identify and describe this marine species."

    else:
        # Build user message
        img_b64 = pil_to_b64(st.session_state.pending_image)
        st.session_state.messages.append({
            "role": "user",
            "text": query,
            "image_b64": img_b64,
        })

        # Call backend
        image_bytes = st.session_state.pending_image_bytes
        image_name  = st.session_state.pending_image_name

        # Clear pending
        st.session_state.pending_image = None
        st.session_state.pending_image_bytes = None
        st.session_state.show_attach = False
        st.session_state.input_key += 1   # reset text area

        with st.spinner("🔍 Analyzing marine life..."):
            try:
                files = {"image": (image_name, image_bytes, "image/jpeg")}
                data  = {"query": query}
                resp  = requests.post(BACKEND_URL, files=files, data=data, timeout=180)
                resp.raise_for_status()
                result = resp.json()

                detected = result.get("detected_labels", [])
                answer   = result.get("answer", "No answer returned.")

                st.session_state.messages.append({
                    "role": "ai",
                    "text": answer,
                    "detected": detected,
                    "image_b64": img_b64,
                })

            except requests.exceptions.ConnectionError:
                st.session_state.messages.append({
                    "role": "ai",
                    "text": "❌ Cannot reach the backend. Start it with: <code>uvicorn backend.app:app --reload</code>",
                    "detected": [],
                    "image_b64": None,
                })
            except requests.exceptions.HTTPError:
                st.session_state.messages.append({
                    "role": "ai",
                    "text": f"❌ Backend error ({resp.status_code}): {resp.text}",
                    "detected": [],
                    "image_b64": None,
                })
            except Exception as e:
                st.session_state.messages.append({
                    "role": "ai",
                    "text": f"❌ Unexpected error: {e}",
                    "detected": [],
                    "image_b64": None,
                })

        st.rerun()