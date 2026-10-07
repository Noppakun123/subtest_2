"""แยกให้ถูก (Sort It Right) — RAG chatbot ตอบคำถามการคัดแยกขยะและรีไซเคิล"""
import os

import streamlit as st

from rag_core import (
    NOT_FOUND,
    REWRITE_PROMPT,
    SYSTEM_PROMPT,
    HybridIndex,
    build_user_prompt,
    chunk_documents,
    cited_numbers,
    load_documents,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
EMBED_MODEL = "intfloat/multilingual-e5-small"  # ~470MB, รองรับไทย/อังกฤษ เหมาะกับหน่วยความจำจำกัด
LLM_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
REWRITE_MODEL = "openai/gpt-oss-20b"

EXAMPLES = [
    "ขวดน้ำพลาสติกต้องทิ้งถังสีอะไร และต้องเตรียมอย่างไร",
    "ถ่านไฟฉายหมดแล้วทิ้งยังไง",
    "กล่องนม UHT รีไซเคิลได้ไหม",
    "พลาสติกเบอร์ 6 คืออะไร",
    "How should I dispose of an old phone?",
    "ราคารับซื้อกระป๋องอะลูมิเนียมวันนี้เท่าไร",
]

st.set_page_config(page_title="แยกให้ถูก | Sort It Right", page_icon="♻️", layout="centered")

st.markdown(
    """
    <style>
    .bin-row {display:flex; gap:.5rem; flex-wrap:wrap; margin:.25rem 0 1rem;}
    .bin {padding:.25rem .65rem; border-radius:999px; font-size:.85rem; color:#fff; font-weight:600;}
    .src-card {border-left:4px solid #2e7d32; padding:.4rem .75rem; margin:.4rem 0;
               background:rgba(46,125,50,.06); border-radius:4px; font-size:.9rem;}
    .src-meta {font-size:.8rem; opacity:.75;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Resources: โหลดโมเดลและสร้าง index เพียงครั้งเดียว (cache_resource)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="กำลังโหลด Embedding Model และสร้าง Vector Index (ครั้งแรกเท่านั้น)…")
def get_index():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL, device="cpu")

    def embed(texts, is_query):
        prefix = "query: " if is_query else "passage: "  # e5 ต้องใส่ prefix
        return model.encode([prefix + t for t in texts], normalize_embeddings=True,
                            batch_size=32, show_progress_bar=False)

    docs = load_documents(DATA_DIR)
    chunks = chunk_documents(docs, size=600, overlap=120)
    return HybridIndex(chunks, embed), docs


def get_client():
    try:
        key = st.secrets["GROQ_API_KEY"]
    except Exception:
        key = os.environ.get("GROQ_API_KEY")
    if not key:
        return None
    from groq import Groq
    return Groq(api_key=key)


def chat_kwargs(model):
    kw = {"model": model, "temperature": 0.1}
    if model.startswith("openai/gpt-oss"):
        kw.update(reasoning_effort="low", include_reasoning=False)
    return kw


def rewrite_query(client, history, question):
    """ทำให้คำถามต่อเนื่อง (เช่น 'แล้วอันนั้นล่ะ') กลายเป็นคำถามสมบูรณ์ก่อนค้นหา"""
    turns = [m for m in history if m["role"] in ("user", "assistant")][-4:]
    if not turns:
        return question
    convo = "\n".join(f"{'ผู้ใช้' if m['role'] == 'user' else 'ผู้ช่วย'}: {m['content'][:400]}" for m in turns)
    try:
        r = client.chat.completions.create(
            messages=[{"role": "system", "content": REWRITE_PROMPT},
                      {"role": "user", "content": f"ประวัติ:\n{convo}\n\nคำถามล่าสุด: {question}"}],
            max_completion_tokens=400, **chat_kwargs(REWRITE_MODEL))
        out = (r.choices[0].message.content or "").strip().split("\n")[0]
        return out or question
    except Exception:
        return question


def stream_answer(client, model, history, question, results):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    # ส่งบทสนทนาก่อนหน้า (ข้อความเท่านั้น) เพื่อให้คุยต่อเนื่องได้
    for m in history[-6:]:
        msgs.append({"role": m["role"], "content": m["content"]})
    msgs.append({"role": "user", "content": build_user_prompt(question, results)})
    stream = client.chat.completions.create(messages=msgs, stream=True,
                                            max_completion_tokens=1500, **chat_kwargs(model))
    for part in stream:
        delta = part.choices[0].delta.content if part.choices else None
        if delta:
            yield delta


def render_sources(sources, answer, show_scores):
    used = cited_numbers(answer)
    not_found = answer.strip().startswith(NOT_FOUND)
    label = ("🔎 เอกสารที่ค้นหาแล้ว (ไม่พบคำตอบ)" if not_found
             else f"📚 แหล่งอ้างอิง ({len(used) or len(sources)} รายการ)")
    with st.expander(label, expanded=not not_found):
        for n, s in enumerate(sources, 1):
            if used and n not in used and not not_found:
                continue
            score = (f" · dense={s['dense']:.3f}" if s.get("dense") is not None else "") + \
                    (f" · bm25={s['bm25']:.2f}" if s.get("bm25") is not None else "")
            st.markdown(
                f"<div class='src-card'><b>[{n}] {s['title']}</b> › {s['section']}<br>"
                f"<span class='src-meta'>📄 {s['source']}{score if show_scores else ''}</span></div>",
                unsafe_allow_html=True)
            snippet = s["text"] if len(s["text"]) <= 280 else s["text"][:280] + "…"
            st.caption(snippet)
        if used:
            others = [n for n in range(1, len(sources) + 1) if n not in used]
            if others:
                st.caption(f"ค้นพบแต่ไม่ได้ใช้ตอบ: {', '.join(f'[{n}]' for n in others)}")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
index, docs = get_index()

with st.sidebar:
    st.header("♻️ แยกให้ถูก")
    st.caption("ผู้ช่วย AI ตอบคำถามการคัดแยกขยะและรีไซเคิล จากคลังเอกสารความรู้ (ไทย/อังกฤษ) ด้วยเทคนิค RAG")
    st.subheader("ลองถามดู")
    for q in EXAMPLES:
        if st.button(q, use_container_width=True):
            st.session_state.pending = q
    st.divider()
    with st.expander("⚙️ ตั้งค่า"):
        model = st.selectbox("LLM (Groq)", LLM_MODELS, index=0)
        top_k = st.slider("จำนวน chunk ที่ค้นคืน (top-k)", 3, 8, 5)
        use_rewrite = st.toggle("เขียนคำถามต่อเนื่องใหม่ก่อนค้นหา", value=True)
        show_scores = st.toggle("แสดงคะแนนการค้นหา", value=False)
    with st.expander(f"📂 เอกสารความรู้ ({len(docs)} ไฟล์ · {len(index.chunks)} chunks)"):
        for d in docs:
            st.markdown(f"- `{d['source']}` ({len(d['text']):,} ตัวอักษร)")
    if st.button("🗑️ ล้างการสนทนา", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Main chat
# ---------------------------------------------------------------------------
st.title("♻️ แยกให้ถูก")
st.markdown(
    "ถามได้เลยว่าขยะชิ้นนี้ **ทิ้งถังไหน** รีไซเคิลได้ไหม หรือต้องจัดการอย่างไร — "
    "ทุกคำตอบอ้างอิงจากเอกสาร และจะตอบว่า *ไม่พบข้อมูล* ถ้าเอกสารไม่มีคำตอบ")
st.markdown(
    "<div class='bin-row'>"
    "<span class='bin' style='background:#2e7d32'>เขียว · อินทรีย์</span>"
    "<span class='bin' style='background:#f9a825'>เหลือง · รีไซเคิล</span>"
    "<span class='bin' style='background:#1565c0'>น้ำเงิน · ทั่วไป</span>"
    "<span class='bin' style='background:#c62828'>แดง · อันตราย</span></div>",
    unsafe_allow_html=True)

client = get_client()
if client is None:
    st.error("ยังไม่ได้ตั้งค่า `GROQ_API_KEY` ใน Secrets ของ Streamlit "
             "(Manage app → Settings → Secrets) จึงยังตอบคำถามไม่ได้")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar="🧑" if m["role"] == "user" else "♻️"):
        st.markdown(m["content"])
        if m.get("sources"):
            if m.get("search_query") and m["search_query"] != m.get("question"):
                st.caption(f"🔁 ค้นหาด้วย: {m['search_query']}")
            render_sources(m["sources"], m["content"], show_scores)

typed = st.chat_input("พิมพ์คำถาม เช่น หลอดไฟเสียทิ้งที่ไหน?", disabled=client is None)
question = typed or st.session_state.pop("pending", None)

if question and client is not None:
    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="♻️"):
        with st.spinner("กำลังค้นหาเอกสาร…"):
            search_q = rewrite_query(client, history, question) if (use_rewrite and history) else question
            results = index.search(search_q, k=top_k)
        if search_q != question:
            st.caption(f"🔁 ค้นหาด้วย: {search_q}")
        try:
            answer = st.write_stream(stream_answer(client, model, history, question, results))
        except Exception as e:  # noqa: BLE001
            answer = f"เกิดข้อผิดพลาดในการเรียก LLM: `{e}`"
            st.error(answer)
        sources = [{"source": r["chunk"].source, "title": r["chunk"].title, "section": r["chunk"].section,
                    "text": r["chunk"].text, "dense": r["dense"], "bm25": r["bm25"]} for r in results]
        render_sources(sources, answer, show_scores)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources,
                                      "question": question, "search_query": search_q})
