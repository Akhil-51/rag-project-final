import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pypdf import PdfReader

# ---- CONFIG ----
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "your-groq-api-key-here")

# ---- PAGE SETUP ----
st.set_page_config(
    page_title="DocMind AI",
    page_icon="🧠",
    layout="wide"
)

# ---- CUSTOM CSS + 3D ANIMATIONS ----
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;600&display=swap');

    * { margin: 0; padding: 0; box-sizing: border-box; }

    .stApp {
        background: #000000;
        color: white;
        font-family: 'Inter', sans-serif;
        overflow-x: hidden;
    }

    /* Animated background canvas */
    .bg-animation {
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        z-index: 0;
        pointer-events: none;
    }

    /* Floating orbs */
    .orb {
        position: fixed;
        border-radius: 50%;
        filter: blur(80px);
        opacity: 0.15;
        animation: float 8s ease-in-out infinite;
        pointer-events: none;
        z-index: 0;
    }
    .orb1 { width: 600px; height: 600px; background: #7c3aed; top: -200px; left: -200px; animation-delay: 0s; }
    .orb2 { width: 500px; height: 500px; background: #2563eb; top: 50%; right: -200px; animation-delay: 2s; }
    .orb3 { width: 400px; height: 400px; background: #059669; bottom: -150px; left: 30%; animation-delay: 4s; }
    .orb4 { width: 300px; height: 300px; background: #dc2626; top: 30%; left: 50%; animation-delay: 6s; }

    @keyframes float {
        0%, 100% { transform: translateY(0px) scale(1); }
        50% { transform: translateY(-30px) scale(1.05); }
    }

    /* Grid lines background */
    .grid-bg {
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        background-image:
            linear-gradient(rgba(124,58,237,0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(124,58,237,0.03) 1px, transparent 1px);
        background-size: 50px 50px;
        z-index: 0;
        pointer-events: none;
    }

    /* Main content wrapper */
    .main-wrapper {
        position: relative;
        z-index: 10;
    }

    /* Hero title */
    .hero-title {
        text-align: center;
        font-family: 'Orbitron', monospace;
        font-size: 4rem;
        font-weight: 900;
        background: linear-gradient(135deg, #a78bfa 0%, #60a5fa 30%, #34d399 60%, #f59e0b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        animation: titleGlow 3s ease-in-out infinite;
        text-shadow: none;
        letter-spacing: 4px;
    }

    @keyframes titleGlow {
        0%, 100% { filter: brightness(1); }
        50% { filter: brightness(1.3); }
    }

    .hero-subtitle {
        text-align: center;
        color: #64748b;
        font-size: 1.1rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 3rem;
    }

    /* 3D Card effect */
    .card-3d {
        background: linear-gradient(135deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0.08) 100%);
        border: 1px solid rgba(167,139,250,0.2);
        border-radius: 20px;
        padding: 2rem;
        backdrop-filter: blur(20px);
        box-shadow:
            0 0 0 1px rgba(167,139,250,0.1),
            0 20px 40px rgba(0,0,0,0.5),
            inset 0 1px 0 rgba(255,255,255,0.1);
        transform: perspective(1000px) rotateX(0deg);
        transition: all 0.3s ease;
        margin-bottom: 1.5rem;
    }

    /* Feature cards */
    .feature-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.02), rgba(255,255,255,0.06));
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.05);
        transition: all 0.3s ease;
        height: 100%;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }

    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 0.8rem;
        display: block;
        animation: iconPulse 2s ease-in-out infinite;
    }

    @keyframes iconPulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.1); }
    }

    /* Success badge */
    .success-badge {
        background: linear-gradient(135deg, rgba(16,185,129,0.1), rgba(5,150,105,0.2));
        border: 1px solid rgba(16,185,129,0.4);
        border-radius: 50px;
        padding: 0.8rem 2rem;
        text-align: center;
        color: #34d399;
        font-weight: 600;
        letter-spacing: 1px;
        margin-bottom: 1.5rem;
        box-shadow: 0 0 20px rgba(16,185,129,0.2);
        animation: successPulse 2s ease-in-out infinite;
    }

    @keyframes successPulse {
        0%, 100% { box-shadow: 0 0 20px rgba(16,185,129,0.2); }
        50% { box-shadow: 0 0 40px rgba(16,185,129,0.4); }
    }

    /* Chat messages */
    .user-msg {
        background: linear-gradient(135deg, rgba(37,99,235,0.15), rgba(59,130,246,0.08));
        border: 1px solid rgba(59,130,246,0.3);
        border-radius: 16px 16px 4px 16px;
        padding: 1rem 1.5rem;
        margin: 0.8rem 0;
        color: #e2e8f0;
        box-shadow: 0 4px 15px rgba(37,99,235,0.1);
        animation: slideInRight 0.3s ease;
    }

    .ai-msg {
        background: linear-gradient(135deg, rgba(124,58,237,0.15), rgba(139,92,246,0.08));
        border: 1px solid rgba(139,92,246,0.3);
        border-radius: 16px 16px 16px 4px;
        padding: 1rem 1.5rem;
        margin: 0.8rem 0;
        color: #e2e8f0;
        box-shadow: 0 4px 15px rgba(124,58,237,0.1);
        animation: slideInLeft 0.3s ease;
    }

    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(20px); }
        to { opacity: 1; transform: translateX(0); }
    }

    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }

    /* Neon divider */
    .neon-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #7c3aed, #2563eb, #059669, transparent);
        margin: 2rem 0;
        box-shadow: 0 0 10px rgba(124,58,237,0.5);
    }

    /* Glowing label */
    .glow-label {
        font-family: 'Orbitron', monospace;
        font-size: 0.8rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #a78bfa;
        margin-bottom: 1rem;
        text-shadow: 0 0 10px rgba(167,139,250,0.5);
    }

    /* Input styling */
    .stTextInput input {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(167,139,250,0.3) !important;
        border-radius: 12px !important;
        color: white !important;
        padding: 1rem !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
    }
    .stTextInput input:focus {
        border-color: rgba(167,139,250,0.8) !important;
        box-shadow: 0 0 20px rgba(167,139,250,0.2) !important;
    }

    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #7c3aed, #2563eb) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.8rem 2rem !important;
        font-weight: 600 !important;
        letter-spacing: 1px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(124,58,237,0.4) !important;
    }
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(124,58,237,0.6) !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(0,0,0,0.8) !important;
        border-right: 1px solid rgba(167,139,250,0.1) !important;
        backdrop-filter: blur(20px) !important;
    }

    /* File uploader */
    .stFileUploader {
        border: 2px dashed rgba(167,139,250,0.3) !important;
        border-radius: 16px !important;
        padding: 1rem !important;
        transition: all 0.3s ease !important;
    }

    /* Spinner */
    .stSpinner > div {
        border-top-color: #a78bfa !important;
    }

   #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Intro animation overlay */
    .intro-overlay {
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        background: #000;
        z-index: 9999;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-direction: column;
        animation: introFade 0.5s ease 3s forwards;
    }

    @keyframes introFade {
        to { opacity: 0; pointer-events: none; }
    }

    .intro-text {
        font-family: 'Orbitron', monospace;
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: introScale 1s ease forwards;
        letter-spacing: 8px;
    }

    .intro-bar {
        width: 0%;
        height: 3px;
        background: linear-gradient(90deg, #7c3aed, #2563eb, #34d399);
        margin-top: 1.5rem;
        border-radius: 2px;
        animation: introBar 2.5s ease forwards;
        box-shadow: 0 0 20px rgba(124,58,237,0.8);
    }

    .intro-sub {
        color: #475569;
        font-size: 0.9rem;
        letter-spacing: 4px;
        margin-top: 1rem;
        text-transform: uppercase;
        animation: introFadeIn 1s ease 0.5s forwards;
        opacity: 0;
    }

    @keyframes introScale {
        from { transform: scale(0.5); opacity: 0; }
        to { transform: scale(1); opacity: 1; }
    }

    @keyframes introBar {
        from { width: 0%; }
        to { width: 300px; }
    }

    @keyframes introFadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Mouse cursor glow */
    .cursor-glow {
        position: fixed;
        width: 300px;
        height: 300px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(124,58,237,0.08) 0%, transparent 70%);
        pointer-events: none;
        z-index: 5;
        transform: translate(-50%, -50%);
        transition: all 0.1s ease;
    }

    /* Magnetic card effect */
    .card-3d:hover {
        transform: perspective(1000px) rotateX(2deg) rotateY(2deg) translateY(-5px);
        border-color: rgba(167,139,250,0.5);
        box-shadow:
            0 0 0 1px rgba(167,139,250,0.2),
            0 30px 60px rgba(0,0,0,0.6),
            0 0 40px rgba(124,58,237,0.15),
            inset 0 1px 0 rgba(255,255,255,0.15);
    }

    .feature-card:hover {
        transform: translateY(-8px) scale(1.02);
        border-color: rgba(167,139,250,0.3);
        box-shadow: 0 20px 40px rgba(0,0,0,0.4), 0 0 30px rgba(124,58,237,0.1);
    }
</style>

<!-- Intro Overlay -->
<div class="intro-overlay" id="introOverlay">
    <div class="intro-text">DOCMIND AI</div>
    <div class="intro-bar"></div>
    <div class="intro-sub">Initializing Neural Engine...</div>
</div>

<!-- Mouse cursor glow -->
<div class="cursor-glow" id="cursorGlow"></div>

<!-- Three.js 3D Particles -->
<canvas id="bg-canvas" style="position:fixed;top:0;left:0;width:100%;height:100%;z-index:1;pointer-events:none;opacity:0.4;"></canvas>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
(function() {
    // ---- CURSOR GLOW ----
    const glow = document.getElementById('cursorGlow');
    document.addEventListener('mousemove', (e) => {
        if (glow) {
            glow.style.left = e.clientX + 'px';
            glow.style.top = e.clientY + 'px';
        }
    });

    // ---- 3D PARTICLE SYSTEM ----
    const canvas = document.getElementById('bg-canvas');
    if (!canvas) return;
    const renderer = new THREE.WebGLRenderer({ canvas, alpha: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth/window.innerHeight, 0.1, 1000);
    camera.position.z = 5;

    // Particles
    const geometry = new THREE.BufferGeometry();
    const count = 3000;
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);

    for(let i = 0; i < count * 3; i += 3) {
        positions[i] = (Math.random() - 0.5) * 25;
        positions[i+1] = (Math.random() - 0.5) * 25;
        positions[i+2] = (Math.random() - 0.5) * 25;

        const c = Math.random();
        if(c < 0.33) {
            colors[i] = 0.6; colors[i+1] = 0.3; colors[i+2] = 1.0;
        } else if(c < 0.66) {
            colors[i] = 0.2; colors[i+1] = 0.5; colors[i+2] = 1.0;
        } else {
            colors[i] = 0.1; colors[i+1] = 0.9; colors[i+2] = 0.6;
        }
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
        size: 0.04,
        vertexColors: true,
        transparent: true,
        opacity: 0.9
    });

    const particles = new THREE.Points(geometry, material);
    scene.add(particles);

    // Mouse tracking for 3D rotation
    let mouseX = 0, mouseY = 0;
    document.addEventListener('mousemove', (e) => {
        mouseX = (e.clientX / window.innerWidth - 0.5) * 2;
        mouseY = (e.clientY / window.innerHeight - 0.5) * 2;
    });

    // Animate
    function animate() {
        requestAnimationFrame(animate);
        particles.rotation.x += 0.0002 + mouseY * 0.0005;
        particles.rotation.y += 0.0004 + mouseX * 0.0005;
        camera.position.x += (mouseX * 0.5 - camera.position.x) * 0.02;
        camera.position.y += (-mouseY * 0.5 - camera.position.y) * 0.02;
        camera.lookAt(scene.position);
        renderer.render(scene, camera);
    }
    animate();

    window.addEventListener('resize', () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    });
})();
</script>

<!-- 3D Floating Orbs -->
<div class="orb orb1"></div>
<div class="orb orb2"></div>
<div class="orb orb3"></div>
<div class="orb orb4"></div>
<div class="grid-bg"></div>

<!-- Three.js 3D Particles -->
<canvas id="bg-canvas" style="position:fixed;top:0;left:0;width:100%;height:100%;z-index:1;pointer-events:none;opacity:0.4;"></canvas>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
(function() {
    const canvas = document.getElementById('bg-canvas');
    if (!canvas) return;
    const renderer = new THREE.WebGLRenderer({ canvas, alpha: true });
    renderer.setSize(window.innerWidth, window.innerHeight);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth/window.innerHeight, 0.1, 1000);
    camera.position.z = 5;

    // Create particles
    const geometry = new THREE.BufferGeometry();
    const count = 2000;
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);

    for(let i = 0; i < count * 3; i += 3) {
        positions[i] = (Math.random() - 0.5) * 20;
        positions[i+1] = (Math.random() - 0.5) * 20;
        positions[i+2] = (Math.random() - 0.5) * 20;

        const colorChoice = Math.random();
        if(colorChoice < 0.33) {
            colors[i] = 0.6; colors[i+1] = 0.3; colors[i+2] = 1.0;
        } else if(colorChoice < 0.66) {
            colors[i] = 0.2; colors[i+1] = 0.5; colors[i+2] = 1.0;
        } else {
            colors[i] = 0.1; colors[i+1] = 0.9; colors[i+2] = 0.6;
        }
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
        size: 0.05,
        vertexColors: true,
        transparent: true,
        opacity: 0.8
    });

    const particles = new THREE.Points(geometry, material);
    scene.add(particles);

    // Animate
    function animate() {
        requestAnimationFrame(animate);
        particles.rotation.x += 0.0003;
        particles.rotation.y += 0.0005;
        renderer.render(scene, camera);
    }
    animate();

    window.addEventListener('resize', () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    });
})();
</script>
""", unsafe_allow_html=True)

# ---- SESSION STATE ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "docs_processed" not in st.session_state:
    st.session_state.docs_processed = False

# ---- SIDEBAR ----
with st.sidebar:
    st.markdown('<p class="glow-label"> DocMind AI</p>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("###  How it works")
    st.markdown("1.  Upload your PDFs")
    st.markdown("2.  AI analyzes them")
    st.markdown("3.  Ask anything")
    st.markdown("4.  Get instant answers")
    st.markdown("---")
    st.markdown("###  Powered By")
    st.markdown("-  Llama 3.3 70B")
    st.markdown("-  LangChain RAG")
    st.markdown("-  FAISS Vector DB")
    st.markdown("-  HuggingFace")
    st.markdown("---")
    if st.button(" Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    if st.button(" Reset All", use_container_width=True):
        st.session_state.vector_store = None
        st.session_state.docs_processed = False
        st.session_state.chat_history = []
        st.rerun()

# ---- HERO SECTION ----
st.markdown('<div style="position:relative;z-index:10;padding-top:2rem;">', unsafe_allow_html=True)
st.markdown('<h1 class="hero-title">DOCMIND AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Next Generation Document Intelligence</p>', unsafe_allow_html=True)
st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

# ---- UPLOAD SECTION ----
if not st.session_state.docs_processed:
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon"></span>
            <div style="color:#a78bfa;font-weight:700;font-size:1.1rem;margin-bottom:0.4rem">Multiple PDFs</div>
            <div style="color:#475569;font-size:0.85rem">Upload several documents at once</div>
        </div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon"></span>
            <div style="color:#60a5fa;font-weight:700;font-size:1.1rem;margin-bottom:0.4rem">Chat History</div>
            <div style="color:#475569;font-size:0.85rem">Remembers entire conversation</div>
        </div>""", unsafe_allow_html=True)
    with col_c:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon"></span>
            <div style="color:#34d399;font-weight:700;font-size:1.1rem;margin-bottom:0.4rem">Instant Answers</div>
            <div style="color:#475569;font-size:0.85rem">Powered by Llama 3.3 70B</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="card-3d">', unsafe_allow_html=True)
    st.markdown('<p class="glow-label">📁 Upload Documents</p>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Drop your PDFs here",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    if uploaded_files:
        st.markdown(f"<p style='color:#34d399;margin-top:0.5rem'>✅ {len(uploaded_files)} file(s) selected</p>", unsafe_allow_html=True)
        if st.button("⚡ Process Documents", use_container_width=True):
            with st.spinner("🔍 Analyzing your documents..."):
                raw_text = ""
                for f in uploaded_files:
                    pdf_reader = PdfReader(f)
                    for page in pdf_reader.pages:
                        raw_text += page.extract_text()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks = text_splitter.split_text(raw_text)
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                st.session_state.vector_store = FAISS.from_texts(chunks, embeddings)
                st.session_state.docs_processed = True
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ---- CHAT SECTION ----
if st.session_state.docs_processed:
    st.markdown('<div class="success-badge">✅ Documents Ready — Start Chatting Below</div>', unsafe_allow_html=True)

    prompt = PromptTemplate.from_template("""
    You are DocMind, an expert AI assistant. Use the context below to answer accurately.
    If the answer is not in the context, say "I couldn't find that in the document."
    Be clear, concise and helpful.

    Context: {context}
    Question: {question}
    Answer:
    """)

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=0.3
    )

    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Chat history
    if st.session_state.chat_history:
        st.markdown('<p class="glow-label">💬 Conversation</p>', unsafe_allow_html=True)
        for chat in st.session_state.chat_history:
            st.markdown(f'<div class="user-msg">🙋 <strong>You</strong><br>{chat["question"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="ai-msg">🧠 <strong>DocMind</strong><br>{chat["answer"]}</div>', unsafe_allow_html=True)

    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)
    st.markdown('<p class="glow-label">✏️ Ask a Question</p>', unsafe_allow_html=True)
    question = st.text_input("", placeholder="What is this document about? Summarize key points...")

    if st.button("🚀 Ask DocMind", use_container_width=True):
        if question:
            with st.spinner("🤔 Thinking..."):
                answer = chain.invoke(question)
            st.session_state.chat_history.append({"question": question, "answer": answer})
            st.rerun()

st.markdown('</div>', unsafe_allow_html=True)
