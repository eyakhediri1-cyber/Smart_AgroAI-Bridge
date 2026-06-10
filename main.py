"""
Olive AI Assistant - Backend
Stack: FastAPI + Whisper + FAISS + Claude API (LLM) + edge-tts
Anti-hallucination: strict RAG with relevance threshold
"""

import os, io, base64, json, asyncio, tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import faiss
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import anthropic

# ── Optional heavy deps (graceful import) ─────────────────────────────────────
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    import edge_tts
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

try:
    import torch
    import torchvision.transforms as T
    from torchvision import models
    from PIL import Image
    CNN_AVAILABLE = True
except ImportError:
    CNN_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# ── Configuration ──────────────────────────────────────────────────────────────
RELEVANCE_THRESHOLD = 0.25   # tuned for cross-lingual AR↔FR/EN
CHUNK_SIZE = 200             # smaller = tighter semantic focus
TOP_K = 7                    # more candidates for cross-lingual gap
EMBED_MODEL = "paraphrase-multilingual-mpnet-base-v2"  # stronger than MiniLM
TTS_VOICE = "ar-TN-HediNeural"  # Tunisian Arabic voice (edge-tts)
CNN_CLASSES = ["healthy", "peacock_eye", "anthracnose", "verticillium", "sooty_mold", "leaf_spot", "not_olive"]

CORPUS_DIR = Path("corpus")
INDEX_PATH = Path("faiss_index.bin")
CHUNKS_PATH = Path("chunks.json")
CNN_MODEL_PATH = Path("olive_cnn.pth")

CORPUS_DIR.mkdir(exist_ok=True)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Olive AI Assistant", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Globals (loaded once at startup) ──────────────────────────────────────────
whisper_model = None
embedder = None
faiss_index = None
chunks_data = []   # list of {"text": ..., "source": ..., "page": ...}
cnn_model = None
anthropic_client = None

# ══════════════════════════════════════════════════════════════════════════════
# STARTUP — load all models
# ══════════════════════════════════════════════════════════════════════════════
@app.on_event("startup")
async def startup():
    global whisper_model, embedder, faiss_index, chunks_data, cnn_model, anthropic_client

    print("🚀 Loading models...")

    # Anthropic client
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if api_key:
        try:
            anthropic_client = anthropic.Anthropic(api_key=api_key)
            print("✅ Anthropic client initialized")
        except Exception as e:
            print(f"⚠️  Anthropic client error: {e}")

    # Whisper ASR — SKIP on startup to avoid hangs (load on first use instead)
    # if WHISPER_AVAILABLE:
    #     try:
    #         whisper_model = whisper.load_model("base")
    #         print("✅ Whisper loaded")
    #     except Exception as e:
    #         print(f"⚠️  Whisper load error: {e}")
    
    print("⏭️  Skipping Whisper on startup (will load on first use)")

    # Sentence embedder
    if ST_AVAILABLE:
        try:
            embedder = SentenceTransformer(EMBED_MODEL)
            print("✅ Embedder loaded")
        except Exception as e:
            print(f"⚠️  Embedder load error: {e}")

    # FAISS index
    if CHUNKS_PATH.exists() and INDEX_PATH.exists():
        try:
            with open(CHUNKS_PATH) as f:
                temp_chunks = json.load(f)
            temp_index = faiss.read_index(str(INDEX_PATH))
            
            # Check dimension mismatch
            if embedder is not None:
                expected_dim = embedder.get_sentence_embedding_dimension()
                if temp_index.d != expected_dim:
                    print(f"⚠️ FAISS dimension mismatch: index={temp_index.d}, model={expected_dim}. Clearing index.")
                    INDEX_PATH.unlink(missing_ok=True)
                    CHUNKS_PATH.unlink(missing_ok=True)
                else:
                    faiss_index = temp_index
                    chunks_data = temp_chunks
                    print(f"✅ FAISS index loaded — {len(chunks_data)} chunks (dim={temp_index.d})")
            else:
                faiss_index = temp_index
                chunks_data = temp_chunks
                print(f"✅ FAISS index loaded (embedder unavailable)")
        except Exception as e:
            print(f"⚠️ FAISS load error: {e}. Clearing corrupted index.")
            INDEX_PATH.unlink(missing_ok=True)
            CHUNKS_PATH.unlink(missing_ok=True)
    else:
        print("⚠️ No FAISS index found. Run /admin/index-corpus first.")

    # CNN for olive disease
    if CNN_AVAILABLE and CNN_MODEL_PATH.exists():
        try:
            cnn_model = _load_cnn(CNN_MODEL_PATH)
            print("✅ CNN loaded")
        except Exception as e:
            print(f"⚠️  CNN load error: {e}")
    else:
        print("⚠️  No CNN model found — using mock classifier")

    print("✅ Initialization complete!")

# ══════════════════════════════════════════════════════════════════════════════
# CORPUS INDEXING
# ══════════════════════════════════════════════════════════════════════════════
def _chunk_text(text: str, source: str, page: int, size: int = CHUNK_SIZE):
    """Chunk with 20% overlap so context isn't cut at boundaries."""
    words = text.split()
    overlap = size // 5
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+size])
        if len(chunk.strip()) > 40:
            chunks.append({"text": chunk, "source": source, "page": page})
        i += size - overlap  # slide with overlap
    return chunks

@app.post("/admin/index-corpus")
async def index_corpus():
    """Index all PDFs in ./corpus/ directory."""
    global faiss_index, chunks_data

    if not PDF_AVAILABLE:
        raise HTTPException(500, "PyMuPDF not installed. pip install pymupdf")
    if not ST_AVAILABLE:
        raise HTTPException(500, "sentence-transformers not installed.")

    all_chunks = []
    pdf_files = list(CORPUS_DIR.glob("**/*.pdf"))
    txt_files = list(CORPUS_DIR.glob("**/*.txt"))

    for pdf_path in pdf_files:
        try:
            doc = fitz.open(str(pdf_path))
            for page_num, page in enumerate(doc):
                text = page.get_text()
                all_chunks.extend(_chunk_text(text, pdf_path.name, page_num + 1))
            doc.close()
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")

    for txt_path in txt_files:
        text = txt_path.read_text(errors="ignore")
        all_chunks.extend(_chunk_text(text, txt_path.name, 0))

    if not all_chunks:
        return {"status": "error", "message": "No documents found in ./corpus/"}

    # Embed all chunks
    texts = [c["text"] for c in all_chunks]
    embeddings = embedder.encode(texts, batch_size=32, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product = cosine after normalization
    index.add(embeddings)

    faiss_index = index
    chunks_data = all_chunks

    faiss.write_index(index, str(INDEX_PATH))
    with open(CHUNKS_PATH, "w") as f:
        json.dump(all_chunks, f, ensure_ascii=False)

    return {"status": "ok", "chunks_indexed": len(all_chunks), "documents": len(pdf_files + txt_files)}

# ══════════════════════════════════════════════════════════════════════════════
# RAG RETRIEVAL  (anti-hallucination core)
# ══════════════════════════════════════════════════════════════════════════════
def _translate_query_for_retrieval(query: str) -> str:
    """
    Translate the Tunisian Arabic query into English + French keywords
    so it matches the FAO/EPPO corpus language.
    Uses a lightweight LLM call — result is ONLY used for retrieval, not shown to user.
    Falls back to original query if LLM unavailable.
    """
    if anthropic_client is None:
        return query
    try:
        msg = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=120,
            system=(
                "You are a translation assistant. "
                "Translate the following Tunisian Arabic agricultural question into "
                "a short list of English AND French keywords (no full sentences, just key terms). "
                "Output ONLY the keywords, comma-separated, nothing else."
            ),
            messages=[{"role": "user", "content": query}]
        )
        translated = msg.content[0].text.strip()
        # Combine original + translation for hybrid retrieval
        return f"{query} {translated}"
    except Exception as e:
        print(f"⚠️ Translation Error: {e}")
        return query


def _retrieve(query: str, k: int = TOP_K):
    """
    Retrieve top-k chunks.
    Translates query to EN/FR first to bridge the cross-lingual gap
    between Tunisian Arabic questions and EN/FR corpus.
    """
    if faiss_index is None or embedder is None:
        return []

    # Cross-lingual bridge: embed translated query
    search_query = _translate_query_for_retrieval(query)

    q_emb = embedder.encode([search_query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, indices = faiss_index.search(q_emb, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            results.append((float(score), chunks_data[idx]))
    return results

def _is_in_corpus(results) -> bool:
    if not results:
        return False
    return results[0][0] >= RELEVANCE_THRESHOLD

# ══════════════════════════════════════════════════════════════════════════════
# CNN — olive disease classifier
# ══════════════════════════════════════════════════════════════════════════════
def _load_cnn(model_path: Path):
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CNN_CLASSES))
    model.load_state_dict(torch.load(str(model_path), map_location="cpu"))
    model.eval()
    return model

def _classify_leaf(image_bytes: bytes) -> dict:
    if cnn_model is None:
        # ── Simple Visual Heuristic for Demo ──
        # Check if the image has enough "green" to be a leaf
        try:
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            # Downsample for speed
            img.thumbnail((64, 64))
            pixels = list(img.getdata())
            
            green_pixels = 0
            for r, g, b in pixels:
                # Basic green check: G is dominant and reasonably bright
                if g > r * 1.1 and g > b * 1.1 and g > 40:
                    green_pixels += 1
            
            green_ratio = green_pixels / len(pixels)
            if green_ratio < 0.05: # Less than 5% green
                return {"class": "not_olive", "confidence": 0.99, "mock": True}
            
            # Heuristic for healthy vs anomalous
            if green_ratio > 0.3:
                g_values = [p[1] for p in pixels if p[1] > p[0] and p[1] > p[2]]
                import statistics
                if len(g_values) > 10:
                    g_std = statistics.stdev(g_values)
                    # Low variance + bright green = Healthy
                    if g_std < 25:
                        return {"class": "healthy", "confidence": 0.95, "mock": True}
                    # High variance = Anomalous (Sick)
                    if g_std > 40:
                        import hashlib
                        h = int(hashlib.md5(image_bytes).hexdigest(), 16)
                        choices = ["peacock_eye", "anthracnose", "verticillium"]
                        return {"class": choices[h % len(choices)], "confidence": 0.88, "mock": True}

        except Exception:
            pass

        # Smart Mock Fallback
        import hashlib
        h = int(hashlib.md5(image_bytes).hexdigest(), 16)
        # Even balance for fallback
        choices = ["healthy", "peacock_eye", "anthracnose", "verticillium", "not_olive"]
        mock_class = choices[h % len(choices)]
        conf = 0.80 + (h % 150) / 1000.0
        return {"class": mock_class, "confidence": conf, "mock": True}

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        logits = cnn_model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        top_idx = int(probs.argmax())
    return {
        "class": CNN_CLASSES[top_idx],
        "confidence": float(probs[top_idx]),
        "all_scores": {c: float(p) for c, p in zip(CNN_CLASSES, probs)},
        "mock": False
    }

# ══════════════════════════════════════════════════════════════════════════════
# LLM — strict RAG prompt (anti-hallucination)
# ══════════════════════════════════════════════════════════════════════════════
STRICT_SYSTEM_PROMPT = """أنت مساعد زراعي متخصص في زراعة الزيتون في تونس. تتحدث بالدارجة التونسية.

قواعد صارمة جداً:
1. إذا كانت نتيجة تحليل الصورة "ورقة سليمة"، طمئن الفلاح وقل له أن الورقة سليمة وشجعه على مواصلة العناية.
2. إذا كانت هناك إصابة بمرض، حدد اسم المرض بوضوح وقدم نصائح عامة بناءً على المقاطع المرفقة.
3. أجب فقط بناءً على المقاطع المُقدَّمة أدناه. لا تستخدم معرفتك الخاصة أبداً.
4. إذا لم تجد الإجابة في المقاطع، قل بدقة: "ما عنديش المعلومة في قاعدة بياناتي، اتصل بمستشار زراعي."
5. دائماً اذكر المصدر (اسم الوثيقة والصفحة) في نهاية إجابتك.
6. لا تعطي جرعات مبيدات دقيقة — أحل دائماً على الفيشة التقنية أو المستشار الزراعي.
7. تحدث بالدارجة التونسية دائماً.

You are a strict RAG assistant. NEVER hallucinate. ONLY use the provided excerpts."""

def _build_rag_prompt(question: str, retrieved: list, disease_info: Optional[dict] = None) -> str:
    context_parts = []
    for i, (score, chunk) in enumerate(retrieved):
        context_parts.append(
            f"[مقطع {i+1} — المصدر: {chunk['source']}, الصفحة: {chunk['page']}, الصلة: {score:.2f}]\n{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    disease_section = ""
    if disease_info:
        ar_names = {
            "peacock_eye": "عين الطاووس (Cycloconium oleaginum)",
            "anthracnose": "الأنثراكنوز (Colletotrichum spp.)",
            "verticillium": "الذبول الفيرتيسيلي (Verticillium dahliae)",
            "sooty_mold": "السواد العفني",
            "leaf_spot": "تبقع الأوراق",
            "healthy": "ورقة سليمة",
            "not_olive": "ليست ورقة زيتون"
        }
        ar_name = ar_names.get(disease_info["class"], disease_info["class"])
        disease_section = f"\n\n[نتيجة تحليل الصورة: {ar_name} — نسبة الثقة: {disease_info['confidence']*100:.0f}%]\n"

    return f"""{disease_section}

المقاطع المرجعية:
{context}

السؤال: {question}

الإجابة (بالدارجة التونسية، مع ذكر المصدر):"""

DEMO_RESPONSES = {
    "peacock_eye": "تحليل الصورة يوري اللي فمة احتمال كبير متاع إصابة بمرض 'عين الطاووس' (Spilocaea oleagina). هذا مرض فطري منتشر برشة في تونس. ننصحك تداوي بمركبات النحاس (bouillie bordelaise) وتنحي الأغصان اللي فيها برشة عدوى باش تزيد التهوية. شوف الوثائق المرجعية لوطة فيها تفاصيل أكثر.",
    "anthracnose": "الصورة تشير لإصابة بـ 'الأنثراكنوز'. هذا الفطر يضرب الثمار والأوراق خاصة في وقت الرطوبة. لازمك تنحي الثمار الخامجة وتداوي بمبيد فطري مصادق عليه. ارجع للمصادر المرجعية لمزيد من المعلومات التقنية.",
    "verticillium": "هذا يبان 'الذبول الفيرتيسيلي'. هو مرض صعيب شوية خاطر الفطر يعيش في التربة. أحسن حل هو الوقاية، وتقليع الأشجار الميتة، وما تفرطش في الري. شوف المصادر المرجعية لوطة فيها نصايح دقيقة.",
    "sooty_mold": "هذا 'السواد العفني'. هو عادة يجي جرة الحشرات القشرية اللي تسيب مادة حلوة. لازمك تداوي الحشرات قبل كل شيء باش يتنحى السواد. ارجع للفيشة التقنية لوطة.",
    "healthy": "الورقة تبان سليمة (Saine) والحمد لله. واصل العناية بالشجرة وتبع برنامج التسميد والري العادي. المصادر المرجعية فيها نصايح عامة لزيادة الإنتاج.",
    "not_olive": "عذراً، الصورة هذي ما تبانش ورقة زيتون! (Ce n'est pas un olivier) 🛑 المساعد هذا مخصص فقط لأشجار الزيتون في تونس. من فضلك صور ورقة زيتون واضحة باش نجم نعاونك."
}

def _call_llm(question: str, retrieved: list, disease_info: Optional[dict] = None) -> str:
    # ── Demo Mode (No API Key) ────────────────────────────────────────────────
    if anthropic_client is None:
        q = question.lower()
        
        # Out-of-scope check (Tomato example)
        if "طماطم" in q or "tomat" in q:
            return "آسف، أنا مساعد مختص فقط في أشجار الزيتون. ما عنديش معلومات على زراعة الطماطم أو الشركات اللي تشريها. ننصحك تتصل بالمجمع المهني للمصبرات الغذائية."
        
        # Pruning check (وقتاش نحشش)
        if "نحشش" in q or "taille" in q or "وقتاش" in q and "زيتون" in q:
            return "الحشّان (التلييم) متاع الزيتون في تونس أحسن وقت ليه هو بعد الجني مباشرة، يعني بين ديسمبر وفيفري. المهم يكون قبل ما تبدا الشجرة تخرج في النوار. ارجع للوثيقة CIHEAM_olive_tunisia.pdf الصفحة 12 فيها تفاصيل أكثر."

        # Treatment check (كيفاش نعالج)
        if "نعالج" in q or "traitement" in q:
            if "طاووس" in q or (disease_info and disease_info["class"] == "peacock_eye"):
                return "علاج عين الطاووس يعتمد أساساً على مركبات النحاس (النحاس المعدني) في الخريف والربيع. لازمك زادة تليّم الشجرة باش تدخلها التهوية والشمس. ثبت في فيشة EPPO المرفقة لوطة."

        # Default disease-based responses
        if disease_info:
            cls = disease_info.get("class", "healthy")
            return DEMO_RESPONSES.get(cls, "أنا مساعدك الزراعي، تفضل اسألني أي سؤال على الزيتون.")
        
        return "أنا مساعدك الزراعي المختص في الزيتون. تفضل اسألني على الأمراض، الحشّان، أو الجني."

    # ── Real LLM Mode ─────────────────────────────────────────────────────────
    user_content = _build_rag_prompt(question, retrieved, disease_info)
    try:
        message = anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1024,
            system=STRICT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}]
        )
        return message.content[0].text
    except Exception as e:
        print(f"⚠️ LLM Error: {e}")
        return "آسف، فمة مشكلة تقنية في الاتصال بالذكاء الاصطناعي. المصادر المرجعية موجودة لوطة."

# ══════════════════════════════════════════════════════════════════════════════
# TTS — text to speech in Tunisian Arabic
# ══════════════════════════════════════════════════════════════════════════════
async def _text_to_speech(text: str) -> Optional[bytes]:
    if not TTS_AVAILABLE:
        return None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            out_path = f.name
        communicate = edge_tts.Communicate(text, TTS_VOICE)
        await communicate.save(out_path)
        audio_bytes = Path(out_path).read_bytes()
        Path(out_path).unlink(missing_ok=True)
        return audio_bytes
    except Exception as e:
        print(f"⚠️ TTS Error: {e}")
        return None

# ══════════════════════════════════════════════════════════════════════════════
# ASR — Whisper transcription
# ══════════════════════════════════════════════════════════════════════════════
def _transcribe(audio_bytes: bytes) -> str:
    global whisper_model
    if not WHISPER_AVAILABLE:
        return ""
    
    # Lazy load Whisper on first use
    if whisper_model is None:
        print("Loading Whisper model on first use...")
        try:
            whisper_model = whisper.load_model("base")
            print("✅ Whisper loaded")
        except Exception as e:
            print(f"❌ Failed to load Whisper: {e}")
            return ""
    
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
        f.write(audio_bytes)
        audio_path = f.name
    try:
        result = whisper_model.transcribe(audio_path, language="ar")
        return result["text"].strip()
    finally:
        Path(audio_path).unlink(missing_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENDPOINT — /query
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/query")
async def query(
    audio: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    text_question: Optional[str] = Form(None),
):
    """
    Main multimodal endpoint.
    Accepts: audio (webm/wav) + image (jpg/png) + optional text override.
    Returns: JSON with answer text + base64 audio + disease info.
    """
    # 1. ASR
    question = text_question or ""
    if audio and not question:
        audio_bytes = await audio.read()
        question = _transcribe(audio_bytes)

    # Allow image-only queries
    if not question and not image:
        raise HTTPException(400, "No question or image provided")
    
    # If no question, use a default one based on image
    if not question and image:
        question = "ما هو هذا المرض وكيفية علاجه؟" 

    # 2. CNN — classify leaf image
    disease_info = None
    if image:
        image_bytes = await image.read()
        disease_info = _classify_leaf(image_bytes)
        
        # If it's not an olive leaf, stop here and inform the user
        if disease_info["class"] == "not_olive":
            answer = DEMO_RESPONSES["not_olive"]
            audio_b64 = None
            if TTS_AVAILABLE:
                audio_bytes_out = await _text_to_speech(answer)
                if audio_bytes_out:
                    audio_b64 = base64.b64encode(audio_bytes_out).decode()
            return JSONResponse({
                "transcription": question,
                "answer": answer,
                "in_corpus": False,
                "disease": disease_info,
                "sources": [],
                "audio_b64": audio_b64
            })

    # 3. Enrich query with disease name for better retrieval
    search_query = question
    if disease_info and disease_info["class"] != "healthy":
        en_names = {
            "peacock_eye": "peacock eye cycloconium olive leaf",
            "anthracnose": "anthracnose colletotrichum olive",
            "verticillium": "verticillium wilt olive",
            "sooty_mold": "sooty mold olive",
            "leaf_spot": "olive leaf spot cercospora"
        }
        search_query += " " + en_names.get(disease_info["class"], "")

    # 4. RAG retrieval
    retrieved = _retrieve(search_query)

    # 5. Anti-hallucination gate
    in_corpus = _is_in_corpus(retrieved)

    if not in_corpus:
        refusal_ar = (
            "آسف، ما عنديش معلومات كافية في قاعدة بياناتي باش نجاوبك على هذا السؤال. "
            "ننصحك تتصل بمستشار زراعي متخصص أو بالإرشاد الفلاحي."
        )
        audio_b64 = None
        if TTS_AVAILABLE:
            audio_bytes_out = await _text_to_speech(refusal_ar)
            if audio_bytes_out:
                audio_b64 = base64.b64encode(audio_bytes_out).decode()
        return JSONResponse({
            "transcription": question,
            "answer": refusal_ar,
            "in_corpus": False,
            "top_score": retrieved[0][0] if retrieved else 0,
            "disease": disease_info,
            "sources": [],
            "audio_b64": audio_b64
        })

    # 6. LLM — generate answer from retrieved context only
    answer = _call_llm(question, retrieved, disease_info)

    # 7. TTS
    audio_b64 = None
    if TTS_AVAILABLE:
        audio_bytes_out = await _text_to_speech(answer)
        if audio_bytes_out:
            audio_b64 = base64.b64encode(audio_bytes_out).decode()

    # 8. Build sources list
    sources = list({c["source"] for _, c in retrieved[:3]})

    return JSONResponse({
        "transcription": question,
        "answer": answer,
        "in_corpus": True,
        "top_score": retrieved[0][0],
        "disease": disease_info,
        "sources": sources,
        "audio_b64": audio_b64
    })

# ══════════════════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/debug/search")
async def debug_search(q: str, k: int = 10):
    """
    Debug endpoint — shows retrieval scores for any query.
    Use this to tune RELEVANCE_THRESHOLD.
    Example: GET /debug/search?q=عين الطاووس
    """
    translated = _translate_query_for_retrieval(q)
    results = _retrieve(q, k=k)
    return {
        "query_original": q,
        "query_translated": translated,
        "threshold": 0.25,
        "results": [
            {
                "score": round(score, 4),
                "above_threshold": score >= 0.25,
                "source": chunk["source"],
                "page": chunk["page"],
                "text_preview": chunk["text"][:200]
            }
            for score, chunk in results
        ]
    }



@app.get("/health")
def health():
    return {
        "status": "ok",
        "whisper": WHISPER_AVAILABLE,
        "embedder": ST_AVAILABLE and embedder is not None,
        "faiss_chunks": len(chunks_data),
        "cnn": CNN_AVAILABLE and cnn_model is not None,
        "tts": TTS_AVAILABLE,
        "llm": anthropic_client is not None
    }

# Serve frontend
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
