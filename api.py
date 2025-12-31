"""
FastAPI Backend - Student Assistant
متصل به model.AIManager و front.html داشبورد
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict
import uuid

import pdfplumber
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from model import AIManager  # همین model.py که ساختی

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Student Assistant API",
    description="دستیار هوشمند دانشجویی",
    version="1.0.0",
)

# ============== CORS ==============

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # اگر بعداً خواستی محدود کن
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============== Paths & Static ==============

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

STATIC_DIR = BASE_DIR / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}  # برای شروع

# ============== Global State (بدون DB) ==============

class AppState:
    def __init__(self):
        self.ai_manager: Optional[AIManager] = None
        self.start_time = datetime.now()
        self.conversations: Dict[str, List[Dict[str, str]]] = {}
        self.last_file_paths: Dict[str, Path] = {}  # file_id -> path

    def uptime(self) -> float:
        return (datetime.now() - self.start_time).total_seconds()


state = AppState()

# ============== Schemas ==============

class FileUploadResponse(BaseModel):
    file_id: str
    filename: str
    file_type: str
    size: int
    upload_time: str
    status: str = "uploaded"


class SummarizeRequest(BaseModel):
    file_id: str
    max_length: int = Field(default=512, ge=50, le=1024)
    min_length: int = Field(default=50, ge=10, le=500)


class SummarizeResponse(BaseModel):
    file_id: str
    summary: str
    original_length: int
    summary_length: int
    processing_time: float


class QuestionRequest(BaseModel):
    file_id: str
    question: str = Field(..., min_length=3, max_length=500)


class QuestionResponse(BaseModel):
    file_id: str
    question: str
    answer: str
    processing_time: float


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    conversation_id: Optional[str] = None
    context_file_id: Optional[str] = None


class ChatResponse(BaseModel):
    conversation_id: str
    message: str
    response: str
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: Dict[str, bool]
    uptime: float

# ============== Startup / Shutdown ==============

@app.on_event("startup")
async def on_startup():
    logger.info("🚀 API startup: init AIManager")
    try:
        # AIManager تو خودش هر دو مدل را eager بارگذاری می‌کند
        state.ai_manager = AIManager()
        logger.info("✅ AIManager آماده است")
    except Exception as e:
        logger.error(f"❌ خطا در راه‌اندازی AIManager: {e}", exc_info=True)
        state.ai_manager = None


@app.on_event("shutdown")
async def on_shutdown():
    logger.info("🛑 API shutdown")
    try:
        if state.ai_manager:
            state.ai_manager.cleanup()
            logger.info("✅ مدل‌ها آزاد شدند")
    except Exception as e:
        logger.error(f"❌ خطا در shutdown: {e}", exc_info=True)

# ============== Frontend Routes ==============

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    frontend_path = BASE_DIR / "front.html"
    if frontend_path.exists():
        return FileResponse(frontend_path)
    return HTMLResponse(
        """
        <html><body dir="rtl" style="font-family:sans-serif;">
        <h1>🎓 دستیار دانشجویی</h1>
        <p>فایل <code>front.html</code> را کنار <code>api.py</code> قرار بده.</p>
        <a href="/docs">/docs</a> · <a href="/health">/health</a>
        </body></html>
        """,
        status_code=200,
    )


@app.get("/index")
@app.get("/home")
async def redirect_to_frontend():
    frontend_path = BASE_DIR / "front.html"
    return FileResponse(frontend_path) if frontend_path.exists() else await serve_frontend()

# ============== Health ==============

@app.get("/health", response_model=HealthResponse)
async def health_check():
    ai_ready = state.ai_manager is not None
    models_loaded = {
        "main_model": bool(ai_ready and state.ai_manager.is_main_loaded()),
        "summary_model": bool(ai_ready and state.ai_manager.is_summary_loaded()),
    }
    status = "healthy" if ai_ready else "ai_manager_not_initialized"
    return HealthResponse(
        status=status,
        models_loaded=models_loaded,
        uptime=state.uptime(),
    )

# ============== Helpers ==============

def validate_file(file: UploadFile) -> str:
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"فرمت مجاز نیست. فرمت‌های مجاز: {ALLOWED_EXTENSIONS}",
        )
    return ext


def read_file_text(file_path: Path) -> str:
    """
    متن را از روی فایل می‌خواند (txt + pdf)
    """
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="فایل روی دیسک یافت نشد.")

    suffix = file_path.suffix.lower()

    if suffix == ".txt":
        return file_path.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".pdf":
        try:
            texts = []
            with pdfplumber.open(str(file_path)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    texts.append(page_text)
            text = "\n\n".join(texts).strip()
            return text or "متنی از این PDF استخراج نشد."
        except Exception as e:
            logger.error(f"Error reading PDF {file_path}: {e}", exc_info=True)
            return "خطا در خواندن متن PDF."

    # برای docx می‌توان بعداً parser اضافه کرد
    return f"این یک فایل {file_path.suffix} است. فعلاً فقط txt و pdf به‌درستی خوانده می‌شوند."

# ============== Upload ==============

@app.post("/api/upload", response_model=FileUploadResponse)
async def upload_file(file: UploadFile = File(...)):
    if not state.ai_manager:
        raise HTTPException(status_code=503, detail="مدل AI آماده نیست")

    try:
        ext = validate_file(file)
        content = await file.read()
        size = len(content)

        if size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"حجم فایل بیش از حد. حداکثر: {MAX_FILE_SIZE / 1024 / 1024}MB",
            )

        file_id = str(uuid.uuid4())
        file_path = UPLOAD_DIR / f"{file_id}{ext}"
        file_path.write_bytes(content)

        # در حافظه نگه می‌داریم
        state.last_file_paths[file_id] = file_path

        return FileUploadResponse(
            file_id=file_id,
            filename=file.filename,
            file_type=ext,
            size=size,
            upload_time=datetime.now().isoformat(),
            status="uploaded",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in upload_file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"خطا: {str(e)}")

# ============== Summarize ==============

@app.post("/api/summarize", response_model=SummarizeResponse)
async def summarize_document(request: SummarizeRequest):
    if not state.ai_manager:
        raise HTTPException(status_code=503, detail="مدل AI آماده نیست")

    start = datetime.now()

    try:
        file_path = state.last_file_paths.get(request.file_id)
        if not file_path:
            raise HTTPException(status_code=404, detail="فایل پیدا نشد")

        content = read_file_text(file_path)

        summary_model = state.ai_manager.get_summary_model()
        summary = summary_model.summarize(
            content,
            max_length=request.max_length,
            min_length=request.min_length,
        )

        processing_time = (datetime.now() - start).total_seconds()

        return SummarizeResponse(
            file_id=request.file_id,
            summary=summary,
            original_length=len(content),
            summary_length=len(summary),
            processing_time=processing_time,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in summarize_document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"خطا: {str(e)}")

# ============== Question Answering ==============

@app.post("/api/question", response_model=QuestionResponse)
async def answer_question(request: QuestionRequest):
    if not state.ai_manager:
        raise HTTPException(status_code=503, detail="مدل AI آماده نیست")

    start = datetime.now()

    try:
        file_path = state.last_file_paths.get(request.file_id)
        if not file_path:
            raise HTTPException(status_code=404, detail="فایل پیدا نشد")

        context = read_file_text(file_path)

        main_model = state.ai_manager.get_main_model()
        answer = main_model.answer_question(context, request.question)

        processing_time = (datetime.now() - start).total_seconds()

        return QuestionResponse(
            file_id=request.file_id,
            question=request.question,
            answer=answer,
            processing_time=processing_time,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in answer_question: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"خطا: {str(e)}")

# ============== Chat ==============

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not state.ai_manager:
        raise HTTPException(status_code=503, detail="مدل AI آماده نیست")

    import uuid as _uuid

    try:
        conv_id = request.conversation_id or str(_uuid.uuid4())
        history = state.conversations.get(conv_id, [])

        messages = history + [{"role": "user", "content": request.message}]

        main_model = state.ai_manager.get_main_model()
        response = main_model.chat(messages, max_tokens=256)

        messages.append({"role": "assistant", "content": response})
        state.conversations[conv_id] = messages[-10:]

        return ChatResponse(
            conversation_id=conv_id,
            message=request.message,
            response=response,
            timestamp=datetime.now().isoformat(),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"خطا: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
