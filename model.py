"""
AI Models Module - OpenRouter-based
MainModel و SummaryModel و AIManager بدون مدل لوکال، فقط با API

- MainModel: چت و QA با OpenRouter
- SummaryModel: خلاصه‌سازی با OpenRouter
"""

import logging
from typing import List, Dict, Optional
import json
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== تنظیمات کلی ====================

API_KEY = ""
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

# می‌توانی برای چت/QA و خلاصه‌سازی مدل‌های مختلف بگذاری
MAIN_MODEL_NAME = "google/gemma-3n-e4b-it:free"
SUM_MODEL_NAME = "google/gemma-3n-e4b-it:free"


# ==================== Helper برای کال OpenRouter ====================

def call_openrouter(model: str, messages: List[Dict[str, str]]) -> str:
    """
    یک رپر ساده روی OpenRouter
    """
    body = {
        "model": model,
        "messages": messages,
        "reasoning": {"enabled": True},
    }

    logger.info(f"🔗 Calling OpenRouter model={model} with {len(messages)} messages")

    resp = requests.post(
        OPENROUTER_ENDPOINT,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "Student Assistant Backend",
        },
        data=json.dumps(body).encode("utf-8"),
        timeout=60,
    )

    logger.info(f"OpenRouter status: {resp.status_code}")

    if resp.status_code != 200:
        logger.error(f"OpenRouter error: {resp.text}")
        raise RuntimeError(f"OpenRouter error {resp.status_code}")

    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    logger.debug(f"OpenRouter raw content: {content}")
    return content.strip()


# ==================== مدل اصلی: Q&A / Chat ====================

class MainModel:
    """
    مدل اصلی برای چت و QA (روی OpenRouter)
    اسم کلاس حفظ شده؛ زیرش فقط API است.
    """

    def __init__(self, device: Optional[object] = None):
        # device دیگر معنایی ندارد، فقط برای سازگاری نگه می‌داریم
        self.device = device
        logger.info("MainModel (OpenRouter) ایجاد شد")

    def load(self):
        """
        برای سازگاری با نسخه‌های قبلی.
        چون مدل لوکال نداریم، فقط لاگ می‌زنیم.
        """
        logger.info(f"⚠️ MainModel: استفاده از OpenRouter ({MAIN_MODEL_NAME}) به‌جای مدل لوکال")

    def config_generation(
        self,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ):
        """
        این تنظیمات روی خود API قابل ست‌کردن است، ولی فعلاً فقط لاگ می‌گیریم.
        """
        logger.info(
            f"⚙️ تنظیمات تولید (غیر فعال در API مستقیم): "
            f"max_new_tokens={max_new_tokens}, "
            f"temperature={temperature}, top_p={top_p}, do_sample={do_sample}"
        )

    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 512) -> str:
        """
        چت آزاد با مدل؛ messages همان فرمت قبلی است.
        """
        try:
            # می‌توانی max_tokens را در body بگذاری؛ اینجا برای سادگی فقط لاگ می‌کنیم
            logger.info(f"💬 MainModel.chat with {len(messages)} messages")
            return call_openrouter(MAIN_MODEL_NAME, messages)
        except Exception as e:
            logger.exception(f"❌ خطا در chat: {e}")
            return f"خطا در چت: {e}"

    def answer_question(self, context: str, question: str) -> str:
        """
        QA بر اساس context؛ همان امضای قبلی، زیرش API.
        """
        prompt = (
            "تو یک دستیار هوشمند فارسی هستی. فقط بر اساس متن زیر به سوال پاسخ بده "
            "و اگر جواب در متن نبود، بگو 'در متن داده شده نیست'.\n\n"
            f"متن:\n{context}\n\nسوال: {question}"
        )
        messages = [
            {"role": "user", "content": prompt}
        ]
        try:
            logger.info("❓ MainModel.answer_question called")
            return call_openrouter(MAIN_MODEL_NAME, messages)
        except Exception as e:
            logger.exception(f"❌ خطا در answer_question: {e}")
            return f"خطا در QA: {e}"

    def unload(self):
        """
        مدل لوکال نداریم؛ فقط لاگ.
        """
        logger.info("🗑️ MainModel (OpenRouter) چیزی برای unload ندارد")


# ==================== مدل خلاصه‌سازی ====================

class SummaryModel:
    """
    مدل خلاصه‌سازی (روی OpenRouter)
    اسم و API شبیه نسخهٔ قبلی، زیرش API است.
    """

    def __init__(self, device: Optional[object] = None):
        self.device = device
        self.gen_config = {
            "max_length": 512,
            "min_length": 50,
        }
        logger.info("SummaryModel (OpenRouter) ایجاد شد")

    def load(self):
        logger.info(f"⚠️ SummaryModel: استفاده از OpenRouter ({SUM_MODEL_NAME}) برای خلاصه‌سازی")

    def config_generation(
        self,
        max_length: int = 512,
        min_length: int = 50,
        num_beams: int = 4,
        length_penalty: float = 2.0,
    ):
        self.gen_config = {
            "max_length": max_length,
            "min_length": min_length,
        }
        logger.info(f"⚙️ تنظیمات خلاصه‌سازی (OpenRouter): {self.gen_config}")

    def summarize(self, text: str, max_length: int = 512, min_length: int = 50) -> str:
        """
        خلاصه‌سازی متن از طریق OpenRouter (Gemma)
        """
        try:
            logger.info("📝 SummaryModel.summarize called")
            prompt = (
                "تو یک خلاصه‌ساز حرفه‌ای هستی. متن زیر را به بهترین شکل ممکن خلاصه کن "
                "و زبان پاسخ را همان زبان متن ورودی قرار بده.\n\n"
                f"متن:\n{text}"
            )
            messages = [
                {"role": "user", "content": prompt}
            ]
            return call_openrouter(SUM_MODEL_NAME, messages)
        except Exception as e:
            logger.exception(f"❌ خطا در خلاصه‌سازی: {e}")
            return f"خطا در خلاصه‌سازی: {e}"

    def unload(self):
        logger.info("🗑️ SummaryModel (OpenRouter) چیزی برای unload ندارد")


# ==================== AIManager ====================

class AIManager:
    """
    مدیر مدل‌ها (هر دو روی OpenRouter)
    اسم و API مثل قبل است.
    """

    def __init__(self, device: Optional[object] = None):
        self.device = device
        self.main_model = MainModel(self.device)
        self.summary_model = SummaryModel(self.device)
        logger.info("✅ AIManager (OpenRouter) ایجاد شد")

        # می‌توانی اینجا load را صدا بزنی برای لاگ
        self.main_model.load()
        self.summary_model.load()

    def get_main_model(self) -> MainModel:
        logger.info("AIManager.get_main_model called")
        return self.main_model

    def get_summary_model(self) -> SummaryModel:
        logger.info("AIManager.get_summary_model called")
        return self.summary_model

    def is_main_loaded(self) -> bool:
        # همیشه True، چون مدل لوکال نداریم؛ برای سازگاری:
        return True

    def is_summary_loaded(self) -> bool:
        return True

    def cleanup(self):
        logger.info("🧹 AIManager.cleanup called")
        self.main_model.unload()
        self.summary_model.unload()
        logger.info("🧹 AIManager.cleanup finished")


# ==================== تست مستقل ====================

if __name__ == "__main__":
    """
    تست سریع:
      python model.py
    """

    manager = AIManager()

    # تست چت
    print("\n=== تست چت ===")
    mm = manager.get_main_model()
    resp = mm.chat(
        [{"role": "user", "content": "سلام! یک جمله انگیزشی کوتاه به فارسی بگو."}],
        max_tokens=64,
    )
    print("پاسخ چت:", resp)

    # تست خلاصه‌سازی
    print("\n=== تست خلاصه‌سازی ===")
    sm = manager.get_summary_model()
    long_text = (
        "یادگیری ماشین شاخه‌ای از هوش مصنوعی است که در آن الگوریتم‌ها "
        "با استفاده از داده‌ها آموزش می‌بینند تا الگوها را کشف کنند و "
        "بدون برنامه‌نویسی صریح، پیش‌بینی یا تصمیم‌گیری انجام دهند."
    )
    s = sm.summarize(long_text, max_length=128, min_length=32)
    print("خلاصه:", s)

    manager.cleanup()

