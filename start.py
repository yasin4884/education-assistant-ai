"""
start.py
لانچر کامل دستیار دانشجویی:
- تست خیلی سریع مدل‌ها (اختیاری)
- اجرای FastAPI backend (api:app)
- سرو کردن front.html از طریق api.py روی /
"""

import logging
import sys
from pathlib import Path

import uvicorn

from model import AIManager  # اطمینان از سالم بودن مدل‌ها قبل از استارت

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("starter")


def quick_ai_smoke_test() -> bool:
    """
    تست خیلی سبک:
    - AIManager را می‌سازد
    - یک چت خیلی کوتاه می‌زند
    - اگر اوکی بود، cleanup می‌کند
    """
    try:
        logger.info("🧪 تست سبک AIManager...")
        manager = AIManager()

        main = manager.get_main_model()
        resp = main.chat(
            [{"role": "user", "content": "سلام. اگر آماده‌ای فقط بنویس: آماده‌ام."}],
            max_tokens=16,
        )
        logger.info(f"✅ پاسخ تست مدل: {resp}")

        manager.cleanup()
        logger.info("✅ تست AIManager با موفقیت تمام شد")
        return True
    except Exception as e:
        logger.error(f"❌ خطا در تست AIManager: {e}", exc_info=True)
        return False


def main():
    base_dir = Path(__file__).parent.resolve()
    sys.path.insert(0, str(base_dir))
    logger.info("🎓 لانچر دستیار دانشجویی")
    logger.info(f"📁 مسیر پروژه: {base_dir}")

    # ۱) تست اختیاری مدل‌ها (اگر fail شد، جلوی استارت را می‌گیریم)
    if not quick_ai_smoke_test():
        logger.error("❌ به‌دلیل خطای مدل‌ها، سرور استارت داده نشد")
        sys.exit(1)

    # ۲) اجرای سرور FastAPI
    logger.info("🌐 در حال اجرای FastAPI backend روی api:app ...")
    logger.info("➡️  فرانت روی: http://localhost:8000")
    logger.info("➡️  مستندات روی: http://localhost:8000/docs")
    logger.info("➡️  health روی: http://localhost:8000/health")

    try:
        uvicorn.run(
            "api:app",          # خود api.py اپ را تعریف کرده
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info",
        )
    except KeyboardInterrupt:
        logger.info("⚠️ سرور با سیگنال کاربر متوقف شد")
    except Exception as e:
        logger.error(f"❌ خطا در اجرای سرور: {e}", exc_info=True)
    finally:
        logger.info("👋 پایان اجرای start.py")


if __name__ == "__main__":
    main()
