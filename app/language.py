"""Language detection and conservative translation helpers."""

import logging
import re
from typing import Any

from pydantic import BaseModel, Field

from app.llm import LLMError, complete_json

logger = logging.getLogger(__name__)


class TranslationResult(BaseModel):
    """Validated translation response."""

    text: str


class LanguageDecision(BaseModel):
    """Supported output of the language detector."""

    language: str = Field(pattern="^(en|te|hi)$")


def detect_language(message: str) -> str:
    """Detect English, Telugu, Hindi, or supported Latin-script code-mixing."""
    if re.search(r"[\u0c00-\u0c7f]", message):
        return "te"
    if re.search(r"[\u0900-\u097f]", message):
        return "hi"
    lower = message.lower()
    telugu_markers = (
        "chati",
        "gunde",
        "noppi",
        "oopiri",
        "kadupu",
        "teesukunna",
        "nenu",
        "cheyocha",
        "chesukovacha",
        "ela",
        "emiti",
    )
    hindi_markers = (
        "seene",
        "saans",
        "khoon",
        "behosh",
        "pet dard",
        "goli",
        "sakta hoon",
        "karun",
        "kitne din",
        "meri",
        "kya main",
        "ke baad",
    )
    if any(term in lower for term in telugu_markers):
        return "te"
    if any(term in lower for term in hindi_markers):
        return "hi"
    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0
        detected = detect(message)
        return detected if detected in {"en", "te", "hi"} else "en"
    except Exception:
        logger.info("Language detector unavailable; treating Latin text as English.")
        return "en"


async def detect_language_async(message: str) -> str:
    """Use the LLM adapter to resolve Latin-script Telugu/Hindi code-mixed text."""
    heuristic = detect_language(message)
    is_latin = not re.search(r"[\u0900-\u097f\u0c00-\u0c7f]", message)
    if heuristic == "en" or not is_latin:
        return heuristic

    def local_decision(payload: dict[str, Any]) -> dict[str, str]:
        return {"language": detect_language(str(payload["message"]))}

    try:
        result = await complete_json(
            "Detect whether this message is English, Telugu, or Hindi. "
            "For code-mixed text choose the patient's base language.",
            {"message": message},
            LanguageDecision,
            temperature=0.0,
            local_handler=local_decision,
        )
        return result.language
    except LLMError:
        logger.warning(
            "Code-mixed language detection failed; using conservative script/phrase heuristic."
        )
        return heuristic


async def translate_to_english(message: str, language: str) -> str:
    """Translate a supported-language patient message to English, or fail safely."""
    if language == "en":
        return message

    def unavailable(_: dict[str, Any]) -> dict[str, str]:
        raise ValueError(f"Local translation for {language} is not available.")

    result = await complete_json(
        "Translate the patient's message faithfully to English without adding meaning.",
        {"language": language, "text": message},
        TranslationResult,
        temperature=0.1,
        local_handler=unavailable,
    )
    return result.text.strip()


async def translate_from_english(message: str, language: str) -> str:
    """Translate a grounded English answer while preserving citation markers."""
    if language == "en":
        return message
    result = await complete_json(
        "Translate faithfully. Preserve citation markers like [1] exactly.",
        {"language": language, "text": message},
        TranslationResult,
        temperature=0.1,
        local_handler=lambda _: (_ for _ in ()).throw(
            ValueError(f"Local translation for {language} is not available.")
        ),
    )
    return result.text.strip()


def urgent_message(language: str, emergency_contact_text: str | None = None) -> str:
    """Return a fixed, reviewed-for-demo emergency handoff message."""
    messages = {
        "en": (
            "This may need urgent attention. Contact your locally verified emergency services "
            "or clinic immediately. The nurse team has been alerted. CareBridge cannot contact "
            "emergency services for you."
        ),
        "te": (
            "ఇది అత్యవసర శ్రద్ధ అవసరమైన విషయం కావచ్చు. వెంటనే మీ స్థానిక అత్యవసర సేవలను లేదా "
            "క్లినిక్‌ను సంప్రదించండి. నర్సు బృందానికి సమాచారం ఇచ్చాము. CareBridge మీ తరపున "
            "అత్యవసర సేవలను సంప్రదించదు."
        ),
        "hi": (
            "इस पर तुरंत ध्यान देने की जरूरत हो सकती है। अपने स्थानीय सत्यापित आपातकालीन सेवा "
            "या क्लिनिक से तुरंत संपर्क करें। नर्स टीम को सूचना दे दी गई है। CareBridge आपकी "
            "ओर से आपातकालीन सेवा से संपर्क नहीं कर सकता।"
        ),
    }
    message = messages.get(language, messages["en"])
    if (
        emergency_contact_text
        and emergency_contact_text != "your locally verified emergency services"
    ):
        message += f" Emergency contact: {emergency_contact_text}."
    return message
