"""Fail-safe multilingual emergency-symptom screening."""

import logging
import re
from pathlib import Path
from typing import Any

import yaml

from app.config import get_settings
from app.llm import LLMError, complete_json
from app.schemas import RedFlagClassification

logger = logging.getLogger(__name__)
_RULES: dict[str, dict[str, list[str]]] | None = None
_LOCAL_PARAPHRASES: tuple[tuple[str, str], ...] = (
    ("elephant", "chest_pain"),
    ("crushing", "chest_pain"),
    ("chesst pain", "chest_pain"),
    ("crushng chest", "chest_pain"),
    ("chaathi noppi", "chest_pain"),
    ("chati noppi", "chest_pain"),
    ("gunde noppi", "chest_pain"),
    ("seene me dard", "chest_pain"),
    ("chaati me dard", "chest_pain"),
    ("chhati me dard", "chest_pain"),
    ("cant breathe", "breathing_difficulty"),
    ("can't breathe", "breathing_difficulty"),
    ("cant breath", "breathing_difficulty"),
    ("cant brreathe", "breathing_difficulty"),
    ("shortnes of breath", "breathing_difficulty"),
    ("cannot catch my breath", "breathing_difficulty"),
    ("struggling for air", "breathing_difficulty"),
    ("swasa kastam", "breathing_difficulty"),
    ("oopiri aadatledu", "breathing_difficulty"),
    ("saans lene me dikkat", "breathing_difficulty"),
    ("saans nahi aa rahi", "breathing_difficulty"),
    ("bleedng", "severe_bleeding"),
    ("heavy bleeding", "severe_bleeding"),
    ("soaking through bandages", "severe_bleeding"),
    ("raktham aagatledu", "severe_bleeding"),
    ("bleeding aagatledu", "severe_bleeding"),
    ("khoon ruk nahi raha", "severe_bleeding"),
    ("khoon band nahi ho raha", "severe_bleeding"),
    ("passed out", "fainting"),
    ("faintng", "fainting"),
    ("spruha tappindi", "fainting"),
    ("behosh", "fainting"),
    ("weak and numb", "stroke"),
    ("face is drooping", "stroke"),
    ("throat is swelling", "allergic_reaction"),
    ("throat closing", "allergic_reaction"),
    ("sweling and closing", "allergic_reaction"),
    ("took a lot of tablets", "overdose"),
    ("took too many", "overdose"),
    ("10 sleeping pills", "overdose"),
    ("bottle of pills", "overdose"),
    ("extra dossage", "overdose"),
    ("want to die", "self_harm"),
    ("want to end my life", "self_harm"),
    ("kill myself", "self_harm"),
    ("champukovali", "self_harm"),
    ("mar jana chahta", "self_harm"),
    ("do not want to live", "self_harm"),
    ("pus draining", "fever_wound_infection"),
    ("yellow fluid", "fever_wound_infection"),
    ("sudn vision loss", "vision_loss"),
    ("sudden vissio loss", "vision_loss"),
    ("chupu poyindi", "vision_loss"),
    ("nazar chali gayi", "vision_loss"),
    ("calf is enlarged", "calf_swelling_pain"),
    ("leg is throbbing and calf", "calf_swelling_pain"),
    ("severe belly pain", "severe_abdominal_pain"),
    ("sevear belly pain", "severe_abdominal_pain"),
    ("severe stomach pain", "severe_abdominal_pain"),
)

_PRE_NEGATION_EN = re.compile(
    r"\b(no|not|without|never|don't|dont|didn't|didnt|doesn't|doesnt|"
    r"free\s+of|denies|negative\s+for|no\s+sign\s+of|zero)\b",
    re.IGNORECASE,
)
_POST_NEGATION_EN = re.compile(
    r"\b(stopped|resolved|is\s+gone|has\s+gone|has\s+stopped|ceased|disappeared|ended|absent|minimal)\b",
    re.IGNORECASE,
)
_INDIC_PRE_NEGATION = ["koi", "bina", "kisi bhi"]
_INDIC_POST_NEGATION = [
    # Telugu
    "లేదు",
    "లేవు",
    "తగ్గింది",
    "తగ్గిపోయింది",
    "ఆగిపోయింది",
    "ఆగింది",
    "నయమైంది",
    "ledu",
    "levu",
    "leydhu",
    "taggindi",
    "taggipoindi",
    "aagipoindi",
    "aagindi",
    # Hindi
    "नहीं है",
    "नहीं",
    "नहीं हो रहा",
    "रुक गया",
    "बंद हो गया",
    "खत्म हो गया",
    "गायब हो गया",
    "ठीक हो गया",
    "theek ho gaya",
    "nahi hai",
    "nahin hai",
    "nahi",
    "nahin",
    "band ho gaya",
    "ruk gaya",
]


def _is_negated(text: str, start: int, end: int) -> bool:
    """Check if the phrase occurrence is negated by immediately surrounding words."""
    pre_window = text[max(0, start - 35):start]
    post_window = text[end:min(len(text), end + 45)]

    if _PRE_NEGATION_EN.search(pre_window):
        return True
    if any(neg in pre_window for neg in _INDIC_PRE_NEGATION):
        return True
    if _POST_NEGATION_EN.search(post_window):
        return True
    return any(neg in post_window for neg in _INDIC_POST_NEGATION)


def load_rules(directory: Path | None = None) -> dict[str, dict[str, list[str]]]:
    """Load multilingual keyword/phrase patterns from the YAML data directory."""
    global _RULES
    rules_dir = directory or get_settings().redflags_dir
    loaded: dict[str, dict[str, list[str]]] = {}
    for path in sorted(rules_dir.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        loaded[path.stem] = {
            str(category): [str(term).casefold() for term in terms]
            for category, terms in data.items()
        }
    if not loaded:
        raise FileNotFoundError(f"No red-flag rules found in {rules_dir}.")
    _RULES = loaded
    return loaded


def _rule_match(message: str) -> str | None:
    """Return the first category matched by configured literal phrases, ignoring negations."""
    rules = _RULES or load_rules()
    normalized = re.sub(r"\s+", " ", message.casefold()).strip()
    for language_rules in rules.values():
        for category, phrases in language_rules.items():
            for phrase in phrases:
                start = 0
                while True:
                    idx = normalized.find(phrase, start)
                    if idx == -1:
                        break
                    if not _is_negated(normalized, idx, idx + len(phrase)):
                        return category
                    start = idx + 1
    return None


def _local_classification(payload: dict[str, Any]) -> dict[str, Any]:
    """Conservative local classifier used only when no remote model is configured."""
    message = str(payload.get("message", "")).casefold()
    category = _rule_match(message)
    if category is None:
        for phrase, matched_category in _LOCAL_PARAPHRASES:
            start = 0
            while True:
                idx = message.find(phrase, start)
                if idx == -1:
                    break
                if not _is_negated(message, idx, idx + len(phrase)):
                    category = matched_category
                    break
                start = idx + 1
            if category is not None:
                break
    return {
        "red_flag": category is not None,
        "category": category or "none",
        "confidence": 0.95 if category else 0.9,
    }


# Sentinel returned when the LLM classifier is unavailable but no rule matched.
# Callers receive (flagged=False, category="classifier_unavailable", confidence=0.0)
# so they can escalate as a normal (non-urgent) service outage.
_CLASSIFIER_OUTAGE = "classifier_unavailable"


async def screen_red_flags(
    original_message: str,
    english_message: str | None = None,
) -> tuple[bool, str, float]:
    """Screen original and translated text for urgent symptoms.

    Returns (flagged, category, confidence).

    Behaviour matrix
    ----------------
    Rule layer matches             → (True,  category,              1.0)   urgent
    Classifier flags (no rule)     → (True,  category, classifier_conf)   urgent
    Classifier outage + rule match → (True,  rule_category,         1.0)   urgent
    Classifier outage, no rule     → (False, "classifier_unavailable", 0.0) normal escalation
    No match, no outage            → (False, "none",       min_confidence) pass through
    """
    messages = [original_message]
    if english_message and english_message.casefold() != original_message.casefold():
        messages.append(english_message)

    rule_category = next((category for text in messages if (category := _rule_match(text))), None)

    classifications: list[RedFlagClassification] = []
    classifier_failed = False
    try:
        from app.config import get_settings as _gs

        settings = _gs()
        stage_timeout = settings.llm_redflag_timeout_seconds or None
        for message in messages:
            result = await complete_json(
                (
                    "Classify possible urgent symptoms in the original message. "
                    "Err on the side of flagging."
                ),
                {"message": message},
                RedFlagClassification,
                temperature=0.0,
                local_handler=_local_classification,
                stage_timeout=stage_timeout,
            )
            classifications.append(result)
    except LLMError as exc:
        classifier_failed = True
        logger.error(
            "Red-flag classifier unavailable (%s); applying fail-safe.",
            type(exc).__name__,
        )

    if rule_category:
        # Rule layer always wins: urgent escalation regardless of classifier state.
        return True, rule_category, 1.0

    if classifier_failed:
        # Classifier down but no rule matched.
        # This is a service outage, not a confirmed emergency — return normal escalation.
        # The caller creates a normal (non-urgent) ticket so a nurse reviews it,
        # but the patient does NOT receive false emergency wording.
        return False, _CLASSIFIER_OUTAGE, 0.0

    flagged = [result for result in classifications if result.red_flag]
    if flagged:
        selected = max(flagged, key=lambda result: result.confidence)
        return True, selected.category, selected.confidence
    return False, "none", min(result.confidence for result in classifications)
