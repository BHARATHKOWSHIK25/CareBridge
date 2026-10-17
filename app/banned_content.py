"""Safety checks for prohibited clinical statements in generated or translated answers."""

import re

# Dose modification regexes (English)
_DOSE_CHANGE_EN = re.compile(
    r"\b("
    r"(increase|decrease|change|double|halve|adjust|modify)\s+(the\s+|your\s+|this\s+|a\s+|an\s+)?(dose|dosage|amount|medication|tablets|pills)"
    r"|(higher|lower)\s+(dose|dosage)"
    r"|take\s+\d+\s*(mg|tablets|pills)"
    r"|change\s+your\s+(dose|dosage)"
    r"|dosage\s+(change|increase|decrease)"
    r")\b",
    re.IGNORECASE,
)

# Stop taking medication (English)
_STOP_TAKING_EN = re.compile(
    r"\b("
    r"(stop|discontinue|cease|quit)\s+(taking|using)"
    r"|(stop|discontinue)\s+(the\s+|your\s+|this\s+|a\s+)?(medication|medicine|tablets|pills|antibiotic|antibiotics|painkillers)"
    r"|do\s+not\s+take\s+(your\s+|the\s+|this\s+)?(medication|medicine|tablets|pills)"
    r")\b",
    re.IGNORECASE,
)

# Start taking medication (English)
_START_TAKING_EN = re.compile(
    r"\b("
    r"(start|begin|commence)\s+(taking|using)"
    r"|(start|begin)\s+(the\s+|your\s+|a\s+|new\s+)?(medication|medicine|tablets|pills|antibiotic|antibiotics)"
    r"|start\s+new\s+medication"
    r")\b",
    re.IGNORECASE,
)

# Diagnosis statements (English)
_DIAGNOSIS_EN = re.compile(
    r"\b("
    r"you\s+(have|may\s+have|suffer\s+from)\s+(pneumonia|infection|appendicitis|cancer|diabetes|asthma|depression|hypertension|embolism|sepsis|fracture|dvt|covid|stroke)"
    r"|i\s+diagnose\s+(you\s+with)?"
    r"|my\s+diagnosis\s+is"
    r"|you\s+are\s+diagnosed\s+with"
    r"|you\s+are\s+suffering\s+from"
    r")\b",
    re.IGNORECASE,
)

# Multilingual keywords and phrases (Telugu & Hindi, script + transliteration)
_MULTILINGUAL_BANNED: list[tuple[str, list[str]]] = [
    (
        "dose_change",
        [
            "మోతాదు మార్చండి",
            "మోతాదు పెంచండి",
            "మోతాదు తగ్గించండి",
            "మోతాదు మార్పు",
            "mothadu marchandi",
            "mothadu penchandi",
            "mothadu tagginchandi",
            "dose marchandi",
            "खुराक बढ़ाएं",
            "खुराक कम करें",
            "खुराक बदलें",
            "खुराक में बदलाव",
            "khuraak badhayein",
            "khuraak kam karein",
            "khuraak badlein",
            "dose badal dein",
        ],
    ),
    (
        "stop_taking",
        [
            "తీసుకోవడం ఆపండి",
            "మందులు ఆపండి",
            "మాత్రలు ఆపండి",
            "వాడటం ఆపండి",
            "teesukovadam aapandi",
            "mandulu aapandi",
            "maathralu aapandi",
            "లేనా బంద్ కరేం",
            "लेना बंद करें",
            "दवा बंद करें",
            "गोलियां बंद करें",
            "दवा लेना बंद कर दें",
            "lena band karein",
            "dawa band karein",
            "band kar dein",
        ],
    ),
    (
        "start_taking",
        [
            "తీసుకోవడం ప్రారంభించండి",
            "తీసుకోవడం మొదలుపెట్టండి",
            "మందులు మొదలుపెట్టండి",
            "teesukovadam modalupettandi",
            "teesukovadam praarambhinchandi",
            "mandulu start cheyandi",
            "లేనా శురూ కరేం",
            "लेना शुरू करें",
            "दवा शुरू करें",
            "दवा लेना शुरू कर दें",
            "lena shuru karein",
            "dawa shuru karein",
            "shuru kar dein",
        ],
    ),
    (
        "diagnosis",
        [
            "రోగ నిర్ధారణ",
            "మీకు ఇన్ఫెక్షన్ ఉంది",
            "మీకు వ్యాధి ఉంది",
            "మీకు జబ్బు ఉంది",
            "roga nirdharana",
            "meeku vyadhi undi",
            "meeku rogam undi",
            "रोग का निदान",
            "निदान किया जाता है",
            "आपको बीमारी है",
            "आपको संक्रमण है",
            "आप पीड़ित हैं",
            "aapko bimari hai",
            "aapko sankraman hai",
        ],
    ),
]


def check_banned_content(text: str | None) -> tuple[bool, str | None]:
    """Check text for prohibited medical advice (dose changes, stop/start, diagnosis).

    Returns:
        (True, reason) if prohibited content is detected.
        (False, None) if clean.
    """
    if not text:
        return False, None

    normalized = re.sub(r"\s+", " ", text).strip().casefold()

    # English regex checks
    if _DOSE_CHANGE_EN.search(text):
        return True, "dose_change"
    if _STOP_TAKING_EN.search(text):
        return True, "stop_taking"
    if _START_TAKING_EN.search(text):
        return True, "start_taking"
    if _DIAGNOSIS_EN.search(text):
        return True, "diagnosis"

    # Multilingual phrase checks
    for category, phrases in _MULTILINGUAL_BANNED:
        if any(phrase.casefold() in normalized for phrase in phrases):
            return True, category

    return False, None
