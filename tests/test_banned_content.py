"""Unit tests for banned medical advice detection across English, Telugu, and Hindi."""

import pytest

from app.banned_content import check_banned_content


@pytest.mark.parametrize(
    ("text", "expected_category"),
    [
        # English: Dose changes
        ("You should increase the dose to 20mg.", "dose_change"),
        ("Please decrease your dosage immediately.", "dose_change"),
        ("Double the dose if pain persists.", "dose_change"),
        ("Take 2 tablets instead of 1.", "dose_change"),
        ("A higher dose is recommended.", "dose_change"),
        # English: Stop taking
        ("Stop taking your blood pressure medication.", "stop_taking"),
        ("You should discontinue taking the pills.", "stop_taking"),
        ("Do not take your medication anymore.", "stop_taking"),
        # English: Start taking
        ("Start taking amoxicillin 500mg twice daily.", "start_taking"),
        ("Begin taking the new painkiller today.", "start_taking"),
        ("Start new medication immediately.", "start_taking"),
        # English: Diagnosis
        ("You are diagnosed with severe pneumonia.", "diagnosis"),
        ("I diagnose you with appendicitis.", "diagnosis"),
        ("You may have diabetes based on symptoms.", "diagnosis"),
        ("You are suffering from sepsis.", "diagnosis"),
        # Telugu: Banned phrases
        ("మీరు మోతాదు పెంచండి.", "dose_change"),
        ("దయచేసి మోతాదు మార్చండి.", "dose_change"),
        ("ఈ మందులు తీసుకోవడం ఆపండి.", "stop_taking"),
        ("రోజూ మాత్రలు తీసుకోవడం ప్రారంభించండి.", "start_taking"),
        ("మీకు ఇన్ఫెక్షన్ ఉంది మరియు రోగ నిర్ధారణ చేయబడింది.", "diagnosis"),
        # Telugu: Transliteration
        ("mothadu penchandi", "dose_change"),
        ("mandulu aapandi", "stop_taking"),
        ("mandulu start cheyandi", "start_taking"),
        ("meeku vyadhi undi", "diagnosis"),
        # Hindi: Banned phrases
        ("अपनी दवा की खुराक बढ़ाएं।", "dose_change"),
        ("खुराक में बदलाव करें।", "dose_change"),
        ("यह दवा लेना बंद करें।", "stop_taking"),
        ("गोलियां बंद करें।", "stop_taking"),
        ("नई दवा लेना शुरू करें।", "start_taking"),
        ("आपको बीमारी है और संक्रमण है।", "diagnosis"),
        # Hindi: Hinglish
        ("khuraak badhayein", "dose_change"),
        ("dawa band karein", "stop_taking"),
        ("dawa shuru karein", "start_taking"),
        ("aapko bimari hai", "diagnosis"),
    ],
)
def test_banned_content_detected(text, expected_category):
    flagged, category = check_banned_content(text)
    assert flagged is True
    assert category == expected_category


@pytest.mark.parametrize(
    "text",
    [
        "Keep the surgical wound dry and clean.",
        "You may take a shower after 48 hours according to discharge instructions.",
        "Take prescribed medications exactly as directed by your surgeon.",
        "Call your care team if you have questions.",
        "స్నానం చేసేటప్పుడు గాయం తడవకుండా చూసుకోండి.",
        "घाव को साफ और सूखा रखें।",
    ],
)
def test_clean_discharge_advice_allowed(text):
    flagged, category = check_banned_content(text)
    assert flagged is False
    assert category is None
