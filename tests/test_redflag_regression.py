"""Multilingual red-flag regression test suite with recall and false-alarm rate metrics.

Covers at least 40 English, 20 Telugu, and 20 Hindi/Hinglish test phrasings including:
- Standard clinical symptom phrases
- Typos and spelling variations
- Paraphrases and colloquial expressions
- Transliteration and mixed-script formulations
- Clearly negated phrases that must NOT trigger emergency alerts
"""

import pytest

from app.redflag import screen_red_flags

ENGLISH_CASES: list[tuple[str, bool, str]] = [
    # ---- Positives (25) ----
    ("I have crushing chest pain right now", True, "chest_pain"),
    ("There is severe pressure on my chest", True, "chest_pain"),
    ("I have chesst pain", True, "chest_pain"),  # typo
    ("Crushng chest pain", True, "chest_pain"),  # typo
    ("Feels like an elephant sitting on my chest", True, "chest_pain"),  # paraphrase
    ("I am having severe difficulty breathing", True, "breathing_difficulty"),
    ("I cannot breathe properly", True, "breathing_difficulty"),
    ("Cant breath at all", True, "breathing_difficulty"),  # typo
    ("Shortnes of breath after walking", True, "breathing_difficulty"),  # typo
    ("Struggling for air since this morning", True, "breathing_difficulty"),  # paraphrase
    ("Heavy uncontrolled bleeding from incision", True, "severe_bleeding"),
    ("The bleeding won't stop at all", True, "severe_bleeding"),
    ("Bleedng won't stop", True, "severe_bleeding"),  # typo
    ("Soaking through bandages with dark blood rapidly", True, "severe_bleeding"),  # paraphrase
    ("I fainted when I tried to stand up", True, "fainting"),
    ("Passed out on the floor", True, "fainting"),  # paraphrase
    ("Faintng spells repeatedly today", True, "fainting"),  # typo
    ("My face is drooping on the left side", True, "stroke"),
    ("One side of my body feels completely weak and numb", True, "stroke"),
    ("Severe allergic reaction and my throat is closing", True, "allergic_reaction"),
    ("Throat is sweling and closing up", True, "allergic_reaction"),  # typo
    ("I accidentally took 10 sleeping pills", True, "overdose"),
    ("Took an extra dossage of heart medication", True, "overdose"),  # typo
    ("I want to kill myself", True, "self_harm"),
    ("I do not want to live anymore", True, "self_harm"),
    ("High fever with pus draining from wound", True, "fever_wound_infection"),
    ("Sudden vision loss in right eye", True, "vision_loss"),
    ("Sudn vision loss after waking", True, "vision_loss"),  # typo
    ("My calf is swollen and my leg hurts terribly", True, "calf_swelling_pain"),
    ("Leg is throbbing and calf is enlarged", True, "calf_swelling_pain"),  # paraphrase
    ("Severe belly pain and vomiting non stop", True, "severe_abdominal_pain"),
    ("Sevear belly pain with violent vomiting", True, "severe_abdominal_pain"),  # typo
    # ---- Negatives / Clearly Negated (15) ----
    ("I do not have chest pain today", False, "none"),
    ("No chest pain or pressure observed", False, "none"),
    ("I have no difficulty breathing at rest", False, "none"),
    ("Without any bleeding from the incision", False, "none"),
    ("The bleeding has stopped completely", False, "none"),
    ("My stomach pain has resolved", False, "none"),
    ("No fever and wound is healing nicely", False, "none"),
    ("Never had thoughts of self harm", False, "none"),
    ("Zero vision loss, eyesight is normal", False, "none"),
    ("I did not take extra pills", False, "none"),
    ("Calf swelling is gone and pain is minimal", False, "none"),
    ("The dressing is dry and clean", False, "none"),
    ("When can I safely take a shower?", False, "none"),
    ("What foods should I eat after discharge?", False, "none"),
    ("Is it normal to feel a little tired?", False, "none"),
]

TELUGU_CASES: list[tuple[str, bool, str]] = [
    # ---- Positives (14) ----
    ("నాకు తీవ్రమైన ఛాతి నొప్పి ఉంది", True, "chest_pain"),
    ("గుండె నొప్పి వస్తోంది", True, "chest_pain"),
    ("chaathi noppi ekkuva ga undi", True, "chest_pain"),  # transliteration
    ("chati noppi undi", True, "chest_pain"),  # transliteration
    ("naaku chest pain undi", True, "chest_pain"),  # mixed script
    ("శ్వాస తీసుకోవడం కష్టంగా ఉంది", True, "breathing_difficulty"),
    ("ఊపిరి ఆడటం లేదు", True, "breathing_difficulty"),
    ("swasa kastam ga undi", True, "breathing_difficulty"),  # transliteration
    ("oopiri aadatledu", True, "breathing_difficulty"),  # transliteration
    ("రక్తం ఆగడం లేదు చాలా ఎక్కువగా ఉంది", True, "severe_bleeding"),
    ("raktham aagatledu", True, "severe_bleeding"),  # transliteration
    ("bleeding aagatledu", True, "severe_bleeding"),  # mixed
    ("స్పృహ తప్పిపోయాను", True, "fainting"),
    ("spruha tappindi", True, "fainting"),  # transliteration
    ("తీవ్రమైన కడుపు నొప్పి వాంతులు", True, "severe_abdominal_pain"),
    ("గాయం నుండి చీము మరియు జ్వరం", True, "fever_wound_infection"),
    ("చూపు పోయింది అకస్మాత్తుగా", True, "vision_loss"),
    ("chupu poyindi sudden ga", True, "vision_loss"),  # transliteration
    # ---- Negatives / Clearly Negated (8) ----
    ("నాకు ఛాతి నొప్పి లేదు", False, "none"),
    ("గుండె నొప్పి లేదు అంతా బాగుంది", False, "none"),
    ("chati noppi ledu", False, "none"),  # transliteration negation
    ("శ్వాస తీసుకోవడంలో ఇబ్బంది లేదు", False, "none"),
    ("రక్తం ఆగిపోయింది", False, "none"),
    ("raktham aagipoindi", False, "none"),  # transliteration negation
    ("నేను ఎప్పుడు స్నానం చేయవచ్చు?", False, "none"),
    ("గాయం పొడిగా ఉంది నొప్పి లేదు", False, "none"),
]

HINDI_CASES: list[tuple[str, bool, str]] = [
    # ---- Positives (14) ----
    ("सीने में बहुत तेज दर्द हो रहा है", True, "chest_pain"),
    ("छाती में बहुत दबाव महसूस हो रहा है", True, "chest_pain"),
    ("seene mein dard ho raha hai", True, "chest_pain"),  # transliteration
    ("chaati me dard hai", True, "chest_pain"),  # transliteration
    ("mujhe chest pain ho raha hai", True, "chest_pain"),  # mixed script
    ("सांस लेने में बहुत कठिनाई हो रही है", True, "breathing_difficulty"),
    ("सांस नहीं आ रही है", True, "breathing_difficulty"),
    ("saans lene me dikkat ho rahi hai", True, "breathing_difficulty"),  # transliteration
    ("saans nahi aa rahi", True, "breathing_difficulty"),  # transliteration
    ("खून बंद नहीं हो रहा है बहुत बह रहा है", True, "severe_bleeding"),
    ("khoon ruk nahi raha", True, "severe_bleeding"),  # transliteration
    ("wound se bleeding ho rahi hai ruk nahi rahi", True, "severe_bleeding"),  # mixed
    ("अचानक चक्कर खाकर बेहोश हो गया", True, "fainting"),
    ("behosh ho gaya tha", True, "fainting"),  # transliteration
    ("तेज पेट दर्द और उल्टी हो रही है", True, "severe_abdominal_pain"),
    ("घाव से मवाद और तेज बुखार है", True, "fever_wound_infection"),
    ("अचानक नजर चली गई दिखाई नहीं दे रहा", True, "vision_loss"),
    ("nazar chali gayi achanak", True, "vision_loss"),  # transliteration
    # ---- Negatives / Clearly Negated (8) ----
    ("सीने में कोई दर्द नहीं है", False, "none"),
    ("छाती में दर्द नहीं है", False, "none"),
    ("seene me dard nahi hai", False, "none"),  # transliteration negation
    ("सांस लेने में कोई दिक्कत नहीं है", False, "none"),
    ("खून बहना बंद हो गया है", False, "none"),
    ("khoon ruk gaya hai", False, "none"),  # transliteration negation
    ("क्या मैं आज नहा सकता हूँ?", False, "none"),
    ("पट्टी सूखी है कोई परेशानी नहीं है", False, "none"),
]


async def _evaluate_suite(
    suite_name: str, cases: list[tuple[str, bool, str]]
) -> tuple[float, float]:
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    mismatches = []

    from app.redflag import load_rules
    load_rules()

    for text, should_flag, expected_category in cases:
        flagged, found_category, _ = await screen_red_flags(text)
        if should_flag:
            if flagged:
                tp += 1
            else:
                fn += 1
                mismatches.append(
                    f"FN: '{text}' (expected {expected_category}, got {found_category})"
                )
        else:
            if flagged:
                fp += 1
                mismatches.append(f"FP: '{text}' (expected no flag, got {found_category})")
            else:
                tn += 1

    total_pos = tp + fn
    total_neg = fp + tn
    recall = tp / total_pos if total_pos > 0 else 1.0
    far = fp / total_neg if total_neg > 0 else 0.0

    print(f"\n================ RED-FLAG REGRESSION: {suite_name} ================")
    print(f"Total Cases: {len(cases)} (Positives: {total_pos}, Negatives: {total_neg})")
    print(f"TP: {tp}, FN: {fn}, TN: {tn}, FP: {fp}")
    print(f"Recall: {recall * 100:.1f}%")
    print(f"False Alarm Rate: {far * 100:.1f}%")
    if mismatches:
        print("Mismatches:")
        for m in mismatches:
            safe_m = m.encode("ascii", errors="backslashreplace").decode("ascii")
            print(f"  - {safe_m}")
    print("====================================================================")

    return recall, far


@pytest.mark.asyncio
async def test_english_redflag_regression_suite():
    """Verify >= 40 English phrasings with recall >= 95% and false alarm rate <= 5%."""
    assert len(ENGLISH_CASES) >= 40
    recall, far = await _evaluate_suite("English", ENGLISH_CASES)
    assert recall >= 0.95, f"English recall {recall:.2%} below target"
    assert far <= 0.05, f"English false-alarm rate {far:.2%} above target"


@pytest.mark.asyncio
async def test_telugu_redflag_regression_suite():
    """Verify >= 20 Telugu phrasings with recall >= 95% and false alarm rate <= 5%."""
    assert len(TELUGU_CASES) >= 20
    recall, far = await _evaluate_suite("Telugu", TELUGU_CASES)
    assert recall >= 0.95, f"Telugu recall {recall:.2%} below target"
    assert far <= 0.05, f"Telugu false-alarm rate {far:.2%} above target"


@pytest.mark.asyncio
async def test_hindi_redflag_regression_suite():
    """Verify >= 20 Hindi/Hinglish phrasings with recall >= 95% and false alarm rate <= 5%."""
    assert len(HINDI_CASES) >= 20
    recall, far = await _evaluate_suite("Hindi / Hinglish", HINDI_CASES)
    assert recall >= 0.95, f"Hindi recall {recall:.2%} below target"
    assert far <= 0.05, f"Hindi false-alarm rate {far:.2%} above target"
