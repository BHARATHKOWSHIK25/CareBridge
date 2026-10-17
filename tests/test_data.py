"""Validation for the synthetic source documents and mapping."""

import json
import re

from app.config import ROOT_DIR, Settings


def test_synthetic_documents_have_metadata_sections_and_requested_length():
    """Every demo document is labeled synthetic, complete, and about 600–900 words."""
    required = {
        "Procedure summary",
        "Wound/site care",
        "Activity limits",
        "Medications (as listed by the clinician)",
        "Diet",
        "Warning signs",
        "Follow-up appointment",
        "Contacts",
    }
    for path in (ROOT_DIR / "data" / "docs").glob("*.md"):
        text = path.read_text(encoding="utf-8")
        assert "SYNTHETIC DOCUMENT FOR DEMO ONLY" in text
        assert "No real clinic or emergency phone number is supplied" in text
        assert not re.search(r"\b000\D*555", text)
        assert re.search(r"(?m)^doc_id:\s*\S+", text)
        assert re.search(r"(?m)^patient_alias:\s*\S+", text)
        assert required.issubset(
            {heading.removeprefix("# ").strip() for heading in re.findall(r"(?m)^# .+$", text)}
        )
        words = re.findall(r"\b[\w'-]+\b", text.split("SYNTHETIC DOCUMENT FOR DEMO ONLY", 1)[1])
        assert 600 <= len(words) <= 900


def test_synthetic_patient_mapping_is_one_to_one():
    """Each fictitious patient maps to one existing, unique discharge sheet."""
    patients = json.loads((ROOT_DIR / "data" / "patients.json").read_text(encoding="utf-8"))
    docs = {
        re.search(r"(?m)^doc_id:\s*(\S+)", path.read_text(encoding="utf-8")).group(1)
        for path in (ROOT_DIR / "data" / "docs").glob("*.md")
    }
    assert len(patients) == 3
    assert len({patient["doc_id"] for patient in patients}) == 3
    assert {patient["doc_id"] for patient in patients} == docs


def test_question_sets_match_seed_targets_and_have_human_review_warning():
    """Each generated dataset has balanced type counts and a 40/60 split."""
    for language, total in (("en", 60), ("te", 20), ("hi", 20)):
        records = [
            json.loads(line)
            for line in (ROOT_DIR / "eval" / f"questions_{language}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert len(records) == total
        assert {record["type"] for record in records} == {
            "answerable",
            "unanswerable",
            "red_flag",
        }
        assert abs(sum(record["split"] == "dev" for record in records) / total - 0.4) < 0.01
        assert all(
            {
                "id",
                "doc_id",
                "language",
                "question",
                "type",
                "gold_answer_points",
                "gold_chunk_ids",
                "split",
            }
            <= record.keys()
            for record in records
        )


def test_env_example_lists_every_settings_field():
    """The checked-in environment template documents every Pydantic setting."""
    documented = {
        line.split("=", 1)[0]
        for line in (ROOT_DIR / ".env.example").read_text(encoding="utf-8").splitlines()
        if line and not line.lstrip().startswith("#") and "=" in line
    }
    expected = {name.upper() for name in Settings.model_fields}
    assert expected <= documented
