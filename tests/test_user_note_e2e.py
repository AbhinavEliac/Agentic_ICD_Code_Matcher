import asyncio
import pytest
from medical_coding.graph.pipeline import process_clinical_document

@pytest.mark.asyncio
async def test_user_screenshot_note():
    sample_text = """DIAGNOSIS: Urticaria Gastritis Post frozen embryo transfer (FET)
DRUG ALLERGIES: None
HISTORY OF PRESENT ILLNESS: The patient, a post-frozen embryo transfer (FET) case on hormonal support, was admitted under Medicine for evaluation of acute chest pain, epigastric discomfort, urticarial rash over the chest and neck, and ECG changes while receiving Augmentin and other IVF-related medications.
CLINICAL EXAMINATION ON ADMISSION: On admission, she was hemodynamically stable, afebrile, alert, maintaining oxygen saturation on room air, with no respiratory distress or focal systemic abnormalities.
RELEVANT INVESTIGATIONS: Attached to the file.
COURSE IN THE HOSPITAL AND DISCUSSION: The patient presented with above mentioned complaints. Dermatology evaluation concluded that the rash was urticarial in nature, with additional contact dermatitis over the wrists due to adhesive tape, and advised Levocetirizine 5 mg twice daily along with Flutivate cream twice daily."""

    res = await process_clinical_document(source=sample_text, document_id="user-sample-test")

    print("\n" + "=" * 50)
    print("FORMATTED OUTPUT:")
    print(res.format_output())
    print("=" * 50)

    # Primary diagnosis must not be clubbed
    assert res.primary_diagnosis is not None
    pri_name = (res.primary_diagnosis.normalized_diagnosis or res.primary_diagnosis.raw_term or "").lower()
    assert "urticaria gastritis" not in pri_name, "Urticaria and Gastritis must not be clubbed together!"

    # Secondary diagnoses must be present and captured
    assert len(res.secondary_diagnoses) >= 1, "Secondary diagnoses must not be empty!"
    sec_names = [(s.normalized_diagnosis or s.raw_term or s.description or "").lower() for s in res.secondary_diagnoses]
    print("Secondary names:", sec_names)

    # Check that contact dermatitis or gastritis or urticaria is among diagnoses
    all_names = [pri_name] + sec_names
    print("All diagnosed conditions:", all_names)
    assert any("gastritis" in n for n in all_names), "Gastritis must be diagnosed!"
    assert any("urticaria" in n or "urticarial" in n for n in all_names), "Urticaria must be diagnosed!"
    assert any("dermatitis" in n for n in all_names), "Contact dermatitis must be diagnosed!"

    # Verify no audit strings leaked into evidence
    assert "designated as primary diagnosis" not in (res.primary_diagnosis.evidence_quote or "").lower()
    assert "clinical criteria:" not in (res.primary_diagnosis.evidence_quote or "").lower()

if __name__ == "__main__":
    asyncio.run(test_user_screenshot_note())
