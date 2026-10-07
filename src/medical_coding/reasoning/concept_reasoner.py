"""Clinical Concept Reasoner: Transforms raw candidates into structured ClinicalConcepts.

Conforms to Master Specification Section 5 & 6:
- Enforces CODE_SPECIFICITY <= EVIDENCE_SPECIFICITY.
- Maps candidates into canonical disease families.
- Isolates attributes (biomarkers, laterality, staging, displacement) from disease entities.
- Identifies documented supported attributes vs unknown unevidenced attributes.
"""

import re
from typing import Any

from medical_coding.schemas.enums import ClinicalEntityType
from medical_coding.schemas.reasoning import ClinicalConcept
from medical_coding.utils.logging import get_logger

logger = get_logger(__name__)

# Canonical Disease Families mapped to key entity detection cues
DISEASE_FAMILY_PATTERNS: list[tuple[str, list[str], str]] = [
    ("breast_malignancy", ["breast cancer", "breast carcinoma", "carcinoma of breast", "malignant neoplasm of breast", "ductal carcinoma", "lobular carcinoma", "breast"], "breast"),
    ("lymphoma", ["dlbcl", "diffuse large b-cell lymphoma", "b-cell lymphoma", "hodgkin", "non-hodgkin", "lymphoma"], "lymph nodes"),
    ("renal_calculus", ["renal calculus", "kidney stone", "renal stone", "nephrolithiasis", "calculus of kidney"], "kidney"),
    ("ureteric_calculus", ["ureteric calculus", "ureteral calculus", "ureter stone", "ureteral stone", "calculus of ureter"], "ureter"),
    ("uti", ["urinary tract infection", "uti", "catheter-associated uti", "cauti"], "urinary tract"),
    ("heart_failure", ["heart failure", "congestive heart failure", "chf", "cardiac failure", "systolic heart failure", "diastolic heart failure"], "heart"),
    ("gastritis", ["gastritis", "gastric inflammation", "antral gastritis"], "stomach"),
    ("pyelonephritis", ["pyelonephritis", "acute pyelonephritis", "emphysematous pyelonephritis"], "kidney"),
    ("bronchitis", ["bronchitis", "acute bronchitis", "tracheobronchitis"], "bronchus"),
    ("pneumonia", ["pneumonia", "community-acquired pneumonia", "bronchopneumonia", "aspiration pneumonia"], "lung"),
    ("diabetes", ["diabetes", "type 2 diabetes", "type 1 diabetes", "diabetic", "dm", "t2dm", "t1dm"], "endocrine"),
    ("hypertension", ["hypertension", "essential hypertension", "high blood pressure", "htn"], "cardiovascular"),
    ("ckd", ["chronic kidney disease", "ckd", "renal impairment", "renal failure"], "kidney"),
    ("septic_shock", ["septic shock"], "systemic"),
    ("sepsis", ["sepsis", "urosepsis", "septicemia"], "systemic"),
    ("bacteremia", ["bacteremia", "bloodstream infection", "blood culture positive"], "blood"),
    ("candidiasis", ["candidiasis", "candidal", "candida", "funguria"], "urogenital"),
    ("castleman_disease", ["castleman", "castleman's disease", "angiofollicular"], "lymph nodes"),
    ("kaposi_sarcoma", ["kaposi", "kaposi's sarcoma"], "skin"),
    ("hiv_disease", ["hiv", "human immunodeficiency virus", "aids"], "immune"),
    ("viral_hepatitis", ["hepatitis b", "hepatitis c", "hbv", "hcv", "viral hepatitis"], "liver"),
    ("pleural_metastasis", ["pleural metastasis", "malignant pleural effusion", "metastasis to pleura", "secondary malignant neoplasm of pleura"], "pleura"),
    ("neuroendocrine_tumor", ["neuroendocrine tumor", "net", "carcinoid"], "colon"),
    ("h_pylori", ["h. pylori", "h pylori", "helicobacter pylori", "helicobacter"], "stomach"),
    ("chemotherapy_encounter", ["chemotherapy encounter", "antineoplastic chemotherapy", "chemo encounter", "chemotherapy"], "systemic"),
    ("e_coli_infection", ["e. coli", "escherichia coli"], "systemic"),
    ("acl_injury", ["acl tear", "anterior cruciate ligament", "acl sprain", "acl rupture"], "knee"),
    ("fracture_malleolus", ["lateral malleolus", "malleolus fracture", "malleolar fracture", "fracture of lateral malleolus", "fracture of left lateral malleolus", "fracture of right lateral malleolus", "fibula fracture"], "fibula"),
    ("fracture_tarsal", ["tarsal fracture", "tarsals fracture", "tarsus fracture", "fracture of tarsal", "fracture of tarsals", "fractures of tarsals", "tarsal"], "foot"),
    ("fracture_metatarsal", ["metatarsal fracture", "metatarsals fracture", "fifth metatarsal", "fracture of metatarsal", "fractures of metatarsal", "metatarsal"], "foot"),
    ("sprain_ankle_ligament", ["deltoid ligament", "calcaneofibular ligament", "deltoid and calcaneofibular", "ankle sprain", "ligament sprain", "deltoid sprain", "calcaneofibular sprain"], "ankle"),
    ("urticaria", ["urticaria", "hives"], "skin"),
    ("hernia", ["inguinal hernia", "ventral hernia", "umbilical hernia"], "abdomen"),
    ("asthma", ["asthma", "bronchial asthma", "status asthmaticus"], "lung"),
    ("liver_sarcoma", ["embryonal sarcoma of liver", "sarcoma of liver", "liver sarcoma", "angiosarcoma of liver", "embryonal sarcoma"], "liver"),
    ("liver_metastasis", ["liver metastasis", "liver metastases", "metastasis to liver", "secondary malignant neoplasm of liver"], "liver"),
    ("myocardial_infarction", ["myocardial infarction", "nstemi", "stemi", "acute coronary syndrome", "subendocardial myocardial infarction", "heart attack"], "heart"),
    ("cholecystitis", ["cholecystitis", "acute cholecystitis", "chronic cholecystitis"], "gallbladder"),
    ("cholelithiasis", ["cholelithiasis", "gallstone", "gallstones", "calculus of gallbladder", "biliary calculus"], "gallbladder"),
    ("hyperlipidemia", ["hyperlipidemia", "hypercholesterolemia", "dyslipidemia", "hypertriglyceridemia"], "blood"),
    ("hypokalemia", ["hypokalemia", "hypopotassemia"], "blood"),
    ("hyperkalemia", ["hyperkalemia"], "blood"),
    ("hyponatremia", ["hyponatremia"], "blood"),
    ("anemia", ["anemia", "iron deficiency anemia", "acute blood loss anemia"], "blood"),
    ("appendicitis", ["appendicitis", "acute appendicitis"], "appendix"),
    ("pancreatitis", ["pancreatitis", "acute pancreatitis", "chronic pancreatitis"], "pancreas"),
    ("atrial_fibrillation", ["atrial fibrillation", "a-fib", "afib", "atrial flutter"], "heart"),
    ("coronary_artery_disease", ["coronary artery disease", "cad", "atherosclerotic heart disease", "ashd", "ischemic heart disease"], "heart"),
    ("cerebrovascular_accident", ["cerebrovascular accident", "cva", "cerebral infarction", "ischemic stroke", "stroke"], "brain"),
    ("deep_vein_thrombosis", ["deep vein thrombosis", "dvt", "deep venous thrombosis"], "vein"),
    ("pulmonary_embolism", ["pulmonary embolism", "pe"], "lung"),
    ("gerd", ["gastroesophageal reflux", "gerd", "acid reflux", "esophagitis"], "esophagus"),
    ("cellulitis", ["cellulitis"], "skin"),
    ("osteoarthritis", ["osteoarthritis", "degenerative joint disease", "oa"], "joint"),
    ("hypothyroidism", ["hypothyroidism", "underactive thyroid"], "thyroid"),
    ("acute_kidney_injury", ["acute kidney injury", "aki", "acute renal failure"], "kidney"),
]


class ClinicalConceptReasoner:
    """Transforms raw candidates and narrative evidence into structured, constrained ClinicalConcepts."""

    @classmethod
    def reason_concept(
        cls,
        candidate: Any,
        evidence_text: str = "",
        document_text: str = "",
    ) -> ClinicalConcept:
        """Derive a ClinicalConcept from candidate term and supporting evidence."""
        raw_term = (
            getattr(candidate, "raw_term", None)
            or getattr(candidate, "normalized_diagnosis", None)
            or getattr(candidate, "diagnosis", None)
            or getattr(candidate, "normalized_description", "")
        ).strip()

        quote = (
            evidence_text
            or getattr(candidate, "primary_evidence_quote", None)
            or (
                candidate.context.evidence.quote
                if hasattr(candidate, "context") and hasattr(candidate.context, "evidence") and candidate.context.evidence
                else ""
            )
            or getattr(candidate, "evidence_quote", None)
            or getattr(candidate, "evidence", "")
            or ""
        )

        term_lower = raw_term.lower()

        # ISOLATION PER DIAGNOSIS (Master Prompt Section 41)
        # Build local diagnosis context strictly from candidate term and its own evidence quote
        local_context = f"{raw_term} {quote}".strip().lower()

        # If evidence quote is minimal, supplement only with document sentences referencing this specific entity
        if len(local_context.split()) < 5 and document_text:
            target_words = [
                w for w in re.findall(r"\b[a-z0-9]{3,}\b", term_lower)
                if w not in ("with", "without", "and", "the", "for", "left", "right", "acute", "chronic", "mild", "severe")
            ]
            if target_words:
                targeted_sents: list[str] = []
                for s in re.split(r"(?<=[.!?\n])\s+", document_text):
                    s_lower = s.lower()
                    if any(tw in s_lower for tw in target_words):
                        targeted_sents.append(s.strip())
                if targeted_sents:
                    local_context = f"{local_context} {' '.join(targeted_sents[:2])}".lower()

        # 1. Infer Disease Family and Default Body Site
        matched_family = "unspecified_condition"
        detected_site = None

        for fam_name, patterns, site in DISEASE_FAMILY_PATTERNS:
            if any(re.search(rf"\b{re.escape(p)}\b", term_lower) for p in patterns):
                matched_family = fam_name
                detected_site = site
                break

        if matched_family == "unspecified_condition":
            # Fallback scan across candidate's isolated evidence
            for fam_name, patterns, site in DISEASE_FAMILY_PATTERNS:
                if any(re.search(rf"\b{re.escape(p)}\b", local_context) for p in patterns):
                    matched_family = fam_name
                    detected_site = site
                    break

        # 2. Extract Laterality (left, right, bilateral)
        detected_laterality = None
        if re.search(r"\b(?:left|lt)\b", term_lower) or re.search(r"\b(?:left|lt)\s+(?:breast|kidney|ureter|fibula|malleolus|foot|ankle|knee)\b", local_context):
            detected_laterality = "left"
        elif re.search(r"\b(?:right|rt)\b", term_lower) or re.search(r"\b(?:right|rt)\s+(?:breast|kidney|ureter|fibula|malleolus|foot|ankle|knee)\b", local_context):
            detected_laterality = "right"
        elif re.search(r"\bbilateral\b", term_lower) or re.search(r"\bbilateral\b", local_context):
            detected_laterality = "bilateral"

        # 3. Metastatic Status and Direction (Master Prompt Section 7 & 38)
        # Distinguish primary cancer with distant metastasis from metastatic lesions to an organ
        is_metastatic = bool(
            re.search(r"\b(?:metastatic|metastasis|metastasized|stage\s+iv)\b", local_context)
            and not re.search(r"\bwithout\s+metastasis\b", local_context)
        )
        metastatic_sites: list[str] = []
        if is_metastatic:
            for site_cand in ["pleura", "pleural", "lung", "liver", "bone", "brain"]:
                if re.search(rf"\b{site_cand}\s+metastasis\b", local_context) or re.search(rf"\bmetastasis\s+to\s+{site_cand}\b", local_context):
                    metastatic_sites.append(site_cand)

        # 4. Histology & Etiology
        histology = None
        if "dlbcl" in local_context or "diffuse large b-cell" in local_context:
            histology = "diffuse large b-cell"
        elif "carcinoma" in local_context:
            histology = "carcinoma"
        elif "sarcoma" in local_context:
            histology = "sarcoma"

        etiology = None
        if "h. pylori" in local_context or "helicobacter" in local_context:
            etiology = "helicobacter_pylori"
        elif "e. coli" in local_context or "escherichia" in local_context:
            etiology = "escherichia_coli"
        elif "candida" in local_context:
            etiology = "candida"

        # 5. Extract Supported vs Unknown Attributes
        supported_attributes: list[str] = []
        unknown_attributes: list[str] = []

        if detected_laterality:
            supported_attributes.append(detected_laterality)
        else:
            unknown_attributes.append("laterality")

        if is_metastatic:
            supported_attributes.append("metastatic")

        # Specificity qualifiers
        if "nondisplaced" in local_context or "no displacement" in local_context:
            supported_attributes.append("nondisplaced")
        elif "displaced" in local_context and "no displacement" not in local_context and "nondisplaced" not in local_context:
            supported_attributes.append("displaced")
        else:
            unknown_attributes.append("displacement")

        if "bleeding" in local_context or "bleed" in local_context or "hemorrhage" in local_context:
            from medical_coding.validation.reverse_attributes import has_positive_mention
            if (
                has_positive_mention("bleeding", local_context)
                or has_positive_mention("bleed", local_context)
                or has_positive_mention("hemorrhage", local_context)
            ):
                supported_attributes.append("with_bleeding")
            else:
                supported_attributes.append("without_bleeding")
        else:
            unknown_attributes.append("bleeding_status")

        # Oncology specifics: quadrant is unknown if not specified
        if matched_family == "breast_malignancy":
            quadrants = ["upper outer", "upper inner", "lower outer", "lower inner", "central"]
            if not any(q in local_context for q in quadrants):
                unknown_attributes.append("quadrant")

        # 6. Assertion & Temporality
        assertion = "CONFIRMED"
        if "ruled out" in quote.lower() or "no evidence" in quote.lower():
            assertion = "RULED_OUT"
        elif "?" in raw_term or "suspected" in quote.lower() or "possible" in quote.lower():
            assertion = "SUSPECTED"

        temporality = "CURRENT"
        if getattr(candidate, "temporality", None) in ("HISTORICAL", "RESOLVED"):
            temporality = str(candidate.temporality)

        entity_type = getattr(candidate, "entity_type", None) or "DIAGNOSIS"
        if isinstance(entity_type, str):
            try:
                entity_type = ClinicalEntityType(entity_type)
            except ValueError:
                entity_type = ClinicalEntityType.DIAGNOSIS

        cid = getattr(candidate, "diagnosis_id", None) or getattr(candidate, "condition_id", None) or getattr(candidate, "id", "") or ""

        return ClinicalConcept(
            concept_id=str(cid) if cid else None,
            id=str(cid) if cid else None,
            canonical_name=raw_term,
            canonical_concept=raw_term,
            normalized_concept=getattr(candidate, "normalized_diagnosis", raw_term),
            entity_type=entity_type,
            disease_family=matched_family,
            body_site=detected_site,
            anatomy=detected_site,
            laterality=detected_laterality,
            etiology=etiology,
            histology=histology,
            severity=None,
            complication=None,
            acuity=getattr(candidate, "acuity", None),
            temporal_status=temporality,
            temporality=temporality,
            assertion_status=assertion,
            role_hint=str(getattr(candidate, "role", "SECONDARY")),
            role=str(getattr(candidate, "role", "SECONDARY")),
            metastatic_status=is_metastatic,
            metastatic_sites=metastatic_sites,
            evidence_spans=[quote] if quote else [],
            supporting_evidence=[quote] if quote else [],
            supporting_sections=[getattr(candidate, "source_section", "DISCHARGE_DIAGNOSES") or "DISCHARGE_DIAGNOSES"],
            source_sections=[getattr(candidate, "source_section", "DISCHARGE_DIAGNOSES") or "DISCHARGE_DIAGNOSES"],
            supported_attributes=supported_attributes,
            unknown_attributes=unknown_attributes,
            contradictory_attributes=[],
            contradiction_evidence=[],
            confidence=float(getattr(candidate, "confidence_score", 1.0) or 1.0),
            clinical_confidence=float(getattr(candidate, "confidence_score", 1.0) or 1.0),
        )
