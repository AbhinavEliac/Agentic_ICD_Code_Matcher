"""MatchSpec Builder: Builds deterministic database retrieval constraints from ClinicalConcept.

Conforms to Master Specification Sections 7 and 8:
- Hard-filters database candidate space by allowed_code_families.
- Implements strict attribute firewalls (forbidden_attributes, required_attributes).
- Prevents cross-candidate contamination and database poisoning.
"""

import re
from typing import Any

from medical_coding.schemas.reasoning import ClinicalConcept, MatchSpec
from medical_coding.utils.logging import get_logger

logger = get_logger(__name__)

# Mapping from canonical disease family to authoritative ICD-10-CM code category prefixes
FAMILY_TO_CODE_PREFIXES: dict[str, list[str]] = {
    "breast_malignancy": ["C50"],
    "lymphoma": ["C83", "C81", "C82", "C85"],
    "pyelonephritis": ["N10"],
    "renal_calculus": ["N20.0", "N20"],
    "ureteric_calculus": ["N20.1", "N20"],
    "uti": ["N39.0", "N39"],
    "heart_failure": ["I50"],
    "gastritis": ["K29"],
    "bronchitis": ["J20"],
    "pneumonia": ["J18", "J15", "J13"],
    "diabetes": ["E11", "E10"],
    "hypertension": ["I10"],
    "ckd": ["N18"],
    "septic_shock": ["R65.21", "R65"],
    "bacteremia": ["R78.81", "R78"],
    "candidiasis": ["B37"],
    "castleman_disease": ["D47.Z2", "D47"],
    "kaposi_sarcoma": ["C46"],
    "hiv_disease": ["B20"],
    "viral_hepatitis": ["B18"],
    "pleural_metastasis": ["C78.2", "C78"],
    "neuroendocrine_tumor": ["C7A.02", "C7A", "C7B", "D3A"],
    "h_pylori": ["B96.81", "B96"],
    "chemotherapy_encounter": ["Z51.11", "Z51.1", "Z51"],
    "e_coli_infection": ["B96.20", "B96.2", "A41.51"],
    "acl_injury": ["S83.51", "S83.5", "M23.61"],
    "fracture_malleolus": ["S82.6", "S82"],
    "fracture_tarsal": ["S92.2", "S92"],
    "fracture_metatarsal": ["S92.35", "S92.3", "S92"],
    "sprain_ankle_ligament": ["S93.4"],
    "urticaria": ["L50"],
    "hernia": ["K40", "K41", "K42", "K43"],
    "asthma": ["J45"],
    "liver_sarcoma": ["C22.4", "C22"],
    "liver_metastasis": ["C78.7", "C78", "C7B.02"],
    "myocardial_infarction": ["I21", "I22"],
    "cholecystitis": ["K81", "K80"],
    "cholelithiasis": ["K80"],
    "hyperlipidemia": ["E78"],
    "hypokalemia": ["E87.6", "E87"],
    "hyperkalemia": ["E87.5", "E87"],
    "hyponatremia": ["E87.1", "E87"],
    "anemia": ["D50", "D64"],
    "appendicitis": ["K35", "K36", "K37"],
    "pancreatitis": ["K85"],
    "atrial_fibrillation": ["I48"],
    "coronary_artery_disease": ["I25"],
    "cerebrovascular_accident": ["I63", "I67"],
    "deep_vein_thrombosis": ["I82"],
    "pulmonary_embolism": ["I26"],
    "gerd": ["K21"],
    "cellulitis": ["L03"],
    "osteoarthritis": ["M15", "M16", "M17", "M19"],
    "hypothyroidism": ["E03"],
    "acute_kidney_injury": ["N17"],
}


class MatchSpecBuilder:
    """Builds deterministic MatchSpec constraints from a validated ClinicalConcept."""

    @classmethod
    def build_match_spec(cls, concept: ClinicalConcept) -> MatchSpec:
        """Construct MatchSpec with allowed code families and attribute firewalls."""
        allowed_prefixes = list(FAMILY_TO_CODE_PREFIXES.get(concept.disease_family, []))
        required_attrs: dict[str, Any] = {}
        forbidden_attrs: list[str] = []

        # 1. Laterality Constraint
        if concept.laterality:
            required_attrs["laterality"] = concept.laterality
            if concept.laterality == "left":
                forbidden_attrs.extend(["right laterality", "right"])
            elif concept.laterality == "right":
                forbidden_attrs.extend(["left laterality", "left"])

        # 2. Oncology / Metastatic Direction Firewall (Section 7 & 21)
        if concept.disease_family == "breast_malignancy":
            # Primary breast cancer MUST NOT match secondary breast malignancy (C79.81)
            forbidden_attrs.extend([
                "secondary malignant neoplasm of breast",
                "c79.81",
                "secondary neoplasm of breast",
                "metastasis to breast",
            ])
            # If not explicitly documented as male, forbid male breast codes (C50.*2*)
            evidence_str = f"{concept.canonical_name} {' '.join(concept.evidence_spans)}".lower()
            if not re.search(r"\b(?:male|man|men|boy|gentleman)\b", evidence_str):
                forbidden_attrs.extend(["male breast", "male"])
            required_attrs["primary_site"] = "breast"

        elif concept.disease_family == "acl_injury":
            # ACL must not match posterior cruciate ligament (Master Prompt Section 5)
            forbidden_attrs.extend(["posterior cruciate", "pcl"])

        elif concept.disease_family == "pleural_metastasis":
            # Pleural metastasis MUST NOT match lung cancer (C78.00) or primary pleura (C38.4)
            forbidden_attrs.extend([
                "lung",
                "c78.0",
                "bronchus",
            ])
            required_attrs["site"] = "pleura"

        elif concept.disease_family == "liver_metastasis":
            # Liver metastasis MUST NOT match primary liver neoplasm or unspecified primary/secondary (C22.9)
            forbidden_attrs.extend([
                "not specified as primary or secondary",
                "c22.9",
                "primary malignant neoplasm of liver",
                "hepatocellular",
            ])
            required_attrs["site"] = "liver"
            required_attrs["metastatic"] = True

        # 3. Symptom Firewall (Section 20 & CMS Guideline I.B.4)
        if concept.disease_family in ("renal_calculus", "ureteric_calculus"):
            # Calculus must not become symptom renal colic (N23)
            forbidden_attrs.extend(["n23", "unspecified renal colic", "renal colic nos"])

        # 4. Fracture Displacement Guardrail (Section 22 & 26)
        if "fracture" in concept.disease_family:
            if "nondisplaced" in concept.supported_attributes:
                forbidden_attrs.append("displaced")
                required_attrs["displacement"] = "nondisplaced"
            elif "displaced" in concept.supported_attributes:
                forbidden_attrs.append("nondisplaced")
                required_attrs["displacement"] = "displaced"
            else:
                # Ambiguous displacement: forbidden from picking displaced unless documented
                forbidden_attrs.append("displaced")

        # 5. Gastritis Bleeding Guardrail
        if concept.disease_family == "gastritis":
            if "without_bleeding" in concept.supported_attributes:
                forbidden_attrs.append("with bleeding")
                required_attrs["bleeding"] = False
            elif "with_bleeding" in concept.supported_attributes:
                forbidden_attrs.append("without bleeding")
                required_attrs["bleeding"] = True

        return MatchSpec(
            concept_id=concept.concept_id,
            canonical_name=concept.canonical_name,
            disease_family=concept.disease_family,
            allowed_code_families=allowed_prefixes,
            required_attributes=required_attrs,
            forbidden_attributes=forbidden_attrs,
            unknown_attributes=concept.unknown_attributes,
            target_coding_system="ICD-10-CM",
            concept_family=concept.disease_family,
            anatomical_site=concept.body_site or concept.anatomy,
            laterality=concept.laterality,
            etiology=concept.etiology,
            histology=concept.histology,
            severity=concept.severity,
            complication=concept.complication,
            temporal_status=concept.temporality.value if hasattr(concept.temporality, "value") else str(concept.temporality),
            encounter_context=concept.encounter_context,
            relationship_constraints=[],
            coding_system="ICD-10-CM",
        )
