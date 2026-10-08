"""Reverse Attribute Checker for Database-Authoritative ICD-10-CM & CPT Validation.

Extracts clinically meaningful qualifiers and attributes from candidate code descriptions
and validates that every positive attribute is explicitly supported by documented patient evidence.
Enforces the fundamental clinical coding invariant:
    CODE_SPECIFICITY <= EVIDENCE_SPECIFICITY
"""

import re

from medical_coding.retrieval.tokenizer import CLINICAL_MORPHOLOGY
from medical_coding.utils.logging import get_logger

logger = get_logger(__name__)

# Non-specificity words, grammatical tokens, and broad taxonomic terms
NON_SPECIFICITY_WORDS: set[str] = {
    "a", "an", "the", "and", "or", "of", "in", "with", "without", "due", "to",
    "for", "on", "by", "as", "at", "from", "unspecified", "nos", "nec", "not",
    "elsewhere", "classified", "organism", "site", "type", "side", "cause",
    "causes", "other", "part", "parts", "multiple", "single", "disease", "diseases",
    "disorder", "disorders", "condition", "conditions", "tract", "lower", "upper",
    "congestive", "mention", "evidence", "specified", "syndrome", "status", "post",
    "body", "area", "region", "agent", "agents", "manifestation", "manifestations",
    "infection", "infections", "primary", "mellitus", "initial", "encounter",
    "closed", "bone", "bones", "female",
    "acute", "chronic", "subacute", "spontaneous", "disruption", "injury", "sites",
    "secondary", "neoplasm", "neoplasms", "tumor", "tumour", "intrahepatic", "bile", "duct",
    "coronary", "artery", "arteries", "vessel", "vessels", "wall", "vein", "veins",
    "essential", "uncomplicated",
}

# Clinical attribute qualifiers mapped to documentary confirmation tokens/synonyms
CLINICAL_QUALIFIERS: dict[str, list[str]] = {
    # Fractures & Displacement
    "nondisplaced": ["nondisplaced", "non-displaced", "no displacement", "undisplaced", "not displaced"],
    "displaced": ["displaced", "displacement"],

    # Complications & Manifestations
    "bleeding": ["bleed", "bleeding", "hemorrhage", "hemorrhagic", "melena", "hematemesis", "blood in stool"],
    "perforation": ["perforat", "rupture", "free air", "hole"],
    "obstruction": ["obstruct", "blockage", "stricture", "impaction", "occlu"],
    "gangrene": ["gangrene", "gangrenous", "necrosis", "necrotic"],
    "ulceration": ["ulcer", "ulcerated", "ulcerative", "erosion"],
    "abscess": ["abscess", "empyema", "purulent collection"],
    "fistula": ["fistula", "fistulous"],
    "hemorrhage": ["hemorrhage", "hemorrhagic", "bleed"],
    "colic": ["colic", "colicky"],

    # Pathological Subtypes & Features
    "atrophic": ["atrophic", "atrophy"],
    "superficial": ["superficial"],
    "eosinophilic": ["eosinophil", "eosinophilic"],
    "hypertrophic": ["hypertrophic", "hypertrophy", "thickened"],
    "granulomatous": ["granuloma", "granulomatous"],
    "metaplastic": ["metaplasia", "metaplastic"],
    "polypoid": ["polyp", "polypoid"],

    # Heart Failure Subtypes & Specific Features
    "systolic": ["systolic", "hfref", "reduced ejection", "systolic dysfunction"],
    "diastolic": ["diastolic", "hfpef", "preserved ejection", "diastolic dysfunction"],
    "right heart": ["right heart", "right ventricular", "right side heart"],
    "left heart": ["left heart", "left ventricular", "left side heart"],
    "high output": ["high output"],
    "end stage": ["end stage", "stage d", "terminal heart failure"],
    "biventricular": ["biventricular"],

    # Underlying Etiology & Special Circumstances
    "transplant": ["transplant", "allograft", "graft", "organ donor"],
    "rheumatic": ["rheumatic", "rheumatic fever"],
    "hypertensive": ["hypertension", "hypertensive", "htn", "high blood pressure"],
    "postprocedural": ["postprocedural", "postoperative", "post-op", "complication of surgery", "following surgery"],
    "congenital": ["congenital", "birth defect", "anomaly", "inborn"],
    "toxic": ["toxic", "toxicity", "poisoning", "chemical exposure"],
    "alcoholic": ["alcohol", "alcoholic", "etoh", "substance abuse"],
    "diabetic": ["diabetes", "diabetic", "dm", "t1dm", "t2dm"],

    # Diabetic Complications & Specificity
    "nephropathy": ["nephropathy", "kidney disease", "renal disease", "ckd", "proteinuria"],
    "retinopathy": ["retinopathy", "maculopathy", "proliferative"],
    "neuropathy": ["neuropathy", "polyneuropathy", "mononeuropathy", "autonomic"],
    "ketoacidosis": ["ketoacidosis", "dka", "ketosis"],
    "hyperglycemia": ["hyperglycemia", "high blood sugar", "uncontrolled"],
    "hypoglycemia": ["hypoglycemia", "low blood sugar", "hypoglycemic"],
    "dermatitis": ["dermatitis", "skin lesion", "necrobiosis"],
    "cataract": ["cataract"],
    "amyotrophy": ["amyotrophy"],
    "arthropathy": ["arthropathy", "charcot"],

    # Specific Anatomical Sites & Lymph Node Regions
    "intrathoracic": ["intrathoracic", "mediastinal", "mediastinum", "hilum", "hilar"],
    "head, face, and neck": ["head", "face", "neck", "cervical"],
    "intra-abdominal": ["intra-abdominal", "abdominal lymph", "mesenteric", "retroperitoneal"],
    "axilla": ["axilla", "axillary"],
    "inguinal": ["inguinal", "groin"],
    "spleen": ["spleen", "splenic"],

    # Specific Pathogens & Infectious Etiologies
    "chlamydial": ["chlamydia", "chlamydial"],
    "rubella": ["rubella"],
    "salmonella": ["salmonella"],
    "typhoid": ["typhoid"],
    "gonococcal": ["gonococcal", "gonorrhea"],
    "streptococcal": ["streptococc", "strep", "streptococcus"],
    "streptococcus": ["streptococc", "strep", "streptococcus"],
    "pseudomonas": ["pseudomonas"],
    "klebsiella": ["klebsiella"],
    "mycoplasma": ["mycoplasma"],
    "adenoviral": ["adenovirus", "adenoviral"],
    "varicella": ["varicella", "chickenpox"],
    "staphylococcal": ["staphylococc", "staph", "mrsa", "mssa"],
    "influenza": ["influenza", "flu"],
    "fungal": ["fungal", "fungus", "mycotic", "aspergillus", "candida"],
    "aspiration": ["aspiration", "aspirated"],
    "tuberculous": ["tuberculosis", "tb", "tuberculous"],
    "echovirus": ["echovirus"],
    "coxsackievirus": ["coxsackievirus", "coxsackie"],
    "rhinovirus": ["rhinovirus"],
}


def has_positive_mention(synonym: str, text: str) -> bool:
    """Check if synonym appears in text without being negated."""
    syn_esc = re.escape(synonym)
    pattern = rf"\b{syn_esc}\b"
    for match in re.finditer(pattern, text, re.IGNORECASE):
        start = match.start()
        lookback_start = max(0, start - 45)
        preceding = text[lookback_start:start]
        # Inspect after last punctuation if present
        last_punct = max(preceding.rfind("."), preceding.rfind(";"), preceding.rfind(","))
        if last_punct != -1:
            preceding = preceding[last_punct + 1 :]

        # Check if preceding snippet contains negation words
        if re.search(
            r"\b(?:no|not|without|denies|denied|negative\s+for|no\s+evidence\s+of|no\s+active|ruled\s+out|free\s+of)\b",
            preceding,
            re.IGNORECASE,
        ):
            continue
        if preceding.rstrip().endswith("non-") or preceding.rstrip().endswith("non"):
            continue

        return True
    return False


class ReverseAttributeChecker:
    """Evaluates whether attributes in a database code description are supported by evidence."""

    @classmethod
    def extract_code_attributes(cls, code_description: str) -> dict[str, bool]:
        """Extract explicit clinical attributes implied by a database description.

        Returns a mapping of attribute -> is_positive (True = must be documented).
        Attributes preceded by 'without' or 'unspecified' are NOT required to be documented.
        """
        desc_lower = code_description.lower()
        attributes: dict[str, bool] = {}

        for attr, _ in CLINICAL_QUALIFIERS.items():
            if re.search(rf"\b{re.escape(attr)}\b", desc_lower):
                without_pattern = rf"\bwithout\s+(?:mention\s+of\s+|evidence\s+of\s+|any\s+)?{re.escape(attr)}\b"
                if re.search(without_pattern, desc_lower):
                    attributes[attr] = False
                else:
                    attributes[attr] = True

        return attributes

    @classmethod
    def validate_laterality(
        cls,
        code_description: str,
        evidence_text: str,
        diagnosis_term: str = "",
    ) -> tuple[bool, str | None]:
        """Verify that anatomical laterality in code description is documented in evidence."""
        desc_lower = code_description.lower()
        combined = f"{diagnosis_term} {evidence_text}".lower()

        # Remove 'right heart' / 'left heart' to avoid confusion with limb/organ laterality
        desc_clean = re.sub(r"\b(?:right|left)\s+heart\b", "", desc_lower)

        req_bilateral = bool(
            re.search(r"\bbilateral\b", desc_clean)
            and not re.search(r"\bwithout\s+bilateral\b", desc_clean)
        )
        req_right = bool(
            re.search(
                r"\b(?:right\s+side|of\s+right|in\s+right|right\s+(?:kidney|ureter|leg|arm|lung|flank|ear|eye|ovary|testis|foot|hand|shoulder|hip|knee))\b",
                desc_clean,
            )
            or desc_clean.endswith(", right")
            or desc_clean.endswith(" right side")
            or ", right " in desc_clean
            or " right," in desc_clean
        )
        req_left = bool(
            re.search(
                r"\b(?:left\s+side|of\s+left|in\s+left|left\s+(?:kidney|ureter|leg|arm|lung|flank|ear|eye|ovary|testis|foot|hand|shoulder|hip|knee))\b",
                desc_clean,
            )
            or desc_clean.endswith(", left")
            or desc_clean.endswith(" left side")
            or ", left " in desc_clean
            or " left," in desc_clean
        )

        if not req_right and not req_left and not req_bilateral:
            return True, None

        doc_right = has_positive_mention("right", combined) or has_positive_mention("rt", combined)
        doc_left = has_positive_mention("left", combined) or has_positive_mention("lt", combined)
        doc_bilateral = has_positive_mention("bilateral", combined) or (doc_right and doc_left)

        if req_bilateral:
            if doc_bilateral:
                return True, None
            return False, "bilateral laterality"

        if req_right:
            if doc_right and not doc_left:
                return True, None
            if not doc_right:
                return False, "right laterality"
            if doc_left:
                return False, "conflicting left laterality"

        if req_left:
            if doc_left and not doc_right:
                return True, None
            if not doc_left:
                return False, "left laterality"
            if doc_right:
                return False, "conflicting right laterality"

        return True, None

    @classmethod
    def validate_code_attributes(
        cls,
        code_description: str,
        evidence_text: str,
        diagnosis_term: str = "",
    ) -> tuple[bool, list[str]]:
        """Validate that all positive attributes in the code description are evidenced.

        Args:
            code_description: Exact description from the database.
            evidence_text: Verbatim clinical documentation / evidence quote.
            diagnosis_term: Extracted normalized clinical diagnosis term.

        Returns:
            Tuple of (is_valid: bool, unsupported_attributes: list[str]).
        """
        combined_evidence = f"{diagnosis_term} {evidence_text}".lower()
        implied_attributes = cls.extract_code_attributes(code_description)
        unsupported: list[str] = []

        for attr, is_positive in implied_attributes.items():
            if not is_positive:
                continue

            synonyms = CLINICAL_QUALIFIERS.get(attr, [attr])
            has_support = any(
                has_positive_mention(s, combined_evidence) for s in synonyms
            )
            if not has_support:
                unsupported.append(attr)

        # Laterality validation
        lat_valid, lat_error = cls.validate_laterality(
            code_description=code_description,
            evidence_text=evidence_text,
            diagnosis_term=diagnosis_term,
        )
        if not lat_valid and lat_error:
            unsupported.append(lat_error)

        # Diabetes Type 1 vs Type 2 distinction
        desc_lower = code_description.lower()
        if re.search(r"\btype\s+1\b", desc_lower) and not re.search(
            r"\b(?:type\s+1|t1dm|type\s+i\b)", combined_evidence
        ):
            unsupported.append("type 1")
        if re.search(r"\btype\s+2\b", desc_lower) and not re.search(
            r"\b(?:type\s+2|t2dm|type\s+ii\b)", combined_evidence
        ):
            unsupported.append("type 2")

        # Male vs Female breast/genitourinary specificity (Master Prompt Section 5)
        if re.search(r"\bmale\b", desc_lower) and not re.search(
            r"\b(?:male|man|men|boy|gentleman)\b", combined_evidence
        ):
            unsupported.append("male")

        # Universal Clinical Modifier Entailment (CODE_SPECIFICITY <= EVIDENCE_SPECIFICITY)
        # Strips negative 'without' expressions so absences are not required
        desc_clean = re.sub(
            r"\bwithout\s+(?:mention\s+of\s+|evidence\s+of\s+|any\s+)?[a-z0-9\-]+(?:\s+[a-z0-9\-]+)?\b",
            "",
            desc_lower,
        )
        # Strip parenthetical nonessential modifiers and bracketed synonyms (CMS Guidelines I.A.4 & I.A.7)
        desc_clean = re.sub(r"[\(\[][^\)\]]*[\)\]]", "", desc_clean)
        # Strip parent taxonomy phrases for well-defined syndromic terms
        # E.g. "Severe sepsis with septic shock" -> septic shock is the clinical manifestation
        if re.search(r"\bseptic\s+shock\b", combined_evidence):
            desc_clean = re.sub(r"\bsevere\s+sepsis\s+with\s+septic\s+shock\b", "septic shock", desc_clean)
        raw_code_tokens = re.findall(r"[a-z0-9]+", desc_clean)
        code_modifiers = [
            w for w in raw_code_tokens if len(w) > 2 and w not in NON_SPECIFICITY_WORDS
        ]

        from medical_coding.retrieval.tokenizer import CLINICAL_ABBREVIATIONS

        for mod in code_modifiers:
            if mod in unsupported:
                continue
            variants = [mod] + CLINICAL_MORPHOLOGY.get(mod, [])
            if mod in CLINICAL_ABBREVIATIONS:
                variants.extend(CLINICAL_ABBREVIATIONS[mod].split())
            if mod.endswith("s") and len(mod) > 4:
                variants.append(mod[:-1])
            if mod.endswith("ic") and len(mod) > 4:
                variants.append(mod[:-2])
            if mod.endswith("al") and len(mod) > 4:
                variants.append(mod[:-2])

            supported = any(has_positive_mention(v, combined_evidence) for v in variants)
            if not supported:
                unsupported.append(mod)

        return (len(unsupported) == 0, unsupported)

    @classmethod
    def get_specificity_penalty(
        cls,
        code_description: str,
        evidence_text: str,
        diagnosis_term: str = "",
    ) -> float:
        """Calculate penalty (0.0 to 0.95) for unsupported attributes.

        If any positive attribute is unsupported, returns a severe penalty (0.90)
        effectively disqualifying the candidate in competition against supported codes.
        """
        is_valid, unsupported = cls.validate_code_attributes(
            code_description=code_description,
            evidence_text=evidence_text,
            diagnosis_term=diagnosis_term,
        )
        if not is_valid:
            return 0.90
        return 0.0
