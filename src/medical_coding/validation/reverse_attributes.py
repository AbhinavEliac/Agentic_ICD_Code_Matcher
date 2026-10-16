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
    "native", "atherosclerotic", "atherosclerosis", "arteriosclerotic", "gallbladder",
    "collapse", "failure", "pneumonia",
    "systemic",
}

# Clinical attribute qualifiers mapped to documentary confirmation tokens/synonyms
CLINICAL_QUALIFIERS: dict[str, list[str]] = {
    # Fractures & Displacement
    "nondisplaced": ["nondisplaced", "non-displaced", "no displacement", "undisplaced", "not displaced"],
    "displaced": ["displaced", "displacement"],

    # Complications & Manifestations
    "bleeding": ["bleed", "bleeding", "hemorrhage", "hemorrhagic", "melena", "hematemesis", "blood in stool"],
    "perforation": ["perforation", "perforated", "perforat", "rupture", "ruptured", "free air", "hole"],
    "obstruction": ["obstruction", "obstructive", "obstructed", "obstruct", "blockage", "stricture", "impaction", "occlusion", "occluded", "occlu"],
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
        lookback_start = max(0, start - 60)
        preceding = text[lookback_start:start]

        # Sentence boundaries stop negation scope, but commas in lists do not
        sent_boundary = max(preceding.rfind("."), preceding.rfind(";"), preceding.rfind("\n"))
        clause = preceding[sent_boundary + 1 :] if sent_boundary != -1 else preceding

        # Check if preceding snippet or clause contains negation words
        if re.search(
            r"\b(?:no|not|without|denies|denied|negative\s+for|no\s+evidence\s+of|no\s+active|ruled\s+out|free\s+of)\b",
            clause,
            re.IGNORECASE,
        ):
            continue
        if clause.rstrip().endswith("non-") or clause.rstrip().endswith("non"):
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
                r"\b(?:right\s+side|of\s+right|in\s+right|right\s+(?:female\s+)?breast|right\s+(?:kidney|ureter|leg|arm|lung|flank|ear|eye|ovary|testis|foot|hand|shoulder|hip|knee|fallopian|tube|extremity|quadrant|lobe|adrenal|femur|humerus))\b",
                desc_clean,
            )
            or re.search(r"\b(?:quadrant\s+of\s+right|of\s+right\s+female\s+breast)\b", desc_clean)
            or desc_clean.endswith(", right")
            or desc_clean.endswith(" right side")
            or ", right " in desc_clean
            or " right," in desc_clean
        )
        req_left = bool(
            re.search(
                r"\b(?:left\s+side|of\s+left|in\s+left|left\s+(?:female\s+)?breast|left\s+(?:kidney|ureter|leg|arm|lung|flank|ear|eye|ovary|testis|foot|hand|shoulder|hip|knee|fallopian|tube|extremity|quadrant|lobe|adrenal|femur|humerus))\b",
                desc_clean,
            )
            or re.search(r"\b(?:quadrant\s+of\s+left|of\s+left\s+female\s+breast)\b", desc_clean)
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
            if doc_right or doc_left:
                return False, "conflicting unilateral documentation (code requires bilateral laterality)"
            return False, "unsupported bilateral laterality"

        if req_right:
            if doc_right and not doc_left:
                return True, None
            if not doc_right:
                if doc_left:
                    return False, "conflicting left laterality (code requires right)"
                return False, "unsupported right laterality"
            if doc_left:
                return False, "conflicting left laterality"

        if req_left:
            if doc_left and not doc_right:
                return True, None
            if not doc_left:
                if doc_right:
                    return False, "conflicting right laterality (code requires left)"
                return False, "unsupported left laterality"
            if doc_right:
                return False, "conflicting right laterality"

        return True, None

    @classmethod
    def validate_anatomical_site(
        cls,
        code_description: str,
        evidence_text: str,
        diagnosis_term: str = "",
    ) -> tuple[bool, str | None]:
        """Verify that anatomical organ and quadrant in code description matches documented diagnosis."""
        desc_lower = code_description.lower()
        combined = f"{diagnosis_term} {evidence_text}".lower()

        organ_patterns: dict[str, list[str]] = {
            "breast": [r"\bbreasts?\b", r"\bmammary\b", r"\bnipple\b", r"\bareola\b"],
            "ovary": [r"\bovary\b", r"\bovaries\b", r"\bovarian\b", r"\badnexa\b", r"\badnexal\b", r"\boophor\w*"],
            "fallopian_tube": [r"\bfallopian\b", r"\bsalpinx\b", r"\bsalpingo\w*"],
            "uterus": [r"\buterus\b", r"\buterine\b", r"\bendometr\w*", r"\bmyometr\w*"],
            "cervix": [r"\bcervix\b", r"\bcervical\b"],
            "prostate": [r"\bprostate\b", r"\bprostatic\b"],
            "testis": [r"\btestis\b", r"\btesticle\b", r"\btesticular\b", r"\bscrotum\b"],
            "lung": [r"\blung\b", r"\blungs\b", r"\bbronch\w*", r"\bpleura\w*"],
            "colon": [r"\bcolon\b", r"\bcolorectal\b", r"\brectum\b", r"\brectal\b", r"\bsigmoid\b", r"\bcecum\b", r"\bappendix\b", r"\bappendiceal\b"],
            "kidney": [r"\bkidney\b", r"\brenal\b", r"\bnephr\w*"],
            "bladder": [r"\bbladder\b", r"\burinary\s+bladder\b"],
            "skin": [r"\bskin\b", r"\bcutaneous\b", r"\bmelanoma\b", r"\bepiderm\w*"],
            "thyroid": [r"\bthyroid\b"],
            "stomach": [r"\bstomach\b", r"\bgastric\b"],
            "liver": [r"\bliver\b", r"\bhepatic\b"],
            "pancreas": [r"\bpancreas\b", r"\bpancreatic\b"],
            "bone": [r"\bbones?\b", r"\bfemur\b", r"\btibia\b", r"\bhumerus\b", r"\bfibula\b", r"\bradius\b", r"\bulna\b", r"\bspines?\b", r"\blumbar\b", r"\bpelvis\b", r"\bpelvic\s+bone\b", r"\bsacrum\b", r"\bvertebra\w*\b", r"\bribs?\b", r"\bskeletal\b", r"\bosseous\b", r"\bosteoblastic\b"],
            "lymph_node": [r"\blymph\s*nodes?\b", r"\blymphadenopathy\b", r"\bretroperitoneal\b", r"\bintra-abdominal\b", r"\bintrapelvic\b", r"\bmesenteric\b", r"\baxillary\b", r"\bmediastinal\b", r"\binguinal\b"],
            "brain": [r"\bbrain\b", r"\bcerebral\b", r"\bcerebellum\b"],
            "adrenal": [r"\badrenal\b"],
            "oral": [r"\boral\b", r"\bmouth\b", r"\bbuccal\b", r"\blips?\b", r"\btongue\b", r"\bgingiv\w*", r"\bgums?\b", r"\bpalate\b", r"\bstomatitis\b"],
            "nasal": [r"\bnasal\b", r"\bnose\b", r"\bnares\b", r"\bseptum\b", r"\bsinus(?:es)?\b", r"\bsinusitis\b", r"\bturbinate\w*"],
            "pharynx": [r"\bpharyn\w*", r"\bthroat\b", r"\btonsil\w*", r"\badenoid\w*"],
            "ear": [r"\bear\b", r"\botic\b", r"\baural\b", r"\btympanic\b"],
            "eye": [r"\beye\b", r"\beyes\b", r"\bocular\b", r"\bophthalm\w*", r"\bcornea\w*", r"\bretina\w*", r"\bconjunctiv\w*"],
        }

        # Determine documented organs in diagnosis term (primary) and combined text
        doc_term_organs = {
            organ for organ, pats in organ_patterns.items()
            if any(re.search(pat, diagnosis_term.lower()) for pat in pats)
        }
        doc_organs = {
            organ for organ, pats in organ_patterns.items()
            if any(re.search(pat, combined) for pat in pats)
        }

        # Determine organs specified in code description
        code_organs = {
            organ for organ, pats in organ_patterns.items()
            if any(re.search(pat, desc_lower) for pat in pats)
        }

        # If diagnosis term itself specifies an organ, candidate code MUST match that organ if code specifies an organ
        if doc_term_organs and code_organs and not doc_term_organs.intersection(code_organs):
            gyn_organs = {"ovary", "fallopian_tube", "uterus", "cervix"}
            if not (doc_term_organs.issubset(gyn_organs) and code_organs.issubset(gyn_organs)):
                return (
                    False,
                    f"anatomical site mismatch: code is for {', '.join(sorted(code_organs))} "
                    f"but condition is for {', '.join(sorted(doc_term_organs))}",
                )

        # If documented organ is specific and code organ is entirely disjoint
        if doc_organs and code_organs and not doc_organs.intersection(code_organs):
            gyn_organs = {"ovary", "fallopian_tube", "uterus", "cervix"}
            if not (doc_organs.issubset(gyn_organs) and code_organs.issubset(gyn_organs)):
                return (
                    False,
                    f"anatomical site mismatch: code is for {', '.join(sorted(code_organs))} "
                    f"but documented condition is {', '.join(sorted(doc_organs))}",
                )

        # Enforce CODE_SPECIFICITY <= EVIDENCE_SPECIFICITY:
        # If code specifies an organ/site, evidence must substantiate at least one organ/site in that code
        if code_organs and not code_organs.intersection(doc_organs):
            return (
                False,
                f"unsupported anatomical site: code specifies '{', '.join(sorted(code_organs))}' but evidence does not document this site",
            )

        # Breast quadrants check: If code specifies a specific quadrant, evidence must support it
        quadrant_specs = {
            "upper-outer quadrant": [r"upper[- ]outer\s+quadrant", r"\buoq\b", r"upper\s+and\s+outer"],
            "upper-inner quadrant": [r"upper[- ]inner\s+quadrant", r"\buiq\b", r"upper\s+and\s+inner"],
            "lower-outer quadrant": [r"lower[- ]outer\s+quadrant", r"\bloq\b", r"lower\s+and\s+outer"],
            "lower-inner quadrant": [r"lower[- ]inner\s+quadrant", r"\bliq\b", r"lower\s+and\s+inner"],
            "nipple and areola": [r"\bnipple\b", r"\bareola\b"],
            "axillary tail": [r"axillary\s+tail", r"tail\s+of\s+spence"],
        }
        for q_name, q_pats in quadrant_specs.items():
            if any(re.search(pat, desc_lower) for pat in q_pats):
                has_q = any(re.search(pat, combined) for pat in q_pats)
                if not has_q:
                    return False, f"unsupported quadrant specificity: code specifies '{q_name}'"

        return True, None

    @classmethod
    def validate_code_attributes(
        cls,
        code_description: str,
        evidence_text: str,
        diagnosis_term: str = "",
        code: str = "",
    ) -> tuple[bool, list[str]]:
        """Validate that all positive attributes in the code description are evidenced.

        Args:
            code_description: Exact description from the database.
            evidence_text: Verbatim clinical documentation / evidence quote.
            diagnosis_term: Extracted normalized clinical diagnosis term.
            code: Optional ICD-10-CM / CPT code string.

        Returns:
            Tuple of (is_valid: bool, unsupported_attributes: list[str]).
        """
        combined_evidence = f"{diagnosis_term} {evidence_text}".lower()
        clean_code = (code or "").strip().upper()
        desc_lower = code_description.lower()
        unsupported: list[str] = []

        # 1. Demographic & Perinatal Guardrail (ICD-10-CM Chapter 16: P00-P96)
        # Chapter 16 codes are strictly for perinatal conditions (birth through day 28 of life).
        is_perinatal_code = clean_code.startswith("P") or bool(
            re.search(r"\b(?:neonatal|perinatal|newborn|fetus|fetal)\b", desc_lower)
        )
        if is_perinatal_code:
            has_adult_context = bool(
                re.search(
                    r"\b(?:adult|\d+\s*(?:years?[\s\-]old|yo|y/o|year)|type\s+2|t2dm|geriatric|elderly|gentleman|lady|man|woman)\b",
                    combined_evidence,
                )
            )
            has_neonate_evidence = bool(
                re.search(
                    r"\b(?:neonate|neonatal|newborn|day\s+of\s+life|dol\s*\d+|birth\s+weight|premature\s+infant|gestational\s+age)\b",
                    combined_evidence,
                )
            )
            if has_adult_context and not has_neonate_evidence:
                unsupported.append("demographic mismatch: perinatal/neonatal code (Chapter 16) for adult patient")
            elif not has_neonate_evidence and re.search(r"\b(?:neonatal|newborn)\b", desc_lower):
                unsupported.append("unsupported neonatal/newborn status")

        # 2. Obstetric & Pregnancy Guardrail (ICD-10-CM Chapter 15: O00-O9A, Z33-Z34)
        is_pregnancy_code = (
            clean_code.startswith("O")
            or clean_code.startswith("Z33")
            or clean_code.startswith("Z34")
            or bool(re.search(r"\b(?:incidental\s+pregnancy|in\s+pregnancy|during\s+pregnancy|childbirth|puerperium|antepartum|postpartum|gestational)\b", desc_lower))
        )
        if is_pregnancy_code:
            has_pregnancy_evidence = bool(
                re.search(
                    r"\b(?:pregnant|pregnancy|gravid|gravida|gestation|intrauterine\s+pregnancy|trimester|postpartum|puerperal|delivery|labor)\b",
                    combined_evidence,
                )
            )
            if not has_pregnancy_evidence:
                unsupported.append("unsupported pregnancy/obstetric status in non-pregnant documentation")

        # 3. Clinical Polarity & Hemodynamic Contradiction Guardrail
        # Hypotension vs Hypertension / Hypertensive Emergency
        has_hypotension = (
            has_positive_mention("hypotension", combined_evidence)
            or has_positive_mention("hypotensive", combined_evidence)
            or has_positive_mention("low blood pressure", combined_evidence)
            or bool(re.search(r"\bbp\s*(?:of\s*)?(?:6\d|7\d|8\d)/\d+", combined_evidence))
        )
        is_hypertension_code = (
            clean_code.startswith("I16")
            or clean_code == "I10"
            or bool(re.search(r"\b(?:hypertensive\s+emergency|hypertensive\s+crisis|urgency|essential\s+hypertension)\b", desc_lower))
        )
        if is_hypertension_code:
            has_hypertension_doc = (
                has_positive_mention("hypertension", combined_evidence)
                or has_positive_mention("hypertensive", combined_evidence)
                or has_positive_mention("htn", combined_evidence)
                or has_positive_mention("high blood pressure", combined_evidence)
            )
            if clean_code.startswith("I16") or "emergency" in desc_lower or "crisis" in desc_lower:
                has_crisis_doc = bool(re.search(r"\b(?:hypertensive\s+(?:emergency|crisis|urgency)|malignant\s+hypertension)\b", combined_evidence))
                if not has_crisis_doc:
                    unsupported.append("unsupported hypertensive emergency/crisis (requires documented crisis)")
            if has_hypotension and not has_hypertension_doc:
                unsupported.append("clinical polarity conflict: hypertension code assigned when evidence documents hypotension")

        # Shock confirmation vs documented contradiction / stability
        is_shock_code = (
            clean_code.startswith("R57")
            or bool(re.search(r"\bshock\b", desc_lower) and not re.search(r"\b(?:without|no)\s+shock\b", desc_lower))
        )
        if is_shock_code:
            has_shock_doc = has_positive_mention("shock", combined_evidence)
            has_shock_contradiction = bool(
                re.search(
                    r"\b(?:no\s+shock|normotensive\s+without\s+shock|hemodynamically\s+stable|ruled\s+out\s+shock|shock\s+ruled\s+out|no\s+evidence\s+of\s+shock|stable\s+without\s+shock|not\s+in\s+shock|stable\s+hemodynamics|bp\s+normalized|normotensive)\b",
                    combined_evidence,
                )
            )
            if not has_shock_doc:
                unsupported.append("unsupported shock: shock code assigned without affirmative diagnosis of shock in evidence")
            elif has_shock_contradiction:
                unsupported.append("contradictory clinical evidence: shock documented as absent/ruled out or patient hemodynamically stable")

        # Glycemic Polarity: Hypoglycemia vs Hyperglycemia
        if "hypoglycemia" in desc_lower and (has_positive_mention("hyperglycemia", combined_evidence) or has_positive_mention("dka", combined_evidence)):
            if not has_positive_mention("hypoglycemia", combined_evidence):
                unsupported.append("clinical polarity conflict: hypoglycemia code assigned when evidence documents hyperglycemia")

        # 4. Specific Pathogen & Organism Hallucination Guardrail
        # A code for a specific pathogen (e.g. Hepatitis B/C) MUST require evidence of that pathogen
        is_hepatitis_code = (
            clean_code.startswith("B15")
            or clean_code.startswith("B16")
            or clean_code.startswith("B17")
            or clean_code.startswith("B18")
            or clean_code.startswith("B19")
            or bool(re.search(r"\b(?:viral\s+hepatitis|hepatitis\s+[abcde])\b", desc_lower))
        )
        if is_hepatitis_code:
            has_hepatitis_doc = (
                has_positive_mention("hepatitis", combined_evidence)
                or has_positive_mention("hbv", combined_evidence)
                or has_positive_mention("hcv", combined_evidence)
                or has_positive_mention("hav", combined_evidence)
            )
            if not has_hepatitis_doc:
                unsupported.append("unsupported pathogen specificity: viral hepatitis code assigned without documented hepatitis diagnosis")

        # 5. Core Anatomical Entity vs General Constitutional Symptom Guardrail
        # e.g., R39.14 (Feeling of incomplete bladder emptying) derived from "feverish feeling"
        is_urinary_code = clean_code.startswith("R39") or clean_code.startswith("N30") or clean_code.startswith("N39") or bool(
            re.search(r"\b(?:bladder|urinary|emptying|micturition|voiding|urine|dysuria)\b", desc_lower)
        )
        is_constitutional_term = bool(
            re.search(r"^\s*(?:fever|feverish|feverish\s+feeling|chills?|rigors?|malaise|fatigue|weakness|body\s+aches?)\s*$", diagnosis_term.lower().strip())
        )
        if is_urinary_code and is_constitutional_term:
            has_urinary_doc = bool(
                re.search(r"\b(?:bladder|urinary|urine|dysuria|frequency|urgency|micturition|voiding|emptying|retention)\b", combined_evidence)
            )
            if not has_urinary_doc:
                unsupported.append("core anatomical entity mismatch: urinary/bladder code assigned for constitutional symptom")

        implied_attributes = cls.extract_code_attributes(code_description)

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

        # Anatomical Site & Quadrant validation
        anat_valid, anat_error = cls.validate_anatomical_site(
            code_description=code_description,
            evidence_text=evidence_text,
            diagnosis_term=diagnosis_term,
        )
        if not anat_valid and anat_error:
            unsupported.append(anat_error)

        # Diabetes Type 1 vs Type 2 distinction
        if re.search(r"\btype\s+1\b", desc_lower) and not re.search(
            r"\b(?:type\s+1|t1dm|type\s+i\b)", combined_evidence
        ):
            unsupported.append("type 1")
        if re.search(r"\btype\s+2\b", desc_lower) and not re.search(
            r"\b(?:type\s+2|t2dm|type\s+ii\b)", combined_evidence
        ):
            unsupported.append("type 2")

        # Universal Clinical Modifier Entailment (CODE_SPECIFICITY <= EVIDENCE_SPECIFICITY)
        without_matches = re.findall(r"\bwithout\s+([a-z0-9\-]+(?:\s+[a-z0-9\-]+)?)", desc_lower)
        for w_phrase in without_matches:
            w_phrase_clean = w_phrase.strip()
            if w_phrase_clean in ("mention", "evidence", "complication", "complications"):
                continue
            if has_positive_mention(w_phrase_clean, combined_evidence):
                unsupported.append(f"without {w_phrase_clean}")

        # Strips negative 'without' expressions so absences are not required
        desc_clean = re.sub(
            r"\bwithout\s+(?:mention\s+of\s+|evidence\s+of\s+|any\s+)?[a-z0-9\-]+(?:\s+[a-z0-9\-]+)?\b",
            "",
            desc_lower,
        )
        # Strip parenthetical nonessential modifiers and bracketed synonyms (CMS Guidelines I.A.4 & I.A.7)
        desc_clean = re.sub(r"[\(\[][^\)\]]*[\)\]]", "", desc_clean)
        # Strip parent taxonomy phrases for well-defined syndromic terms
        if re.search(r"\bseptic\s+shock\b", combined_evidence):
            desc_clean = re.sub(r"\bsevere\s+sepsis\s+with\s+septic\s+shock\b", "septic shock", desc_clean)
        raw_code_tokens = re.findall(r"[a-z0-9]+", desc_clean)
        code_modifiers = [
            w for w in raw_code_tokens if len(w) > 2 and w not in NON_SPECIFICITY_WORDS
        ]

        from medical_coding.retrieval.tokenizer import CLINICAL_ABBREVIATIONS

        ORGANISM_EQUIVALENCES: dict[str, list[str]] = {
            "escherichia": ["e. coli", "e coli", "escherichia", "coliform"],
            "coli": ["e. coli", "e coli", "escherichia coli", "coli"],
            "helicobacter": ["h. pylori", "h pylori", "helicobacter"],
            "pylori": ["h. pylori", "h pylori", "helicobacter pylori", "pylori"],
            "streptococcus": ["strep", "streptococcal", "s. pneumoniae", "s. pyogenes", "streptococcus"],
            "staphylococcus": ["staph", "staphylococcal", "s. aureus", "mrsa", "mssa", "staphylococcus"],
            "pseudomonas": ["p. aeruginosa", "pseudomonal", "pseudomonas"],
            "klebsiella": ["k. pneumoniae", "klebsiella"],
            "clostridioides": ["c. diff", "c. difficile", "c difficile", "clostridioides", "clostridium"],
            "clostridium": ["c. diff", "c. difficile", "c difficile", "clostridium", "clostridioides"],
            "mycobacterium": ["tb", "tuberculosis", "m. tuberculosis", "mycobacterium"],
        }

        # Check bracketed synonyms in code description (e.g. [E. coli], [H. pylori])
        bracketed_synonyms = [b.strip() for b in re.findall(r"\[(.*?)\]", desc_lower) if b.strip()]
        bracket_supported = any(has_positive_mention(b, combined_evidence) for b in bracketed_synonyms)

        # Identify constituent tokens of qualifiers that were already verified as supported
        already_supported_qualifier_tokens: set[str] = set()
        for attr, is_positive in implied_attributes.items():
            if is_positive and attr not in unsupported:
                for tok in re.findall(r"[a-z0-9]+", attr):
                    if len(tok) > 2:
                        already_supported_qualifier_tokens.add(tok)

        ANATOMICAL_MODIFIER_EQUIVALENCES: dict[str, list[str]] = {
            "intra": ["retroperitoneal", "intra-abdominal", "intraabdominal", "abdominal"],
            "abdominal": ["retroperitoneal", "intra-abdominal", "intraabdominal", "abdomen", "abdominal", "pelvic", "pelvis"],
            "bone": ["spine", "lumbar", "pelvis", "vertebra", "vertebral", "sacrum", "rib", "femur", "tibia", "skeletal", "osseous"],
            "osseous": ["bone", "spine", "lumbar", "pelvis", "vertebra", "vertebral"],
        }

        for mod in code_modifiers:
            if mod in unsupported or mod in already_supported_qualifier_tokens:
                continue
            variants = [mod] + CLINICAL_MORPHOLOGY.get(mod, [])
            if mod in CLINICAL_ABBREVIATIONS:
                variants.extend(CLINICAL_ABBREVIATIONS[mod].split())
            if mod in ORGANISM_EQUIVALENCES:
                variants.extend(ORGANISM_EQUIVALENCES[mod])
            if mod in ANATOMICAL_MODIFIER_EQUIVALENCES:
                variants.extend(ANATOMICAL_MODIFIER_EQUIVALENCES[mod])
            if mod.endswith("s") and len(mod) > 4:
                variants.append(mod[:-1])
            if mod.endswith("ic") and len(mod) > 4:
                variants.append(mod[:-2])
            if mod.endswith("al") and len(mod) > 4:
                variants.append(mod[:-2])

            supported = any(has_positive_mention(v, combined_evidence) for v in variants)
            if not supported and bracket_supported and mod in ("escherichia", "helicobacter", "streptococcus", "staphylococcus", "pseudomonas", "klebsiella", "clostridium", "clostridioides"):
                supported = True
            if not supported:
                unsupported.append(mod)

        return (len(unsupported) == 0, unsupported)

    @classmethod
    def get_specificity_penalty(
        cls,
        code_description: str,
        evidence_text: str,
        diagnosis_term: str = "",
        code: str = "",
    ) -> float:
        """Calculate penalty (0.0 to 0.95) for unsupported attributes.

        If any positive attribute is unsupported, returns a severe penalty (0.90)
        effectively disqualifying the candidate in competition against supported codes.
        """
        is_valid, unsupported = cls.validate_code_attributes(
            code_description=code_description,
            evidence_text=evidence_text,
            diagnosis_term=diagnosis_term,
            code=code,
        )
        if not is_valid:
            return 0.90
        return 0.0
