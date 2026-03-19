import io
import re
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from PyPDF2 import PdfReader


def extract_text_from_pdf(file_bytes: bytes, max_chars: int = 12000) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    parts = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text:
            parts.append(text)
        if sum(len(p) for p in parts) >= max_chars:
            break
    combined = "\n".join(parts).strip()
    return combined[:max_chars]


def extract_text_from_csv(file_bytes: bytes, max_chars: int = 12000) -> str:
    # Try a couple encodings to reduce chance of failure.
    for enc in ("utf-8", "utf-8-sig", "latin1"):
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding=enc)
            break
        except Exception:
            df = None
    if df is None:
        # Last attempt without specifying encoding.
        df = pd.read_csv(io.BytesIO(file_bytes))

    # Convert to a compact representation for NLP.
    # If the CSV is huge, limit rows/cols.
    df = df.head(50)
    text = df.to_csv(index=False)
    text = text.strip()
    return text[:max_chars]


def _maybe_number(x: Any) -> Optional[float]:
    try:
        s = str(x).strip()
        # Normalize 12,3 to 12.3
        s = s.replace(",", ".")
        return float(s)
    except Exception:
        return None


def extract_medical_values(text: str) -> Dict[str, str]:
    """
    Extract a few common medical values using regex.
    Returns string values (for UI/PDF friendliness).
    """
    t = (text or "").lower()

    values: Dict[str, str] = {}

    # Blood Pressure: e.g. 120/80 or BP: 120 / 80
    bp_match = re.search(r"\b(bp|blood pressure)\b[^0-9]{0,10}(\d{2,3})\s*/\s*(\d{2,3})", t)
    if bp_match:
        values["blood_pressure"] = f"{bp_match.group(2)}/{bp_match.group(3)}"
    else:
        # Sometimes just "120/80" appears.
        bp_match2 = re.search(r"\b(\d{2,3})\s*/\s*(\d{2,3})\b", t)
        if bp_match2:
            values["blood_pressure"] = f"{bp_match2.group(1)}/{bp_match2.group(2)}"

    # Glucose (fasting/random) numbers
    glucose_match = re.search(
        r"\b(glucose|blood sugar|fasting blood sugar|random blood sugar)\b[^0-9]{0,15}(\d{2,3}(\.\d+)?)",
        t,
    )
    if glucose_match:
        values["glucose_mg_dl"] = glucose_match.group(2)

    # HbA1c
    hba1c_match = re.search(r"\b(hba1c|a1c)\b[^0-9]{0,15}(\d{1,2}(\.\d+)?)", t)
    if hba1c_match:
        values["hba1c_percent"] = hba1c_match.group(2)

    # Cholesterol (total)
    chol_match = re.search(
        r"\b(total cholesterol|cholesterol)\b[^0-9]{0,15}(\d{2,3}(\.\d+)?)",
        t,
    )
    if chol_match:
        values["cholesterol_mg_dl"] = chol_match.group(2)

    # Triglycerides
    tg_match = re.search(r"\b(triglycerides|triglyceride)\b[^0-9]{0,15}(\d{2,3}(\.\d+)?)", t)
    if tg_match:
        values["triglycerides_mg_dl"] = tg_match.group(2)

    # Creatinine
    creat_match = re.search(r"\b(creatinine|serum creatinine)\b[^0-9]{0,15}(\d{1,2}(\.\d+)?)", t)
    if creat_match:
        values["creatinine_mg_dl"] = creat_match.group(2)

    # Blood Urea
    urea_match = re.search(r"\b(urea|blood urea)\b[^0-9]{0,15}(\d{1,3}(\.\d+)?)", t)
    if urea_match:
        values["blood_urea"] = urea_match.group(2)

    # Hemoglobin
    hb_match = re.search(r"\b(hemoglobin|haemoglobin|hb)\b[^0-9]{0,15}(\d{1,2}(\.\d+)?)", t)
    if hb_match:
        values["hemoglobin_g_dl"] = hb_match.group(2)

    # Sodium / Potassium
    sodium_match = re.search(r"\b(sodium|na)\b[^0-9]{0,15}(\d{2,3}(\.\d+)?)", t)
    if sodium_match:
        values["sodium_mmol_l"] = sodium_match.group(2)

    pot_match = re.search(r"\b(potassium|k)\b[^0-9]{0,15}(\d{1,3}(\.\d+)?)", t)
    if pot_match:
        values["potassium_mmol_l"] = pot_match.group(2)

    return {k: str(v).strip() for k, v in values.items() if str(v).strip()}


def build_report_summary(values: Dict[str, str], raw_text: str, max_chars: int = 1200) -> str:
    lines = []
    if values:
        lines.append("Extracted key medical values (from report text):")
        for k, v in values.items():
            lines.append(f"- {k.replace('_', ' ').title()}: {v}")
    else:
        lines.append("No key values detected by regex. Using report text excerpt for analysis.")

    excerpt = (raw_text or "").strip().replace("\r", "\n")
    if excerpt:
        excerpt = excerpt[:max_chars]
        lines.append("\nReport text excerpt:")
        lines.append(excerpt)

    return "\n".join(lines).strip()


def prepare_report_context(
    file_bytes: bytes,
    file_name: str,
    max_report_chars: int = 6000,
) -> Tuple[str, str, Dict[str, str]]:
    """
    Returns:
    - uploaded_report_summary: human readable
    - extracted_text_for_ai: text passed to the chatbot
    - extracted_values: key-value dict
    """
    lower = (file_name or "").lower()
    if lower.endswith(".pdf"):
        extracted_text = extract_text_from_pdf(file_bytes)
    elif lower.endswith(".csv"):
        extracted_text = extract_text_from_csv(file_bytes)
    else:
        extracted_text = file_bytes.decode("utf-8", errors="ignore")[:max_report_chars]

    extracted_text = extracted_text[:max_report_chars]
    values = extract_medical_values(extracted_text)
    summary = build_report_summary(values, extracted_text)
    return summary, extracted_text, values

