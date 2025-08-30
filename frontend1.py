
from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import streamlit as st
from PIL import Image

# Optional imports guarded for environments without system deps
try:
    import pytesseract  # type: ignore
except Exception:  # pragma: no cover
    pytesseract = None

try:
    from pdf2image import convert_from_bytes  # type: ignore
except Exception:  # pragma: no cover
    convert_from_bytes = None

try:
    from rapidfuzz import process as rf_process, fuzz as rf_fuzz  # type: ignore
except Exception:  # pragma: no cover
    rf_process = None
    rf_fuzz = None

# -----------------------------
# Domain data (toy formulary)
# -----------------------------
FORMULARY: Dict[str, Dict] = {
    "amoxicillin": {
        "max_daily_mg": 4000,
        "indications": ["otitis media", "sinusitis", "pharyngitis"],
        "adult_usual_mg_per_dose": (250, 1000),
        "interactions": ["warfarin"],
        "contra": ["severe penicillin allergy"],
    },
    "ibuprofen": {
        "max_daily_mg": 3200,
        "indications": ["pain", "fever", "inflammation"],
        "adult_usual_mg_per_dose": (200, 800),
        "interactions": ["warfarin"],
        "contra": ["pregnancy (3rd trimester)", "active GI bleed"],
    },
    "warfarin": {
        "max_daily_mg": 10,
        "indications": ["AFib", "DVT/PE"],
        "adult_usual_mg_per_dose": (1, 10),
        "interactions": ["amoxicillin", "ibuprofen"],
        "contra": ["pregnancy"],
    },
    "paracetamol": {
        "aka": ["acetaminophen", "pcm", "tylenol"],
        "max_daily_mg": 4000,
        "indications": ["fever", "pain"],
        "adult_usual_mg_per_dose": (325, 1000),
        "interactions": [],
        "contra": ["severe liver disease"],
    },
}

CONTROLLED_FLAGS = {
    # toy list; in real app wire to a national controlled substances DB
    "tramadol": "Schedule H1 (varies by country)",
}

# -----------------------------
# Data structures
# -----------------------------
@dataclass
class DrugOrder:
    raw: str
    name: str
    dose_mg: Optional[int] = None
    route: Optional[str] = None
    freq: Optional[str] = None
    duration: Optional[str] = None

@dataclass
class Prescription:
    patient_name: Optional[str]
    prescriber: Optional[str]
    date: Optional[datetime]
    dx: Optional[str]
    drugs: List[DrugOrder]
    notes: Optional[str] = None

# -----------------------------
# Utility functions
# -----------------------------
def clean_text(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()


def ocr_extract_text(file_bytes: bytes, file_type: str) -> str:
    """OCR images or PDFs to text. Falls back to empty string if OCR not available."""
    if file_type in ("png", "jpg", "jpeg", "webp"):
        if pytesseract is None:
            return ""
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return pytesseract.image_to_string(img)

    if file_type == "pdf":
        if convert_from_bytes is None or pytesseract is None:
            return ""
        pages = convert_from_bytes(file_bytes)
        texts = [pytesseract.image_to_string(p.convert("RGB")) for p in pages]
        return "\n\n".join(texts)

    return ""


MED_NAME_CANON = {k: k for k in FORMULARY.keys()}
for k, v in FORMULARY.items():
    for aka in v.get("aka", []):
        MED_NAME_CANON[aka] = k


def fuzzy_med_lookup(token: str) -> Optional[str]:
    token_l = token.lower()
    if token_l in MED_NAME_CANON:
        return MED_NAME_CANON[token_l]
    # fuzzy fallback
    if rf_process and rf_fuzz:
        choices = list(MED_NAME_CANON.keys())
        best, score, _ = rf_process.extractOne(token_l, choices, scorer=rf_fuzz.token_sort_ratio)
        if score >= 85:
            return MED_NAME_CANON[best]
    return None


DRUG_LINE_RE = re.compile(
    r"^(?P<name>[A-Za-z][A-Za-z\-/ ]{1,40})\s*(?P<dose>\d{2,4})?\s*(?P<unit>mg|mcg|g)?\s*(?P<route>po|iv|im|top|pr|sc)?\s*(?P<freq>od|bd|tds|qid|q\d+h|prn)?\s*(?P<duration>\d+\s*(day|days|wk|wks|week|weeks))?",
    re.IGNORECASE,
)


def parse_prescription(text: str) -> Prescription:
    # Basic fields
    name = None
    prescriber = None
    dx = None
    date_val = None

    name_m = re.search(r"Patient[:\-]\s*([A-Za-z ]{2,60})", text, re.IGNORECASE)
    if name_m:
        name = clean_text(name_m.group(1))

    doc_m = re.search(r"(Dr\.?\s*[A-Za-z .]{2,60})", text)
    if doc_m:
        prescriber = clean_text(doc_m.group(1))

    dx_m = re.search(r"Dx[:\-]\s*([A-Za-z ,]{2,120})", text, re.IGNORECASE)
    if dx_m:
        dx = clean_text(dx_m.group(1))

    date_m = re.search(r"(\d{1,2}[\-/]\d{1,2}[\-/]\d{2,4})", text)
    if date_m:
        for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%d-%m-%y", "%d/%m/%y"):
            try:
                date_val = datetime.strptime(date_m.group(1), fmt)
                break
            except ValueError:
                continue

    # Drug lines: look for lines starting with a med-like token
    drugs: List[DrugOrder] = []
    for line in text.splitlines():
        m = DRUG_LINE_RE.match(line.strip())
        if not m:
            continue
        raw_name = clean_text(m.group("name"))
        canon = fuzzy_med_lookup(raw_name)
        if not canon:
            continue
        dose = m.group("dose")
        unit = (m.group("unit") or "").lower()
        dose_mg = None
        if dose:
            try:
                dose_val = int(dose)
                if unit == "g":
                    dose_val *= 1000
                elif unit == "mcg":
                    dose_val = max(1, dose_val // 1000)
                dose_mg = dose_val
            except Exception:
                pass
        drugs.append(
            DrugOrder(
                raw=line.strip(),
                name=canon,
                dose_mg=dose_mg,
                route=(m.group("route") or "").lower() or None,
                freq=(m.group("freq") or "").lower() or None,
                duration=(m.group("duration") or "").lower() or None,
            )
        )

    return Prescription(
        patient_name=name,
        prescriber=prescriber,
        date=date_val,
        dx=dx,
        drugs=drugs,
        notes=None,
    )


def check_interactions(drugs: List[DrugOrder]) -> List[str]:
    issues = []
    names = [d.name for d in drugs]
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            a_int = set(FORMULARY.get(a, {}).get("interactions", []))
            if b in a_int:
                issues.append(f"Potential interaction between {a} and {b} — review clinically.")
    return issues


def check_dose_limits(drugs: List[DrugOrder]) -> List[str]:
    issues = []
    for d in drugs:
        if d.dose_mg is None:
            issues.append(f"{d.name}: dose missing or unreadable.")
            continue
        max_daily = FORMULARY.get(d.name, {}).get("max_daily_mg")
        if max_daily and d.dose_mg > max_daily:
            issues.append(
                f"{d.name}: dose {d.dose_mg} mg exceeds adult max daily ({max_daily} mg)."
            )
        usual = FORMULARY.get(d.name, {}).get("adult_usual_mg_per_dose")
        if usual:
            lo, hi = usual
            if not (lo <= d.dose_mg <= hi):
                issues.append(
                    f"{d.name}: {d.dose_mg} mg is outside usual per-dose range {lo}-{hi} mg."
                )
    return issues


def check_admin(drugs: List[DrugOrder]) -> List[str]:
    issues = []
    for d in drugs:
        if not d.freq:
            issues.append(f"{d.name}: missing frequency (e.g., bd/tds/q12h/prn).")
        if d.route and d.route not in {"po", "iv", "im", "top", "pr", "sc"}:
            issues.append(f"{d.name}: unrecognized route '{d.route}'.")
        if CONTROLLED_FLAGS.get(d.name):
            issues.append(f"{d.name}: controlled substance flag — {CONTROLLED_FLAGS[d.name]}.")
    return issues


def check_metadata(p: Prescription) -> List[str]:
    issues = []
    if not p.patient_name:
        issues.append("Missing patient name.")
    if not p.prescriber:
        issues.append("Missing prescriber name.")
    if not p.date:
        issues.append("Missing or unreadable date.")
    else:
        if p.date > datetime.now() + timedelta(days=1):
            issues.append("Date appears in the future.")
        if p.date < datetime.now() - timedelta(days=180):
            issues.append("Prescription older than 180 days — may be invalid.")
    if len(p.drugs) == 0:
        issues.append("No recognizable medications found.")
    return issues

def ai_reasoning_summary(text: str, p: Prescription) -> str:
    # Heuristic summary as a stand-in for LLM
    lines = [
        "AI reasoning (heuristic demo):",
        f"- Parsed {len(p.drugs)} medication(s).",
    ]
    if p.dx:
        lines.append(f"- Diagnosis noted: {p.dx}.")
    # naive alignment check between Dx and common indications
    if p.dx and p.drugs:
        off_label = []
        for d in p.drugs:
            inds = FORMULARY.get(d.name, {}).get("indications", [])
            if not any(term.lower() in p.dx.lower() for term in inds):
                off_label.append(d.name)
        if off_label:
            lines.append("- Potential off-label or unclear indication: " + ", ".join(sorted(set(off_label))) + ".")
        else:
            lines.append("- All meds plausibly align with the documented diagnosis.")
    if len(text) < 20:
        lines.append("- OCR text very short; confidence low.")
    return "\n".join(lines)

def build_report(p: Prescription, source_text: str) -> Dict:
    issues = (
        check_metadata(p)
        + check_dose_limits(p.drugs)
        + check_interactions(p.drugs)
        + check_admin(p.drugs)
    )
    status = "PASS" if len(issues) == 0 else "REVIEW"
    return {
        "status": status,
        "extracted": {
            "patient_name": p.patient_name,
            "prescriber": p.prescriber,
            "date": p.date.isoformat() if p.date else None,
            "diagnosis": p.dx,
            "drugs": [d.__dict__ for d in p.drugs],
        },
        "issues": issues,
        "ai_summary": ai_reasoning_summary(source_text, p),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "disclaimer": "This tool is for triage/verification assistance only and is not a substitute for professional judgment.",
    }


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="AI Prescription Verification", page_icon="🩺", layout="wide")

st.title("🩺 AI Medical Prescription Verification (Demo)")
st.caption(
    "Upload a prescription (image/PDF) or paste text. The app will OCR, parse, and run safety checks."
)

with st.sidebar:
    st.header("Input source")
    uploaded = st.file_uploader("Upload prescription (PDF/PNG/JPG)", type=["pdf", "png", "jpg", "jpeg", "webp"])  # noqa: E501
    st.markdown("**— or —**")
    text_input = st.text_area("Paste prescription text", height=160, placeholder=(
        "e.g.\nPatient: Jane Doe\nDx: otitis media\nDr. A. Physician\n12/08/2025\n\nAmoxicillin 500 mg po tds 7 days\nIbuprofen 400 mg po tds prn pain 5 days"
    ))

    st.divider()
    st.subheader("Export")
    want_json = st.checkbox("Enable JSON report download", value=True)

# Determine text
ocr_text = ""
file_type = None
if uploaded is not None:
    file_type = uploaded.name.split(".")[-1].lower()
    file_bytes = uploaded.read()
    with st.spinner("Running OCR…"):
        ocr_text = ocr_extract_text(file_bytes, file_type)
        if not ocr_text:
            st.info("OCR unavailable or produced no text. You can paste text manually in the sidebar.")

source_text = text_input.strip() or ocr_text.strip()

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Source Text")
    if source_text:
        st.code(source_text, language="text")
    else:
        st.warning("Provide an input to proceed.")

with col2:
    st.subheader("Parsed Summary")
    if source_text:
        p = parse_prescription(source_text)
        st.markdown(
            f"**Patient:** {p.patient_name or '—'}  \n**Prescriber:** {p.prescriber or '—'}  \n**Date:** {p.date.date() if p.date else '—'}  \n**Diagnosis:** {p.dx or '—'}"
        )
        if p.drugs:
            st.markdown("**Medications**")
            for d in p.drugs:
                st.markdown(
                    f"• **{d.name}** — {d.dose_mg or '?'} mg {d.route or ''} {d.freq or ''} {d.duration or ''}  \n"
                    f"  _line:_ `{d.raw}`"
                )
        else:
            st.info("No medications parsed yet.")

st.divider()

if source_text:
    report = build_report(p, source_text)
    status = report["status"]
    if status == "PASS":
        st.success("No issues detected in automated checks. Please still review clinically.")
    else:
        st.error("Review required — potential issues detected. See details below.")

    with st.expander("Details: Checks & Flags", expanded=True):
        for i, issue in enumerate(report["issues"], 1):
            st.markdown(f"{i}. {issue}")

    with st.expander("AI Reasoning", expanded=False):
        st.text(report["ai_summary"]) 

    st.caption(report["disclaimer"])

    if want_json:
        st.download_button(
            "Download JSON Report",
            data=json.dumps(report, indent=2).encode("utf-8"),
            file_name="prescription_verification_report.json",
            mime="application/json",
        )

# Footer
st.markdown("---")
st.markdown(
    "Built as an example. For production: add authentication, audit logging, secure PHI handling, and connect to authoritative drug/interaction databases."
)
