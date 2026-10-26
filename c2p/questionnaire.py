"""Keep the questionnaire answers in one JSON file and write them into a filled copy of the PDF.

results/answers.json is the source of truth. The blank docs/questionnaire.pdf is never modified:
every call to fill_pdf() rebuilds docs/questionnaire_filled.pdf from all answers recorded so far,
so earlier answers always stay in it.
"""
import json

from pypdf import PdfReader, PdfWriter
from pypdf.generic import NameObject

from .data import RESULTS_DIR, ROOT

BLANK_PDF = ROOT / "docs" / "questionnaire.pdf"
FILLED_PDF = ROOT / "docs" / "questionnaire_filled.pdf"
ANSWERS_JSON = RESULTS_DIR / "answers.json"

# Radio options on every question, left to right: area 1 / area 2 / area 3 / not enough data
# (for Q5, Q6, Q12 they are the area pairs or trial segments in the order printed on the form).
CHOICES = ["A", "B", "C", "D"]


def load_answers():
    return json.loads(ANSWERS_JSON.read_text()) if ANSWERS_JSON.exists() else {}


def record_answer(q, choice, rows, explanation):
    """Store the answer to question q.

    choice: index 0-3 of the box to tick, in the order printed on the form.
    rows: three (mean, ci, units) tuples of strings, in the order of the table rows.
    """
    assert choice in range(4) and len(rows) == 3 and all(len(r) == 3 for r in rows)
    answers = load_answers()
    answers[str(q)] = {"choice": choice, "rows": [list(map(str, r)) for r in rows], "explanation": explanation}
    RESULTS_DIR.mkdir(exist_ok=True)
    ANSWERS_JSON.write_text(json.dumps(answers, indent=2, ensure_ascii=False))
    return answers[str(q)]


def _text_fields(q, page):
    """Field names of question q on its page, in creation order: 9 table cells (row by row), then the explanation."""
    names = []
    for a in page.get("/Annots", []):
        name = a.get_object().get("/T")
        if name and name.startswith(f"q{q}field"):
            names.append(name)
    return names


# Section 2 (meta-information): one grid of checkboxes per page, rows a..o = Q1..Q15,
# boxes numbered 1-10 (confidence, page 16), 11-20 (prior experience, page 17), 21-30 (expected consensus, page 18).
META = {"confidence": (16, 0), "experience": (17, 10), "consensus": (18, 20)}
ROWS = "abcdefghijklmno"


def record_meta(q, **ratings):
    """Store 1-10 ratings for question q, e.g. record_meta(3, confidence=5, consensus=4)."""
    answers = load_answers()
    meta = answers.setdefault("meta", {})
    for kind, value in ratings.items():
        assert kind in META and value in range(1, 11), (kind, value)
        meta.setdefault(kind, {})[str(q)] = value
    ANSWERS_JSON.write_text(json.dumps(answers, indent=2, ensure_ascii=False))
    return meta


def harvest_manual_ratings(kinds=("experience",)):
    """Copy ratings entered directly in the filled PDF into answers.json, so that a rebuild preserves them."""
    if not FILLED_PDF.exists():
        return {}
    fields = PdfReader(FILLED_PDF).get_fields() or {}
    found = {}
    for kind in kinds:
        _, offset = META[kind]
        for i, row in enumerate(ROWS):
            for r in range(1, 11):
                f = fields.get(f"{row}{offset + r}")
                if f is not None and f.get("/V") == "/Yes":
                    found.setdefault(kind, {})[i + 1] = r
    for kind, per_q in found.items():
        for q, r in per_q.items():
            record_meta(q, **{kind: r})
    return found


def _tick_meta(writer, meta):
    for kind, per_q in meta.items():
        page_no, offset = META[kind]
        page = writer.pages[page_no - 1]
        on = {f"{ROWS[int(q) - 1]}{offset + r}" for q, r in per_q.items()}
        row_of = {ROWS[int(q) - 1] for q in per_q}
        for a in page["/Annots"]:
            o = a.get_object()
            name = o.get("/T")
            if name and name[0] in row_of:          # reset the rated rows, then tick the chosen box
                state = NameObject("/Yes" if name in on else "/Off")
                o[NameObject("/V")] = state
                o[NameObject("/AS")] = state


# The form's font (Helvetica, WinAnsi encoding) has no Greek letters or math minus; swap them for plain text.
_PDF_REPLACE = {"Δt": "dt", "Δ": "delta ", "−": "-", "≥": ">=", "≤": "<=", "±": "+/-", "≈": "~", "×": "x",
                "⇔": "<->", "⇒": "->", "→": "->", "↔": "<->"}


def _pdf_safe(text):
    for k, v in _PDF_REPLACE.items():
        text = text.replace(k, v)
    return text.encode("cp1252", errors="replace").decode("cp1252")


def fill_pdf():
    """Write every recorded answer into a fresh copy of the blank questionnaire."""
    harvest_manual_ratings()                      # preserve ratings entered directly in the PDF
    answers = load_answers()
    meta = answers.pop("meta", {})
    writer = PdfWriter(clone_from=PdfReader(BLANK_PDF))
    _tick_meta(writer, meta)
    for q, ans in answers.items():
        page = writer.pages[int(q) - 1]          # question q is on page q
        fields = _text_fields(q, page)
        values = [_pdf_safe(v) for row in ans["rows"] for v in row] + [_pdf_safe(ans["explanation"])]
        writer.update_page_form_field_values(page, dict(zip(fields, values)), auto_regenerate=False)
        _tick(page, f"q{q}choice", ans["choice"])
    writer.set_need_appearances_writer(True)
    with open(FILLED_PDF, "wb") as f:
        writer.write(f)
    return FILLED_PDF


def _tick(page, name, choice):
    """Select one radio button.

    The form (made with LaTeX hyperref) has four separate widgets that all share the field name, e.g. "q1choice",
    each with its own "on" state. The chosen one is switched on and the others off.
    """
    kids = [a.get_object() for a in page["/Annots"] if a.get_object().get("/T") == name]
    assert len(kids) == 4, f"expected 4 boxes for {name}, found {len(kids)}"
    kids.sort(key=lambda k: (-round(float(k["/Rect"][3])), float(k["/Rect"][0])))   # reading order: rows, then left to right
    for i, k in enumerate(kids):
        if i == choice:
            on = NameObject(list(k["/AP"]["/N"].keys())[0])
            k[NameObject("/AS")] = on
            k[NameObject("/V")] = on
        else:                                      # leave unticked boxes as in the blank form
            k.pop(NameObject("/AS"), None)
