"""The report placeholder filler: replaces known ⟦CODE⟧ tokens, removes their highlight, leaves the rest."""
from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

from fairreturn.experiments import report_numbers as R

DOC = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/'
       'wordprocessingml/2006/main"><w:body><w:p>'
       '<w:r><w:rPr><w:b/></w:rPr><w:t xml:space="preserve">Flip rate: </w:t></w:r>'
       '<w:r><w:rPr><w:highlight w:val="yellow"/></w:rPr><w:t xml:space="preserve">⟦A2⟧</w:t></w:r>'
       '<w:r><w:rPr><w:highlight w:val="yellow"/></w:rPr><w:t xml:space="preserve">⟦A8⟧</w:t></w:r>'
       '</w:p></w:body></w:document>')


def test_fill_docx_replaces_known_codes_only():
    d = Path(tempfile.mkdtemp())
    src, dst = d / "r.docx", d / "r_filled.docx"
    with zipfile.ZipFile(src, "w") as z:
        z.writestr("[Content_Types].xml", "<Types/>")
        z.writestr("word/document.xml", DOC)
    n, still = R.fill_docx(src, dst, {"A2": "23.0%"})
    xml = zipfile.ZipFile(dst).read("word/document.xml").decode("utf-8")
    assert n == 1 and still == {"A8"}
    assert "23.0%" in xml and "⟦A2⟧" not in xml and "⟦A8⟧" in xml
    assert xml.count('w:val="yellow"') == 1          # the filled run lost its highlight, the open one kept it


def test_mock_numbers_are_refused_by_default():
    rows, warn = R.collect(allow_mock=False)
    values = {c: v for c, v, _ in rows}
    assert all(code in values for code in ("T3", "A2", "S4", "P1"))
    # whatever exists on disk now comes from the mock backend or is missing, so nothing may be filled
    assert all(values[c] is None for c in ("T3", "A2", "S4"))


def test_report_placeholders_are_known_codes():
    """Every ⟦CODE⟧ in the Word report must be one that report_numbers can fill, and each run holds whole tokens."""
    from fairreturn import config as cfg
    import re
    doc = cfg.ROOT / "report" / "CDS521_FairReturns_Report.docx"
    if not doc.exists():
        return
    xml = zipfile.ZipFile(doc).read("word/document.xml").decode("utf-8")
    in_doc = set(R.TOKEN.findall(xml))
    known = {c for c, _, _ in R.collect(allow_mock=False)[0]}
    assert in_doc and in_doc <= known, sorted(in_doc - known)
    # a token split across runs could not be filled
    assert not re.search(r"⟦[A-Z]\d*</w:t>", xml)
