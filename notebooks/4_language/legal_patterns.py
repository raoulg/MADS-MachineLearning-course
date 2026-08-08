"""Designed features for US Supreme Court opinions.

This module exists for the same reason `2_convolutions/cnn.py` does: two
notebooks (04_composition) need the same definition, and pasting it
into both guarantees they drift apart silently (lesson2.yml C10).

It is the US counterpart of vectormesh's Dutch `build_legal_reference_pattern`
(the `aktes` example): the same two-function protocol — a pattern that finds
candidate matches, and a harmonizer that collapses surface variants into one
feature name — aimed at what a US opinion actually contains:

- case citations by reporter    "347 U.S. 483"       -> cite:US
- statutes                      "28 U.S.C. § 1291"   -> cite:USC
- constitutional amendments     "Fourteenth Amendment" -> amend:fourteenth
- procedural posture            "certiorari", "remanded", "dissenting"
- doctrinal vocabulary          "due process", "antitrust", "equal protection"

Note what the harmonizer throws away on purpose: "347 U.S. 483" and
"410 U.S. 113" both become `cite:US`. The feature is *that a reporter is
cited*, not which case — keeping every citation distinct would give a feature
per case, and no feature would clear `min_doc_frequency`.
"""

import re


def build_scotus_pattern() -> re.Pattern:
    """Match citations, statutes, amendments and legal vocabulary."""
    # 347 U.S. 483  |  67 S.Ct. 1  |  91 L.Ed. 22  |  410 F.2d 701
    reporter = r"\d+\s+(U\.\s?S\.|S\.\s?Ct\.|L\.\s?Ed\.(?:\s?2d)?|F\.\s?[23]d|F\.\s?Supp\.)\s+\d+"
    # 28 U.S.C. § 1291  |  42 U. S. C. §1983
    statute = r"\d+\s+U\.\s?S\.\s?C\.\s?§+\s?\d+"
    amendment = (
        r"(First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth|Ninth|Tenth|"
        r"Eleventh|Twelfth|Thirteenth|Fourteenth|Fifteenth|Sixteenth)\s+Amendment"
    )
    # what the Court is doing, procedurally
    posture = (
        r"(certiorari|writ of habeas corpus|per curiam|remanded?|affirmed?|"
        r"reversed?|vacated?|dissent(?:ing)?|concurr(?:ing|ence)|"
        r"summary judgment|preliminary injunction|class action|"
        r"stare decisis|de novo|harmless error|standing to sue)"
    )
    # subject-matter vocabulary that separates issue areas
    subject = (
        r"(due process|equal protection|commerce clause|establishment clause|"
        r"free exercise|freedom of speech|search and seizure|probable cause|"
        r"double jeopardy|self-incrimination|antitrust|patent|trademark|"
        r"collective bargaining|labor union|taxable income|deportation|"
        r"immigration|sovereign immunity|eminent domain|voting rights|"
        r"capital punishment|death penalty|attorney's fees|punitive damages)"
    )
    return re.compile(
        rf"({reporter})|({statute})|({amendment})|\b{posture}\b|\b{subject}\b",
        re.IGNORECASE,
    )


def harmonize_scotus(match: tuple) -> str:
    """Collapse surface variants of a match into one readable feature name."""
    parts = [p for p in match if p]
    if not parts:
        return ""
    text = parts[0].strip()
    low = text.lower()

    if re.match(r"^\d+\s+u\.\s?s\.\s?c\.", low):
        return "cite:USC"
    m = re.search(
        r"\d+\s+(u\.\s?s\.|s\.\s?ct\.|l\.\s?ed\.(?:\s?2d)?|f\.\s?[23]d|f\.\s?supp\.)",
        low,
    )
    if m:
        return "cite:" + re.sub(r"[\s.]", "", m.group(1)).upper()
    if low.endswith("amendment"):
        return "amend:" + low.split()[0]
    for stem, canon in (
        ("remand", "remanded"),
        ("affirm", "affirmed"),
        ("revers", "reversed"),
        ("vacat", "vacated"),
        ("dissent", "dissent"),
        ("concurr", "concurrence"),
    ):
        if low.startswith(stem):
            return canon
    return re.sub(r"\s+", " ", low)
