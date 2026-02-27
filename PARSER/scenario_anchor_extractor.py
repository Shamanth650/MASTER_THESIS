# scenario_anchor_extractor.py
"""
Scenario anchor extraction utilities.

Purpose:
- Extract likely scenario "anchors" (scenario code + scenario name/title) from protocol text.
- Attach lightweight context (raw_block_text) and evidence (page + match) to each anchor.
- Classify anchors into ADAS vs NON-ADAS using conservative heuristics.

Design goals:
- Conservative: prefer missing an anchor over inventing one.
- Deterministic: no randomness; no LLM use here.
- Robust to OCR noise and mixed text sources.

IMPORTANT FIX (2026-01-02):
- Do NOT treat glossary abbreviations (e.g., TFCW, FCW) as scenario codes.
- Scenario codes must START WITH real Euro NCAP scenario families.

VRU NOTE (2026-01-11):
- Euro NCAP VRU Assessment Protocols commonly use scenario codes like:
    CPNA-75, CPFA-50, CPLA-25 (pedestrian)
    CBNA-50, CBDA (bicyclist)
    CMRb, CMoncoming (motorcyclist)
  These are valid scenario codes even though they don't start with "VRU".
  Therefore, strict acceptance must also include CP/CB/CM families.

IMPORTANT FIX (2026-01-02 / follow-up):
- Handle PDF text extraction line breaks where headings split across lines, e.g.:
    "Car-to-Car Rear Stationary"
    "(CCRs)"
  by merging the next line if it is a standalone "(CODE)".
- Handle the Euro NCAP scenario list format:
    "Car-to-Car Rear Stationary (CCRs) – a collision in which ..."
  (i.e., Title (CODE) followed by dash/description on the same line)
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

# Codes/labels that commonly appear as variables/events in docs (NOT scenario codes)
REJECT_CODES = {
    "ABS", "ESC", "TTC", "NHTSA", "UNECE", "ISO", "EU", "ECE", "UN",
    "KMH", "MPS", "M/S", "KPH",
    "FCW", "LKA", "LSS", "ELK", "LDW", "BSM", "DIM",
    "ACC", "AEB", "AEBS",  # system features, not scenario codes
}

# STRICT: Real Euro NCAP scenario code families (what you actually want as scenarios)
# - Car-to-Car / AEB: CCR*, CCF*, CCB*, C2C*, C2P*, C2B*
# - VRU (Vulnerable Road User): CP* (pedestrian), CB* (bicyclist), CM* (motorcyclist)
#   plus any internal synthetic "VRU_*" codes we create later.
SCENARIO_PREFIXES = ("CCR", "CCF", "CCB", "C2C", "C2P", "C2B", "VRU", "CP", "CB", "CM")

# Strict code regex: must start with one of the scenario prefixes (case-insensitive)
# and then optional letters/digits/underscore/hyphen.
# Allow slightly longer suffix to survive OCR/export quirks, while staying conservative.
STRICT_SCENARIO_CODE = re.compile(
    r"^(CCR|CCF|CCB|C2C|C2P|C2B|VRU|CP|CB|CM)[A-Za-z0-9_-]{0,20}$",
    re.IGNORECASE,
)


def _adas_family_from_code(code: str) -> Optional[str]:
    """Map a scenario code to an ADAS family label used downstream."""
    c = (code or "").strip().upper()
    if not c:
        return None
    if c.startswith(("LSS_",)):
        return "LSS"
    if c.startswith(("VRU_", "CP", "CB", "CM")):
        return "VRU"
    if c.startswith(("CCR", "CCF", "CCB", "C2C", "C2P", "C2B")):
        return "AEB"
    return None


# Page markers used by parser_1.py text export
_PAGE_MARKERS = [
    re.compile(r"---\s*PAGE\s*(\d+)\s*---", re.IGNORECASE),
    re.compile(r"\[\s*PAGE\s*(\d+)\s*\]", re.IGNORECASE),
    re.compile(r"^\s*PAGE\s*(\d+)\s*$", re.IGNORECASE),
]

# Standalone "(CODE)" line, used for two-line merge handling
_STANDALONE_PAREN_CODE = re.compile(r"^\(\s*([A-Za-z0-9_-]{3,24})\s*\)$")

# ----------------------------
# LSS anchor extraction (NEW)
# ----------------------------
# Euro NCAP LSS protocols often do NOT provide compact scenario codes like "CCRs".
# Instead, they define scenario families and sub-scenarios using headings such as:
#   - Emergency Lane Keeping (ELK): Road Edge / Solid Line / Oncoming / Overtaking
#   - Lane Keep Assist (LKA): Dashed Line / Solid Line
#   - Lane Departure Warning (LDW)
#   - Blind Spot Monitoring (BSM)
#
# We treat these as ADAS anchors, but we DO NOT relax STRICT_SCENARIO_CODE.
# Instead, we generate a stable synthetic scenario_code with prefix "LSS_...".
#
_LSS_SYSTEM_RX = re.compile(
    r"\b("
    r"Emergency\s+Lane\s+Keeping|ELK|"
    r"Lane\s+Keep\s+Assist|LKA|"
    r"Lane\s+Departure\s+Warning|LDW|"
    r"Blind\s+Spot\s+Monitoring|BSM"
    r")\b",
    re.IGNORECASE,
)

# Common sub-scenario cues in LSS protocols
_LSS_SUBSCEN_RX = re.compile(
    r"\b("
    r"Road\s*Edge|"
    r"Solid\s*Line|"
    r"Dashed\s*Line|"
    r"Oncoming\s+Vehicle|"
    r"Overtaking\s+Vehicle|"
    r"Blind[-\s]*Spot"
    r")\b",
    re.IGNORECASE,
)

# Strip numeric prefixes like "7.2.1 - Road Edge Tests"
_LSS_NUM_PREFIX_RX = re.compile(r"^\s*(?:\d+(?:\.\d+){0,3})\s*[:\-–—]\s*(.+?)\s*$")

# Strip numeric prefixes like "7.2.4.1 Road Edge tests" (no ":" / dash)
_LSS_NUM_PREFIX_BARE_RX = re.compile(r"^\s*\d+(?:\.\d+){1,4}\s+(.+?)\s*$")


# ----------------------------
# LSS anchor filtering (IMPROVED, generic)
# ----------------------------
# LSS documents contain many headings/definitions/parameter bullets mentioning ELK/LKA/LDW/BSM.
# We only want "scenario container" anchors (e.g., ELK Road Edge, LKA Solid Line, BSM scenario),
# not glossary text (e.g., "LDW means ...", "• TLDW ...", "The end of an ... test is considered ...").
#
# This gate is deliberately "scoring-based" (not PDF-specific). It prefers precision over recall
# and can be tuned via _LSS_MIN_SCORE.

_LSS_POSITIVE_PHRASES = (
    "scenario", "scenarios", "scenario path", "scenario paths",
    "test", "tests", "tests will be performed", "will be performed",
)

_LSS_NEGATIVE_PHRASES = (
    "means", "shall", "is considered", "where applicable",
    "the end of", "end of a", "end of an",
    "parameter", "time where", "width", "incremental steps", "incremental step",
    "figure", "table",
    # New (2026-01-11): common eligibility/criteria language that must NOT become an LSS scenario
    "criteria", "fulfil", "fulfills", "fulfils", "dossier", "evidence of the effectiveness",
)

_LSS_MIN_SCORE = 6


def _lss_score_title(title: str, system: Optional[str]) -> int:
    t = _clean(title).lower()
    score = 0

    # Structural / context cues
    if re.search(r"\b\d+\.\d+(\.\d+)*\b", t):  # section numbers like 7.2.4.3
        score += 2
    if any(p in t for p in _LSS_POSITIVE_PHRASES):
        score += 2

    # NOTE: "figure"/"table" captions are hard-rejected upstream; do not boost them here.

    # Strong signals: system + a concrete sub-scenario cue
    if system == "ELK" and any(x in t for x in ("road edge", "solid line", "oncoming", "overtaking")):
        score += 4
    if system == "LKA" and any(x in t for x in ("dashed line", "solid line", "single line", "dashed", "solid")):
        score += 4
    if system == "BSM" and any(x in t for x in ("blind spot", "blind spot monitoring", "target vehicle")):
        score += 4

    # Penalties: definition / variable / termination rule language
    if any(n in t for n in _LSS_NEGATIVE_PHRASES):
        score -= 4

    # Bullet parameter lines (e.g., "• TLDW ...") are almost never scenario anchors
    if t.startswith("•") or " - • " in t or "•" in t:
        score -= 3

    # LDW-only lines should not become standalone scenarios
    if system == "LDW" and not any(x in t for x in ("scenario", "scenarios", "test scenario", "tests")):
        score -= 6

    return score


def _is_valid_lss_anchor_title(title: str, system: Optional[str]) -> bool:
    # Keep it deterministic and conservative.
    return _lss_score_title(title, system) >= _LSS_MIN_SCORE


def _lss_system_from_line(line: str, prev_system: Optional[str]) -> Optional[str]:
    """Infer current LSS system context from a line and previous context."""
    line_c = _clean(line)
    if not line_c:
        return prev_system
    m = _LSS_SYSTEM_RX.search(line_c)
    if not m:
        return prev_system
    token = (m.group(1) or "").lower()
    if "emergency" in token or token == "elk":
        return "ELK"
    if "keep assist" in token or token == "lka":
        return "LKA"
    if "departure warning" in token or token == "ldw":
        return "LDW"
    if "blind" in token or token == "bsm":
        return "BSM"
    return prev_system


def _slug(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", (s or "").strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s.upper()


def _extract_lss_title(line: str, system_ctx: Optional[str] = None) -> Optional[str]:
    """Extract a clean LSS scenario title from a line."""
    line_c = _clean(line)
    if not line_c or len(line_c) < 6:
        return None

    # Hard reject captions (they create container-only outputs downstream)
    # Examples: "Figure 7-2: ELK Road Edge scenarios", "Table 7-1 ..."
    if re.match(r"(?i)^\s*(figure|table)\b", line_c):
        return None

    m = _LSS_NUM_PREFIX_RX.match(line_c)
    if m:
        line_c = _clean(m.group(1))
    else:
        m2 = _LSS_NUM_PREFIX_BARE_RX.match(line_c)
        if m2:
            line_c = _clean(m2.group(1))

    # Keep only plausible cue lines
    if not (_LSS_SUBSCEN_RX.search(line_c) or _LSS_SYSTEM_RX.search(line_c) or "test scenario" in line_c.lower()):
        return None

    # Avoid overly generic headings
    if line_c.lower() in {"test scenarios", "test scenario", "evaluation scenarios", "scenarios"}:
        return None

    # NEW: LSS-only precision gate (generic, not PDF-specific)
    # IMPORTANT: Use the caller-provided system context when available.
    # Many LSS headings (e.g., "7.2.4.1 Road Edge tests") do not contain "ELK" on the same line.
    sys_ctx = system_ctx or _lss_system_from_line(line_c, prev_system=None)
    if not _is_valid_lss_anchor_title(line_c, sys_ctx):
        return None

    return line_c


def _build_lss_code(system: str, title: str) -> str:
    # Stable and compact synthetic code (kept short to survive downstream tooling)
    return f"LSS_{_slug(system)[:8]}_{_slug(title)[:18]}"


def extract_lss_anchors(
    text: str,
    *,
    context_lines: int = 40,
    max_block_lines: int = 120,
) -> List[Dict]:
    """Extract LSS anchors from plain text using heading cues."""
    anchors: List[Dict] = []
    if not isinstance(text, str) or not text.strip():
        return anchors

    lines = text.splitlines()

    # Track page numbers across lines using markers (reuse same logic as in extract_scenario_anchors)
    pages_for_line: List[Optional[int]] = []
    current_page: Optional[int] = None
    for ln in lines:
        p = _page_from_line(ln)
        if p is not None:
            current_page = p
        pages_for_line.append(current_page)

    current_system: Optional[str] = None
    found: List[Tuple[int, str, str, str]] = []  # (line_idx, code, title, system)

    for i, raw_line in enumerate(lines):
        line = _clean(raw_line)
        if not line:
            continue

        # Update system context when we see ELK/LKA/LDW/BSM headings
        current_system = _lss_system_from_line(line, current_system)

        title = _extract_lss_title(line, current_system)
        if not title:
            continue

        # Require a system context for sub-scenarios. If none, try to infer from the same line.
        sys_guess = _lss_system_from_line(line, current_system)
        if not sys_guess:
            continue

        # Avoid anchors for generic feature mentions (e.g., "LKA system shall...")
        if " shall " in f" {title.lower()} " and not _LSS_SUBSCEN_RX.search(title):
            continue

        code = _build_lss_code(sys_guess, title)
        found.append((i, code, title, sys_guess))

    # De-duplicate by (code, title)
    seen = set()
    deduped: List[Tuple[int, str, str, str]] = []
    for i, code, title, system in found:
        key = (code.upper(), title.lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append((i, code, title, system))

    # Build lightweight blocks
    for idx, (i, code, title, system) in enumerate(deduped):
        next_i = deduped[idx + 1][0] if idx + 1 < len(deduped) else len(lines)
        start = max(0, i - context_lines)
        end = min(len(lines), i + max_block_lines)
        end = min(end, next_i)

        block = "\n".join(lines[start:end]).strip()
        page = pages_for_line[i]

        anchors.append(
            {
                "anchor_type": "LSS",
                "adas_family": "LSS",
                "lss_system": system,
                "scenario_code": code,
                "scenario_name": f"{system} - {title}" if not title.upper().startswith(system) else title,
                "line": _clean(lines[i]),
                "page": page,
                "raw_block_text": block,
                "evidence": [
                    {
                        "field": "scenario_details.extra.lss",
                        "page": page,
                        "match": _clean(lines[i])[:300],
                    }
                ],
            }
        )

    return anchors


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _page_from_line(line: str) -> Optional[int]:
    for rx in _PAGE_MARKERS:
        m = rx.search(line or "")
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
    return None


def _looks_like_scenario_code(code: str) -> bool:
    """
    STRICT acceptance:
    - Must match STRICT_SCENARIO_CODE (starts with CCR/CCF/CCB/C2C/C2P/C2B/VRU/CP/CB/CM)
    - Must not be in REJECT_CODES
    """
    code = (code or "").strip()
    if not code:
        return False
    if code.upper() in REJECT_CODES:
        return False
    return bool(STRICT_SCENARIO_CODE.match(code))


def _extract_code_and_title_from_line(line: str) -> Optional[Tuple[str, str]]:
    """
    Try patterns like:
      - "Car-to-Car Rear Stationary (CCRs)"
      - "Car-to-Car Rear Stationary (CCRs) – a collision in which ..."
      - "CCRs – Car-to-Car Rear Stationary"
      - "CCRs: Car-to-Car Rear Stationary"
    Returns (code, title) if found and plausible.
    """
    line = _clean(line)
    if not line or len(line) < 8:
        return None

    # Pattern 1: Title (CODE)  [CODE at end of line]
    m = re.search(r"(.+?)\s*\(\s*([A-Za-z0-9_-]{3,24})\s*\)\s*$", line)
    if m:
        title = _clean(m.group(1))
        code = _clean(m.group(2))
        if _looks_like_scenario_code(code) and len(title) >= 6:
            return code, title

    # Pattern 1b: Title (CODE) – description  [Euro NCAP scenario list format]
    # Example: "Car-to-Car Rear Stationary (CCRs) – a collision in which ..."
    m = re.search(r"(.+?)\s*\(\s*([A-Za-z0-9_-]{3,24})\s*\)\s*[-:–—]\s*(.+)$", line)
    if m:
        title = _clean(m.group(1))
        code = _clean(m.group(2))
        # ignore m.group(3) (description), we only need anchor title+code
        if _looks_like_scenario_code(code) and len(title) >= 6:
            return code, title

    # Pattern 2: CODE - Title  (allow _ and - in code, allow longer codes)
    m = re.search(r"^\s*([A-Za-z0-9_-]{3,24})\s*[-:–—]\s*(.+)$", line)
    if m:
        code = _clean(m.group(1))
        title = _clean(m.group(2))
        if _looks_like_scenario_code(code) and len(title) >= 6:
            return code, title

    return None



def _count_numeric_tokens(s: str) -> int:
    # Count numeric tokens like 50, 50.0, 1.5, 10%, etc.
    if not s:
        return 0
    return len(re.findall(r"\b\d+(?:\.\d+)?%?\b", s))


def _anchor_occurrence_score(code: str, title: str, block: str, page: Optional[int]) -> float:
    """Heuristic score to choose the *best* occurrence of a scenario anchor.

    Why: many protocols mention the scenario early in 'definitions/overview' and later in the
    'scenario parameters' section where tables live. We want the later, parameter-rich block.

    Scoring principles:
    - Prefer blocks that look like parameter sections: VUT/XVUT/VT, km/h, overlap, offset, etc.
    - Prefer blocks that reference tables/figures.
    - Prefer blocks containing lots of numbers.
    - Prefer later pages (small bonus).
    - Penalize glossary/definitions/intro sections.
    """
    b = (block or "").lower()
    c = (code or "").strip().lower()
    t = (title or "").strip().lower()

    score = 0.0

    # Strong parameter keywords (tables usually contain these)
    param_keywords = [
        "vut", "xvut", "vt", "speed", "km/h", "kph", "m/s",
        "overlap", "offset", "lateral", "longitudinal", "headway",
        "deceleration", "acceleration", "brake", "t0", "t1", "ttc",
        "initial distance", "target", "stationary", "moving", "braking",
        "test speed", "desired", "range",
    ]
    for kw in param_keywords:
        if kw in b:
            score += 2.0

    # Table/figure references are strong indicators of the right section
    table_like = ["table", "tables", "figure", "fig.", "shown in", "see table", "as in table"]
    for kw in table_like:
        if kw in b:
            score += 3.0

    # Lots of numbers usually means parameter tables / conditions
    nnums = _count_numeric_tokens(block)
    score += min(25.0, nnums * 0.6)

    # Bonus if the block explicitly mentions the scenario code/title
    if c and c in b:
        score += 6.0
    if t and t in b:
        score += 4.0

    # Bonus for section-style numbering (often scenario sections)
    if re.search(r"\b\d+\.\d+(?:\.\d+)?\b", b):
        score += 2.0
    # Extra bonus for later 'chapter 8/9/10' which is common for test scenarios in Euro NCAP docs
    if re.search(r"\b(8|9|10)\.\d+\b", b):
        score += 3.0

    # Penalties for glossary/definitions/intro
    glossary_hits = 0
    glossary_terms = [
        "abbreviation", "abbreviations", "definitions", "definition",
        "terminology", "introduction", "scope", "overview",
        "purpose", "this document", "general", "references",
    ]
    for kw in glossary_terms:
        if kw in b:
            glossary_hits += 1
    score -= glossary_hits * 3.5

    # Small bonus for later pages (do not dominate)
    if isinstance(page, int) and page > 0:
        score += min(8.0, page * 0.25)

    return score


def _pick_best_anchor_occurrences(
    occurrences: List[Tuple[int, str, str, Optional[int], str]],
) -> List[Tuple[int, str, str, Optional[int], str]]:
    """Pick the best occurrence per scenario code.

    Problem:
      Many protocols mention the scenario early in definitions/overview, and later in the
      scenario-parameter section where tables live. We must prefer the later parameter section.

    Strategy:
      - Score each occurrence using _anchor_occurrence_score().
      - If there are MULTIPLE strong occurrences, prefer the one on the LATEST page.
        (This is critical for Euro NCAP docs where the real tables are later.)
      - Otherwise pick by highest score, with a weak tie-break on later page/line.

    occurrences items:
      (line_idx, code, title, page, block_text)

    Returns the chosen items in document order.
    """
    # Tunables (kept local to avoid leaking global config)
    STRONG_DELTA = 4.0          # how close to top_score counts as "also strong"
    MIN_STRONG_SCORE = 18.0     # floor for being considered "strong" at all

    by_code: Dict[str, List[Tuple[int, str, str, Optional[int], str]]] = {}
    for occ in occurrences:
        i, code, title, page, block = occ
        key = (code or "").strip().upper()
        if not key:
            continue
        by_code.setdefault(key, []).append(occ)

    chosen: List[Tuple[int, str, str, Optional[int], str]] = []
    for code_u, occs in by_code.items():
        scored: List[Tuple[float, Tuple[int, str, str, Optional[int], str]]] = []
        for (i, code, title, page, block) in occs:
            s = _anchor_occurrence_score(code, title, block, page)
            scored.append((s, (i, code, title, page, block)))

        # Highest score
        top_score = max(s for s, _ in scored)

        # Strong candidates: close to top OR above a minimum floor
        strong_cutoff = max(MIN_STRONG_SCORE, top_score - STRONG_DELTA)
        strong = [(s, occ) for (s, occ) in scored if s >= strong_cutoff]

        def _page_val(p: Optional[int]) -> int:
            return p if isinstance(p, int) else -1

        if strong and len(strong) >= 2:
            # IMPORTANT: Prefer the latest page among strong candidates.
            # If same page, prefer later occurrence in text.
            best_s, best_occ = max(
                strong,
                key=lambda x: (_page_val(x[1][3]), x[1][0])
            )
            chosen.append(best_occ)
            continue

        # Otherwise: choose by score primarily; tie-break by later page then later line.
        best_s, best_occ = max(
            scored,
            key=lambda x: (x[0], _page_val(x[1][3]), x[1][0])
        )
        chosen.append(best_occ)

    # Keep document order for downstream page selection
    chosen.sort(key=lambda x: x[0])
    return chosen

def _reanchor_aeb_early_definition(
    lines: List[str],
    pages_for_line: List[Optional[int]],
    occ: Tuple[int, str, str, Optional[int], str],
    *,
    context_lines: int,
    max_block_lines: int,
) -> Tuple[int, str, str, Optional[int], str]:
    """If an AEB (CCR/CCF/CCB/C2*) scenario anchor is detected very early in the document,
    it is often a *definition/overview* mention. The real numeric tables typically appear later
    inside the main scenario/test-conditions chapter (often Chapter 8 in Euro NCAP AEB C2C protocols).

    This function deterministically re-anchors to a later, more useful location by searching for:
      1) Later occurrences of the SCENARIO TITLE (best signal)
      2) Later occurrences of the CODE as a standalone word (e.g., "CCRs scenario")
      3) Fallback to the first Chapter-8 style heading after page >= 15

    Returns a possibly-updated occurrence tuple (line_idx, code, title, page, block).
    """
    i, code, title, page, block = occ

    # Only AEB-style car-to-car families; VRU/LSS should not be forced to Chapter 8.
    fam = _adas_family_from_code(code)
    if fam != "AEB":
        return occ
    if not isinstance(page, int) or page >= 15:
        return occ

    title_l = (title or "").strip().lower()
    code_u = (code or "").strip().upper()
    if not title_l or not code_u:
        return occ

    # Scan forward from the original line to find better anchor candidates on later pages.
    best_idx: Optional[int] = None
    best_page: Optional[int] = None
    best_kind: int = 999  # lower is better (0=title hit, 1=code hit, 2=chapter8 hit)

    def _update(idx2: int, kind: int):
        nonlocal best_idx, best_page, best_kind
        p2 = pages_for_line[idx2]
        if not isinstance(p2, int):
            return
        if p2 < 15:
            return
        # Prefer better kind; if same kind, prefer later pages; if same page, earlier line in that page.
        if (best_idx is None) or (kind < best_kind) or (kind == best_kind and (p2 > (best_page or -1))) or (kind == best_kind and p2 == best_page and idx2 < best_idx):
            best_idx = idx2
            best_page = p2
            best_kind = kind

    # 1) TITLE hit (case-insensitive substring).
    if title_l:
        for j in range(i + 1, len(lines)):
            p2 = pages_for_line[j]
            if isinstance(p2, int) and p2 >= 15:
                if title_l in _clean(lines[j]).lower():
                    _update(j, 0)
                    break

    # 2) CODE hit as word boundary (e.g., "CCRs scenario")
    code_pat = re.compile(r"\b" + re.escape(code_u) + r"\b", re.IGNORECASE)
    for j in range(i + 1, len(lines)):
        p2 = pages_for_line[j]
        if isinstance(p2, int) and p2 >= 15:
            if code_pat.search(_clean(lines[j])):
                _update(j, 1)
                break

    # 3) Chapter-8 fallback: first "8", "8.1", "8.2" heading after page>=15
    chap8_pat = re.compile(r"^\s*8(\.[0-9]{1,2}){0,3}\b")
    for j in range(i + 1, len(lines)):
        p2 = pages_for_line[j]
        if isinstance(p2, int) and p2 >= 15:
            if chap8_pat.search(_clean(lines[j])):
                _update(j, 2)
                break

    if best_idx is None or best_page is None:
        return occ

    # Rebuild block around the new anchor line, bounded similarly to extract_scenario_anchors().
    start = max(0, best_idx - context_lines)
    end = min(len(lines), best_idx + max_block_lines)
    new_block = "\n".join(lines[start:end]).strip()
    return (best_idx, code, title, best_page, new_block)

def extract_scenario_anchors(
    text: str,
    *,
    context_lines: int = 40,
    max_block_lines: int = 120,
) -> List[Dict]:
    """
    Extract anchors from plain text.

    Output items include:
      {
        "scenario_code": str,
        "scenario_name": str,
        "line": str,
        "page": int | null,
        "raw_block_text": str,
        "evidence": [{"field": str, "page": int|null, "match": str}]
      }

    IMPORTANT:
    - A scenario code may appear multiple times in a protocol:
        * early: definitions/overview (usually no tables)
        * later: scenario parameter section (tables + numeric ranges)
      This function selects the *best* occurrence per code using heuristics so Stage-2
      gets pages near the numeric tables.
    """
    anchors: List[Dict] = []
    if not isinstance(text, str) or not text.strip():
        return anchors

    lines = text.splitlines()

    # Track page numbers across lines using markers
    pages_for_line: List[Optional[int]] = []
    current_page: Optional[int] = None
    for ln in lines:
        p = _page_from_line(ln)
        if p is not None:
            current_page = p
        pages_for_line.append(current_page)

    # Find anchor lines (allow duplicates; we will pick best later)
    found: List[Tuple[int, str, str]] = []  # (line_idx, code, title)
    for i, raw_line in enumerate(lines):
        line = _clean(raw_line)

        # Handle common PDF export where heading splits:
        #   "Some Scenario Title"
        #   "(CCRs)"
        candidate = line
        if i + 1 < len(lines):
            nxt = _clean(lines[i + 1])
            if nxt and _STANDALONE_PAREN_CODE.fullmatch(nxt):
                candidate = f"{line} {nxt}"

        got = _extract_code_and_title_from_line(candidate)
        if not got:
            continue
        code, title = got
        found.append((i, code, title))

    if not found:
        return anchors

    # Sort by document order
    found.sort(key=lambda x: x[0])

    # Build blocks for every occurrence (bounded by next anchor or max_block_lines)
    occurrences: List[Tuple[int, str, str, Optional[int], str]] = []
    for idx, (i, code, title) in enumerate(found):
        next_i = found[idx + 1][0] if idx + 1 < len(found) else len(lines)

        start = max(0, i - context_lines)
        end = min(len(lines), i + max_block_lines)
        end = min(end, next_i)

        block = "\n".join(lines[start:end]).strip()
        page = pages_for_line[i]

        occurrences.append((i, code, title, page, block))

    # Pick the best occurrence per code (prefers parameter/table-rich blocks)
    chosen = _pick_best_anchor_occurrences(occurrences)

    # Euro NCAP AEB protocols often mention scenario codes early in definitions,
    # while numeric tables appear later (often Chapter 8). If an AEB anchor is very early,
    # deterministically re-anchor to a later, more table-rich section.
    chosen = [
        _reanchor_aeb_early_definition(lines, pages_for_line, occ, context_lines=context_lines, max_block_lines=max_block_lines)
        for occ in chosen
    ]


    # Build anchor items from chosen occurrences
    for (i, code, title, page, block) in chosen:
        adas_family = _adas_family_from_code(code)
        anchor_type = "VRU" if adas_family == "VRU" else None

        item = {
            "scenario_code": code,
            "scenario_name": title,
            "line": _clean(lines[i]),
            "page": page,
            "raw_block_text": block,
            "evidence": [
                {
                    "field": "scenario_details.scenario",
                    "page": page,
                    "match": _clean(lines[i])[:300],
                }
            ],
        }
        if adas_family:
            item["adas_family"] = adas_family
        if anchor_type:
            item["anchor_type"] = anchor_type

        anchors.append(item)

    return anchors



def classify_adas_vs_non_adas(anchor: Dict) -> str:
    """
    Minimal rule (UPDATED for LSS/VRU):
    - If we have a STRICT scenario code (CCR/CCF/CCB/C2C/C2P/C2B/VRU/CP/CB/CM), call it ADAS.
    - If the anchor is explicitly marked as LSS/VRU, call it ADAS.
    - Else NON_ADAS.
    """
    at = (anchor.get("anchor_type") or "").upper()
    if at in {"LSS", "VRU"}:
        return "ADAS"
    fam = (anchor.get("adas_family") or "").upper()
    if fam in {"LSS", "VRU", "AEB"}:
        return "ADAS"
    code = (anchor.get("scenario_code") or "").strip()
    if code and _looks_like_scenario_code(code):
        return "ADAS"
    return "NON_ADAS"