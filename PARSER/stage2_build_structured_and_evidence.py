# stage2_build_structured_and_evidence.py
"""
STAGE 2 (MERGED): Build structured_scenarios.json + scenario_evidence.json from knowledge_base_raw.json

Merges responsibilities of:
- scenario_anchor_extractor.py (anchors + strict scenario code logic)
- parse_knowledge_base.py      (structured scenario skeleton schema)
- filter_json_raw.py           (keep/drop heuristics, but applied safely)

Design goals:
- Generic across PDFs (10–15 PDFs): no hardcoding to one protocol version.
- Deterministic Stage 2 (no LLM).
- Evidence-first: create per-scenario evidence packs so Stage 3 (LLM) can fill important numeric fields.

Outputs:
- structured_scenarios.json
- scenario_evidence.json
- scenario_evidence_report.json

NEW (2026-01):
- Knowledge base may contain IMAGE METADATA entries:
    {"type":"image","page":21,"source_type":"table_image","meta":{"path":"images/table_image/...png"}}

IMPORTANT FIX (2026-01-xx):
- Previously image_candidates were selected ONLY by selected_pages derived from text evidence.
- When selected_pages became None/empty, image_candidates always became empty -> tables=0/images=0.
- Now Stage 2 falls back to anchor evidence pages (from structured_scenarios extra.evidence),
  and if still unavailable, falls back to a capped list of pages that contain ANY images.

IMPORTANT CHANGE (2026-01-xx):
- We DO NOT use page_image anymore.
- Stage 2 attaches only:
    - table_image
    - diagram_image
- We also attach image candidate metadata (type/page/path) to support the Step-1 LLM matcher
  (anchor_pages ± 1) and avoid sending huge image sets.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, DefaultDict
from collections import defaultdict

from PARSER.scenario_anchor_extractor import extract_scenario_anchors, classify_adas_vs_non_adas

# ---------------------------
# Anchor filtering (LSS safety)
# ---------------------------
_LSS_REAL_SCENARIO_HINTS = (
    "road edge",
    "solid line",
    "dashed line",
    "oncoming",
    "overtaking",
    "lane departure",
    "blind-spot",
    "blind spot",
    "bsm",
)

_LSS_NEGATIVE_ANCHOR_PHRASES = (
    "criteria",
    "fulfil",
    "fulfills",
    "fulfils",
    "eligib",
    "shall",
    "means",
    "is considered",
    "dossier",
    "evidence of the effectiveness",
)

def _is_probable_container_anchor(a: Dict[str, Any]) -> bool:
    at = (a.get("anchor_type") or "").upper().strip()
    if at != "LSS":
        return False

    name = (a.get("scenario_name") or "").strip().lower()
    line = (a.get("line") or "").strip().lower()

    if "figure" in name or "table" in name or "figure" in line or "table" in line:
        return True

    if any(p in name for p in _LSS_NEGATIVE_ANCHOR_PHRASES) or any(p in line for p in _LSS_NEGATIVE_ANCHOR_PHRASES):
        return True

    if any(h in name for h in _LSS_REAL_SCENARIO_HINTS) or any(h in line for h in _LSS_REAL_SCENARIO_HINTS):
        return False

    if "system" in name or "test scenarios" in name or name.endswith("tests") is False:
        return True

    return True

def _filter_noise_anchors(anchors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [a for a in anchors if not _is_probable_container_anchor(a)]


# ---------------------------
# Config (override via env)
# ---------------------------
KB_PATH = os.getenv("KB_PATH", "knowledge_base_raw.json")

OUT_STRUCTURED = os.getenv("OUT_STRUCTURED", "structured_scenarios.json")
OUT_EVIDENCE = os.getenv("OUT_EVIDENCE", "scenario_evidence.json")
OUT_REPORT = os.getenv("OUT_REPORT", "scenario_evidence_report.json")

TOP_K_ITEMS = int(os.getenv("EVID_TOP_K_ITEMS", "18"))
NEIGHBORS = int(os.getenv("EVID_NEIGHBORS", "1"))
MAX_DOC_CHARS = int(os.getenv("EVID_MAX_DOC_CHARS", "55000"))

# When fallback is needed, limit how many image pages we attach to avoid bloat.
MAX_FALLBACK_IMAGE_PAGES = int(os.getenv("MAX_FALLBACK_IMAGE_PAGES", "6"))
ANCHOR_PAGE_EXPAND = int(os.getenv("ANCHOR_PAGE_EXPAND", "1"))  # +/- N pages

PARAM_KEYWORDS = [
    "speed", "km/h", "kph", "m/s", "m/s2",
    "ttc", "time-to-collision",
    "overlap", "offset", "lateral", "headway", "gap", "distance",
    "decel", "deceler", "brak", "braking",
    "increment", "step", "range",
    "table", "figure", "graph",
    "gvt", "target", "actor", "vehicle", "pedestrian", "vru",
    "stationary", "moving", "rear", "front", "crossing", "turn", "lane", "lane change",
    "elk", "lka", "ldw", "bsm",
    "lane support", "lane support systems",
    "road edge", "solid line", "dashed line", "oncoming", "overtaking",
    "lateral velocity", "vlat", "v_lat", "v_lat,vut", "vut",
    "lane invasion", "lane departure",
]


# Negative hints that usually indicate "definitions / overview" pages, not parameter tables.
GLOSSARY_NEGATIVE_HINTS = [
    "abbreviation",
    "abbreviations",
    "terminology",
    "definitions",
    "introduction",
    "overview",
    "scope of application",
    "general",
    "this document",
    "shall be",
    "means that",
    "is defined as",
]

# Positive hints that usually indicate the real parameter section in Euro NCAP protocols.
PARAM_SECTION_HINTS = [
    "test conditions",
    "test procedure",
    "scenario",
    "test parameters",
    "parameter",
    "shown in the table",
    "shown in the tables below",
    "table ",
    "figure ",
    "vut",
    "xvut",
]

# Pattern hints for the AEB C2C protocol (commonly uses section 8.x / 8.2.x for CCR/CCF).
AEB_SECTION_RX = re.compile(r"\b8\.\d+(?:\.\d+)?\b")
FIGURE_TABLE_RX = re.compile(r"\b(?:figure|table)\s+\d+[-–]\d+\b", re.IGNORECASE)

SCENARIO_FAMILY_RX = re.compile(
    r"\b(CCR|CCF|CCB|C2C|C2P|C2B|VRU|CP|CB|CM|LSS|ELK|LKA|LDW|BSM)\b",
    re.IGNORECASE,
)
NUM_RX = re.compile(r"\b\d+(?:\.\d+)?\b")
SPEED_PAIR_RX = re.compile(r"\b(\d{2,3})\s*/\s*(\d{2,3})\b")


# ---------------------------
# Helpers (KB → text)
# ---------------------------
LEGACY_TEXT_KEYS = ("filtered", "text", "combined_text", "raw_text", "full_text")

def _as_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for x in content:
            if isinstance(x, dict):
                parts.append(" ".join([f"{k}: {v}" for k, v in x.items()]))
            else:
                parts.append(str(x))
        return "\n".join(parts)
    if isinstance(content, dict):
        for k in LEGACY_TEXT_KEYS:
            v = content.get(k)
            if isinstance(v, str) and v.strip():
                return v
        return json.dumps(content, ensure_ascii=False)
    return str(content)

def _load_kb(path: str) -> List[Dict[str, Any]]:
    candidates = [path, "knowledge_base_raw.json", "knowledge_base.json"]
    kb_path = None
    for p in candidates:
        if p and os.path.exists(p):
            kb_path = p
            break
    if kb_path is None:
        raise FileNotFoundError(f"KB file not found. Tried: {candidates}")

    with open(kb_path, "r", encoding="utf-8") as f:
        kb = json.load(f)

    if isinstance(kb, list):
        out = []
        for x in kb:
            if isinstance(x, dict):
                out.append(x)
            elif isinstance(x, str):
                out.append({"type": "text", "content": x, "page": None})
        return out

    if isinstance(kb, dict):
        for k in ("items", "chunks", "pages"):
            if isinstance(kb.get(k), list):
                return [x for x in kb[k] if isinstance(x, dict)]
        return [kb]

    if isinstance(kb, str):
        return [{"type": "text", "content": kb, "page": None}]

    return []

def _kb_to_full_text(kb_items: List[Any]) -> str:
    out: List[str] = []
    last_page = None

    for it in kb_items:
        if not isinstance(it, dict):
            continue

        it_type = (it.get("type") or "text").lower().strip()
        if it_type == "image":
            continue

        page = it.get("page")
        if isinstance(page, str) and page.isdigit():
            page = int(page)

        if isinstance(page, int) and page != last_page:
            out.append(f"[PAGE {page}]")
            last_page = page

        blob = _as_text(it.get("content") if it.get("content") is not None else it.get("text"))
        if blob and blob.strip():
            out.append(blob.strip())

    return "\n".join(out).strip()


# ---------------------------
# Image indexing + candidate selection (NO page_image)
# ---------------------------
def _norm_page(page: Any) -> Optional[int]:
    if page is None:
        return None
    if isinstance(page, int):
        return page
    if isinstance(page, str) and page.isdigit():
        return int(page)
    return None

def _get_image_rel_path(it: Dict[str, Any]) -> Optional[str]:
    meta = it.get("meta") if isinstance(it.get("meta"), dict) else {}
    p = meta.get("path")
    if isinstance(p, str) and p.strip():
        return p.strip()
    p2 = it.get("path")
    if isinstance(p2, str) and p2.strip():
        return p2.strip()
    f = it.get("file")
    return f if isinstance(f, str) and f.strip() else None

def _index_images_by_page(kb_items: List[Dict[str, Any]]) -> Dict[int, Dict[str, List[str]]]:
    """
    Index only table_image + diagram_image by page.
    """
    out: DefaultDict[int, Dict[str, List[str]]] = defaultdict(
        lambda: {"table_image": [], "diagram_image": []}
    )
    for it in kb_items:
        if not isinstance(it, dict):
            continue
        if (it.get("type") or "").lower().strip() != "image":
            continue

        page = _norm_page(it.get("page"))
        if page is None:
            continue

        src = (it.get("source_type") or "").strip()
        if src not in ("table_image", "diagram_image"):
            continue

        rel = _get_image_rel_path(it)
        if not rel:
            continue

        out[page][src].append(rel)

    return dict(out)

def _select_image_candidates(
    selected_pages: Optional[List[int]],
    img_index: Dict[int, Dict[str, List[str]]]
) -> Dict[str, List[str]]:
    """
    Return legacy format:
      { "table_images": [...], "diagram_images": [...] }
    """
    if not selected_pages:
        return {"table_images": [], "diagram_images": []}

    tables: List[str] = []
    diags: List[str] = []

    for p in selected_pages:
        bucket = img_index.get(p)
        if not bucket:
            continue
        tables.extend(bucket.get("table_image", []))
        diags.extend(bucket.get("diagram_image", []))

    def _dedupe(seq: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in seq:
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    return {"table_images": _dedupe(tables), "diagram_images": _dedupe(diags)}

def _select_image_candidates_meta(
    selected_pages: Optional[List[int]],
    img_index: Dict[int, Dict[str, List[str]]]
) -> List[Dict[str, Any]]:
    """
    Return metadata format for LLM matcher:
      [ { "type": "table"|"diagram", "page": int, "path": str }, ... ]
    """
    if not selected_pages:
        return []

    out: List[Dict[str, Any]] = []
    seen = set()

    for p in selected_pages:
        bucket = img_index.get(p)
        if not bucket:
            continue

        for rel in bucket.get("table_image", []):
            key = ("table", p, rel)
            if key in seen:
                continue
            seen.add(key)
            out.append({"type": "table", "page": p, "path": rel})

        for rel in bucket.get("diagram_image", []):
            key = ("diagram", p, rel)
            if key in seen:
                continue
            seen.add(key)
            out.append({"type": "diagram", "page": p, "path": rel})

    return out


# ---------------------------
# NEW: fallback page selection for images
# ---------------------------
def _extract_anchor_pages_from_structured(scenario: Dict[str, Any]) -> List[int]:
    """
    Read pages from:
      structured scenario -> scenario_details -> extra -> evidence -> [{page: N, ...}]
    """
    pages: List[int] = []
    details = scenario.get("scenario_details") if isinstance(scenario.get("scenario_details"), dict) else {}
    extra = details.get("extra") if isinstance(details.get("extra"), dict) else {}
    evid = extra.get("evidence")
    if isinstance(evid, list):
        for e in evid:
            if not isinstance(e, dict):
                continue
            p = _norm_page(e.get("page"))
            if p is not None:
                pages.append(p)
    out = []
    seen = set()
    for p in pages:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out

def _expand_pages(pages: List[int], expand: int) -> List[int]:
    if not pages:
        return []
    out = set()
    for p in pages:
        for q in range(p - expand, p + expand + 1):
            if q > 0:
                out.add(q)
    return sorted(out)

def _fallback_pages_with_any_images(img_index: Dict[int, Dict[str, List[str]]], cap: int) -> List[int]:
    """
    If we cannot localize pages at all, at least attach some images rather than 0/0 everywhere.
    (Only pages that have table/diagram images.)
    """
    pages = sorted(img_index.keys())
    if not pages:
        return []
    return pages[:max(1, cap)]


# ---------------------------
# Structured skeleton
# ---------------------------
def _extract_speed_pairs(block: str) -> List[Dict[str, int]]:
    out: List[Dict[str, int]] = []
    seen = set()
    for m in SPEED_PAIR_RX.finditer(block or ""):
        a = int(m.group(1))
        b = int(m.group(2))
        if (a, b) in seen:
            continue
        seen.add((a, b))
        out.append({"ego_speed_kmh": a, "target_speed_kmh": b})
    return out

def _dedupe_by_code_or_name(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for s in items:
        code = (s.get("scenario_code") or "").strip().lower()
        name = re.sub(r"\s+", " ", (s.get("scenario_name") or "").strip()).lower()
        key = code or name
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out

def _drop_umbrella_parent_scenarios(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Some Euro NCAP headings (e.g. "Car-to-Car Front Head-On (CCFho)") are umbrella
    sections that group two or more specific sub-scenarios (e.g. CCFhos, CCFhol)
    rather than being a distinct testable scenario in their own right.

    _aliases_for_code() already encodes which parent code each specific sibling
    implies (CCFHOS/CCFHOL -> "CCFho"). If a scenario's own code equals a parent
    alias implied by OTHER, more specific scenario codes present in this same
    batch, it is a redundant umbrella entry and gets dropped here.

    This intentionally lives here (post-extraction, batch-aware) rather than in
    the per-scenario enrichment prompt, because the enrichment LLM only ever
    sees one scenario at a time and has no way to know its siblings exist.
    """
    codes_present = {
        (s.get("scenario_code") or "").strip().upper()
        for s in items
        if s.get("scenario_code")
    }

    implied_parents = set()
    for s in items:
        code = (s.get("scenario_code") or "").strip().upper()
        if not code:
            continue
        for alias in _aliases_for_code(code):
            alias_u = alias.upper()
            if alias_u != code:
                implied_parents.add(alias_u)

    out = []
    for s in items:
        code = (s.get("scenario_code") or "").strip().upper()
        if code and code in implied_parents and code in codes_present:
            continue  # redundant umbrella entry — a more specific sibling covers it
        out.append(s)
    return out


def _make_structured_from_anchor(a: Dict[str, Any]) -> Dict[str, Any]:
    scenario_type = classify_adas_vs_non_adas(a)
    code = (a.get("scenario_code") or "").strip() or None
    name = (a.get("scenario_name") or "").strip() or (code or "UNNAMED")
    block = a.get("raw_block_text", "") or ""
    evidence = a.get("evidence") or []

    if scenario_type == "ADAS":
        extra: Dict[str, Any] = {}
        pairs = _extract_speed_pairs(block)
        if pairs:
            extra["test_points"] = pairs
        if evidence:
            extra["evidence"] = evidence

        anchor_type = (a.get("anchor_type") or "").upper().strip()
        adas_family_hint = (a.get("adas_family") or "").upper().strip()
        code_u = (code or "").upper() if code else ""

        if anchor_type == "LSS":
            extra["adas_family"] = "LSS"
            extra["lss"] = {
                "system": (a.get("lss_system") or None),
                "boundary_type": None,
                "departure_side": None,
                "lateral_speed_mps_min": None,
                "lateral_speed_mps_max": None,
                "ego_speed_kph_min": None,
                "ego_speed_kph_max": None,
                "target_actor": None,
            }
        elif anchor_type == "VRU" or adas_family_hint == "VRU" or code_u.startswith(("VRU_", "CP", "CB", "CM")):
            extra["adas_family"] = "VRU"

            vru_type = None
            if code_u.startswith("CP"):
                vru_type = "pedestrian"
            elif code_u.startswith("CB"):
                vru_type = "cyclist"
            elif code_u.startswith("CM"):
                vru_type = "motorcyclist"

            extra["vru"] = {
                "vru_type": vru_type,
                "scenario_variant": None,
                "adult_child": None,
                "obscured": None,
                "crossing_side": None,
                "vru_speed_mps_min": None,
                "vru_speed_mps_max": None,
                "start_offset_m": None,
            }
        else:
            extra.setdefault("adas_family", "AEB_CCR")

        scenario_details: Dict[str, Any] = {
            "is_scenario": True,
            "scenario": name,
            "ttc": None,
            "ttc_end": None,
            "ego_speed_min": None,
            "ego_speed_max": None,
            "target_speed_min": None,
            "target_speed_max": None,
            "ego_vehicle_type": None,
            "target_vehicle_type": None,
            "overlap_percent": None,
            "lateral_offset_m": None,
            "initial_distance_m": None,
            "headway_m": None,
            "target_decel_mps2": None,
            "ego_decel_mps2": None,
            "road_layout": None,
            "notes": None,
            "extra": extra or None,
        }

        user_config: Dict[str, Any] = {
            "road_selection_mode": None,
            "ego_lane_role": None,
            "spawn": {"ego": None, "target": None},
            "initial_gap_m": None,
            "lateral_offset_m": None,
            "overlap_percent": None,
            "trigger": {"type": None, "distance_m": None, "ttc_s": None},
            "stop_on_collision": True,
            "timeout_s": 60,
            "target_vehicle_blueprint": None,
        }

        item: Dict[str, Any] = {
            "scenario_type": "ADAS",
            "scenario_name": name,
            "script_eligible": True,
            "scenario_details": scenario_details,
            "user_config": user_config,
        }
        if code:
            item["scenario_code"] = code
        return item

    desc = re.sub(r"\s+", " ", block).strip()
    if len(desc) > 500:
        desc = desc[:500] + "…"
    return {
        "scenario_type": "NON_ADAS",
        "scenario_name": name,
        "scenario_description": desc or "Non-ADAS content extracted from the document.",
        "script_eligible": False,
        "disclaimer": "This is a NON-ADAS scenario/item. Script cannot be generated.",
    }


# ---------------------------
# Evidence selection (text-only scoring)
# ---------------------------
def _count_numbers(text: str) -> int:
    return len(NUM_RX.findall(text or ""))

def _keyword_hits(text: str) -> int:
    t = (text or "").lower()
    return sum(1 for kw in PARAM_KEYWORDS if kw in t)

def _aliases_for_code(code: str) -> List[str]:
    if not code:
        return []
    u = code.strip()
    U = u.upper()
    aliases = {u, U}

    m = re.match(r"^(CCR|CCF|CCB|C2C|C2P|C2B|VRU|CP|CB|CM|LSS)", U)
    if m:
        aliases.add(m.group(1))

    if U.startswith(("CP", "CB", "CM")):
        aliases.add("VRU")

    if U.startswith("CCFHOS") or U.startswith("CCFHOL"):
        aliases.add("CCFho")
        aliases.add("CCFHO")

    if U.startswith("LSS_"):
        aliases.add("LSS")
        for tok in ("ELK", "LKA", "LDW", "BSM"):
            if tok in U:
                aliases.add(tok)
    return sorted(aliases)

def _score_item(blob: str, code: Optional[str], name: Optional[str], page: Optional[int]) -> int:
    """Score a KB text item for evidence selection.

    Goal: prefer the *parameter/table* section (usually later in the PDF) over early
    definition/overview mentions.
    """
    t = (blob or "").lower()
    score = 0

    # --- Code / alias matches (strong signal) ---
    if code:
        for a in _aliases_for_code(code):
            al = a.lower()
            score += 10 * len(re.findall(rf"\b{re.escape(al)}\b", t))
            score += 2 * len(re.findall(re.escape(al), t))

    # --- Name token matches (weak/medium signal) ---
    if name:
        toks = [w for w in re.split(r"\W+", name.lower()) if len(w) >= 3]
        for tok in toks[:12]:
            score += 2 * len(re.findall(rf"\b{re.escape(tok)}\b", t))

    # --- Parameter keywords / numeric density (table-ish text) ---
    kw = _keyword_hits(t)
    if kw:
        score += 2 * kw

    nums = _count_numbers(t)
    if nums >= 15:
        score += 6
    if nums >= 35:
        score += 6

    # Extra boost if it explicitly references tables/figures
    if "table" in t:
        score += 6
    if "figure" in t:
        score += 3

    # Euro NCAP AEB protocols often use section 8.x/8.2.x for the core scenario parameters
    if AEB_SECTION_RX.search(t):
        score += 6
    if FIGURE_TABLE_RX.search(t):
        score += 4

    # Positive parameter-section hints
    for ph in PARAM_SECTION_HINTS:
        if ph in t:
            score += 2

    # Penalize glossary/overview style pages
    neg_hits = 0
    for gh in GLOSSARY_NEGATIVE_HINTS:
        if gh in t:
            neg_hits += 1
    if neg_hits:
        score -= 6 * neg_hits

    # Small preference for later pages (tables/parameters tend to be later)
    if isinstance(page, int):
        if page >= 15:
            score += 3
        if page >= 22:
            score += 4
        if page >= 30:
            score += 2

    return score

    if code:
        for a in _aliases_for_code(code):
            al = a.lower()
            score += 10 * len(re.findall(rf"\b{re.escape(al)}\b", t))
            score += 2 * len(re.findall(re.escape(al), t))

    if name:
        toks = [w for w in re.split(r"\W+", name.lower()) if len(w) >= 3]
        for tok in toks[:12]:
            score += 2 * len(re.findall(rf"\b{re.escape(tok)}\b", t))

    kw = _keyword_hits(t)
    if kw:
        score += 2 * kw

    nums = _count_numbers(t)
    if nums >= 15:
        score += 6
    if nums >= 35:
        score += 6

    if "table" in t:
        score += 4

    return score

def _select_evidence_indices(kb_items: List[Any], code: Optional[str], name: Optional[str]) -> Tuple[List[int], Dict[str, Any]]:
    scored: List[Tuple[int, int]] = []
    for idx, it in enumerate(kb_items):
        if not isinstance(it, dict):
            continue

        it_type = (it.get("type") or "text").lower().strip()
        if it_type == "image":
            continue

        blob = _as_text(it.get("content") if it.get("content") is not None else it.get("text"))
        if not blob or len(blob.strip()) < 30:
            continue

        sc = _score_item(blob, code, name, _norm_page(it.get('page')))
        if sc > 0:
            scored.append((sc, idx))

    dbg: Dict[str, Any] = {"hits": len(scored), "aliases": _aliases_for_code(code or "")}

    if not scored:
        # Better fallback than "first pages": take a small window from the start AND the end,
        # because many protocols put the real scenario parameter tables later.
        k = min(TOP_K_ITEMS, len(kb_items))
        if k <= 0:
            dbg["fallback"] = "no_items"
            return [], dbg
        half = max(1, k // 2)
        head = list(range(0, min(half, len(kb_items))))
        tail_start = max(0, len(kb_items) - (k - len(head)))
        tail = list(range(tail_start, len(kb_items)))
        chosen = sorted(set(head + tail))
        dbg["fallback"] = "no_scored_items_head_tail"
        return chosen, dbg

    scored.sort(reverse=True, key=lambda x: x[0])
    top = scored[:TOP_K_ITEMS]
    dbg["top"] = [
        {"score": s, "idx": i, "page": (kb_items[i].get("page") if isinstance(kb_items[i], dict) else None)}
        for s, i in top
    ]

    chosen_set = set()
    for _, i in top:
        for j in range(max(0, i - NEIGHBORS), min(len(kb_items), i + NEIGHBORS + 1)):
            chosen_set.add(j)

    return sorted(chosen_set), dbg

def _build_doc_text(kb_items: List[Any], idxs: List[int]) -> str:
    parts: List[str] = []
    for i in idxs:
        it = kb_items[i]
        if not isinstance(it, dict):
            continue

        it_type = (it.get("type") or "text").lower().strip()
        if it_type == "image":
            continue

        page = it.get("page")
        hdr = f"[PAGE {page}]" if page is not None else "[PAGE ?]"
        blob = _as_text(it.get("content") if it.get("content") is not None else it.get("text")).strip()
        if not blob:
            continue
        parts.append(hdr + "\n" + blob)

    doc = "\n\n".join(parts).strip()
    if len(doc) > MAX_DOC_CHARS:
        doc = doc[:MAX_DOC_CHARS] + "\n\n[TRUNCATED]"
    return doc


def main():
    kb_items = _load_kb(KB_PATH)

    # index images once (tables+diagrams only)
    img_index = _index_images_by_page(kb_items)

    full_text = _kb_to_full_text(kb_items)
    if not full_text.strip():
        raise ValueError(f"No usable text found in {KB_PATH}")

    # 1) anchors
    anchors = extract_scenario_anchors(full_text)
    anchors = _filter_noise_anchors(anchors)

    # 2) structured scenarios
    structured = [_make_structured_from_anchor(a) for a in anchors]
    structured = _dedupe_by_code_or_name(structured)
    structured = _drop_umbrella_parent_scenarios(structured)

    Path(OUT_STRUCTURED).write_text(json.dumps(structured, indent=2, ensure_ascii=False), encoding="utf-8")

    # 3) evidence packs (ADAS only)
    evidence: Dict[str, Any] = {}
    report: List[Dict[str, Any]] = []

    for s in structured:
        if (s.get("scenario_type") or "").upper() != "ADAS":
            continue

        code = (s.get("scenario_code") or "").strip() or None
        name = (s.get("scenario_name") or "").strip() or None

        idxs, dbg = _select_evidence_indices(kb_items, code, name)
        doc_text = _build_doc_text(kb_items, idxs)

        doc_low = doc_text.lower()
        param_hit_count = sum(1 for kw in PARAM_KEYWORDS if kw in doc_low)
        num_count = _count_numbers(doc_text)

        # pages from the selected TEXT items
        selected_pages = sorted({
            kb_items[i].get("page")
            for i in idxs
            if isinstance(kb_items[i], dict)
            and kb_items[i].get("page") is not None
            and (kb_items[i].get("type") or "text").lower().strip() != "image"
        }) or None

        # ===== FIX: fallback pages for images =====
        fallback_used = None

        # if selected_pages is None/empty, use anchor evidence pages
        if not selected_pages:
            anchor_pages = _extract_anchor_pages_from_structured(s)
            if anchor_pages:
                selected_pages = _expand_pages(anchor_pages, ANCHOR_PAGE_EXPAND)
                fallback_used = "anchor_evidence_pages"

        # if still none, pick some pages that contain images (capped)
        if not selected_pages:
            selected_pages = _fallback_pages_with_any_images(img_index, MAX_FALLBACK_IMAGE_PAGES) or None
            if selected_pages:
                fallback_used = "image_pages_fallback"

        # attach image candidates (tables+diagrams only)
        image_candidates = _select_image_candidates(selected_pages, img_index)
        image_candidates_meta = _select_image_candidates_meta(selected_pages, img_index)

        key = code or name or f"scenario_{len(evidence)+1}"
        evidence[key] = {
            "scenario_code": code,
            "scenario_name": name,
            "selected_pages": selected_pages,
            "doc_text": doc_text,
            # legacy structure (paths only)
            "image_candidates": image_candidates,
            # new structure (type/page/path)
            "image_candidates_meta": image_candidates_meta,
            "debug": {
                **dbg,
                "param_hit_count": param_hit_count,
                "num_count": num_count,
                "doc_chars": len(doc_text),
                "fallback_used": fallback_used,
                "image_counts": {
                    "tables": len(image_candidates.get("table_images", [])),
                    "diagrams": len(image_candidates.get("diagram_images", [])),
                },
            },
        }

        report.append({
            "key": key,
            "scenario_code": code,
            "scenario_name": name,
            "selected_pages": selected_pages,
            "param_hit_count": param_hit_count,
            "num_count": num_count,
            "table_images": len(image_candidates.get("table_images", [])),
            "diagram_images": len(image_candidates.get("diagram_images", [])),
            "fallback_used": fallback_used,
            "warning": "LOW_EVIDENCE" if (param_hit_count < 3 and num_count < 40) else None,
        })

    Path(OUT_EVIDENCE).write_text(json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8")
    Path(OUT_REPORT).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved {OUT_STRUCTURED} (items={len(structured)})")
    print(f"Saved {OUT_EVIDENCE} (ADAS scenarios={len(evidence)})")
    print(f"Saved {OUT_REPORT} (ADAS scenarios={len(report)})")


if __name__ == "__main__":
    main()
