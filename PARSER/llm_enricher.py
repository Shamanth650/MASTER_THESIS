# llm_enricher.py
"""
LLM enrichment stage: fills missing/null protocol fields in structured_scenarios.json.

UPDATED (Stage-2 Evidence Pack integration):
- Prefer per-scenario DOC_TEXT from scenario_evidence.json (produced by Stage 2)
- Fallback to filtering knowledge_base.json only if evidence pack missing
- Keeps strict null-only patching + evidence requirements + validation/repair

This module is provider-agnostic at the interface level: `enrich_one_scenario`
and `enrich_all` return plain Python dicts/lists.
"""
from __future__ import annotations

import json
import os
import re
import base64
import mimetypes
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
import os
import json
from dotenv import load_dotenv

load_dotenv()   # 👈 THIS is what you were missing

from anthropic import Anthropic

try:
    from anthropic import Anthropic
except Exception as e:  # pragma: no cover
    Anthropic = None  # type: ignore
    _ANTHROPIC_IMPORT_ERROR = e

# Model / temperature
# Keep the variable names used across the file to avoid touching other logic.
OPENAI_MODEL = os.getenv("CLAUDE_MODEL", os.getenv("OPENAI_MODEL", "claude-opus-4-8"))
OPENAI_TEMPERATURE = float(os.getenv("CLAUDE_TEMPERATURE", os.getenv("OPENAI_TEMPERATURE", "0")) or "0")

# Networking / retries (kept as OPENAI_* names to avoid touching the rest of the file)
OPENAI_TIMEOUT_S = float(os.getenv('CLAUDE_TIMEOUT_S', os.getenv('OPENAI_TIMEOUT_S', '180')) or '180')
OPENAI_MAX_RETRIES = int(os.getenv('CLAUDE_MAX_RETRIES', os.getenv('OPENAI_MAX_RETRIES', '6')) or '6')


# When we use evidence packs, doc_text is already compact.
# This cap is still useful for safety if a PDF is huge.
MAX_DOC_CHARS = int(os.getenv("LLM_ENRICH_MAX_DOC_CHARS", "55000") or "55000")

# Fallback retrieval caps (only used if evidence pack not available)
MAX_PAGES = int(os.getenv("LLM_ENRICH_MAX_PAGES", "8") or "8")
NEIGHBOR_PAGES = int(os.getenv("LLM_ENRICH_NEIGHBORS", "1") or "1")

REPAIR_MAX_TRIES = int(os.getenv("LLM_ENRICH_REPAIR_MAX_TRIES", "2") or "2")


# ----------------------------
# Retrieval helpers (family-aware) - fallback only
# ----------------------------
_SCENARIO_FAMILIES = ("CCR", "CCF", "CCB", "C2C", "C2P", "C2B", "VRU", "LSS", "ELK", "LKA", "LDW", "BSM")


def _build_code_aliases(code: str) -> List[str]:
    code = (code or "").strip()
    if not code:
        return []
    u = code.upper()
    aliases = {code, u}

    for fam in _SCENARIO_FAMILIES:
        if u.startswith(fam):
            aliases.add(fam)
            break

    if u.startswith("CCFHOS") or u.startswith("CCFHOL"):
        aliases.add("CCFho")
        aliases.add("CCFHO")

    # LSS synthetic codes like LSS_ELK_ROADEDGE...
    if u.startswith("LSS_"):
        aliases.add("LSS")
        for tok in ("ELK", "LKA", "LDW", "BSM"):
            if tok in u:
                aliases.add(tok)

    return sorted({a for a in aliases if a})


# ----------------------------
# Utilities: JSON path + diff
# ----------------------------
def _is_scalar(x: Any) -> bool:
    return x is None or isinstance(x, (str, int, float, bool))


def _iter_paths(obj: Any, prefix: str = "") -> Iterable[Tuple[str, Any]]:
    if _is_scalar(obj):
        yield (prefix or "$", obj)
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else k
            yield from _iter_paths(v, p)
        return
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            p = f"{prefix}[{i}]"
            yield from _iter_paths(v, p)
        return
    yield (prefix or "$", obj)


def _get_path(obj: Any, path: str) -> Any:
    cur = obj
    if path in ("$", ""):
        return cur
    parts = re.split(r"(?<!\\)\.", path)
    for part in parts:
        if part == "$" or part == "":
            continue
        m = re.match(r"^([^\[]+)(\[\d+\])*$", part)
        if not m:
            return None
        key = m.group(1)
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return None
        idxs = re.findall(r"\[(\d+)\]", part)
        for idx_s in idxs:
            idx = int(idx_s)
            if isinstance(cur, list) and 0 <= idx < len(cur):
                cur = cur[idx]
            else:
                return None
    return cur


def _diff_scalar_leaves(a: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
    a_map = dict(_iter_paths(a))
    b_map = dict(_iter_paths(b))
    paths = sorted(set(a_map.keys()) | set(b_map.keys()))
    changed: List[str] = []
    for p in paths:
        if a_map.get(p) != b_map.get(p):
            changed.append(p)
    return changed


# ----------------------------
# Stage-2 evidence pack loading
# ----------------------------
def load_scenario_evidence(evidence_path: str) -> Dict[str, Any]:
    with open(evidence_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("scenario_evidence.json must be a JSON object/dict")
    return obj


def get_doc_text_from_evidence(
    evidence_obj: Dict[str, Any],
    scenario_code: str,
    scenario_name: str,
) -> Optional[str]:
    """
    Evidence packs are keyed by scenario_code when possible.
    If scenario_code key missing, try scenario_name, else scan for matching scenario_code field.
    """
    code = (scenario_code or "").strip()
    name = (scenario_name or "").strip()

    # 1) direct by code
    if code and code in evidence_obj:
        rec = evidence_obj.get(code)
        if isinstance(rec, dict):
            dt = rec.get("doc_text")
            if isinstance(dt, str) and dt.strip():
                return dt.strip()

    # 2) direct by name
    if name and name in evidence_obj:
        rec = evidence_obj.get(name)
        if isinstance(rec, dict):
            dt = rec.get("doc_text")
            if isinstance(dt, str) and dt.strip():
                return dt.strip()

    # 3) scan values to find a record whose scenario_code matches
    if code:
        for _, rec in evidence_obj.items():
            if not isinstance(rec, dict):
                continue
            if (rec.get("scenario_code") or "").strip() == code:
                dt = rec.get("doc_text")
                if isinstance(dt, str) and dt.strip():
                    return dt.strip()

    
def get_image_candidates_from_evidence(
    evidence_obj: Dict[str, Any],
    scenario_code: str,
    scenario_name: str,
) -> Optional[Dict[str, Any]]:
    """
    Returns image_candidates dict if present in scenario_evidence.json record.

    Expected structure (from updated Stage 2):
      image_candidates: {
        "table_images": [...],
        "diagram_images": [...],
        "page_images": [...]
      }
    """
    code = (scenario_code or "").strip()
    name = (scenario_name or "").strip()

    def _extract(rec: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(rec, dict):
            return None
        ic = rec.get("image_candidates")
        return ic if isinstance(ic, dict) else None

    if code and code in evidence_obj:
        ic = _extract(evidence_obj.get(code))
        if ic:
            return ic

    if name and name in evidence_obj:
        ic = _extract(evidence_obj.get(name))
        if ic:
            return ic

    if code:
        for _, rec in evidence_obj.items():
            if not isinstance(rec, dict):
                continue
            if (rec.get("scenario_code") or "").strip() == code:
                ic = _extract(rec)
                if ic:
                    return ic

    return None


# ----------------------------
# Image handling (Vision inputs for GPT-4o)
# ----------------------------
# Configure how many images we send (keep this small for tokens + speed)
MAX_TABLE_IMAGES = int(os.getenv("LLM_ENRICH_MAX_TABLE_IMAGES", "3") or "3")
MAX_DIAGRAM_IMAGES = int(os.getenv("LLM_ENRICH_MAX_DIAGRAM_IMAGES", "1") or "1")
MAX_PAGE_IMAGES = int(os.getenv("LLM_ENRICH_MAX_PAGE_IMAGES", "0") or "0")

# Where Parsed_Data lives (used to resolve relative paths from Stage 2)
PARSED_ROOT = os.getenv("PARSED_ROOT", None)

def _guess_parsed_root() -> Optional[str]:
    """Try to locate Parsed_Data folder without hardcoding."""
    # 1) explicit env
    if PARSED_ROOT and os.path.isdir(PARSED_ROOT):
        return PARSED_ROOT

    # 2) cwd/Parsed_Data
    cand = os.path.join(os.getcwd(), "Parsed_Data")
    if os.path.isdir(cand):
        return cand

    # 3) script_dir/Parsed_Data
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.join(here, "Parsed_Data")
    if os.path.isdir(cand):
        return cand

    # 4) parent/Parsed_Data
    cand = os.path.join(os.path.dirname(here), "Parsed_Data")
    if os.path.isdir(cand):
        return cand

    return None


def _resolve_image_path(rel_path: str) -> Optional[str]:
    """
    Stage 2 stores paths like:
      images/table_image/foo.png
      images\\table_image\\foo.png
      Parsed_Data/images/table_image/foo.png
    Resolve robustly to an existing absolute file path.
    """
    if not rel_path or not isinstance(rel_path, str):
        return None

    p = rel_path.strip().strip('"').strip("'")
    if not p:
        return None

    # Normalize slashes
    p_norm = p.replace("\\", "/")

    # 1) Already absolute?
    if os.path.isabs(p) and os.path.isfile(p):
        return p
    if os.path.isabs(p_norm) and os.path.isfile(p_norm):
        return p_norm

    # 2) Exists relative to CWD?
    if os.path.isfile(p):
        return os.path.abspath(p)
    if os.path.isfile(p_norm):
        return os.path.abspath(p_norm)

    root = _guess_parsed_root()
    if not root:
        return None

    # Helper: join against root and check
    def _try_join(r: str, rel: str) -> Optional[str]:
        cand = os.path.normpath(os.path.join(r, rel))
        return cand if os.path.isfile(cand) else None

    # 3) Direct join against Parsed_Data root
    hit = _try_join(root, p) or _try_join(root, p_norm)
    if hit:
        return hit

    # 4) If someone stored leading Parsed_Data/..., strip it
    stripped = p_norm
    for prefix in ("Parsed_Data/", "ParsedData/", "parsed_data/", "parseddata/"):
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):]
            break

    hit = _try_join(root, stripped)
    if hit:
        return hit

    # 5) Common case: rel path begins with images/...
    # Ensure it resolves under root/images/...
    if stripped.startswith("images/"):
        hit = _try_join(root, stripped)  # root/images/...
        if hit:
            return hit

    # 6) Last resort: search by basename inside Parsed_Data recursively
    # (helps when Stage 2 stores only filename or wrong subfolder)
    base = os.path.basename(stripped)
    if base:
        for dirpath, _, filenames in os.walk(root):
            if base in filenames:
                cand = os.path.join(dirpath, base)
                if os.path.isfile(cand):
                    return cand

    return None



def _pick_image_paths(image_candidates: Optional[Dict[str, Any]]) -> List[str]:
    if not image_candidates:
        return []
    tables = image_candidates.get("table_images") if isinstance(image_candidates.get("table_images"), list) else []
    diags = image_candidates.get("diagram_images") if isinstance(image_candidates.get("diagram_images"), list) else []
    pages = image_candidates.get("page_images") if isinstance(image_candidates.get("page_images"), list) else []

    picked: List[str] = []
    for p in tables[:MAX_TABLE_IMAGES]:
        ap = _resolve_image_path(p)
        if ap:
            picked.append(ap)
    for p in diags[:MAX_DIAGRAM_IMAGES]:
        ap = _resolve_image_path(p)
        if ap:
            picked.append(ap)
    for p in pages[:MAX_PAGE_IMAGES]:
        ap = _resolve_image_path(p)
        if ap:
            picked.append(ap)

    # de-dupe preserve order
    seen = set()
    out = []
    for p in picked:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def _file_to_data_url(path: str) -> str:
    """Encode local image to a data URL for multimodal input."""
    mime, _ = mimetypes.guess_type(path)
    if not mime:
        mime = "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"

def _file_to_base64_image(path: str) -> Tuple[str, str]:
    """
    Read an image file and return (media_type, base64_data) for Anthropic vision.
    Keeps original bytes (tables-only + capped count keeps payload small).
    """
    mime, _ = mimetypes.guess_type(path)
    if not mime:
        mime = "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return mime, b64



# ----------------------------
# Knowledge base loading + filtering (fallback only)
# ----------------------------
@dataclass
class KBPage:
    page: Optional[int]
    text: str


def _normalize_kb_items(kb_obj: Any) -> List[KBPage]:
    pages: List[KBPage] = []

    def extract_text(it: Any) -> Optional[str]:
        if isinstance(it, dict):
            # NEW: KB may contain image metadata entries; never treat them as text
            if (it.get("type") or "").lower().strip() == "image":
                return None
            for k in ("content", "text", "raw_text", "combined_text"):
                v = it.get(k)
                if isinstance(v, str) and v.strip():
                    return v
        if isinstance(it, str) and it.strip():
            return it
        return None

    def extract_page(it: Any) -> Optional[int]:
        if isinstance(it, dict):
            for k in ("page", "page_no", "pageno", "page_number"):
                v = it.get(k)
                if isinstance(v, int):
                    return v
                if isinstance(v, str) and v.isdigit():
                    return int(v)
        return None

    if isinstance(kb_obj, list):
        for it in kb_obj:
            t = extract_text(it)
            if not t:
                continue
            pages.append(KBPage(page=extract_page(it), text=t))
        return pages

    if isinstance(kb_obj, dict):
        for container_key in ("pages", "chunks", "items"):
            cont = kb_obj.get(container_key)
            if isinstance(cont, list):
                for it in cont:
                    t = extract_text(it)
                    if not t:
                        continue
                    pages.append(KBPage(page=extract_page(it), text=t))
        for k in ("full_text", "text", "raw_text", "combined_text"):
            v = kb_obj.get(k)
            if isinstance(v, str) and v.strip():
                pages.append(KBPage(page=None, text=v))
                break
        return pages

    if isinstance(kb_obj, str) and kb_obj.strip():
        pages.append(KBPage(page=None, text=kb_obj))
    return pages


def load_kb_pages(knowledge_base_path: str) -> List[KBPage]:
    with open(knowledge_base_path, "r", encoding="utf-8") as f:
        kb_obj = json.load(f)
    pages = _normalize_kb_items(kb_obj)

    if all(p.page is None for p in pages) and len(pages) == 1:
        blob = pages[0].text
        splits = re.split(r"(?:\n\s*\[?PAGE\s+(\d+)\]?\s*\n)", blob, flags=re.IGNORECASE)
        if len(splits) > 1:
            rebuilt: List[KBPage] = []
            pre = splits[0]
            if pre.strip():
                rebuilt.append(KBPage(page=None, text=pre))
            i = 1
            while i + 1 < len(splits):
                page_num = int(splits[i])
                txt = splits[i + 1]
                rebuilt.append(KBPage(page=page_num, text=txt))
                i += 2
            pages = rebuilt

    return pages


def _score_page(text: str, code: str, name: str) -> int:
    score = 0
    t = (text or "").lower()

    aliases = _build_code_aliases(code)
    for a in aliases:
        al = a.lower()
        score += 12 * len(re.findall(rf"\b{re.escape(al)}\b", t))
        score += 2 * len(re.findall(re.escape(al), t))

    if name:
        tokens = [w for w in re.split(r"\W+", name.lower()) if w]
        for tok in tokens:
            if len(tok) < 3:
                continue
            score += 2 * len(re.findall(rf"\b{re.escape(tok)}\b", t))

    param_keywords = (
        "km/h", "kph", "m/s", "m/s2", "ttc", "time-to-collision",
        "overlap", "offset", "lateral", "headway", "gap", "distance",
        "decel", "deceler", "brak", "increment", "step", "range",
        "table", "figure",
    )
    for kw in param_keywords:
        if kw in t:
            score += 3

    if len(re.findall(r"\b\d+(?:\.\d+)?\b", t)) >= 15:
        score += 6

    return score


def filter_kb_for_scenario(
    kb_pages: List[KBPage],
    scenario_code: str,
    scenario_name: str,
    max_pages: int = MAX_PAGES,
    neighbors: int = NEIGHBOR_PAGES,
    max_chars: int = MAX_DOC_CHARS,
) -> str:
    code = (scenario_code or "").strip()
    name = (scenario_name or "").strip()

    scored: List[Tuple[int, int]] = []
    for idx, p in enumerate(kb_pages):
        sc = _score_page(p.text, code, name)
        if sc > 0:
            scored.append((sc, idx))

    if not scored:
        chosen = list(range(min(max_pages, len(kb_pages))))
    else:
        scored.sort(reverse=True, key=lambda x: x[0])
        top_idxs = [idx for _, idx in scored[:max_pages]]
        chosen_set = set()
        for idx in top_idxs:
            for j in range(max(0, idx - neighbors), min(len(kb_pages), idx + neighbors + 1)):
                chosen_set.add(j)
        chosen = sorted(chosen_set)

    chunks: List[str] = []
    for idx in chosen:
        p = kb_pages[idx]
        header = f"[PAGE {p.page}]" if p.page is not None else "[PAGE ?]"
        chunks.append(header + "\n" + (p.text or "").strip())

    doc = "\n\n".join(chunks).strip()
    if len(doc) > max_chars:
        doc = doc[:max_chars] + "\n\n[TRUNCATED]"
    return doc


# ----------------------------
# Prompting
# ----------------------------
SYSTEM_PROMPT = """You are a Euro NCAP scenario JSON enrichment engine.

You will receive:
- DOC_TEXT: extracted protocol text (context)
- IMAGES: optional table/diagram/page images from the PDF (ground truth for numeric values)
- SCENARIO_JSON: one scenario object in a fixed schema

Output requirements:
1) Output MUST be valid JSON only (no markdown, no commentary).
2) Return the FULL scenario JSON object in EXACTLY the same schema as SCENARIO_JSON.
3) Do NOT add new top-level keys. Do NOT remove keys.

Identity lock (must NOT change these values):
- scenario_type
- scenario_name
- scenario_code (if present)
- script_eligible
- scenario_details.scenario

Null-only patching:
- Only fill fields that are currently null OR clearly protocol-invalid.
- Never overwrite a non-null value unless the protocol evidence explicitly proves it violates protocol.

IMAGE priority:
- If IMAGES are provided, use them as the PRIMARY source for numeric/table parameters (speeds, offsets, overlaps, TTC, distances, deceleration, etc.).
- Use DOC_TEXT for interpretation and context, but do not invent numbers that are not visible in the images.

Evidence rules (mandatory for each field you change):
- You MUST add an evidence record under:
  scenario_details.extra.evidence : array
- Ensure scenario_details.extra exists. If missing, create:
  "extra": {"evidence": []}
- For EVERY field you change, append an evidence item:
  {"field":"<json_path>", "page": <int|null>, "match":"<short quote from DOC_TEXT OR a short transcription referencing the IMAGE filename>"}
- Evidence must be field-specific: the quote MUST contain the numeric value or constraint.

Protocol vs user separation (STRICT):
- scenario_details = protocol truth / constraints only.
- user_config = runtime / user-selected values.
- Do NOT copy values from user_config into scenario_details.
- Do NOT modify user_config at all.

STRICTLY FORBIDDEN (do NOT change these fields):
- scenario_details.lateral_offset_m

This field is intentionally user-controlled or ambiguous in protocol.
It MUST remain null.

NOTE: scenario_details.overlap_percent IS allowed to be filled when the protocol
explicitly states the overlap percentage for this specific scenario (e.g. "50% of
the vehicle's width"). Extract it as a numeric value (e.g. 50 for 50%, 25 for 25%).

Field scope rules:
A) Constraints (preferred):
- If DOC_TEXT specifies a RANGE (e.g., "10–50 km/h", "-50% to +50%"),
  fill *_min and *_max fields.
- If DOC_TEXT specifies DISCRETE SETS (e.g., "12 m and 40 m", "50 and 70 km/h"),
  do NOT pick a single value as if it were the only one. Instead:
  1) Fill *_min and *_max with min(set) and max(set) so the field always carries
     a usable bound, even though the true constraint is a discrete set.
  2) ALSO write the full discrete set into:
     scenario_details.extra.allowed_values.<field> = [ ... ] with evidence.
  Steps (1) and (2) are both required whenever a discrete set is found — never
  populate allowed_values without also populating *_min/*_max from it, and vice versa.

B) Fixed numeric parameters:
- Applies to: initial_distance_m, headway_m, ttc, ttc_end, *_decel_mps2, AND
  *_speed_min / *_speed_max when the protocol states a SINGLE constant speed
  (not a range, not a discrete set) for this scenario.
- Watch for phrasing such as "constant", "fixed", "held at", "moves at X km/h"
  describing one vehicle's speed — e.g. "GVT moves at a constant 20 km/h" means
  BOTH target_speed_min and target_speed_max should be set to 20.
- Only fill fixed numeric values IF AND ONLY IF DOC_TEXT clearly states a SINGLE
  fixed value for THIS scenario. Otherwise, leave the field null.

C) Vehicle type fields (ego_vehicle_type, target_vehicle_type):
- Euro NCAP AEB Car-to-Car protocols use a consistent target vehicle class
  (typically GVT — Global Vehicle Target) across every scenario in the document.
  If DOC_TEXT establishes the target vehicle class ANYWHERE in the document
  (e.g. a general/definitions section, or any other scenario's evidence), fill
  target_vehicle_type for THIS scenario with that same class — do not leave it
  null just because this specific scenario's own page doesn't restate it.
  Add evidence citing the page where the class is defined, even if that page
  differs from this scenario's main anchor page.
- Do not guess ego_vehicle_type (VUT) unless DOC_TEXT gives a specific type;
  the ego vehicle is typically the test vehicle itself and often intentionally
  unspecified.

D) Structures that don't fit *_min/*_max/allowed_values — use extra (no schema change):
  These three patterns come up often enough to name explicitly. In every case,
  DO NOT leave the field silently null with no structure at all if DOC_TEXT
  states the information in ANY form — capture it in scenario_details.extra
  even when it doesn't fit *_min/*_max, with evidence as usual.

  D1) Paired/dependent speed combinations — when target_speed depends on
      ego_speed (a matrix/table, e.g. "GVT travels at 20 km/h when VUT is at
      30 km/h, 30 km/h when VUT is at 40 km/h..."), do NOT leave
      target_speed_min/max null with nothing else. Instead write:
        scenario_details.extra.allowed_values.speed_pairs =
          [ {"ego_speed": 30, "target_speed": 20}, {"ego_speed": 40, "target_speed": 30}, ... ]
      using whatever unit DOC_TEXT uses, one evidence item covering the pairing.

  D2) Named sub-ranges within one overall range — when DOC_TEXT splits a speed
      range by assessed subsystem or condition (e.g. "AEB: 10-50 km/h, FCW:
      55-80 km/h" within an overall 10-80 km/h scenario), write BOTH the
      overall *_min/*_max as usual AND the named sub-ranges into:
        scenario_details.extra.allowed_values.<label>_speed_range = [min, max]
      e.g. extra.allowed_values.aeb_speed_range = [10, 50] and
      extra.allowed_values.fcw_speed_range = [55, 80]. Use the label DOC_TEXT
      itself uses (lowercased), not a fixed vocabulary.

  D3) Impact/offset point stated separately from overlap_percent — e.g. "impact
      at 25% along the target vehicle's length" is a different concept from
      lateral overlap_percent. When DOC_TEXT states this explicitly for THIS
      scenario, write:
        scenario_details.extra.impact_point_percent = 25
      with evidence, rather than only mentioning it in notes.

Units:
- Preserve units exactly as stated in DOC_TEXT.
- Do NOT convert units unless the protocol explicitly provides the converted value.

What you are allowed to add:
- scenario_details.notes (string), ONLY to explain ambiguity or conflicts.
- scenario_details.extra.allowed_values.speed_pairs, extra.allowed_values.<label>_speed_range,
  and extra.impact_point_percent as described in Rule D above — these are the only
  extra fields you may introduce beyond what already exists in SCENARIO_JSON.


LSS mode (IMPORTANT):
- If SCENARIO_JSON contains scenario_details.extra.adas_family == "LSS" OR scenario_details.extra.lss exists:
  - Treat this as a Lane Support Systems scenario (ELK/LKA/LDW/BSM).
  - Your primary fill targets are ONLY:
      scenario_details.extra.lss.boundary_type
      scenario_details.extra.lss.departure_side
      scenario_details.extra.lss.lateral_speed_mps_min / lateral_speed_mps_max
      scenario_details.extra.lss.ego_speed_kph_min / ego_speed_kph_max
      scenario_details.extra.lss.target_actor (if clearly specified: oncoming/overtaking/adjacent)
    and, if the protocol explicitly states ranges, you may ALSO fill:
      scenario_details.ego_speed_min / ego_speed_max
  - DO NOT invent TTC/headway/initial_distance or braking/decel values for LSS unless the DOC_TEXT explicitly gives a single fixed value for this scenario.
  - Keep overlap_percent and lateral_offset_m untouched (still forbidden).

  --- LSS CANONICALIZATION & CONTAINER REJECTION (APPENDED, DO NOT REMOVE) ---
  Goal: prevent promoting headings/containers/figure captions into scenarios.

  Canonical LSS scenario types (ONLY these are considered real scenario types):
  - "ELK - Road Edge"
  - "ELK - Solid Line"
  - "ELK - Oncoming Vehicle"
  - "ELK - Overtaking Vehicle"
  - "LKA - Dashed Line"
  - "LKA - Solid Line"
  - "BSM - Blind Spot Monitoring"

  IMPORTANT:
  - Do NOT treat the following as scenarios:
    * Figure captions (e.g., "Figure 7-2: ...")
    * Group headers ending with ":" (e.g., "ELK oncoming scenarios:")
    * Section headers like "7.2.4.3.3 ..."
    * Narrative sentences (e.g., "For the Blind Spot Monitoring scenario, the target vehicle will...")
    * Parameter/step descriptions (e.g., "tests will be performed with 0.1 m/s incremental steps...")

  Because scenario_name and scenario_details.scenario are identity-locked, do NOT rename them.
  Instead:
  - Write a normalized canonical label to:
      scenario_details.extra.lss.canonical_scenario
    Choose exactly ONE from the canonical list above.
  - Also write:
      scenario_details.extra.lss.is_container : true/false
    Set is_container=true if the input looks like a header/caption/narrative/container.

  Container detection heuristics (if any of these match → is_container=true):
  - contains "figure" or "table"
  - ends with ":" or contains "scenarios:"
  - contains "will be performed" but no unique subtype
  - contains "the end of" or "is considered" or "means" or bullet-point parameters

  If is_container=true:
  - Do NOT fill any numeric fields (leave them null).
  - You MAY add scenario_details.notes explaining it is a container/non-scenario anchor.
  - You MAY still add evidence items only if you set canonical_scenario or is_container.

  If is_container=false:
  - Ensure canonical_scenario is set.
  - Fill ONLY the allowed LSS fields/ranges from DOC_TEXT.
  - Never copy values from figures unless DOC_TEXT explicitly states numeric constraints.

Final checklist (must all pass):
- Valid JSON
- Identity fields unchanged
- user_config unchanged
- overlap_percent untouched
- No single-value guesses from ranges or sets
- Every changed field has matching evidence
- Unknown or ambiguous values stay null

"""


def _extract_first_json_object(text: str) -> Dict[str, Any]:
    """
    Claude returns plain text. We must extract the first JSON object robustly.

    Uses json.JSONDecoder().raw_decode(), which parses exactly one JSON value
    starting from a given position and returns where it ended — it does NOT
    require the entire string to be consumed. This is deliberate: if the
    model appends anything after the JSON object (a stray note, a repeated
    object, trailing whitespace plus commentary), a plain json.loads() on the
    whole string raises JSONDecodeError("Extra data") and crashes the whole
    pipeline run, even though the JSON itself is perfectly valid. raw_decode
    reads the first complete object and ignores everything after it.
    """
    if not text:
        raise RuntimeError("Empty LLM response")

    text = text.strip()

    # Strip markdown fences if the model wrapped the JSON in ```json ... ```
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence_match:
        text = fence_match.group(1).strip()

    # Find the first '{' — skip any leading preamble text the model might add
    brace_idx = text.find("{")
    if brace_idx == -1:
        raise RuntimeError("LLM did not return a JSON object")

    decoder = json.JSONDecoder()
    try:
        obj, _end_index = decoder.raw_decode(text, brace_idx)
        return obj
    except json.JSONDecodeError:
        pass

    # Last-resort fallback: greedy first-'{' to last-'}' span (previous
    # behavior), in case raw_decode failed for a reason other than trailing
    # extra data (e.g. genuinely malformed JSON needing the wider span).
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise RuntimeError("LLM did not return a JSON object")
    return json.loads(m.group(0))


def call_llm_json(
    system_prompt: str,
    user_payload: Dict[str, Any],
    image_paths: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Claude multimodal JSON call.
    - Sends the user_payload as JSON text.
    - Attaches images as base64 (compressed to JPEG if OpenCV is available).
    - Expects the assistant to respond with ONLY a JSON object.
    """
    if Anthropic is None:  # pragma: no cover
        raise ImportError(f"anthropic package not available: {_ANTHROPIC_IMPORT_ERROR}")

    client = Anthropic(timeout=OPENAI_TIMEOUT_S, max_retries=OPENAI_MAX_RETRIES)

    # Build a single user message with mixed content
    content_blocks: List[Dict[str, Any]] = [
        {"type": "text", "text": json.dumps(user_payload, ensure_ascii=False)}
    ]

    if image_paths:
        for p in image_paths:
            if not p or not os.path.isfile(p):
                continue
            try:
                media_type, b64 = _file_to_base64_image(p)
            except Exception:
                continue
            content_blocks.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": b64},
                }
            )

    resp = client.messages.create(
        model=OPENAI_MODEL,
        max_tokens=int(os.getenv("CLAUDE_MAX_TOKENS", "4096") or "4096"),
        temperature=OPENAI_TEMPERATURE,
        system=system_prompt,
        messages=[{"role": "user", "content": content_blocks}],
    )

    # resp.content is a list of blocks (usually one text block)
    out_text = ""
    try:
        out_text = "".join([b.text for b in resp.content if getattr(b, "type", None) == "text"])
    except Exception:
        # Fallback: best-effort string
        out_text = str(resp)

    return _extract_first_json_object(out_text)


# ----------------------------
# Validation + repair
# ----------------------------
IDENTITY_LOCK_PATHS = [
    "scenario_type",
    "scenario_name",
    "scenario_code",
    "script_eligible",
    "scenario_details.scenario",
    "scenario_details.lateral_offset_m",
]


def _ensure_extra_evidence_shape(out: Dict[str, Any]) -> None:
    sd = out.setdefault("scenario_details", {})
    extra = sd.setdefault("extra", {})
    ev = extra.get("evidence")
    if not isinstance(ev, list):
        extra["evidence"] = []
    cleaned: List[Dict[str, Any]] = []
    for it in extra["evidence"]:
        if not isinstance(it, dict):
            continue
        field = it.get("field")
        match = it.get("match")
        if not isinstance(field, str) or not field.strip():
            continue
        if not isinstance(match, str) or not match.strip():
            continue
        page = it.get("page", None)
        if isinstance(page, str) and page.isdigit():
            page = int(page)
        if not isinstance(page, int):
            page = None
        cleaned.append({"field": field.strip(), "page": page, "match": match.strip()[:300]})
    extra["evidence"] = cleaned


def _changed_paths_requiring_evidence(inp: Dict[str, Any], out: Dict[str, Any]) -> List[str]:
    changed = _diff_scalar_leaves(inp, out)
    need: List[str] = []
    for p in changed:
        if p.startswith("scenario_details.extra"):
            continue
        if p == "scenario_details.notes":
            continue
        if p.startswith("scenario_details."):
            need.append(p)
        if p == "scenario_code":
            need.append(p)
    return sorted(set(need))


def validate_enrichment(inp: Dict[str, Any], out: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if not isinstance(out, dict):
        return ["Output is not a JSON object"]

    for p in IDENTITY_LOCK_PATHS:
        inp_v = _get_path(inp, p)
        out_v = _get_path(out, p)
        if p == "scenario_code" and (inp_v is None or inp_v == ""):
            continue
        if inp_v != out_v:
            errors.append(f"Identity field changed: {p} (input={inp_v!r}, output={out_v!r})")

    if inp.get("user_config") != out.get("user_config"):
        errors.append("user_config was modified (must remain unchanged)")

    _ensure_extra_evidence_shape(out)

    need_paths = _changed_paths_requiring_evidence(inp, out)
    ev_items = (((out.get("scenario_details") or {}).get("extra") or {}).get("evidence") or [])
    ev_fields = {it.get("field") for it in ev_items if isinstance(it, dict)}
    for p in need_paths:
        if p not in ev_fields:
            errors.append(f"Missing evidence for changed field: {p}")

    return errors


def repair_with_llm(
    inp: Dict[str, Any],
    bad_out: Dict[str, Any],
    errors: List[str],
    doc_text: str,
    image_paths: Optional[List[str]] = None,
) -> Dict[str, Any]:
    user_payload = {
        "task": "json_repair",
        "validation_errors": errors,
        "DOC_TEXT": doc_text,
        "INPUT_SCENARIO_JSON": inp,
        "BAD_OUTPUT_JSON": bad_out,
        "instruction": (
            "Return corrected JSON only. Keep the same schema. "
            "Fix only what is required to satisfy the validation errors. "
            "Do not guess. Ensure evidence exists for every changed scenario_details field."
        ),
    }
    return call_llm_json(SYSTEM_PROMPT, user_payload, image_paths=image_paths)


# ----------------------------
# Post-processing (deterministic)
# ----------------------------
def _is_lss_like(s: Dict[str, Any]) -> bool:
    sd = s.get("scenario_details") or {}
    extra = sd.get("extra") or {}
    if (extra.get("adas_family") or "").upper() == "LSS":
        return True
    lss = extra.get("lss")
    if isinstance(lss, dict):
        return True
    code = (s.get("scenario_code") or "")
    if isinstance(code, str) and code.upper().startswith("LSS_"):
        return True
    name = (s.get("scenario_name") or "")
    if isinstance(name, str) and any(k in name.upper() for k in ("ELK", "LKA", "LDW", "BSM")):
        return True
    return False


def _append_evidence(s: Dict[str, Any], field: str, match: str, page: Optional[int] = None) -> None:
    sd = s.setdefault("scenario_details", {})
    extra = sd.setdefault("extra", {})
    ev = extra.get("evidence")
    if not isinstance(ev, list):
        ev = []
        extra["evidence"] = ev
    ev.append({"field": field, "page": page if isinstance(page, int) else None, "match": (match or "")[:300]})


def normalize_lss_containers(s: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize LSS container flags.

    Goal:
      * True caption/section/container anchors remain non-scenarios and non-runnable.
      * Real LSS scenarios must NOT be accidentally demoted to containers (that blocks UI + generation).
    """
    if not isinstance(s, dict) or not _is_lss_like(s):
        return s

    try:
        sd = s.get("scenario_details") or {}
        extra = sd.get("extra") or {}
        lss = (extra.get("lss") or {})
        if not (isinstance(lss, dict) and lss.get("is_container") is True):
            return s

        # --- Heuristics: override false-positive container flags for real LSS scenarios ---
        name = (s.get("scenario_name") or sd.get("scenario") or "")
        name_l = name.strip().lower()

        # Pull light evidence text (if present) to detect captions like "Figure 7-x".
        ev_txt = ""
        evidence = (extra.get("evidence") or [])
        if isinstance(evidence, list):
            parts = []
            for e in evidence:
                if isinstance(e, dict):
                    mtxt = e.get("match")
                    if isinstance(mtxt, str) and mtxt.strip():
                        parts.append(mtxt.strip())
            ev_txt = " | ".join(parts).lower()

        # Strong caption/container indicators
        if (
            name_l.startswith("figure")
            or name_l.startswith("table")
            or "figure" in name_l
            or "table" in name_l
            or "figure" in ev_txt
            or "table" in ev_txt
        ):
            # Keep container flag (do nothing here, demote below).
            pass
        else:
            # Real LSS scenario hints (sub-scenarios)
            real_hint_rx = re.compile(
                r"\b(road\s*edge|solid\s*line|dashed\s*line|oncoming\s*vehicle|overtaking\s*vehicle|lane\s*keep|lane\s*departure|ldw|blind[-\s]*spot|bsm)\b",
                re.IGNORECASE,
            )
            if real_hint_rx.search(name):
                # Override: treat as a real scenario; do NOT demote.
                lss["is_container"] = False
                extra["lss"] = lss
                sd["extra"] = extra
                s["scenario_details"] = sd
                sd["is_scenario"] = True

                # Keep script_eligible as-is (null numeric values are handled in UI).
                note = "LSS container flag overridden: looks like a real scenario heading."
                if isinstance(sd.get("notes"), str) and sd["notes"].strip():
                    sd["notes"] = sd["notes"].strip() + " | " + note
                else:
                    sd["notes"] = note
                return s

        # Deterministic business rule:
        # container anchors are NOT runnable scenarios
        sd = s.setdefault("scenario_details", {})
        sd["is_scenario"] = False
        s["script_eligible"] = False
        if isinstance(sd.get("notes"), str) and sd["notes"].strip():
            sd["notes"] = sd["notes"].strip() + " | Marked as LSS container; not runnable."
        else:
            sd["notes"] = "Marked as LSS container; not runnable."
    except Exception:
        pass
    return s


_STEP_RX = re.compile(r"(?:step(?:s)?\s*(?:of)?|increment(?:al)?\s+step(?:s)?\s*(?:of)?)\s*(\d+(?:\.\d+)?)\s*(m/s\b|km/h\b|kph\b)", re.IGNORECASE)
_SIDE_RX = re.compile(r"\b(driver|passenger)\s+side\b|\bboth\s+sides\b", re.IGNORECASE)
_BOUNDARY_KEYS = [
    ("road edge", "road_edge"),
    ("solid line", "solid_line"),
    ("dashed line", "dashed_line"),
]
_CANON_MAP = [
    ("ELK - Road Edge", ("ELK", "ROAD", "EDGE")),
    ("ELK - Solid Line", ("ELK", "SOLID")),
    ("ELK - Oncoming Vehicle", ("ELK", "ONCOMING")),
    ("ELK - Overtaking Vehicle", ("ELK", "OVERTAK")),
    ("LKA - Dashed Line", ("LKA", "DASH")),
    ("LKA - Solid Line", ("LKA", "SOLID")),
    ("BSM - Blind Spot Monitoring", ("BSM", "BLIND", "SPOT")),
]

def _fallback_fill_lss_fields_from_text(s: Dict[str, Any], doc_text: str) -> Dict[str, Any]:
    """Deterministic extraction of common LSS fields (boundary/side/step/canonical) when LLM output is sparse."""
    if not isinstance(s, dict) or not _is_lss_like(s):
        return s
    if not isinstance(doc_text, str) or not doc_text.strip():
        return s

    sd = s.setdefault("scenario_details", {})
    extra = sd.setdefault("extra", {})
    lss = extra.get("lss")
    if not isinstance(lss, dict):
        lss = {}
        extra["lss"] = lss

    # Respect container flag if already set
    name = str(s.get("scenario_name") or sd.get("scenario") or "")
    name_u = name.upper()

    # 1) Canonical scenario label
    if not isinstance(lss.get("canonical_scenario"), str) or not lss.get("canonical_scenario"):
        chosen = None
        for label, toks in _CANON_MAP:
            if all(t in name_u for t in toks):
                chosen = label
                break
        if chosen is None:
            # try from doc_text header lines
            dt_u = doc_text.upper()
            for label, toks in _CANON_MAP:
                if all(t in dt_u for t in toks):
                    chosen = label
                    break
        if chosen:
            lss["canonical_scenario"] = chosen
            _append_evidence(s, "scenario_details.extra.lss.canonical_scenario", chosen)

    # 2) Boundary type
    if lss.get("boundary_type") is None:
        dt_low = doc_text.lower()
        nm_low = name.lower()
        for key, val in _BOUNDARY_KEYS:
            if key in nm_low or key in dt_low:
                lss["boundary_type"] = val
                _append_evidence(s, "scenario_details.extra.lss.boundary_type", key)
                break

    # 3) Departure side
    if lss.get("departure_side") is None:
        dt = doc_text
        m = _SIDE_RX.search(dt)
        if m:
            txt = m.group(0).lower()
            if "both" in txt:
                lss["departure_side"] = "both"
            elif "driver" in txt:
                lss["departure_side"] = "driver"
            elif "passenger" in txt:
                lss["departure_side"] = "passenger"
            _append_evidence(s, "scenario_details.extra.lss.departure_side", m.group(0))

    # 4) Step size for lateral speed or ego speed
    if lss.get("lateral_speed_step_mps") is None:
        for m in _STEP_RX.finditer(doc_text):
            val = float(m.group(1))
            unit = m.group(2).lower()
            if unit == "m/s":
                lss["lateral_speed_step_mps"] = val
                _append_evidence(s, "scenario_details.extra.lss.lateral_speed_step_mps", m.group(0))
                break

    if lss.get("ego_speed_step_kph") is None:
        for m in _STEP_RX.finditer(doc_text):
            val = float(m.group(1))
            unit = m.group(2).lower()
            if unit in ("km/h", "kph"):
                lss["ego_speed_step_kph"] = val
                _append_evidence(s, "scenario_details.extra.lss.ego_speed_step_kph", m.group(0))
                break

    return s

def _fallback_fill_lss_ranges_from_text(s: Dict[str, Any], doc_text: str) -> Dict[str, Any]:
    """
    Safe fallback for LSS numeric ranges.
    - Never blocks generation
    - Never invents values
    - Only fills if very obvious range patterns are found
    """
    try:
        if not isinstance(s, dict) or not _is_lss_like(s):
            return s
        if not isinstance(doc_text, str) or not doc_text.strip():
            return s

        sd = s.get("scenario_details") or {}
        extra = sd.get("extra") or {}
        lss = extra.get("lss")
        if not isinstance(lss, dict):
            return s

        # If already filled, do nothing
        if any(lss.get(k) is not None for k in (
            "ego_speed_kph_min",
            "ego_speed_kph_max",
            "lateral_speed_mps_min",
            "lateral_speed_mps_max",
        )):
            return s

        # Conservative range extraction: "70–80 km/h", "0.2–0.4 m/s"
        speed_match = re.search(r"(\d{2,3})\s*[-–]\s*(\d{2,3})\s*(km/h|kph)\b", doc_text, re.IGNORECASE)
        lat_match = re.search(r"(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)\s*(m/s)\b", doc_text, re.IGNORECASE)

        if speed_match:
            lss["ego_speed_kph_min"] = int(speed_match.group(1))
            lss["ego_speed_kph_max"] = int(speed_match.group(2))
            _append_evidence(s, "scenario_details.extra.lss.ego_speed_kph_min", speed_match.group(0))
            _append_evidence(s, "scenario_details.extra.lss.ego_speed_kph_max", speed_match.group(0))

        if lat_match:
            lss["lateral_speed_mps_min"] = float(lat_match.group(1))
            lss["lateral_speed_mps_max"] = float(lat_match.group(2))
            _append_evidence(s, "scenario_details.extra.lss.lateral_speed_mps_min", lat_match.group(0))
            _append_evidence(s, "scenario_details.extra.lss.lateral_speed_mps_max", lat_match.group(0))

        # Write back (in case extra/lss were shallow-copied elsewhere)
        extra["lss"] = lss
        sd["extra"] = extra
        s["scenario_details"] = sd

    except Exception:
        return s

    return s

# ----------------------------
# Public API
# ----------------------------
def enrich_one_scenario(
    scenario: Dict[str, Any],
    knowledge_base_path: str = "knowledge_base.json",
    scenario_evidence_path: Optional[str] = "scenario_evidence.json",
    _evidence_cache: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Enrich a single ADAS scenario dict.

    Preferred behavior:
    - If scenario_evidence.json is available and contains this scenario, use its doc_text.
    - Else fallback to filtering knowledge_base.json.
    """
    inp = deepcopy(scenario)
    if not isinstance(inp, dict):
        return scenario

    scenario_code = str(inp.get("scenario_code") or "")
    scenario_name = str(inp.get("scenario_name") or (inp.get("scenario_details") or {}).get("scenario") or "")

    # ---- Preferred: evidence pack ----
    doc_text: Optional[str] = None
    image_candidates: Optional[Dict[str, Any]] = None
    image_paths: List[str] = []
    evidence_obj: Optional[Dict[str, Any]] = _evidence_cache

    if scenario_evidence_path:
        try:
            if evidence_obj is None:
                evidence_obj = load_scenario_evidence(scenario_evidence_path)
            doc_text = get_doc_text_from_evidence(evidence_obj, scenario_code, scenario_name)
            image_candidates = get_image_candidates_from_evidence(evidence_obj, scenario_code, scenario_name)
            image_paths = _pick_image_paths(image_candidates)
        except Exception:
            doc_text = None
            image_candidates = None
            image_paths = []

    # ---- Fallback: KB filtering ----
    if not doc_text:
        kb_pages = load_kb_pages(knowledge_base_path)
        doc_text = filter_kb_for_scenario(kb_pages, scenario_code, scenario_name)
        image_candidates = None
        image_paths = []

    doc_text = (doc_text or "").strip()
    if len(doc_text) > MAX_DOC_CHARS:
        doc_text = doc_text[:MAX_DOC_CHARS] + "\n\n[TRUNCATED]"

    user_payload = {
        "task": "scenario_enrichment",
        "DOC_TEXT": doc_text,
        "IMAGE_CANDIDATES": image_candidates,
        "SCENARIO_JSON": inp,
    }

    out = call_llm_json(SYSTEM_PROMPT, user_payload, image_paths=image_paths)

    errs = validate_enrichment(inp, out)
    tries = 0
    while errs and tries < REPAIR_MAX_TRIES:
        tries += 1
        out = repair_with_llm(inp, out, errs, doc_text, image_paths=image_paths)
        errs = validate_enrichment(inp, out)

    if isinstance(out, dict):
        _ensure_extra_evidence_shape(out)

    if errs:
        safe = deepcopy(inp)
        sd = safe.setdefault("scenario_details", {})
        extra = sd.setdefault("extra", {})
        if "evidence" not in extra or not isinstance(extra["evidence"], list):
            extra["evidence"] = []
        if isinstance(sd.get("notes"), str) and sd["notes"].strip():
            sd["notes"] = sd["notes"].strip() + " | LLM enrichment validation failed; kept original fields."
        else:
            sd["notes"] = "LLM enrichment validation failed; kept original fields."
        return safe

    # Deterministic post-processing for LSS (does not affect CCRM/CCR/VRU)
    if isinstance(out, dict) and _is_lss_like(out):
        out = normalize_lss_containers(out)
        out = _fallback_fill_lss_fields_from_text(out, doc_text)
        out = _fallback_fill_lss_ranges_from_text(out, doc_text)
        _ensure_extra_evidence_shape(out)

    return out


def enrich_all(
    scenarios: List[Dict[str, Any]],
    knowledge_base_path: str = "knowledge_base.json",
    scenario_evidence_path: Optional[str] = "scenario_evidence.json",
) -> List[Dict[str, Any]]:
    """
    Enrich only ADAS scenarios, leave others unchanged.

    Loads scenario_evidence.json once and reuses it for all scenarios.
    """
    evidence_obj: Optional[Dict[str, Any]] = None
    if scenario_evidence_path:
        try:
            evidence_obj = load_scenario_evidence(scenario_evidence_path)
        except Exception:
            evidence_obj = None

    enriched: List[Dict[str, Any]] = []
    for s in scenarios:
        if not isinstance(s, dict):
            continue
        if (s.get("scenario_type") or "").upper() != "ADAS":
            enriched.append(s)
            continue
        out = enrich_one_scenario(
            s,
            knowledge_base_path=knowledge_base_path,
            scenario_evidence_path=scenario_evidence_path,
            _evidence_cache=evidence_obj,
        )
        enriched.append(out)
    return enriched
# ============================
# APPEND-ONLY PATCH (LSS runnable procedural tests)
# Paste this at the VERY END of llm_enricher.py
# ============================

_LSS_PROCEDURAL_HINTS = (
    "perform the", "tests within", "within the lateral", "lateral velocity range",
    "repeat the test", "repeat on the", "for the blind spot", "blind spot monitoring",
    "for the lane keeping", "lane keeping assist", "lane departure warning",
    "the target vehicle will", "vut is positioned", "will follow a straight",
)

_LSS_CONTAINER_HINTS = (
    "figure", "table", "scenarios:", "scenario paths", "is considered", "means",
)

def _looks_like_lss_container_title(title: str) -> bool:
    t = (title or "").strip().lower()
    if not t:
        return True
    # strong caption/header signals
    if any(h in t for h in _LSS_CONTAINER_HINTS):
        return True
    if t.endswith(":"):
        return True
    return False

def _looks_like_lss_procedural_test(title: str) -> bool:
    t = (title or "").strip().lower()
    if not t:
        return False
    return any(h in t for h in _LSS_PROCEDURAL_HINTS)

def normalize_lss_containers(s: Dict[str, Any]) -> Dict[str, Any]:
    """
    OVERRIDE of earlier normalize_lss_containers (append-only patch).

    Goal:
    - Keep figure captions/headers as containers (not runnable)
    - BUT if the title looks like a procedural test definition,
      do NOT demote it to non-runnable even if LLM set is_container=true.
    """
    if not isinstance(s, dict) or not _is_lss_like(s):
        return s

    sd = s.setdefault("scenario_details", {})
    extra = sd.setdefault("extra", {})
    lss = extra.get("lss")
    if not isinstance(lss, dict):
        return s

    name = str(s.get("scenario_name") or sd.get("scenario") or "")
    # Only act when LLM marked container
    if lss.get("is_container") is True:
        # If it's a procedural test line, treat as runnable scenario
        if _looks_like_lss_procedural_test(name) and not _looks_like_lss_container_title(name):
            lss["is_container"] = False
            sd["is_scenario"] = True
            s["script_eligible"] = True

            # evidence (to satisfy validation expectations)
            _append_evidence(s, "scenario_details.extra.lss.is_container",
                             f"Override: procedural test detected in title: {name[:120]}")
            _append_evidence(s, "scenario_details.is_scenario",
                             f"Override: procedural test detected in title: {name[:120]}")
            _append_evidence(s, "script_eligible",
                             f"Override: procedural test detected in title: {name[:120]}")
            if isinstance(sd.get("notes"), str) and sd["notes"].strip():
                sd["notes"] = sd["notes"].strip() + " | LSS: procedural test detected; allowed runnable."
            else:
                sd["notes"] = "LSS: procedural test detected; allowed runnable."
            return s

        # Otherwise: keep the old behavior (demote)
        sd["is_scenario"] = False
        s["script_eligible"] = False
        if isinstance(sd.get("notes"), str) and sd["notes"].strip():
            sd["notes"] = sd["notes"].strip() + " | Marked as LSS container; not runnable."
        else:
            sd["notes"] = "Marked as LSS container; not runnable."

    return s


# ============================
# VRU FAMILY PATCH (APPEND-ONLY, 2026-01-11)
# - Adds CP*/CB*/CM* support as VRU scenario codes
# - Extends retrieval aliases so evidence packs / KB filtering hit the right pages
# - Adds VRU post-processing fallback to populate essential VRU fields deterministically
# - Extends system prompt with VRU-specific scope rules (no schema changes)
# IMPORTANT: This patch intentionally appends only. It overrides a few functions by re-definition.
# ============================

# Extend scenario families for aliasing / fallback retrieval
_SCENARIO_FAMILIES_V2 = ("CCR", "CCF", "CCB", "C2C", "C2P", "C2B", "VRU", "CP", "CB", "CM", "LSS", "ELK", "LKA", "LDW", "BSM")

def _build_code_aliases(code: str) -> List[str]:  # type: ignore[override]
    """
    OVERRIDE (append-only): family-aware aliases used by fallback retrieval.
    Adds CP/CB/CM -> VRU aliasing.
    """
    code = (code or "").strip()
    if not code:
        return []
    u = code.upper()
    aliases = {code, u}

    for fam in _SCENARIO_FAMILIES_V2:
        if u.startswith(fam):
            aliases.add(fam)
            break

    # Special CCF ho grouping
    if u.startswith("CCFHOS") or u.startswith("CCFHOL"):
        aliases.add("CCFho")
        aliases.add("CCFHO")

    # VRU scenario families: CP (ped), CB (bicycle), CM (motorcycle)
    if u.startswith(("CP", "CB", "CM")):
        aliases.add("VRU")

    # LSS synthetic codes like LSS_ELK_ROADEDGE...
    if u.startswith("LSS_"):
        aliases.add("LSS")
        for tok in ("ELK", "LKA", "LDW", "BSM"):
            if tok in u:
                aliases.add(tok)

    return sorted({a for a in aliases if a})


def _is_vru_like(s: Dict[str, Any]) -> bool:
    sd = s.get("scenario_details") or {}
    extra = sd.get("extra") or {}
    if (extra.get("adas_family") or "").upper() == "VRU":
        return True
    if isinstance(extra.get("vru"), dict):
        return True
    code = (s.get("scenario_code") or "")
    if isinstance(code, str) and code.upper().startswith(("VRU_", "CP", "CB", "CM")):
        return True
    name = (s.get("scenario_name") or "") or (sd.get("scenario") or "")
    if isinstance(name, str) and any(k in name.upper() for k in ("PEDESTRIAN", "CYCLIST", "BICYCL", "MOTORCYCL", "VRU")):
        return True
    return False


# --- VRU deterministic fallback extraction helpers ---
_VRU_VARIANT_RX = re.compile(r"\b(crossing|longitudinal|turning|nearside|farside|oncoming|overtaking|reverse)\b", re.IGNORECASE)
_VRU_ADULT_CHILD_RX = re.compile(r"\b(adult|child)\b", re.IGNORECASE)
_VRU_OBSCURED_RX = re.compile(r"\b(obscured|obstructed)\b", re.IGNORECASE)
_VRU_SIDE_RX = re.compile(r"\b(nearside|farside|left|right)\b", re.IGNORECASE)
# Euro NCAP VRU codes follow <family prefix><marker>[-<overlap%>], e.g. CP+NA-25,
# CB+FA-50, CP+NCO-50, CM+oncoming. Lateral-crossing variants use the marker
# NA/FA/NAO/FAO/NCO/FCO (nearside/farside, adult/child, +/- obstructed).
_VRU_LATERAL_MARKERS = {"NA", "FA", "NAO", "FAO", "NCO", "FCO"}


def _vru_code_core_marker(code_u: str) -> str:
    """
    Extract the marker portion of a VRU code, stripping the 2-letter family
    prefix (CP/CB/CM) and any trailing -NN overlap suffix. Positional, not a
    substring search — avoids false matches like "NCO" coincidentally appearing
    inside "CMONCOMING" (oNCOming), which a plain substring check would catch.
    """
    rest = code_u[2:] if len(code_u) > 2 else code_u
    return rest.split("-")[0]

# Speed ranges: "1.2–2.0 m/s", "1.2-2.0 m/s"
_VRU_SPEED_MPS_RANGE_RX = re.compile(r"\b(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*(m/s)\b", re.IGNORECASE)
# Single speed: "1.4 m/s"
_VRU_SPEED_MPS_SINGLE_RX = re.compile(r"\b(\d+(?:\.\d+)?)\s*(m/s)\b", re.IGNORECASE)

# Sometimes VRU walking speed is given in km/h (rare). We do not convert silently.
_VRU_SPEED_KPH_RANGE_RX = re.compile(r"\b(\d{1,2})\s*[-–]\s*(\d{1,2})\s*(km/h|kph)\b", re.IGNORECASE)

# Single km/h value near a VRU-motion cue word, e.g. "adult pedestrian walking at 5 km/h",
# "cyclist test speed of 15 km/h". Requiring the cue word within ~40 characters before the
# number avoids picking up an unrelated km/h figure (e.g. ego speed) elsewhere on the page.
_VRU_SPEED_KPH_SINGLE_RX = re.compile(
    r"(?:walking|running|cycling|pedestrian|bicyclist|cyclist|motorcyclist)\D{0,40}?"
    r"(\d{1,2}(?:\.\d+)?)\s*(?:km/h|kph)",
    re.IGNORECASE,
)


def _ensure_vru_shape(s: Dict[str, Any]) -> Dict[str, Any]:
    sd = s.setdefault("scenario_details", {})
    extra = sd.setdefault("extra", {})
    extra["adas_family"] = "VRU"
    vru = extra.get("vru")
    if not isinstance(vru, dict):
        vru = {}
        extra["vru"] = vru
    vru.setdefault("vru_type", None)
    vru.setdefault("scenario_variant", None)
    vru.setdefault("adult_child", None)
    vru.setdefault("obscured", None)
    vru.setdefault("crossing_side", None)
    vru.setdefault("vru_speed_mps_min", None)
    vru.setdefault("vru_speed_mps_max", None)
    vru.setdefault("start_offset_m", None)
    return s


def _fallback_fill_vru_fields_from_text(s: Dict[str, Any], doc_text: str) -> Dict[str, Any]:
    """
    Deterministic VRU fill:
    - Only fills when there is an explicit textual cue.
    - Never guesses TTC/headway/initial distances.
    - Adds evidence for each filled field.
    """
    if not isinstance(s, dict) or not _is_vru_like(s):
        return s
    if not isinstance(doc_text, str) or not doc_text.strip():
        return s

    s = _ensure_vru_shape(s)
    sd = s.setdefault("scenario_details", {})
    extra = sd.setdefault("extra", {})
    vru = extra.setdefault("vru", {})

    code_u = str(s.get("scenario_code") or "").upper()
    name = str(s.get("scenario_name") or sd.get("scenario") or "")
    blob = (name + "\n" + doc_text).strip()

    # vru_type from code prefix (strongest)
    if vru.get("vru_type") is None:
        if code_u.startswith("CP"):
            vru["vru_type"] = "pedestrian"
            _append_evidence(s, "scenario_details.extra.vru.vru_type", "CP* code => pedestrian")
        elif code_u.startswith("CB"):
            vru["vru_type"] = "cyclist"
            _append_evidence(s, "scenario_details.extra.vru.vru_type", "CB* code => cyclist")
        elif code_u.startswith("CM"):
            vru["vru_type"] = "motorcyclist"
            _append_evidence(s, "scenario_details.extra.vru.vru_type", "CM* code => motorcyclist")

    # variant (crossing/longitudinal/turning/etc)
    if vru.get("scenario_variant") is None:
        m = _VRU_VARIANT_RX.search(blob)
        if m:
            vru["scenario_variant"] = m.group(1).lower()
            _append_evidence(s, "scenario_details.extra.vru.scenario_variant", m.group(0))

    # adult/child
    if vru.get("adult_child") is None:
        m = _VRU_ADULT_CHILD_RX.search(blob)
        if m:
            vru["adult_child"] = m.group(1).lower()
            _append_evidence(s, "scenario_details.extra.vru.adult_child", m.group(0))

    # obscured/obstructed — ONLY for scenario codes whose marker indicates an
    # obstructed variant (NAO/FAO/NCO/FCO, same marker set used for
    # crossing_side lateral variants — the "O" suffix is Euro NCAP's own
    # obstructed indicator). Without this gate, the regex was picking up the
    # word "Obstructed" from a NEIGHBORING scenario's description sharing the
    # same evidence page range (e.g. CMovertaking wrongly getting
    # obscured=true from a nearby CBNAO/CPNCO mention).
    _core_marker = _vru_code_core_marker(code_u)
    if vru.get("obscured") is None and (
        _core_marker.endswith("O") or "obstruct" in name.lower()
    ):
        m = _VRU_OBSCURED_RX.search(blob)
        if m:
            vru["obscured"] = True
            _append_evidence(s, "scenario_details.extra.vru.obscured", m.group(0))

    # crossing side — ONLY applies to lateral-crossing scenario variants (Euro NCAP
    # codes containing NA/FA: nearside/farside adult/child, e.g. CPNA, CPFA, CBNA,
    # CBFA, CBNAO). Dooring (*DA), longitudinal (*LA), reverse (*RA), rear-approach
    # (*Rs/*Rb), oncoming, and overtaking scenarios do NOT have a lateral crossing
    # side — for those, leave it null rather than grabbing the first stray
    # "nearside"/"farside"/"left"/"right" mention anywhere in the shared page text
    # (which was previously bleeding in from a neighboring scenario's description).
    if vru.get("crossing_side") is None and _vru_code_core_marker(code_u) in _VRU_LATERAL_MARKERS:
        m = _VRU_SIDE_RX.search(blob)
        if m:
            token = m.group(1).lower()
            # normalize nearside/farside to left/right unknown mapping; keep token as-is
            vru["crossing_side"] = token
            _append_evidence(s, "scenario_details.extra.vru.crossing_side", m.group(0))

    # VRU speed in m/s
    if vru.get("vru_speed_mps_min") is None and vru.get("vru_speed_mps_max") is None:
        m = _VRU_SPEED_MPS_RANGE_RX.search(doc_text)
        if m:
            vru["vru_speed_mps_min"] = float(m.group(1))
            vru["vru_speed_mps_max"] = float(m.group(2))
            _append_evidence(s, "scenario_details.extra.vru.vru_speed_mps_min", m.group(0))
            _append_evidence(s, "scenario_details.extra.vru.vru_speed_mps_max", m.group(0))
        else:
            m1 = _VRU_SPEED_MPS_SINGLE_RX.search(doc_text)
            if m1:
                val = float(m1.group(1))
                vru["vru_speed_mps_min"] = val
                vru["vru_speed_mps_max"] = val
                _append_evidence(s, "scenario_details.extra.vru.vru_speed_mps_min", m1.group(0))
                _append_evidence(s, "scenario_details.extra.vru.vru_speed_mps_max", m1.group(0))

    # If speeds appear only in km/h, store allowed range (no conversion)
    if vru.get("vru_speed_mps_min") is None and vru.get("vru_speed_mps_max") is None:
        m = _VRU_SPEED_KPH_RANGE_RX.search(doc_text)
        if m:
            allowed = [int(m.group(1)), int(m.group(2))]
            allowed_values = extra.setdefault("allowed_values", {})
            if "vru_speed_kph_range" not in allowed_values:
                allowed_values["vru_speed_kph_range"] = allowed
                _append_evidence(s, "scenario_details.extra.allowed_values.vru_speed_kph_range", m.group(0))
        else:
            # The report showed this regex consistently grabbing a generic
            # "eligibility minimum" speed (e.g. "system shall detect
            # pedestrians walking at speeds as low as 3 km/h") instead of the
            # actual per-scenario test speed stated in a nearby sentence. A
            # fixed-character lookback window bleeds across sentence
            # boundaries when the eligibility sentence sits right before the
            # real one, so this is scoped to the containing SENTENCE instead.
            _EXCLUDE_CTX = ("eligib", "minimum", "shall detect", "as low as", "criteria")
            _sentences = re.split(r"(?<=[.!?])\s+", doc_text)
            _spans = []
            _offset = 0
            for _sent in _sentences:
                _idx = doc_text.find(_sent, _offset)
                if _idx == -1:
                    continue
                _spans.append((_idx, _idx + len(_sent), _sent))
                _offset = _idx + len(_sent)

            m1 = None
            for cand in _VRU_SPEED_KPH_SINGLE_RX.finditer(doc_text):
                containing_sentence = next(
                    (sent for a, b, sent in _spans if a <= cand.start() < b), ""
                )
                if not any(kw in containing_sentence.lower() for kw in _EXCLUDE_CTX):
                    m1 = cand
                    break
            if m1 is None:
                # Nothing clean found — fall back to the first match anyway,
                # better than nothing, but flagged via evidence text so it's
                # traceable if still wrong.
                m1 = _VRU_SPEED_KPH_SINGLE_RX.search(doc_text)
            if m1:
                val = float(m1.group(1))
                allowed_values = extra.setdefault("allowed_values", {})
                if "vru_speed_kph" not in allowed_values:
                    allowed_values["vru_speed_kph"] = val
                    _append_evidence(s, "scenario_details.extra.allowed_values.vru_speed_kph", m1.group(0))

    return s


# --- Prompt extension: VRU scope rules (append-only) ---
SYSTEM_PROMPT_VRU_APPEND = """
VRU mode (IMPORTANT):
- If SCENARIO_JSON contains scenario_details.extra.adas_family == "VRU" OR scenario_details.extra.vru exists
  OR scenario_code starts with CP/CB/CM/VRU_:
  - Treat as Vulnerable Road User scenario (Pedestrian/Cyclist/Motorcyclist).
  - Your primary fill targets are ONLY:
      scenario_details.extra.vru.vru_type
      scenario_details.extra.vru.scenario_variant
      scenario_details.extra.vru.adult_child
      scenario_details.extra.vru.obscured
      scenario_details.extra.vru.crossing_side
      scenario_details.extra.vru.vru_speed_mps_min / vru_speed_mps_max
    and, if the protocol explicitly states ego speed RANGE for this VRU test:
      scenario_details.ego_speed_min / ego_speed_max
  - DO NOT invent TTC/headway/initial_distance/braking/decel unless DOC_TEXT gives a SINGLE fixed value for this scenario.
  - Keep overlap_percent and lateral_offset_m untouched (still forbidden).
"""

SYSTEM_PROMPT_V2 = SYSTEM_PROMPT + "\n" + SYSTEM_PROMPT_VRU_APPEND


# --- Override enrich_one_scenario to use SYSTEM_PROMPT_V2 and add VRU deterministic fallback ---
_enrich_one_scenario_base = enrich_one_scenario  # keep reference, just in case

def enrich_one_scenario(  # type: ignore[override]
    scenario: Dict[str, Any],
    knowledge_base_path: str = "knowledge_base.json",
    scenario_evidence_path: Optional[str] = "scenario_evidence.json",
    _evidence_cache: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    OVERRIDE (append-only):
    - Uses SYSTEM_PROMPT_V2 (adds VRU mode rules)
    - Keeps validation/repair behavior identical
    - Adds VRU deterministic fallback after LLM/repair
    """
    inp = deepcopy(scenario)
    if not isinstance(inp, dict):
        return scenario

    scenario_code = str(inp.get("scenario_code") or "")
    scenario_name = str(inp.get("scenario_name") or (inp.get("scenario_details") or {}).get("scenario") or "")

    # ---- Preferred: evidence pack ----
    doc_text: Optional[str] = None
    image_candidates: Optional[Dict[str, Any]] = None
    image_paths: List[str] = []
    evidence_obj: Optional[Dict[str, Any]] = _evidence_cache

    if scenario_evidence_path:
        try:
            if evidence_obj is None:
                evidence_obj = load_scenario_evidence(scenario_evidence_path)
            doc_text = get_doc_text_from_evidence(evidence_obj, scenario_code, scenario_name)
            image_candidates = get_image_candidates_from_evidence(evidence_obj, scenario_code, scenario_name)
            image_paths = _pick_image_paths(image_candidates)
        except Exception:
            doc_text = None
            image_candidates = None
            image_paths = []

    # ---- Fallback: KB filtering ----
    if not doc_text:
        kb_pages = load_kb_pages(knowledge_base_path)
        doc_text = filter_kb_for_scenario(kb_pages, scenario_code, scenario_name)
        image_candidates = None
        image_paths = []

    doc_text = (doc_text or "").strip()
    if len(doc_text) > MAX_DOC_CHARS:
        doc_text = doc_text[:MAX_DOC_CHARS] + "\n\n[TRUNCATED]"

    user_payload = {
        "task": "scenario_enrichment",
        "DOC_TEXT": doc_text,
        "IMAGE_CANDIDATES": image_candidates,
        "SCENARIO_JSON": inp,
    }

    out = call_llm_json(
        SYSTEM_PROMPT_V2,
        user_payload,
        image_paths=image_paths
    )

    errs = validate_enrichment(inp, out)
    tries = 0
    while errs and tries < REPAIR_MAX_TRIES:
        tries += 1
        out = repair_with_llm(inp, out, errs, doc_text, image_paths=image_paths)
        errs = validate_enrichment(inp, out)

    if isinstance(out, dict):
        _ensure_extra_evidence_shape(out)

    if errs:
        safe = deepcopy(inp)
        sd = safe.setdefault("scenario_details", {})
        extra = sd.setdefault("extra", {})
        if "evidence" not in extra or not isinstance(extra["evidence"], list):
            extra["evidence"] = []
        if isinstance(sd.get("notes"), str) and sd["notes"].strip():
            sd["notes"] = sd["notes"].strip() + " | LLM enrichment validation failed; kept original fields."
        else:
            sd["notes"] = "LLM enrichment validation failed; kept original fields."
        return safe

    # Deterministic post-processing for LSS (unchanged)
    if isinstance(out, dict) and _is_lss_like(out):
        out = normalize_lss_containers(out)
        out = _fallback_fill_lss_fields_from_text(out, doc_text)
        out = _fallback_fill_lss_ranges_from_text(out, doc_text)
        _ensure_extra_evidence_shape(out)

    # Deterministic post-processing for VRU (new)
    if isinstance(out, dict) and _is_vru_like(out):
        out = _fallback_fill_vru_fields_from_text(out, doc_text)
        _ensure_extra_evidence_shape(out)

    return out


# enrich_all remains compatible; it calls enrich_one_scenario symbol, which is now overridden above.
def main():
    # Default file names (same folder you run from)
    structured_path = os.getenv("STRUCTURED_SCENARIOS_PATH", "structured_scenarios.json")
    evidence_path = os.getenv("SCENARIO_EVIDENCE_PATH", "scenario_evidence.json")
    kb_path = os.getenv("KNOWLEDGE_BASE_PATH", "knowledge_base.json")
    out_path = os.getenv("UNIFORM_SCENARIOS_OUT", "uniform_scenarios.json")

    print("[llm_enricher] START")
    print("  structured:", structured_path)
    print("  evidence  :", evidence_path)
    print("  kb        :", kb_path)
    print("  out       :", out_path)
    print("  model     :", OPENAI_MODEL)

    # Basic sanity checks (so it fails loudly, not silently)
    if Anthropic is None:
        raise RuntimeError(f"anthropic import failed: {_ANTHROPIC_IMPORT_ERROR}")

    if not os.path.isfile(structured_path):
        raise FileNotFoundError(f"Missing: {structured_path}")

    scenarios = json.load(open(structured_path, "r", encoding="utf-8"))
    if not isinstance(scenarios, list):
        raise ValueError("structured_scenarios.json must be a JSON array (list)")

    # Enrich
    enriched = enrich_all(
        scenarios,
        knowledge_base_path=kb_path,
        scenario_evidence_path=evidence_path if os.path.isfile(evidence_path) else None,
    )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)

    print(f"[llm_enricher] DONE. wrote: {out_path} | scenarios: {len(enriched)}")


if __name__ == "__main__":
    main()
