# llm_image_picker.py
# ------------------------------------------------------------
# LLM Call #1: TABLE Image Matcher / Picker (TABLES ONLY)
#
# Inputs:
#  - scenario_evidence.json (from Stage 2)
#  - Parsed_Data/images/table_image/*
#
# Output:
#  - scenario_image_selection.json
#
# Purpose:
#  - Select the correct TABLE images for each scenario using LOW-RES thumbnails
#    to avoid huge payload / WinError 10054.
#
# Key behavior:
#  - ONLY table images are sent to the LLM.
#  - Diagram images may exist in outputs, but this script ignores them entirely.
# ------------------------------------------------------------

from __future__ import annotations

import os
import re
import json
import time
import base64
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

try:
    from PIL import Image
except Exception as e:
    Image = None
    _PIL_IMPORT_ERROR = e

try:
    from openai import OpenAI
except Exception as e:
    OpenAI = None
    _OPENAI_IMPORT_ERROR = e


# -----------------------------
# Env + defaults
# -----------------------------
load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

EVIDENCE_PATH = os.getenv("SCENARIO_EVIDENCE_PATH", "scenario_evidence.json")
OUT_SELECTION_PATH = os.getenv("SCENARIO_IMAGE_SELECTION_PATH", "scenario_image_selection.json")

BASE_DIR = os.getenv("BASE_DIR", os.getcwd())
PARSED_DATA_ROOT = os.getenv("PARSED_DATA_ROOT", os.path.join(BASE_DIR, "Parsed_Data"))

# TABLE-only caps
MAX_TABLE_CANDIDATES = int(os.getenv("LLM_PICKER_MAX_TABLES", "14"))  # total tables sent to Call #1
ANCHOR_PAGE_EXPAND = int(os.getenv("LLM_PICKER_PAGE_EXPAND", "1"))    # +/- page window when available

# Thumbnail settings (small on purpose)
THUMB_MAX_SIDE = int(os.getenv("LLM_PICKER_THUMB_MAX_SIDE", "384"))
THUMB_JPEG_QUALITY = int(os.getenv("LLM_PICKER_THUMB_QUALITY", "45"))

# Network/retry
TIMEOUT_S = int(os.getenv("OPENAI_TIMEOUT_S", "120"))
MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "6"))

# Debug: only run first N scenarios (0 = all)
MAX_SCENARIOS = int(os.getenv("LLM_MAX_SCENARIOS", "0"))


# -----------------------------
# Helpers
# -----------------------------
_PAGE_RX = re.compile(r"(?:^|[_\-])p(\d{1,4})(?:[_\-]|\.|$)", re.IGNORECASE)

def _norm_slashes(p: str) -> str:
    return p.replace("\\", "/")

def _resolve_under_parsed_data(rel_or_abs: str) -> str:
    """
    Resolves image paths saved by Stage2:
    - if absolute path exists, use it
    - else treat it as relative to Parsed_Data root
    """
    if not rel_or_abs:
        return rel_or_abs

    p = rel_or_abs.strip().strip('"').strip("'")
    p = _norm_slashes(p)

    if os.path.isabs(p) and os.path.exists(p):
        return p

    candidate = os.path.normpath(os.path.join(PARSED_DATA_ROOT, p))
    if os.path.exists(candidate):
        return candidate

    alt = os.path.normpath(os.path.join(BASE_DIR, p))
    if os.path.exists(alt):
        return alt

    return candidate  # best effort

def _infer_page_from_path(path: str) -> Optional[int]:
    if not path:
        return None
    m = _PAGE_RX.search(_norm_slashes(path))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _dump_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _safe_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, str) and x.isdigit():
        return int(x)
    return None

def _extract_anchor_pages(evid_entry: Dict[str, Any]) -> List[int]:
    pages = evid_entry.get("selected_pages")
    out: List[int] = []
    if isinstance(pages, list):
        for p in pages:
            ip = _safe_int(p)
            if ip is not None:
                out.append(ip)
    # dedupe
    seen = set()
    final = []
    for p in out:
        if p in seen:
            continue
        seen.add(p)
        final.append(p)
    return final

def _page_window(pages: List[int], expand: int) -> List[int]:
    if not pages:
        return []
    s = set()
    for p in pages:
        for q in range(p - expand, p + expand + 1):
            if q > 0:
                s.add(q)
    return sorted(s)

def _make_thumbnail_data_url(image_path: str) -> str:
    if Image is None:
        raise RuntimeError(f"Pillow import failed: {_PIL_IMPORT_ERROR}. Install: pip install pillow")

    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    scale = min(THUMB_MAX_SIDE / max(w, h), 1.0)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=THUMB_JPEG_QUALITY, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


@dataclass
class TableCandidate:
    idx: int
    page: Optional[int]
    rel_path: str
    abs_path: str

def _gather_table_candidates(evid_entry: Dict[str, Any]) -> List[TableCandidate]:
    """
    Prefer image_candidates_meta (type=table).
    Fallback to legacy image_candidates.table_images.
    """
    out: List[TableCandidate] = []

    meta = evid_entry.get("image_candidates_meta")
    if isinstance(meta, list) and meta:
        i = 0
        for m in meta:
            if not isinstance(m, dict):
                continue
            t = (m.get("type") or "").strip().lower()
            if t != "table":
                continue
            page = _safe_int(m.get("page"))
            rel = m.get("path")
            if not isinstance(rel, str) or not rel.strip():
                continue
            abs_p = _resolve_under_parsed_data(rel)
            out.append(TableCandidate(idx=i, page=page, rel_path=rel, abs_path=abs_p))
            i += 1
        return out

    legacy = evid_entry.get("image_candidates") or {}
    tables = legacy.get("table_images") if isinstance(legacy, dict) else []
    if isinstance(tables, list):
        i = 0
        for rel in tables:
            if not isinstance(rel, str) or not rel.strip():
                continue
            abs_p = _resolve_under_parsed_data(rel)
            out.append(TableCandidate(idx=i, page=_infer_page_from_path(rel), rel_path=rel, abs_path=abs_p))
            i += 1

    return out

def _filter_by_anchor_pages(cands: List[TableCandidate], anchor_pages: List[int]) -> List[TableCandidate]:
    if not cands:
        return []
    if not anchor_pages:
        return cands
    win = set(_page_window(anchor_pages, ANCHOR_PAGE_EXPAND))
    filtered = [c for c in cands if (c.page is None or c.page in win)]
    return filtered if filtered else cands

def _cap_tables(cands: List[TableCandidate]) -> List[TableCandidate]:
    # Prefer known pages first; stable ordering by page then filename
    cands_sorted = sorted(
        cands,
        key=lambda x: (x.page is None, x.page if x.page is not None else 10**9, os.path.basename(_norm_slashes(x.rel_path)))
    )
    capped = cands_sorted[:MAX_TABLE_CANDIDATES]
    # reindex sequentially for the LLM
    out = []
    for new_i, c in enumerate(capped):
        out.append(TableCandidate(idx=new_i, page=c.page, rel_path=c.rel_path, abs_path=c.abs_path))
    return out

def _build_prompt(scenario_code: Optional[str], scenario_name: Optional[str], doc_text: str, tables: List[TableCandidate]) -> str:
    doc_short = (doc_text or "").strip()
    if len(doc_short) > 2200:
        doc_short = doc_short[:2200] + "\n[TRUNCATED]"

    lines = []
    for t in tables:
        fname = os.path.basename(_norm_slashes(t.rel_path))
        lines.append(f"- idx={t.idx} page={t.page} file={fname}")

    return (
        "You are selecting which TABLE images belong to one Euro NCAP scenario.\n"
        "TASK:\n"
        "1) Choose the most relevant TABLE images for this scenario.\n"
        "2) DO NOT extract numeric values yet. Only select the correct tables.\n"
        "3) Prefer tables whose headings/captions mention the scenario code/name, or clearly list VUT/XVUT speeds, offsets, overlaps, TTC, etc.\n\n"
        f"SCENARIO:\n- code: {scenario_code or 'null'}\n- name: {scenario_name or 'null'}\n\n"
        "TEXT CONTEXT (short):\n"
        f"{doc_short}\n\n"
        "TABLE CANDIDATES:\n" + "\n".join(lines) + "\n\n"
        "OUTPUT JSON ONLY in exactly this schema:\n"
        "{\n"
        '  "selected_table_indices": [int, ...],\n'
        '  "confidence": 0.0,\n'
        '  "notes": "short reason"\n'
        "}\n"
        "Rules:\n"
        "- Choose 1-4 tables.\n"
        "- If nothing matches, return empty list and confidence 0.\n"
    )

def _call_openai(client: Any, prompt: str, thumb_data_urls: List[str]) -> Dict[str, Any]:
    content = [{"type": "text", "text": prompt}]
    for du in thumb_data_urls:
        content.append({"type": "image_url", "image_url": {"url": du}})

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": content}],
                temperature=0.0,
            )
            txt = resp.choices[0].message.content or "{}"
            return json.loads(txt)
        except Exception as e:
            last_err = e
            time.sleep(min(8.0, 0.8 * attempt + random.random()))
            # shrink thumbnails if still too many
            if attempt in (2, 3) and len(thumb_data_urls) > 10:
                thumb_data_urls = thumb_data_urls[: max(6, len(thumb_data_urls) // 2)]
                content = [{"type": "text", "text": prompt}]
                for du in thumb_data_urls:
                    content.append({"type": "image_url", "image_url": {"url": du}})

    raise RuntimeError(f"OpenAI picker failed after retries: {last_err}")

def _validate_output(obj: Any, max_idx: int) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return {"selected_table_indices": [], "confidence": 0.0, "notes": "invalid output"}

    def _clean_int_list(x: Any) -> List[int]:
        out: List[int] = []
        if isinstance(x, list):
            for v in x:
                if isinstance(v, int) and 0 <= v <= max_idx:
                    out.append(v)
                elif isinstance(v, str) and v.isdigit():
                    iv = int(v)
                    if 0 <= iv <= max_idx:
                        out.append(iv)
        # dedupe preserve order
        seen = set()
        final = []
        for i in out:
            if i in seen:
                continue
            seen.add(i)
            final.append(i)
        return final

    idxs = _clean_int_list(obj.get("selected_table_indices"))
    conf = obj.get("confidence")
    try:
        conf = float(conf)
    except Exception:
        conf = 0.0

    notes = obj.get("notes")
    if not isinstance(notes, str):
        notes = ""

    return {"selected_table_indices": idxs, "confidence": max(0.0, min(1.0, conf)), "notes": notes[:400]}


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    print("[llm_image_picker] START (TABLES ONLY)")
    print("  evidence :", EVIDENCE_PATH)
    print("  out      :", OUT_SELECTION_PATH)
    print("  model    :", OPENAI_MODEL)
    print("  Parsed_Data:", PARSED_DATA_ROOT)
    print("  max_tables:", MAX_TABLE_CANDIDATES)

    if OpenAI is None:
        raise RuntimeError(f"openai import failed: {_OPENAI_IMPORT_ERROR}")
    if Image is None:
        raise RuntimeError(f"Pillow import failed. Install: pip install pillow. Error: {_PIL_IMPORT_ERROR}")
    if not os.path.isfile(EVIDENCE_PATH):
        raise FileNotFoundError(f"Missing evidence file: {EVIDENCE_PATH}")

    evidence = _load_json(EVIDENCE_PATH)
    if not isinstance(evidence, dict):
        raise ValueError("scenario_evidence.json must be a JSON object/dict keyed by scenario key")

    client = OpenAI(timeout=TIMEOUT_S, max_retries=MAX_RETRIES)

    keys = list(evidence.keys())
    if MAX_SCENARIOS and MAX_SCENARIOS > 0:
        keys = keys[:MAX_SCENARIOS]

    results: Dict[str, Any] = {}

    for n, key in enumerate(keys, start=1):
        entry = evidence.get(key) or {}
        if not isinstance(entry, dict):
            continue

        scenario_code = entry.get("scenario_code")
        scenario_name = entry.get("scenario_name")
        doc_text = entry.get("doc_text") or ""

        anchor_pages = _extract_anchor_pages(entry)

        tables_all = _gather_table_candidates(entry)
        tables_filtered = _filter_by_anchor_pages(tables_all, anchor_pages)
        tables = _cap_tables(tables_filtered)

        # drop missing files
        tables = [t for t in tables if os.path.exists(t.abs_path)]

        print(f"\n[{n}/{len(keys)}] {key} | code={scenario_code} | name={scenario_name}")
        print(f"  tables(all)={len(tables_all)} filtered={len(tables_filtered)} capped={len(tables)} anchor_pages={anchor_pages}")

        if not tables:
            results[key] = {
                "scenario_code": scenario_code,
                "scenario_name": scenario_name,
                "anchor_pages": anchor_pages,
                "candidate_table_count_used": 0,
                "selected_tables": [],
                "confidence": 0.0,
                "notes": "no table candidates"
            }
            continue

        thumb_urls: List[str] = []
        for t in tables:
            try:
                thumb_urls.append(_make_thumbnail_data_url(t.abs_path))
            except Exception as e:
                print(f"  [WARN] thumbnail failed: {t.abs_path} | {e}")

        if not thumb_urls:
            results[key] = {
                "scenario_code": scenario_code,
                "scenario_name": scenario_name,
                "anchor_pages": anchor_pages,
                "candidate_table_count_used": len(tables),
                "selected_tables": [],
                "confidence": 0.0,
                "notes": "thumbnails_failed"
            }
            continue

        prompt = _build_prompt(scenario_code, scenario_name, doc_text, tables)

        try:
            raw = _call_openai(client, prompt, thumb_urls)
        except Exception as e:
            results[key] = {
                "scenario_code": scenario_code,
                "scenario_name": scenario_name,
                "anchor_pages": anchor_pages,
                "candidate_table_count_used": len(tables),
                "selected_tables": [],
                "confidence": 0.0,
                "notes": f"picker_failed: {e}"
            }
            continue

        cleaned = _validate_output(raw, max_idx=len(tables) - 1)

        idx_to_t = {t.idx: t for t in tables}
        selected_tables: List[Dict[str, Any]] = []
        for i in cleaned["selected_table_indices"]:
            t = idx_to_t.get(i)
            if t:
                selected_tables.append({"page": t.page, "path": t.rel_path})

        results[key] = {
            "scenario_code": scenario_code,
            "scenario_name": scenario_name,
            "anchor_pages": anchor_pages,
            "candidate_table_count_used": len(tables),
            "selected_tables": selected_tables,
            "confidence": cleaned["confidence"],
            "notes": cleaned["notes"],
        }

        print(f"  selected tables={len(selected_tables)} conf={cleaned['confidence']:.2f}")

    _dump_json(OUT_SELECTION_PATH, results)
    print(f"\n[llm_image_picker] DONE -> {OUT_SELECTION_PATH} | scenarios={len(results)}")


if __name__ == "__main__":
    main()
