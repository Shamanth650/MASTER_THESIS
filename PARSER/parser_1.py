import os
import sys
import re
import shutil
from typing import Optional, List, Tuple

import fitz  # PyMuPDF
import camelot


# ============================================================
# STABLE ROOTS (OUTPUT ONLY)
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PARSED_ROOT = os.path.join(SCRIPT_DIR, "Parsed_Data")

TEXT_OUTPUT = os.path.join(PARSED_ROOT, "text")

IMAGE_ROOT = os.path.join(PARSED_ROOT, "images")
# NOTE: We keep the folder path for compatibility, but we DO NOT generate page images anymore.
PAGE_IMAGE_OUTPUT = os.path.join(IMAGE_ROOT, "page_image")

TABLE_IMAGE_OUTPUT = os.path.join(IMAGE_ROOT, "table_image")
DIAGRAM_IMAGE_OUTPUT = os.path.join(IMAGE_ROOT, "diagram_image")


# ============================================================
# HELPERS
# ============================================================

def prepare_output_dirs(dirs: List[str]) -> None:
    """
    Clears and recreates output dirs (only those passed).
    This keeps the output neat for each run.
    """
    for folder in dirs:
        if os.path.exists(folder):
            shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder, exist_ok=True)


def extract_text_from_pdf(doc: fitz.Document, pdf_id: str) -> None:
    all_text = ""
    for page in doc:
        all_text += f"\n--- PAGE {page.number + 1} ---\n"
        all_text += page.get_text("text")

    out_path = os.path.join(TEXT_OUTPUT, f"{pdf_id}.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(all_text)


def extract_embedded_images(doc: fitz.Document, pdf_id: str) -> List[Tuple[str, str]]:
    """
    Extract embedded raster images from the PDF.
    We store them under diagram_image because they are usually diagrams/icons/figures.
    """
    os.makedirs(DIAGRAM_IMAGE_OUTPUT, exist_ok=True)

    image_infos: List[Tuple[str, str]] = []
    for page in doc:
        for idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            img_data = doc.extract_image(xref)
            img_bytes = img_data["image"]
            ext = img_data.get("ext", "png")

            name = f"{pdf_id}_p{page.number + 1}_{idx}.{ext}"
            path = os.path.join(DIAGRAM_IMAGE_OUTPUT, name)

            with open(path, "wb") as f:
                f.write(img_bytes)

            image_infos.append((name, path))

    return image_infos


def _safe_int_page(value) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(str(value))
    except Exception:
        m = re.search(r"\d+", str(value))
        return int(m.group()) if m else None


def extract_tables_from_pdf(pdf_path: str, pdf_id: str) -> None:
    """
    Extract tables as IMAGES (no table JSON, no OCR).
    Uses Camelot to locate table bounding boxes (lattice flavor).
    Then crops the corresponding region using PyMuPDF.
    """
    os.makedirs(TABLE_IMAGE_OUTPUT, exist_ok=True)

    try:
        tables = camelot.read_pdf(pdf_path, pages="all", flavor="lattice")
    except Exception as e:
        print(f"[WARN] Camelot failed to read tables: {e}")
        return

    print(f"[DEBUG] Camelot found {len(tables)} tables")

    # Open once for cropping
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"[WARN] PyMuPDF could not open PDF for table cropping: {e}")
        return

    try:
        for i, table in enumerate(tables):
            real_page = _safe_int_page(getattr(table, "page", None))

            # Best effort crop
            try:
                if real_page and hasattr(table, "_bbox") and table._bbox:
                    page = doc[real_page - 1]
                    x1, y1, x2, y2 = table._bbox

                    # Camelot bbox uses PDF coords; convert for PyMuPDF
                    page_height = page.rect.height
                    rect = fitz.Rect(
                        x1,
                        page_height - y2,
                        x2,
                        page_height - y1
                    )

                    pix = page.get_pixmap(clip=rect, dpi=300)
                    img_name = f"{pdf_id}_p{real_page}_table_{i + 1}.png"
                    pix.save(os.path.join(TABLE_IMAGE_OUTPUT, img_name))
                else:
                    # If Camelot cannot provide bbox, skip quietly
                    pass
            except Exception as e:
                print(f"[WARN] Table image failed (table {i + 1}): {e}")

    finally:
        doc.close()


# ============================================================
# MAIN PIPELINE (GENERIC)
# ============================================================

def process_pdf(pdf_path: str) -> None:
    """
    Generic: given a PDF path, generate Parsed_Data outputs:
      - text/<pdf_id>.txt
      - images/diagram_image/*.(png/jpg/...)
      - images/table_image/*.png  (best-effort via Camelot bbox)

    NOTE:
      We NO LONGER generate page images under images/page_image/.
      The folder may exist (for compatibility), but it will remain empty.
    """
    pdf_id = os.path.splitext(os.path.basename(pdf_path))[0].replace(" ", "_")
    print(f"\nProcessing: {pdf_id}")

    # Ensure core dirs exist (even if caller forgot prepare_output_dirs)
    os.makedirs(TEXT_OUTPUT, exist_ok=True)
    os.makedirs(TABLE_IMAGE_OUTPUT, exist_ok=True)
    os.makedirs(DIAGRAM_IMAGE_OUTPUT, exist_ok=True)
    # Keep page_image dir optional/empty
    os.makedirs(PAGE_IMAGE_OUTPUT, exist_ok=True)

    doc = fitz.open(pdf_path)
    try:
        extract_text_from_pdf(doc, pdf_id)
        extract_embedded_images(doc, pdf_id)
    finally:
        doc.close()

    # Table images (separate open inside)
    extract_tables_from_pdf(pdf_path, pdf_id)


def main(pdf_path: str) -> None:
    prepare_output_dirs([
        TEXT_OUTPUT,
        # Keep directory but do not generate any page images
        PAGE_IMAGE_OUTPUT,
        TABLE_IMAGE_OUTPUT,
        DIAGRAM_IMAGE_OUTPUT,
    ])

    process_pdf(pdf_path)

    print("\n✔ Parsed_Data generation completed (NO page images).")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise RuntimeError(
            "parser_1.py must be called with a PDF path.\n"
            "Example:\n"
            "  python parser_1.py /path/to/file.pdf"
        )

    main(sys.argv[1])
