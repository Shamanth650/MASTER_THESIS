import os
import json
import re

# ============================================================
# CONFIG
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PARSED_ROOT = os.path.join(SCRIPT_DIR, "Parsed_Data")

TEXT_DIR = os.path.join(PARSED_ROOT, "text")
IMAGE_ROOT = os.path.join(PARSED_ROOT, "images")

OUTPUT_JSON = os.path.join(SCRIPT_DIR, "knowledge_base_raw.json")


# ============================================================
# HELPERS
# ============================================================

def extract_page_number(filename: str):
    """
    Extract page number from filenames like:
    xxx_p12.png
    xxx_p12_table_1.png
    xxx_p12_0.png
    """
    match = re.search(r"_p(\d+)", filename)
    return int(match.group(1)) if match else None


def load_text_files(folder: str):
    entries = []

    if not os.path.exists(folder):
        return entries

    for file in os.listdir(folder):
        if not file.lower().endswith(".txt"):
            continue

        path = os.path.join(folder, file)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        entries.append({
            "file": file,
            "type": "text",
            "content": content
        })

    return entries


def load_image_metadata(image_dir: str, image_type: str):
    """
    image_type:
      - page_image
      - table_image
      - diagram_image
    """
    entries = []

    if not os.path.exists(image_dir):
        return entries

    for file in os.listdir(image_dir):
        if not file.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        page = extract_page_number(file)

        entries.append({
            "file": file,
            "type": "image",
            "content": None,
            "page": page,
            "source_type": image_type,
            "meta": {
                "path": os.path.join("images", image_type, file)
            }
        })

    return entries


# ============================================================
# MAIN
# ============================================================

def build_knowledge_base():
    knowledge_base = []

    # --------------------
    # TEXT
    # --------------------
    knowledge_base.extend(load_text_files(TEXT_DIR))

    # --------------------
    # IMAGES (metadata only)
    # --------------------
    knowledge_base.extend(
        load_image_metadata(
            os.path.join(IMAGE_ROOT, "page_image"),
            "page_image"
        )
    )

    knowledge_base.extend(
        load_image_metadata(
            os.path.join(IMAGE_ROOT, "table_image"),
            "table_image"
        )
    )

    knowledge_base.extend(
        load_image_metadata(
            os.path.join(IMAGE_ROOT, "diagram_image"),
            "diagram_image"
        )
    )

    # --------------------
    # WRITE OUTPUT
    # --------------------
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(knowledge_base, f, indent=2, ensure_ascii=False)

    print(f" knowledge_base_raw.json created with {len(knowledge_base)} entries")


if __name__ == "__main__":
    build_knowledge_base()
