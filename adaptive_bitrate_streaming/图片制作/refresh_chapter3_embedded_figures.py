from pathlib import Path
from shutil import copy2
from tempfile import NamedTemporaryFile
from zipfile import ZIP_DEFLATED, ZipFile

from lxml import etree


BASE_DIR = Path("/data3/wangxh/NetLLM-master/adaptive_bitrate_streaming")
DOC_PATH = BASE_DIR / "第三章.docx"
COPY_PATH = BASE_DIR / "第三章_插图版.docx"
BACKUP_PATH = BASE_DIR / "第三章_字号放大前备份.docx"
FIG_DIR = BASE_DIR / "论文书写" / "论文图片"

NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
}

CAPTION_TO_IMAGE = {
    "图 3.1": FIG_DIR / "fig3_1_llm_abr_policy_overview_cn.png",
    "图 3.2": FIG_DIR / "fig3_2_semantic_reprogramming_encoder_cn.png",
    "图 3.3": FIG_DIR / "fig3_3_context_alignment_multiscale_history_cn.png",
}


def read_doc_targets(docx_path: Path) -> dict[str, str]:
    with ZipFile(docx_path) as zf:
        document = etree.fromstring(zf.read("word/document.xml"))
        rels = etree.fromstring(zf.read("word/_rels/document.xml.rels"))

    rel_map = {rel.get("Id"): rel.get("Target") for rel in rels}
    paragraphs = document.xpath("//w:body/w:p", namespaces=NS)
    caption_targets = {}

    previous_target = None
    for paragraph in paragraphs:
        embeds = paragraph.xpath(".//a:blip/@r:embed", namespaces=NS)
        if embeds:
            previous_target = "word/" + rel_map[embeds[-1]]
            continue

        text = "".join(paragraph.xpath(".//w:t/text()", namespaces=NS)).strip()
        for caption_prefix in CAPTION_TO_IMAGE:
            if text.startswith(caption_prefix):
                if previous_target is None:
                    raise RuntimeError(f"Cannot find image before caption {caption_prefix}")
                caption_targets[caption_prefix] = previous_target

    missing = set(CAPTION_TO_IMAGE) - set(caption_targets)
    if missing:
        raise RuntimeError(f"Missing figure captions: {sorted(missing)}")
    return caption_targets


def replace_zip_entries(docx_path: Path, replacements: dict[str, Path]) -> None:
    with NamedTemporaryFile(suffix=".docx", dir=docx_path.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)

    with ZipFile(docx_path, "r") as src, ZipFile(tmp_path, "w", ZIP_DEFLATED) as dst:
        replaced = set()
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename in replacements:
                data = replacements[item.filename].read_bytes()
                replaced.add(item.filename)
            dst.writestr(item, data)

    missing = set(replacements) - replaced
    if missing:
        raise RuntimeError(f"Missing media entries in docx: {sorted(missing)}")

    tmp_path.replace(docx_path)


def refresh(docx_path: Path) -> dict[str, str]:
    for image_path in CAPTION_TO_IMAGE.values():
        if not image_path.exists():
            raise FileNotFoundError(image_path)

    caption_targets = read_doc_targets(docx_path)
    replacements = {
        caption_targets[caption_prefix]: image_path
        for caption_prefix, image_path in CAPTION_TO_IMAGE.items()
    }
    replace_zip_entries(docx_path, replacements)
    return caption_targets


def main():
    if not BACKUP_PATH.exists():
        copy2(DOC_PATH, BACKUP_PATH)

    targets = refresh(DOC_PATH)
    copy2(DOC_PATH, COPY_PATH)

    print(f"updated={DOC_PATH}")
    print(f"copy={COPY_PATH}")
    print(f"backup={BACKUP_PATH}")
    for caption, target in targets.items():
        print(f"{caption} -> {target}")


if __name__ == "__main__":
    main()
