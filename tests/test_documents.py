import json

from rcqa.documents import clean_text, label_sections, section_title


def test_clean_text_fixes_pdf_artifacts_only():
    raw = "Falls  preven-\ntion­ in\tolder adults\r\n\n\n\nNext"
    assert clean_text(raw) == "Falls prevention in older adults\n\nNext"


def test_section_title():
    assert section_title("warnings_and_cautions") == "Warnings and Cautions"
    assert section_title("boxed_warning") == "Boxed Warning"


def test_label_sections_keeps_text_in_order_and_drops_metadata():
    label = {
        "boxed_warning": ["WARNING: BLEEDING RISK  Warfarin can cause major bleeding."],
        "openfda": {"generic_name": ["WARFARIN"]},
        "set_id": "abc",
        "adverse_reactions": ["Bleeding.", "Skin necrosis."],
        "adverse_reactions_table": ["<table>...</table>"],
        "effective_time": "20260101",
    }
    assert label_sections(json.dumps(label)) == [
        ("Boxed Warning", "WARNING: BLEEDING RISK Warfarin can cause major bleeding."),
        ("Adverse Reactions", "Bleeding.\nSkin necrosis."),
    ]


def test_clean_text_windows_line_endings_are_one_newline():
    # Databricks reads files as bytes, so \r\n survives; it must not become a blank line
    # (the law parser relies on a heading being directly above its section number).
    assert clean_text("Falls prevention\r\n54. (1) The program\r\nhyphen-\r\nated") == \
        "Falls prevention\n54. (1) The program\nhyphenated"
