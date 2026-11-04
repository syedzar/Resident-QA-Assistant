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
