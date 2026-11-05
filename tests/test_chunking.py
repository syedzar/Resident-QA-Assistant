import pytest

from rcqa.chunking import (ChunkConfig, chunk_label_section, chunk_law, chunk_pdf, chunk_profile,
                           law_sections, pdf_boilerplate, profile_sections, windows)

CFG = ChunkConfig(chunk_size=10, overlap=3)


def words(n, start=0):
    return " ".join(f"w{i}" for i in range(start, start + n))


# ---------------------------------------------------------------- windows
def test_windows_respect_size_and_overlap():
    chunks = windows(words(25), CFG)
    assert all(len(c.split()) <= 10 for c in chunks)
    assert chunks[0].split()[-3:] == chunks[1].split()[:3]          # 3-word overlap
    joined = set(" ".join(chunks).split())
    assert joined == set(words(25).split())                         # nothing lost


def test_windows_pack_short_lines_and_no_overlap_only_tail():
    text = "\n".join(["a b c", "d e f", "g h i", "j k"])
    assert windows(text, CFG) == ["a b c d e f g h i", "g h i j k"]


def test_windows_zero_overlap_and_empty():
    assert windows(words(20), ChunkConfig(10, 0)) == [words(10), words(10, 10)]
    assert windows("", CFG) == []


def test_bad_config_rejected():
    with pytest.raises(ValueError):
        ChunkConfig(chunk_size=10, overlap=10)


# ---------------------------------------------------------------- laws
REG = """ONTARIO REGULATION 246/22
54.
Falls prevention and management
55.
Skin and wound care
Required programs
53. (1) Every licensee shall ensure that the following programs are developed:
1. A falls prevention and management program to reduce the incidence of falls.
2. A skin and wound care program.
(2) Each program must provide for screening protocols. O. Reg. 246/22, s. 53 (2).
Falls prevention and management
 54. (1) The falls prevention and management program must provide for strategies to reduce falls.
(2) Every licensee shall ensure that when a resident has fallen, a post-fall assessment is conducted.
Skin and wound care
55. (1) The skin and wound care program must provide for routine skin care."""

ACT = """Residents' Bill of Rights
3 (1) Every licensee shall ensure that the following rights are respected:
RIGHT TO BE TREATED WITH RESPECT
1. Every resident has the right to be treated with courtesy.
2. Every resident has the right to have their lifestyle respected.
Mission statement
4 (1) Every licensee shall ensure there is a mission statement."""


def test_reg_sections_skip_toc_and_list_items():
    secs = law_sections(REG)
    assert [(n, h) for n, h, _ in secs] == [("53", "Required programs"), ("54", "Falls prevention and management"),
                                             ("55", "Skin and wound care")]
    assert "post-fall assessment" in secs[1][2]
    assert "1. A falls prevention" in secs[0][2]            # list item stays inside s. 53
    assert "Falls prevention and management" not in secs[0][2]  # next heading not glued to s. 53


def test_act_sections_without_period():
    secs = law_sections(ACT)
    assert [(n, h) for n, h, _ in secs] == [("3", "Residents' Bill of Rights"), ("4", "Mission statement")]
    assert "courtesy" in secs[0][2]


def test_chunk_law_citation():
    chunks = chunk_law("oreg_246_22", "O. Reg. 246/22: General", REG, "https://www.ontario.ca/laws/regulation/220246",
                       "2026-11-03", ChunkConfig(200, 40))
    falls = [c for c in chunks if c.section.startswith("s. 54")][0]
    assert falls.citation == "O. Reg. 246/22, s. 54 (Falls prevention and management)"
    assert falls.document_type == "regulation" and falls.resident_id is None
    assert falls.embed_text.startswith("O. Reg. 246/22, s. 54: Falls prevention and management\n")


# ---------------------------------------------------------------- PDFs
def test_pdf_boilerplate_and_section_tabs():
    BODY = "staff should assess every resident for falls risk on admission and after any fall"
    pages = [f"Guide Title Fourth Edition\n{tab}\n{BODY} {i}" for i, tab in
             enumerate(["RECOMMENDATIONS"] * 5 + ["APPENDICES"] * 5)]
    strip, tabs = pdf_boilerplate(pages)
    assert "Guide Title Fourth Edition" in strip
    assert tabs == {"RECOMMENDATIONS", "APPENDICES"}
    chunks = chunk_pdf("rnao_x", "guideline", "RNAO Guide", list(enumerate(pages, 1)), "https://x", "2017-09",
                       ChunkConfig(200, 40))
    assert chunks[0].text == f"{BODY} 0"                  # header and tab stripped from text
    assert chunks[0].citation == "RNAO Guide, PDF p. 1 (Recommendations)"
    assert chunks[0].page == 1


# ---------------------------------------------------------------- labels + residents
def test_label_chunk():
    [c] = chunk_label_section("label_warfarin", "Drug label: warfarin sodium (RxNorm 855332: Warfarin)",
                              "Boxed Warning", "WARNING: BLEEDING RISK", "https://dailymed", "2026-06-03",
                              ChunkConfig(200, 40))
    assert c.citation == "Drug label: warfarin sodium > Boxed Warning"
    assert c.document_type == "drug_label"


PROFILE = """# Resident 204
Sex: F | Age: 68 (age band 65-69) | Profile as of 2026-11-04

## Fall risk and falls
Latest Morse Fall Scale assessment: Low Risk.

## Active medications
- Metformin 500 MG

## Recent encounters (latest 5 of 53)
- 2026-10-18 wellness: General examination
"""


def test_profile_sections_and_chunks():
    assert [n for n, _ in profile_sections(PROFILE)] == ["Fall risk and falls", "Active medications",
                                                        "Recent encounters (latest 5 of 53)"]
    chunks = chunk_profile(204, PROFILE, "2026-11-04", ChunkConfig(200, 40))
    assert {c.resident_id for c in chunks} == {204}
    meds = [c for c in chunks if c.section == "Active medications"][0]
    assert meds.citation == "Resident 204 > Active medications (as of 2026-11-04)"
    assert meds.embed_text == "Resident 204 > Active medications\n- Metformin 500 MG"
    assert [c.section for c in chunks][-1] == "Recent encounters"


def test_chunk_ids_unique_and_depend_on_config():
    a = chunk_profile(204, PROFILE, "2026-11-04", ChunkConfig(200, 40))
    b = chunk_profile(204, PROFILE, "2026-11-04", ChunkConfig(100, 20))
    assert len({c.chunk_id for c in a}) == len(a)
    assert not {c.chunk_id for c in a} & {c.chunk_id for c in b}


def test_law_sections_survive_windows_line_endings_after_cleaning():
    from rcqa.documents import clean_text
    secs = law_sections(clean_text(REG.replace("\n", "\r\n")))
    assert [n for n, _, _ in secs] == ["53", "54", "55"]


def test_near_empty_pdf_pages_are_skipped():
    body = "Falls are common in older adults and many can be prevented by trained staff. Page {}"
    pages = [(1, "157 Endorsements")] + [(i, body.format(i) + f" topic{i} " * i) for i in range(2, 6)]
    chunks = chunk_pdf("rnao_x", "guideline", "RNAO Guide", pages, "https://x", "2017-09", ChunkConfig(200, 40))
    assert [c.page for c in chunks] == [2, 3, 4, 5]
