"""Split every source into retrieval chunks that carry their own citation.

Structure first, size second: each source is cut at its natural boundaries
(law section, label section, PDF page, resident-profile section) and only then
packed into windows of at most ``chunk_size`` words with ``overlap`` words carried
over. So a chunk never straddles two law sections, and every chunk can be cited
precisely ("O. Reg. 246/22, s. 54", "Warfarin label > Boxed Warning",
"RNAO falls guideline, PDF p. 37", "Resident 204 > Active medications").

Sizes are in words (~0.75 words per BGE/MiniLM token); 200 words stays well under
the 512-token limit of the embedding models we'll use. Pure Python, unit-tested.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import asdict, dataclass

LAW_CITES = {"oreg_246_22": "O. Reg. 246/22", "fltca_2021": "Fixing Long-Term Care Act, 2021"}


@dataclass(frozen=True)
class ChunkConfig:
    chunk_size: int = 200   # max words per chunk
    overlap: int = 40       # words repeated from the previous chunk of the same section

    @property
    def config_id(self) -> str:
        return f"w{self.chunk_size}_o{self.overlap}"

    def __post_init__(self):
        if not 0 <= self.overlap < self.chunk_size:
            raise ValueError("need 0 <= overlap < chunk_size")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source_id: str
    document_type: str
    title: str
    section: str | None
    page: int | None
    resident_id: int | None
    source_date: str | None
    citation: str          # human-readable, shown under the answer
    citation_url: str | None
    chunk_index: int       # position within its section
    text: str              # the chunk itself (what the LLM quotes)
    embed_text: str        # context header + text (what gets embedded)
    n_words: int

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- windows
def windows(text: str, cfg: ChunkConfig) -> list[str]:
    """Pack lines into chunks of <= chunk_size words, starting each new chunk with the
    last ``overlap`` words of the previous one. Lines longer than chunk_size are split."""
    units: list[list[str]] = []
    for line in text.splitlines():
        words = line.split()
        while len(words) > cfg.chunk_size:
            units.append(words[:cfg.chunk_size])
            words = words[cfg.chunk_size - cfg.overlap:]
        if words:
            units.append(words)

    chunks, current, fresh = [], [], 0   # fresh = words added since the overlap prefix
    for unit in units:
        if current and len(current) + len(unit) > cfg.chunk_size:
            chunks.append(current)
            current = current[-cfg.overlap:] if cfg.overlap else []
            fresh = 0
            if len(current) + len(unit) > cfg.chunk_size:   # overlap + unit too big: trim overlap
                current = current[len(current) + len(unit) - cfg.chunk_size:]
        current = current + unit
        fresh += len(unit)
    if current and fresh:
        chunks.append(current)
    return [" ".join(c) for c in chunks]


def _cid(*parts) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def _make(cfg, source_id, document_type, title, section, page, resident_id, source_date,
          citation, citation_url, header, body) -> list[Chunk]:
    out = []
    for i, text in enumerate(windows(body, cfg)):
        out.append(Chunk(
            chunk_id=_cid(cfg.config_id, source_id, section, page, resident_id, i),
            source_id=source_id, document_type=document_type, title=title, section=section, page=page,
            resident_id=resident_id, source_date=source_date, citation=citation, citation_url=citation_url,
            chunk_index=i, text=text, embed_text=f"{header}\n{text}", n_words=len(text.split())))
    return out


# --------------------------------------------------------------------------- laws
_SECTION_START = re.compile(r"^(\d+(?:\.\d+)?)\.?\s+(?:\(\d+(?:\.\d+)?\)\s+)?\S")


def _is_heading(line: str) -> bool:
    """e-Laws section headings: short, sentence case, no closing punctuation."""
    s = line.strip()
    return (0 < len(s) <= 120 and not s.isupper() and s[0].isupper()
            and not s[-1] in ".;:," and not re.match(r"^[\d(]", s))


def law_sections(text: str) -> list[tuple[str, str, str]]:
    """[(section number, heading, body)]. A section starts at a numbered line that directly
    follows a heading and whose number is larger than the previous section's (so numbered
    list items like '1. Every resident...' inside a section are not mistaken for sections)."""
    lines = [l.strip() for l in text.splitlines()]   # e-Laws indents section lines with a tab
    out, current, last_num = [], None, -1.0
    for i, line in enumerate(lines):
        m = _SECTION_START.match(line)
        prev = lines[i - 1] if i else ""
        if m and _is_heading(prev) and float(m.group(1)) > last_num:
            if current:
                out.append(current)
            last_num = float(m.group(1))
            current = [m.group(1), prev.strip(), [line]]
        elif current:
            if _is_heading(line) and i + 1 < len(lines) and _SECTION_START.match(lines[i + 1]):
                continue   # next section's heading; it's attached when that section starts
            current[2].append(line)
    if current:
        out.append(current)
    return [(num, heading, "\n".join(body).strip()) for num, heading, body in out]


def chunk_law(source_id, title, text, citation_url, source_date, cfg) -> list[Chunk]:
    cite = LAW_CITES.get(source_id, title)
    doc_type = "regulation" if source_id.startswith("oreg") else "legislation"
    out = []
    for num, heading, body in law_sections(text):
        section = f"s. {num} {heading}"
        out += _make(cfg, source_id, doc_type, title, section, None, None, source_date,
                     f"{cite}, s. {num} ({heading})", citation_url, f"{cite}, s. {num}: {heading}", body)
    return out


# --------------------------------------------------------------------------- PDFs
def pdf_boilerplate(pages: list[str], header_share: float = 0.3, tab_share: float = 0.05):
    """Lines repeated across many pages. Returns (lines to strip, section-tab labels).
    Section tabs are short ALL-CAPS lines like 'RECOMMENDATIONS' that mark which part
    of the guideline a page is in; we keep them as the page's section."""
    counts = Counter(l.strip() for t in pages for l in set(t.splitlines()) if l.strip())
    n = max(len(pages), 1)
    strip = {l for l, c in counts.items() if c / n >= header_share or (l.isupper() and len(l) <= 40 and c / n >= tab_share)}
    tabs = {l for l in strip if l.isupper() and len(l) <= 40 and any(ch.isalpha() for ch in l)
            and tab_share <= counts[l] / n <= 0.6}
    return strip, tabs


MIN_PAGE_WORDS = 10   # cover, image-only and endorsement pages: just a page number + tab label


def chunk_pdf(source_id, document_type, title, pages: list[tuple[int, str]], citation_url, source_date,
              cfg) -> list[Chunk]:
    strip, tabs = pdf_boilerplate([t for _, t in pages])
    out = []
    for page, text in pages:
        lines = text.splitlines()
        section = next((l.strip().title() for l in lines if l.strip() in tabs), None)
        body = "\n".join(l for l in lines if l.strip() not in strip)
        if len(re.findall(r"[A-Za-z]{2,}", body)) < MIN_PAGE_WORDS:
            continue
        cite = f"{title}, PDF p. {page}" + (f" ({section})" if section else "")
        header = f"{title}" + (f" > {section}" if section else "") + f" (p. {page})"
        out += _make(cfg, source_id, document_type, title, section, page, None, source_date,
                     cite, citation_url, header, body)
    return out


# --------------------------------------------------------------------------- labels
def chunk_label_section(source_id, title, section, text, citation_url, source_date, cfg) -> list[Chunk]:
    drug = title.split(" (RxNorm")[0]          # "Drug label: warfarin sodium"
    return _make(cfg, source_id, "drug_label", title, section, None, None, source_date,
                 f"{drug} > {section}", citation_url, f"{drug} > {section}", text)


# --------------------------------------------------------------------------- residents
def profile_sections(profile_text: str) -> list[tuple[str, str]]:
    """Split a rendered profile on '## ' headings. The '# Resident N' header line is
    prepended to every section so each chunk keeps the resident's age/sex context."""
    lines = profile_text.strip().splitlines()
    out, name, body = [], None, []
    for line in lines:
        if line.startswith("## "):
            if name:
                out.append((name, "\n".join(body).strip()))
            name, body = line[3:].strip(), []
        elif name:
            body.append(line)
    if name:
        out.append((name, "\n".join(body).strip()))
    return [(n, b) for n, b in out if b]


def chunk_profile(resident_id: int, profile_text: str, as_of: str, cfg) -> list[Chunk]:
    out = []
    for section, body in profile_sections(profile_text):
        short = re.sub(r"\s*\(.*\)$", "", section)       # "Recent encounters (latest 5 of 53)" -> "Recent encounters"
        out += _make(cfg, f"resident_{resident_id}", "resident_profile", f"Resident {resident_id} profile",
                     short, None, resident_id, as_of, f"Resident {resident_id} > {short} (as of {as_of})",
                     None, f"Resident {resident_id} > {section}", body)
    return out
