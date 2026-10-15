"""Read-only access to an EndNote library's own SQLite files.

An EndNote library `X.enl` keeps its data in `X.Data/`:

    sdb/sdb.eni   SQLite: the `refs` table (one row per reference) and
                  `file_res` (attached files, paths relative to PDF/)
    sdb/pdb.eni   SQLite: `pdf_index`, EndNote's own extracted PDF text
    PDF/          the attachments

The schema is undocumented. Everything here opens those files with
`mode=ro` and never writes to them; `check_schema()` fails loudly if a future
EndNote version renames something we rely on, rather than returning blanks.
"""

from __future__ import annotations

import contextlib
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

DEFAULT_LIBRARY = Path.home() / "Documents" / "My EndNote Library.enl"

# EndNote's numeric reference_type -> name. Derived empirically on 2026-10-15
# by joining refs.reference_type against the type names in an EndNote XML
# export of the same library (all 12 codes in use matched 1:1). Codes not seen
# there are reported as "Type <n>" rather than guessed.
REF_TYPES = {
    0: "Journal Article", 1: "Book", 2: "Thesis", 3: "Conference Proceedings",
    4: "Personal Communication", 6: "Computer Program", 7: "Book Section",
    10: "Report", 16: "Web Page", 31: "Generic", 33: "Conference Paper",
    38: "Ancient Text",
}

# refs columns this module reads; check_schema() verifies each exists.
_COLUMNS = {
    "id": "rec_number", "reference_type": "ref_type_code", "author": "authors",
    "year": "year", "title": "title", "secondary_title": "journal",
    "secondary_author": "editors", "volume": "volume", "number": "issue",
    "pages": "pages", "abstract": "abstract", "keywords": "keywords",
    "electronic_resource_number": "doi", "url": "url",
    "publisher": "publisher", "place_published": "place_published",
    "edition": "edition", "isbn": "isbn", "label": "label", "notes": "notes",
    "research_notes": "research_notes", "short_title": "short_title",
    "record_last_updated": "updated",
}
_LIST_FIELDS = ("authors", "editors", "keywords")
_DOI_RE = re.compile(r"10\.\d{4,9}/\S+")


class SchemaError(RuntimeError):
    pass


@dataclass(frozen=True)
class LibraryPaths:
    enl: Path

    @property
    def data(self) -> Path:
        return self.enl.with_suffix(".Data")

    @property
    def sdb(self) -> Path:
        return self.data / "sdb" / "sdb.eni"

    @property
    def pdb(self) -> Path:
        return self.data / "sdb" / "pdb.eni"

    @property
    def pdf_dir(self) -> Path:
        return self.data / "PDF"


def library_paths(enl: str | os.PathLike | None = None) -> LibraryPaths:
    enl = Path(enl or os.environ.get("ENDNOTE_LIBRARY") or DEFAULT_LIBRARY)
    paths = LibraryPaths(enl.expanduser())
    if not paths.sdb.exists():
        raise FileNotFoundError(
            f"No EndNote database at {paths.sdb}. Set ENDNOTE_LIBRARY to the "
            f"path of your .enl file.")
    return paths


def connect_ro(path: Path) -> sqlite3.Connection:
    """Open an EndNote SQLite file strictly read-only.

    Not `immutable=1`: EndNote may be writing while we read, and immutable
    would let SQLite skip the locking that keeps a read consistent.
    """
    conn = sqlite3.connect(f"file:{quote(str(path))}?mode=ro", uri=True,
                           timeout=15)
    conn.row_factory = sqlite3.Row
    return conn


def check_schema(conn: sqlite3.Connection) -> None:
    have = {r[1] for r in conn.execute("PRAGMA table_info(refs)")}
    missing = [c for c in _COLUMNS if c not in have]
    have_f = {r[1] for r in conn.execute("PRAGMA table_info(file_res)")}
    missing += [f"file_res.{c}" for c in ("refs_id", "file_path", "file_type",
                                          "file_pos") if c not in have_f]
    if missing:
        raise SchemaError(
            "EndNote's database layout has changed; missing: "
            + ", ".join(missing) + ". endnote-local needs updating.")


def _clean(s: str) -> str:
    return s.replace("﻿", "").strip()


def _split(s: str) -> list[str]:
    return [p for p in (_clean(x) for x in s.split("\r")) if p]


def clean_doi(raw: str) -> str:
    m = _DOI_RE.search(raw or "")
    return m.group(0).rstrip(".,;") if m else ""


def read_references(conn: sqlite3.Connection) -> list[dict]:
    """Every reference not in the trash, as plain dicts."""
    cols = ", ".join(_COLUMNS)
    out = []
    for row in conn.execute(f"SELECT {cols} FROM refs WHERE trash_state = 0"):
        ref = {}
        for col, key in _COLUMNS.items():
            v = row[col]
            if key in _LIST_FIELDS:
                v = _split(v or "")
            elif isinstance(v, str):
                v = _clean(v)
            ref[key] = v
        # One author field holding a whole comma-separated list ("A Bône,
        # M Louis, ..."), a data-entry slip in a few records: a single
        # "Family, Given" name has exactly one comma, so 2+ means a list.
        if len(ref["authors"]) == 1 and ref["authors"][0].count(",") >= 2:
            ref["authors"] = [a.strip() for a in ref["authors"][0].split(",")
                              if a.strip()]
        code = ref.pop("ref_type_code")
        ref["ref_type"] = REF_TYPES.get(code, f"Type {code}")
        ref["doi"] = clean_doi(ref["doi"])
        out.append(ref)
    return out


def read_attachments(conn: sqlite3.Connection) -> dict[int, list[str]]:
    """rec_number -> attached file paths (relative to PDF/), in EndNote order."""
    out: dict[int, list[str]] = {}
    for r in conn.execute(
            "SELECT refs_id, file_path FROM file_res WHERE file_type = 1 "
            "ORDER BY refs_id, file_pos"):
        out.setdefault(r["refs_id"], []).append(r["file_path"])
    return out


def endnote_pdf_text(pdb: Path, rec_number: int, rel_path: str) -> str:
    """EndNote's own extracted text for one attachment ('' if none).

    Used only as a fallback when PyMuPDF cannot read a file: it carries no
    page breaks, so it comes back as a single unnumbered page.
    """
    if not pdb.exists():
        return ""
    with contextlib.closing(connect_ro(pdb)) as conn:
        row = conn.execute(
            "SELECT contents FROM pdf_index WHERE refs_id = ? AND subkey = ?",
            (rec_number, rel_path.encode())).fetchone()
    return row["contents"] if row else ""
