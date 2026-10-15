"""MCP server over the local EndNote index (stdio). Run: `enlib serve`.

Tool names and arguments follow the endnote-mcp server this replaces, so
existing habits and instructions carry over; `search_passages` and
`index_status` are new, and `rebuild_index` became `sync_index`.

Stdout is the MCP channel; all logging goes to stderr.
"""

from __future__ import annotations

import time

try:  # mcp >= 2 renamed FastMCP
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:
    from mcp.server.fastmcp import FastMCP as _Server

from . import fmt
from .cite import STYLES, format_bibtex, format_citation
from .index import Index, log

SYNC_EVERY = 60  # seconds; a no-change sync is cheap but not free

mcp = _Server(
    "EndNote (local)",
    instructions=(
        "Search, cite and read PDFs from the user's EndNote library, read "
        "directly from EndNote's own database (always current; no export "
        "step). Record numbers [n] are EndNote's record numbers. Verify a "
        "citation's details with get_reference_details before presenting "
        "it."))

_ix: Index | None = None


def ix() -> Index:
    global _ix
    if _ix is None:
        _ix = Index()
    if time.time() - _ix.last_sync > SYNC_EVERY:
        try:
            _ix.sync(quiet=True)
        except Exception as e:  # serve the existing index rather than fail
            log(f"sync failed: {e}")
            _ix.last_sync = time.time()
    return _ix


def _nums(s: str) -> list[int]:
    return [int(x) for x in s.replace(";", ",").split(",") if x.strip()]


@mcp.tool()
def search_references(query: str, year_from: str | None = None,
                      year_to: str | None = None, author: str | None = None,
                      ref_type: str | None = None, limit: int = 50) -> str:
    """Search the library by title, author, keywords, abstract, journal and notes (BM25-ranked).

    Args:
        query: Search terms. Punctuation is safe; "quoted phrases", prefix*
            and uppercase AND / OR / NOT are honoured. All terms must match;
            if none do, any-term matching is tried. May be empty when a
            filter is given, to list e.g. everything by one author.
        year_from: Optional earliest year (e.g. "2015").
        year_to: Optional latest year.
        author: Optional author filter (partial name match).
        ref_type: Optional type, e.g. "Journal Article", "Book", "Book Section".
        limit: Maximum results (default 50).
    """
    return fmt.references(
        ix().search_references(query, year_from, year_to, author, ref_type, limit),
        f"No references found for: {query}")


@mcp.tool()
def search_fulltext(query: str, limit: int = 50, year_from: str | None = None,
                    year_to: str | None = None, author: str | None = None) -> str:
    """Search inside the attached PDFs for exact words or phrases; returns page-numbered snippets grouped by reference.

    Args:
        query: Terms to find in PDF text (same syntax as search_references;
            all terms must occur on the same page).
        limit: Maximum references (default 50).
        year_from, year_to, author: Optional filters.
    """
    return fmt.fulltext(ix().search_fulltext(query, limit, year_from=year_from,
                                             year_to=year_to, author=author), query)


@mcp.tool()
def search_semantic(query: str, limit: int = 20) -> str:
    """Find references by meaning rather than wording (embeddings of title, abstract and keywords; first pages of the PDF stand in for a missing abstract).

    Args:
        query: A natural-language description of what you are looking for.
        limit: Maximum results (default 20).
    """
    return fmt.semantic(ix().search_semantic(query, limit), query)


@mcp.tool()
def search_passages(query: str, limit: int = 10, per_reference: int = 2) -> str:
    """Find PDF passages by meaning: semantic search over the full text of every attachment, returning the best-matching passages with page numbers.

    Use to locate where a paper discusses something when the wording is unknown.

    Args:
        query: A natural-language description of the content wanted.
        limit: Maximum references (default 10).
        per_reference: Passages shown per reference (default 2).
    """
    return fmt.passages(ix().search_passages(query, limit, per_reference), query)


@mcp.tool()
def search_library(query: str, year_from: str | None = None,
                   year_to: str | None = None, author: str | None = None,
                   ref_type: str | None = None, limit: int = 30) -> str:
    """One-call search combining metadata, PDF full text and semantic similarity (reciprocal-rank fusion); references found several ways rank highest.

    Args:
        query: Search terms or a short description.
        year_from, year_to, author, ref_type: Optional filters.
        limit: Maximum results (default 30).
    """
    return fmt.library(ix().search_library(query, year_from, year_to, author,
                                           ref_type, limit), query)


@mcp.tool()
def list_references_by_topic(topic: str, year_from: str | None = None,
                             year_to: str | None = None,
                             ref_type: str | None = None, limit: int = 50) -> str:
    """List references on a broad topic (same ranking as search_library, without snippets).

    Args:
        topic: Broad topic terms.
        year_from, year_to, ref_type: Optional filters.
        limit: Maximum results (default 50).
    """
    res = ix().search_library(topic, year_from, year_to, None, ref_type, limit)
    return fmt.references(res, f"No references found for topic: {topic}")


@mcp.tool()
def find_related(rec_number: int, limit: int = 10) -> str:
    """Find references most similar in meaning to a given reference.

    Args:
        rec_number: EndNote record number (the [n] in results).
        limit: Maximum results (default 10).
    """
    i = ix()
    r = i.ref(rec_number)
    if r is None:
        return f"Reference #{rec_number} not found."
    return fmt.references(i.find_related(rec_number, limit),
                          f"No related references for #{rec_number}.",
                          f"reference(s) related to [{rec_number}] {r['title']}")


@mcp.tool()
def get_reference_details(rec_number: int) -> str:
    """All fields of one reference: authors, journal, volume, pages, DOI, abstract, keywords, notes, and its attachments.

    Args:
        rec_number: EndNote record number.
    """
    i = ix()
    return fmt.details(i.ref(rec_number), i.attachments(rec_number), rec_number)


@mcp.tool()
def read_pdf_section(rec_number: int, start_page: int = 1, end_page: int = 5,
                     attachment: int = 1) -> str:
    """Read pages of a reference's PDF (at most 30 pages per call).

    Args:
        rec_number: EndNote record number.
        start_page: First page, 1-based PDF page (default 1).
        end_page: Last page (default 5).
        attachment: Which attachment, 1-based, as listed by get_reference_details (default 1).
    """
    end_page = min(end_page, start_page + 29)
    i = ix()
    att, pages, total = i.pages(rec_number, attachment, start_page, end_page)
    return fmt.pdf_pages(i.ref(rec_number), att, pages, total, start_page,
                         end_page, rec_number, attachment)


@mcp.tool()
def get_citation(rec_number: int, style: str = "apa7") -> str:
    """Format one reference as a citation.

    Args:
        rec_number: EndNote record number.
        style: apa7, harvard, vancouver, chicago or ieee.
    """
    r = ix().ref(rec_number)
    if r is None:
        return f"Reference #{rec_number} not found."
    try:
        return f"[{style.upper()}] {format_citation(r, style)}"
    except ValueError as e:
        return str(e)


@mcp.tool()
def get_bibliography(rec_numbers: str, style: str = "apa7",
                     sort: str = "author") -> str:
    """Formatted reference list for several references.

    Args:
        rec_numbers: Comma-separated record numbers, e.g. "12,45,78".
        style: apa7, harvard, vancouver, chicago or ieee.
        sort: "author" or "year".
    """
    if style.lower() not in STYLES:
        return f"Unknown style '{style}'. Choose from: {', '.join(STYLES)}"
    i = ix()
    nums = _nums(rec_numbers)
    refs = [r for r in (i.ref(n) for n in nums) if r]
    key = (lambda r: (r.get("year") or "", r["authors"][:1])) if sort == "year" \
        else (lambda r: (r["authors"][:1], r.get("year") or ""))
    lines = [f"{n}. {format_citation(r, style)}"
             for n, r in enumerate(sorted(refs, key=key), 1)]
    missing = sorted(set(nums) - {r["rec_number"] for r in refs})
    if missing:
        lines.append(f"\nNot found: {', '.join(map(str, missing))}")
    return "\n".join(lines) or "No references found."


@mcp.tool()
def get_bibtex(rec_numbers: str) -> str:
    """BibTeX entries for one or more references.

    Args:
        rec_numbers: Comma-separated record numbers.
    """
    i = ix()
    nums = _nums(rec_numbers)
    refs = [r for r in (i.ref(n) for n in nums) if r]
    out = "\n\n".join(format_bibtex(r) for r in refs)
    missing = sorted(set(nums) - {r["rec_number"] for r in refs})
    if missing:
        out += f"\n\n% Not found: {', '.join(map(str, missing))}"
    return out


@mcp.tool()
def sync_index() -> str:
    """Bring the index up to date with EndNote now (normally automatic, at most once a minute). Large changes, e.g. hundreds of new PDFs, can take minutes."""
    i = ix()
    stats = i.sync(quiet=True, wait=True)
    return "Sync complete:\n" + "\n".join(f"  {k}: {v}" for k, v in stats.items())


@mcp.tool()
def index_status() -> str:
    """Counts of indexed references, attachments, pages and embeddings, and when the index last synced."""
    return "\n".join(f"  {k}: {v}" for k, v in ix().stats().items())


def main():
    mcp.run()
