"""Text rendering shared by the CLI and the MCP server, so both say the same."""

from __future__ import annotations

from .cite import short_authors


def _doi(r) -> str:
    return f"  https://doi.org/{r['doi']}" if r.get("doi") else ""


def ref_line(r: dict, prefix: str = "") -> str:
    j = f" *{r['journal']}*." if r.get("journal") else ""
    return (f"  [{r['rec_number']}] {prefix}{short_authors(r['authors'])} "
            f"({r.get('year') or 'n.d.'}). {r['title']}.{j}{_doi(r)}")


def references(results, empty: str, header: str = "reference(s)") -> str:
    if not results:
        return empty
    return "\n".join([f"Found {len(results)} {header}:", ""]
                     + [ref_line(r) for r in results])


def _page(p) -> str:
    return "p. ?" if p == 0 else f"p. {p}"


def fulltext(results, query) -> str:
    if not results:
        return f"No full-text matches for: {query}"
    n = sum(len(r["snippets"]) for r in results)
    lines = [f"Found {n} match(es) in {len(results)} reference(s):", ""]
    for r in results:
        lines.append(ref_line(r))
        for s in r["snippets"]:
            lines.append(f"      {_page(s['page'])}: {s['snippet']}")
    return "\n".join(lines)


def semantic(results, query) -> str:
    if not results:
        return f"No semantic matches for: {query}"
    return "\n".join([f"Found {len(results)} reference(s) by meaning:", ""]
                     + [ref_line(r, f"({r['similarity']:.0%}) ") for r in results])


def passages(results, query) -> str:
    if not results:
        return (f"No passage matches for: {query}. (Passage embeddings may not "
                f"be built yet: run `enlib sync`.)")
    lines = [f"Best-matching passages in {len(results)} reference(s):", ""]
    for r in results:
        lines.append(ref_line(r))
        for p in r["passages"]:
            lines.append(f"      ({p['similarity']:.0%}) {_page(p['page'])}: "
                         f"{p['text']}")
        lines.append("")
    return "\n".join(lines).rstrip()


def library(results, query) -> str:
    if not results:
        return f"No references found for: {query}"
    lines = [f"Found {len(results)} reference(s) (metadata, full text and "
             f"meaning combined):", ""]
    for r in results:
        lines.append(ref_line(r))
        for s in r.get("snippets", []):
            lines.append(f"      {_page(s['page'])}: {s['snippet']}")
    return "\n".join(lines)


def details(r: dict | None, atts: list[dict], rn: int) -> str:
    if r is None:
        return f"Reference #{rn} not found."
    L = [f"Reference #{rn}:", f"  Type: {r['ref_type']}", f"  Title: {r['title']}"]
    if r["authors"]:
        L.append(f"  Authors: {'; '.join(r['authors'])}")
    L.append(f"  Year: {r.get('year', '')}")
    for key, label in (("journal", "Journal / secondary title"),
                       ("volume", "Volume"), ("issue", "Issue"),
                       ("pages", "Pages"), ("doi", "DOI"), ("url", "URL"),
                       ("publisher", "Publisher"),
                       ("place_published", "Place published"),
                       ("edition", "Edition"), ("isbn", "ISBN"),
                       ("label", "Label")):
        if r.get(key):
            L.append(f"  {label}: {r[key]}")
    if r.get("editors"):
        L.append(f"  Editors: {'; '.join(r['editors'])}")
    if r.get("keywords"):
        L.append(f"  Keywords: {'; '.join(r['keywords'])}")
    if r.get("abstract"):
        L.append(f"  Abstract: {r['abstract']}")
    if r.get("research_notes"):
        L.append(f"  Research notes: {r['research_notes']}")
    if r.get("notes"):
        L.append(f"  Notes: {r['notes']}")
    if atts:
        L.append("  Attachments:")
        for i, a in enumerate(atts, 1):
            how = {"pymupdf": f"{a['n_pages']} pages of text",
                   "endnote": "text from EndNote's index, no page numbers",
                   "missing": "FILE MISSING", "no text": "no extractable text"
                   }.get(a["source"], a["source"])
            L.append(f"    {i}. {a['full_path']} ({how})")
    else:
        L.append("  Attachments: none")
    return "\n".join(L)


def pdf_pages(r, att, pages, total, start, end, rn, attachment) -> str:
    if r is None:
        return f"Reference #{rn} not found."
    if att is None:
        return f"Reference #{rn} has no attachment number {attachment}."
    if not pages:
        return (f"No text for pages {start}-{end} of attachment {attachment} "
                f"({att['source']}; {total} page(s) with text).")
    L = [f"{r['title']} — {att['path']}, pages {start}-{end} of {total}", ""]
    for n, text in pages:
        L.append("--- Page ? (EndNote text has no page breaks) ---" if n == 0
                 else f"--- Page {n} ---")
        L.append(text)
        L.append("")
    return "\n".join(L)
