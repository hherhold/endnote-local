"""`enlib` — command-line access to the EndNote library.

Every query first runs an incremental sync (instant when nothing changed);
`--no-sync` skips it. `--json` prints machine-readable results.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import fmt
from .cite import STYLES, format_bibtex, format_citation
from .index import Index


def _filters(p):
    p.add_argument("--from", dest="year_from")
    p.add_argument("--to", dest="year_to")
    p.add_argument("--author")
    p.add_argument("--type", dest="ref_type")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="enlib", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--json", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search", help="metadata search (title, authors, abstract, keywords)")
    s.add_argument("query", nargs="*"); _filters(s)
    s.add_argument("-n", "--limit", type=int, default=50)

    s = sub.add_parser("fulltext", help="exact words/phrases inside PDFs")
    s.add_argument("query", nargs="+"); _filters(s)
    s.add_argument("-n", "--limit", type=int, default=50)

    s = sub.add_parser("semantic", help="references by meaning")
    s.add_argument("query", nargs="+")
    s.add_argument("-n", "--limit", type=int, default=20)

    s = sub.add_parser("passages", help="PDF passages by meaning")
    s.add_argument("query", nargs="+")
    s.add_argument("-n", "--limit", type=int, default=10)
    s.add_argument("--per-ref", type=int, default=2)

    s = sub.add_parser("library", help="combined metadata + full text + semantic")
    s.add_argument("query", nargs="+"); _filters(s)
    s.add_argument("-n", "--limit", type=int, default=30)

    s = sub.add_parser("related", help="references similar to a record")
    s.add_argument("rec", type=int)
    s.add_argument("-n", "--limit", type=int, default=10)

    s = sub.add_parser("show", help="all fields of a record")
    s.add_argument("rec", type=int)

    s = sub.add_parser("read", help="read pages of a record's PDF")
    s.add_argument("rec", type=int)
    s.add_argument("pages", nargs="?", default="1-5", help="e.g. 3 or 3-7")
    s.add_argument("-a", "--attachment", type=int, default=1)

    s = sub.add_parser("cite", help="formatted citation(s)")
    s.add_argument("recs", type=int, nargs="+")
    s.add_argument("-s", "--style", default="apa7", choices=STYLES)

    s = sub.add_parser("bibtex", help="BibTeX entries")
    s.add_argument("recs", type=int, nargs="+")

    s = sub.add_parser("sync", help="update the index (progress on stderr)")
    s.add_argument("--no-passages", action="store_true",
                   help="skip passage embeddings")

    sub.add_parser("status", help="index statistics")
    sub.add_parser("serve", help="run the MCP server on stdio")

    a = ap.parse_args(argv)
    q = " ".join(getattr(a, "query", []) or [])

    if a.cmd == "serve":
        from .server import main as serve
        return serve()

    ix = Index()
    if a.cmd == "sync":
        stats = ix.sync(passages=not a.no_passages, wait=True)
        print(json.dumps(stats, indent=1))
        return
    if not a.no_sync:
        ix.sync(quiet=False)

    def out(data, text):
        print(json.dumps(data, indent=1, ensure_ascii=False) if a.json else text)

    flt = {k: getattr(a, k, None) for k in ("year_from", "year_to", "author", "ref_type")}
    if a.cmd == "search":
        r = ix.search_references(q, limit=a.limit, **flt)
        out(r, fmt.references(r, f"No references found for: {q}"))
    elif a.cmd == "fulltext":
        r = ix.search_fulltext(q, a.limit, **flt)
        out(r, fmt.fulltext(r, q))
    elif a.cmd == "semantic":
        r = ix.search_semantic(q, a.limit)
        out(r, fmt.semantic(r, q))
    elif a.cmd == "passages":
        r = ix.search_passages(q, a.limit, a.per_ref)
        out(r, fmt.passages(r, q))
    elif a.cmd == "library":
        r = ix.search_library(q, limit=a.limit, **flt)
        out(r, fmt.library(r, q))
    elif a.cmd == "related":
        r = ix.find_related(a.rec, a.limit)
        out(r, fmt.references(r, f"No related references for #{a.rec}."))
    elif a.cmd == "show":
        r, atts = ix.ref(a.rec), ix.attachments(a.rec)
        out({"reference": r, "attachments": atts}, fmt.details(r, atts, a.rec))
    elif a.cmd == "read":
        lo, _, hi = a.pages.partition("-")
        lo = int(lo); hi = min(int(hi or lo), lo + 29)
        att, pages, total = ix.pages(a.rec, a.attachment, lo, hi)
        out({"attachment": att, "pages": pages, "total": total},
            fmt.pdf_pages(ix.ref(a.rec), att, pages, total, lo, hi, a.rec,
                          a.attachment))
    elif a.cmd in ("cite", "bibtex"):
        refs = [ix.ref(n) for n in a.recs]
        for n, r in zip(a.recs, refs):
            if r is None:
                print(f"Reference #{n} not found.", file=sys.stderr)
            elif a.cmd == "cite":
                print(f"[{n}] {format_citation(r, a.style)}")
            else:
                print(format_bibtex(r) + "\n")
    elif a.cmd == "status":
        st = ix.stats()
        out(st, "\n".join(f"{k}: {v}" for k, v in st.items()))


if __name__ == "__main__":
    main()
