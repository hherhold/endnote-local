"""Citation strings and BibTeX from reference dicts (see library.read_references).

Plain-text formatting, good enough to paste and check; not a CSL engine.
Italics are marked with *asterisks*.
"""

from __future__ import annotations

import re
import unicodedata

STYLES = ("apa7", "harvard", "vancouver", "chicago", "ieee")


# ---------------------------------------------------------------- names
def _split_name(name: str) -> tuple[str, str]:
    """'Loza, Cleopatra Mara' or 'Cleopatra Mara Loza' -> (family, given)."""
    name = name.strip()
    if "," in name:
        fam, given = name.split(",", 1)
        return fam.strip(), given.strip()
    bits = name.split()
    if len(bits) == 1:
        return name, ""
    return bits[-1], " ".join(bits[:-1])


def _initials(given: str, sep: str = ". ", end: str = ".") -> str:
    parts = [p for p in re.split(r"[\s.]+", given) if p]
    out = ["-".join(s[0] for s in p.split("-") if s) for p in parts]
    return (sep.join(out) + end) if out else ""


def _inverted(name: str) -> str:            # Loza, C. M.
    fam, given = _split_name(name)
    ini = _initials(given)
    return f"{fam}, {ini}" if ini else fam


def _direct(name: str) -> str:              # C. M. Loza
    fam, given = _split_name(name)
    ini = _initials(given)
    return f"{ini} {fam}" if ini else fam


def short_authors(authors: list[str]) -> str:
    if not authors:
        return "Unknown"
    fams = [_split_name(a)[0] for a in authors]
    if len(fams) == 1:
        return fams[0]
    if len(fams) == 2:
        return f"{fams[0]} & {fams[1]}"
    return f"{fams[0]} et al."


# --------------------------------------------------------------- pieces
def _is_article(r) -> bool:
    return r["ref_type"] in ("Journal Article", "Magazine Article",
                             "Newspaper Article", "Electronic Article")


def _is_book(r) -> bool:
    return r["ref_type"] in ("Book", "Edited Book", "Thesis", "Report")


def _doi_url(r) -> str:
    return f"https://doi.org/{r['doi']}" if r.get("doi") else (r.get("url") or "")


def _vol_issue(r, fmt="{v}({i})") -> str:
    v, i = r.get("volume", ""), r.get("issue", "")
    if v and i:
        return fmt.format(v=v, i=i)
    return v or (f"({i})" if i else "")


def _pages(r) -> str:
    return (r.get("pages") or "").replace("--", "–").replace("-", "–")


def _end(s: str) -> str:
    s = s.strip()
    return s if not s or s[-1] in ".?!" or s.endswith((".”", "?”", "!”")) \
        else s + "."


def _pub(r) -> str:
    return ": ".join(x for x in (r.get("place_published"), r.get("publisher")) if x)


# --------------------------------------------------------------- styles
def _apa7(r):
    a = r["authors"]
    if not a:
        who = ""
    elif len(a) <= 20:
        names = [_inverted(x) for x in a]
        who = names[0] if len(names) == 1 else ", ".join(names[:-1]) + ", & " + names[-1]
    else:
        who = ", ".join(_inverted(x) for x in a[:19]) + ", … " + _inverted(a[-1])
    year = r.get("year") or "n.d."
    head = f"{who} ({year}). " if who else f"{_end(r['title'])} ({year}). "
    title = r["title"] if who else ""
    if _is_article(r):
        tail = f"*{r['journal']}*" if r.get("journal") else ""
        vi = _vol_issue(r, "*{v}*({i})") if r.get("volume") else _vol_issue(r)
        tail = ", ".join(x for x in (tail, vi, _pages(r)) if x)
        s = head + _end(title) + (" " + _end(tail) if tail else "")
    elif r["ref_type"] == "Book Section":
        eds = r.get("editors") or []
        ed = ""
        if eds:
            ed = " " + ", ".join(_direct(e) for e in eds) + (" (Eds.)," if len(eds) > 1 else " (Ed.),")
        bk = f"In{ed} *{r['journal']}*" if r.get("journal") else ""
        pp = f" (pp. {_pages(r)})" if r.get("pages") else ""
        s = head + _end(title) + (" " + _end(bk + pp) if bk else "") + \
            (" " + _end(r["publisher"]) if r.get("publisher") else "")
    else:
        s = head + (_end(f"*{title}*") if title else "") + \
            (" " + _end(r["publisher"]) if r.get("publisher") else "")
    url = _doi_url(r)
    return (s + " " + url).strip() if url else s.strip()


def _harvard(r):
    a = r["authors"]
    names = [_inverted(x) for x in a]
    who = names[0] if len(names) == 1 else (", ".join(names[:-1]) + " and " + names[-1] if names else "")
    s = f"{who} ({r.get('year') or 'n.d.'}) "
    if _is_article(r):
        s += f"'{r['title']}'" + (f", *{r['journal']}*" if r.get("journal") else "")
        vi = _vol_issue(r)
        if vi:
            s += f", {vi}"
        if r.get("pages"):
            s += f", pp. {_pages(r)}"
        s = _end(s)
    else:
        s += _end(f"*{r['title']}*")
        if _pub(r):
            s += " " + _end(_pub(r))
    if r.get("doi"):
        s += f" doi:{r['doi']}."
    return s.strip()


def _vancouver(r):
    def vn(name):
        fam, given = _split_name(name)
        return f"{fam} {_initials(given, sep='', end='')}".strip()
    a = [vn(x) for x in r["authors"]]
    who = ", ".join(a[:6]) + (", et al" if len(a) > 6 else "")
    s = (_end(who) + " " if who else "") + _end(r["title"]) + " "
    if _is_article(r):
        s += (f"{r['journal']}. " if r.get("journal") else "") + (r.get("year") or "")
        vi = _vol_issue(r)
        s += f";{vi}" if vi else ""
        s += f":{_pages(r)}" if r.get("pages") else ""
        s = _end(s)
    else:
        if _pub(r):
            s += _pub(r) + "; "
        s = _end(s + (r.get("year") or ""))
    if r.get("doi"):
        s += f" doi:{r['doi']}"
    return s.strip()


def _chicago(r):
    a = r["authors"]
    if not a:
        who = ""
    elif len(a) == 1:
        who = _split_name(a[0])[0] + ", " + _split_name(a[0])[1]
    else:
        first = _split_name(a[0])[0] + ", " + _split_name(a[0])[1]
        rest = [" ".join(reversed(_split_name(x))).strip() for x in a[1:]]
        who = first + ", " + ", ".join(rest[:-1]) + (", " if rest[:-1] else "") + "and " + rest[-1]
    s = (_end(who) + " " if who else "") + f"{r.get('year') or 'n.d.'}. "
    if _is_article(r):
        s += f"“{_end(r['title'])}”" + (f" *{r['journal']}*" if r.get("journal") else "")
        if r.get("volume"):
            s += f" {r['volume']}"
        if r.get("issue"):
            s += f" ({r['issue']})"
        if r.get("pages"):
            s += f": {_pages(r)}"
        s = _end(s)
    else:
        s += _end(f"*{r['title']}*") + (" " + _end(_pub(r)) if _pub(r) else "")
    url = _doi_url(r)
    return (s + " " + url).strip() if url else s.strip()


def _ieee(r):
    a = [_direct(x) for x in r["authors"]]
    if len(a) > 6:
        who = a[0] + " et al."
    elif len(a) > 2:
        who = ", ".join(a[:-1]) + ", and " + a[-1]
    else:
        who = " and ".join(a)
    s = (who + ", " if who else "")
    if _is_article(r):
        s += f"“{r['title']},”" + (f" *{r['journal']}*" if r.get("journal") else "")
        if r.get("volume"):
            s += f", vol. {r['volume']}"
        if r.get("issue"):
            s += f", no. {r['issue']}"
        if r.get("pages"):
            s += f", pp. {_pages(r)}"
        s += (" " if s.endswith("”") else ", ") + (r.get("year") or "")
    else:
        s += f"*{r['title']}*"
        if _pub(r):
            s += f". {_pub(r)}"
        s += f", {r.get('year', '')}"
    s = _end(s)
    if r.get("doi"):
        s += f" doi: {r['doi']}."
    return s.strip()


def format_citation(ref: dict, style: str = "apa7") -> str:
    style = style.lower().strip()
    fn = {"apa7": _apa7, "harvard": _harvard, "vancouver": _vancouver,
          "chicago": _chicago, "ieee": _ieee}.get(style)
    if fn is None:
        raise ValueError(f"Unknown style '{style}'. Choose from: {', '.join(STYLES)}")
    return fn(ref)


# ---------------------------------------------------------------- BibTeX
_BIBTYPE = {"Journal Article": "article", "Book": "book",
            "Book Section": "incollection", "Thesis": "phdthesis",
            "Conference Proceedings": "inproceedings",
            "Conference Paper": "inproceedings", "Report": "techreport",
            "Web Page": "misc", "Computer Program": "misc"}


def _ascii(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def _bib_escape(s: str) -> str:
    return s.replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")


def format_bibtex(ref: dict) -> str:
    fam = _split_name(ref["authors"][0])[0] if ref["authors"] else "anon"
    word = next((w for w in re.findall(r"[A-Za-z]{4,}", ref["title"])
                 if w.lower() not in {"with", "from", "that", "this", "their"}), "")
    key = re.sub(r"[^A-Za-z0-9]", "", _ascii(fam)).lower() + \
        re.sub(r"\D", "", ref.get("year") or "")[:4] + word.lower()
    fields = {
        "author": " and ".join(ref["authors"]),
        "editor": " and ".join(ref.get("editors") or []) if ref["ref_type"] == "Book Section" else "",
        "title": "{" + ref["title"] + "}",
        "journal": ref["journal"] if ref["ref_type"] == "Journal Article" else "",
        "booktitle": ref["journal"] if ref["ref_type"] in ("Book Section", "Conference Paper", "Conference Proceedings") else "",
        "year": ref.get("year", ""), "volume": ref.get("volume", ""),
        "number": ref.get("issue", ""),
        "pages": (ref.get("pages") or "").replace("–", "--").replace("-", "--").replace("----", "--"),
        "publisher": ref.get("publisher", ""), "address": ref.get("place_published", ""),
        "doi": ref.get("doi", ""), "url": ref.get("url", "") if not ref.get("doi") else "",
        "isbn": ref.get("isbn", ""),
        "note": f"EndNote record {ref['rec_number']}",
    }
    body = ",\n".join(f"  {k} = {{{_bib_escape(v) if k not in ('url', 'doi') else v}}}"
                      for k, v in fields.items() if v)
    return f"@{_BIBTYPE.get(ref['ref_type'], 'misc')}{{{key},\n{body}\n}}"
