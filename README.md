# endnote-local

Search, cite and read an EndNote library straight from EndNote's own SQLite
database, with no XML export and no manual indexing. A command-line tool
(`enlib`) and an MCP server (`enlib serve`, for Claude Desktop) share one
index.

Replaces the `endnote-mcp` connector (2026-10-15). Same tool names and record
numbers, plus semantic search over PDF passages. Possible improvements, such as
a better embedding model, are in `REFINEMENTS.md`.

## How it works

- **Reads EndNote read-only.** `X.Data/sdb/sdb.eni` holds the references and
  attachment paths; nothing in `X.Data/` is ever written.
- **Keeps a private index** at `~/Library/Application Support/endnote-local/index.db`:
  - an FTS5 index of the reference fields;
  - per-page PDF text (PyMuPDF), with an FTS5 index;
  - embeddings (`all-MiniLM-L6-v2`) of each reference (title + abstract +
    keywords, with the first PDF pages standing in for a missing abstract)
    and of ~900-character passages of every PDF.
- **Syncs itself.** Before answering, it updates whatever changed: references
  by a hash of their fields, PDFs by mtime + size, and embeddings by a hash of
  the text. With nothing to do this takes well under a second. The MCP server
  syncs at most once a minute.
- **First build:** about 4 minutes for text plus reference embeddings, and
  another 4 for passage embeddings (2,384 references, 2,330 PDFs, Apple
  Silicon).

EndNote's schema is undocumented. `library.check_schema()` stops with a clear
error if an EndNote update renames a field this tool reads. The type-code map
in `library.REF_TYPES` was derived empirically; unknown codes show as
`Type <n>`.

## CLI

```
enlib search [terms] [--author X --from 2015 --to 2023 --type "Book"]
enlib fulltext "olfactory bulb" width      # exact words in PDFs, page-numbered
enlib semantic "brain shape vs skull shape in carnivores"
enlib passages "ossified tentorium absent in skunks"
enlib library frontal cortex sociality     # all three combined
enlib related 2416
enlib show 2416                            # all fields + attachments
enlib read 2416 3-5 [-a 2]                 # PDF pages (attachment 2)
enlib cite 2416 2386 -s apa7               # apa7 harvard vancouver chicago ieee
enlib bibtex 2416
enlib sync | status | serve
```

Global flags: `--no-sync`, `--json`. Unquoted terms of 4 or more characters
match as prefixes (`pinniped` finds *Pinnipedia*), and quoted terms match
exactly. AND, OR and NOT are honoured.

## Install

```
conda create -n endnote-local python=3.12 pip
~/miniconda3/envs/endnote-local/bin/pip install -e ~/Projects/endnote-local
~/miniconda3/envs/endnote-local/bin/enlib sync
```

Claude Desktop (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
"endnote-local": {
  "command": "/Users/hherhold/miniconda3/envs/endnote-local/bin/enlib",
  "args": ["serve"]
}
```

Settings:

- `ENDNOTE_LIBRARY` overrides the library path (default
  `~/Documents/My EndNote Library.enl`).
- `ENDNOTE_LOCAL_INDEX` overrides the index directory.
- After a schema-version change, delete `index.db` and run `enlib sync`.
