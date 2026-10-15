"""The sidecar index: a private SQLite file kept in step with the EndNote library.

Nothing here writes to EndNote's files. `sync()` reads them, compares against
what it indexed last time, and updates only what changed:

    references   by a hash of their fields (edits, additions, trash, deletions)
    PDF pages    by attachment file mtime + size, extracted with PyMuPDF
    embeddings   by a hash of the exact text embedded

A sync with nothing to do costs well under a second, so the CLI and the MCP
server run one before answering. The embedding model is loaded only when
something actually needs embedding, or a semantic query arrives.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

from . import library as lib

INDEX_DIR = Path(os.environ.get(
    "ENDNOTE_LOCAL_INDEX",
    Path.home() / "Library" / "Application Support" / "endnote-local"))
SCHEMA_VERSION = "1"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_PAGES = 1500            # per attachment; a cap for pathological files
CHUNK_CHARS = 900           # passage size for semantic full-text search
CHUNK_STEP = 700            # i.e. 200 characters of overlap
NO_ABSTRACT_CHARS = 1500    # PDF text stands in for a missing abstract

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS refs (
    rec_number INTEGER PRIMARY KEY, stamp TEXT, year TEXT, ref_type TEXT,
    authors_text TEXT, data TEXT);
CREATE VIRTUAL TABLE IF NOT EXISTS refs_fts USING fts5(
    title, authors, abstract, keywords, journal, notes,
    tokenize = 'unicode61 remove_diacritics 2');
CREATE TABLE IF NOT EXISTS files (
    rec_number INTEGER, path TEXT, pos INTEGER, fkey TEXT, n_pages INTEGER,
    source TEXT, PRIMARY KEY (rec_number, path));
CREATE TABLE IF NOT EXISTS pages (
    id INTEGER PRIMARY KEY, rec_number INTEGER, path TEXT, page INTEGER,
    text TEXT);
CREATE INDEX IF NOT EXISTS pages_ref ON pages (rec_number, path, page);
CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts USING fts5(
    text, content = 'pages', content_rowid = 'id',
    tokenize = 'unicode61 remove_diacritics 2');
CREATE TRIGGER IF NOT EXISTS pages_ai AFTER INSERT ON pages BEGIN
    INSERT INTO pages_fts (rowid, text) VALUES (new.id, new.text); END;
CREATE TRIGGER IF NOT EXISTS pages_ad AFTER DELETE ON pages BEGIN
    INSERT INTO pages_fts (pages_fts, rowid, text)
    VALUES ('delete', old.id, old.text); END;
CREATE TABLE IF NOT EXISTS ref_vecs (
    rec_number INTEGER PRIMARY KEY, stamp TEXT, vec BLOB);
CREATE TABLE IF NOT EXISTS passage_vecs (
    rec_number INTEGER, path TEXT, stamp TEXT, spans TEXT, vecs BLOB,
    PRIMARY KEY (rec_number, path));
"""


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _sha(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode()).hexdigest()


# --------------------------------------------------------------------------
# Embedding model (lazy)
# --------------------------------------------------------------------------
_model = None


def model():
    global _model
    if _model is None:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        # huggingface_hub reads HF_HUB_OFFLINE once, at import: decide before
        # importing. Offline once the model is cached, so a query never waits
        # on (or fails for want of) the network.
        hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
        if (hub / ("models--" + MODEL_NAME.replace("/", "--"))).exists():
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        from sentence_transformers import SentenceTransformer
        from transformers.utils import logging as tf_logging
        tf_logging.disable_progress_bar()
        tf_logging.set_verbosity_error()
        import logging
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed(texts: list[str], batch_size: int = 64) -> np.ndarray:
    return model().encode(texts, batch_size=batch_size,
                          normalize_embeddings=True, show_progress_bar=False,
                          convert_to_numpy=True).astype(np.float32)


# --------------------------------------------------------------------------
# The index
# --------------------------------------------------------------------------
class Index:
    def __init__(self, enl: str | None = None, index_dir: Path | None = None):
        self.paths = lib.library_paths(enl)
        self.dir = Path(index_dir or INDEX_DIR)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.dir / "index.db", timeout=60,
                                  check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode = WAL")
        self.db.executescript(_SCHEMA)
        v = self.db.execute(
            "SELECT value FROM meta WHERE key = 'schema'").fetchone()
        if v is None:
            self._set_meta("schema", SCHEMA_VERSION)
            self._set_meta("library", str(self.paths.enl))
        elif v[0] != SCHEMA_VERSION:
            raise RuntimeError(
                f"Index at {self.dir} was built by another version; delete "
                f"{self.dir / 'index.db'} and run `enlib sync`.")
        self._passages = None   # (vecs, keys) cache for passage search
        self._refvecs = None
        self.last_sync = 0.0

    def _set_meta(self, key, value):
        self.db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)",
                        (key, str(value)))
        self.db.commit()

    def meta(self, key, default=None):
        r = self.db.execute("SELECT value FROM meta WHERE key = ?",
                            (key,)).fetchone()
        return r[0] if r else default

    # ------------------------------------------------------------------ sync
    @contextlib.contextmanager
    def _lock(self, wait: bool):
        f = open(self.dir / "sync.lock", "w")
        try:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
            except BlockingIOError:
                yield False
                return
            yield True
        finally:
            f.close()

    def sync(self, *, passages: bool = True, quiet: bool = False,
             wait: bool = False) -> dict:
        """Bring the index in step with EndNote. Returns counts of changes.

        If another process is already syncing, returns immediately (unless
        wait=True) and queries see the index as it stands.
        """
        say = (lambda m: None) if quiet else log
        with self._lock(wait) as got:
            if not got:
                return {"skipped": "another sync in progress"}
            t0 = time.time()
            with contextlib.closing(lib.connect_ro(self.paths.sdb)) as en:
                lib.check_schema(en)
                refs = lib.read_references(en)
                atts = lib.read_attachments(en)
            stats = self._sync_refs(refs, say)
            stats.update(self._sync_files(atts, say))
            stats["ref_embeddings"] = self._sync_ref_vecs(say)
            if passages:
                stats["passage_embeddings"] = self._sync_passage_vecs(say)
            if any(v for k, v in stats.items()):
                self._passages = self._refvecs = None
            self._set_meta("last_sync", time.strftime("%Y-%m-%d %H:%M:%S"))
            self.last_sync = time.time()
            stats["seconds"] = round(time.time() - t0, 1)
            return stats

    def _sync_refs(self, refs, say) -> dict:
        old = dict(self.db.execute("SELECT rec_number, stamp FROM refs"))
        new_ids = set()
        changed = 0
        for r in refs:
            rn = r["rec_number"]
            new_ids.add(rn)
            stamp = _sha(r)
            if old.get(rn) == stamp:
                continue
            changed += 1
            self._delete_ref_rows(rn, files=False)
            self.db.execute(
                "INSERT INTO refs VALUES (?, ?, ?, ?, ?, ?)",
                (rn, stamp, r["year"], r["ref_type"],
                 "; ".join(r["authors"]), json.dumps(r)))
            self.db.execute(
                "INSERT INTO refs_fts (rowid, title, authors, abstract, "
                "keywords, journal, notes) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (rn, r["title"], "; ".join(r["authors"]), r["abstract"],
                 "; ".join(r["keywords"]), r["journal"],
                 " ".join(filter(None, [r["notes"], r["research_notes"]]))))
        gone = set(old) - new_ids
        for rn in gone:
            self._delete_ref_rows(rn, files=True)
            self.db.execute("DELETE FROM refs_fts WHERE rowid = ?", (rn,))
            self.db.execute("DELETE FROM ref_vecs WHERE rec_number = ?", (rn,))
        self.db.commit()
        if changed or gone:
            say(f"references: {changed} new/changed, {len(gone)} removed")
        return {"references_changed": changed, "references_removed": len(gone)}

    def _delete_ref_rows(self, rn, files: bool):
        self.db.execute("DELETE FROM refs WHERE rec_number = ?", (rn,))
        self.db.execute("DELETE FROM refs_fts WHERE rowid = ?", (rn,))
        if files:
            self.db.execute("DELETE FROM pages WHERE rec_number = ?", (rn,))
            self.db.execute("DELETE FROM files WHERE rec_number = ?", (rn,))
            self.db.execute("DELETE FROM passage_vecs WHERE rec_number = ?",
                            (rn,))

    def _sync_files(self, atts, say) -> dict:
        import pymupdf
        pymupdf.TOOLS.mupdf_display_errors(False)
        old = {(r["rec_number"], r["path"]): r["fkey"]
               for r in self.db.execute("SELECT * FROM files")}
        want = {}
        for rn, paths in atts.items():
            for pos, p in enumerate(paths):
                want[(rn, p)] = pos
        todo = []
        for (rn, p), pos in want.items():
            full = self.paths.pdf_dir / p
            try:
                st = full.stat()
                fkey = f"{st.st_mtime_ns}:{st.st_size}"
            except FileNotFoundError:
                fkey = "missing"
            if old.get((rn, p)) != fkey:
                todo.append((rn, p, pos, fkey))
            else:
                self.db.execute(
                    "UPDATE files SET pos = ? WHERE rec_number = ? AND path = ?",
                    (pos, rn, p))
        gone = set(old) - set(want)
        for rn, p in gone:
            self._drop_file(rn, p)
        if todo:
            say(f"extracting text from {len(todo)} attachment(s)...")
        for i, (rn, p, pos, fkey) in enumerate(todo, 1):
            self._drop_file(rn, p)
            pages, source = self._extract(pymupdf, rn, p, fkey)
            for n, text in pages:
                self.db.execute(
                    "INSERT INTO pages (rec_number, path, page, text) "
                    "VALUES (?, ?, ?, ?)", (rn, p, n, text))
            self.db.execute("INSERT INTO files VALUES (?, ?, ?, ?, ?, ?)",
                            (rn, p, pos, fkey, len(pages), source))
            if i % 50 == 0 or i == len(todo):
                self.db.commit()
                say(f"  {i}/{len(todo)}")
        self.db.commit()
        return {"attachments_indexed": len(todo), "attachments_removed": len(gone)}

    def _drop_file(self, rn, p):
        self.db.execute("DELETE FROM pages WHERE rec_number = ? AND path = ?",
                        (rn, p))
        self.db.execute("DELETE FROM files WHERE rec_number = ? AND path = ?",
                        (rn, p))
        self.db.execute(
            "DELETE FROM passage_vecs WHERE rec_number = ? AND path = ?", (rn, p))

    def _extract(self, pymupdf, rn, p, fkey):
        """[(page_number, text)], source. Page numbers are 1-based PDF pages."""
        if fkey == "missing":
            return [], "missing"
        full = self.paths.pdf_dir / p
        pages = []
        if full.suffix.lower() == ".pdf":
            try:
                with pymupdf.open(full) as doc:
                    for i, page in enumerate(doc):
                        if i >= MAX_PAGES:
                            break
                        t = page.get_text().strip()
                        if t:
                            pages.append((i + 1, t))
            except Exception:
                pages = []
        if pages:
            return pages, "pymupdf"
        text = lib.endnote_pdf_text(self.paths.pdb, rn, p).strip()
        if text:   # no page breaks in EndNote's copy: one unnumbered page
            return [(0, text)], "endnote"
        return [], "no text"

    # ----------------------------------------------------------- embeddings
    def _ref_embed_text(self, r: dict) -> str:
        parts = [r["title"], r["abstract"]]
        if r["keywords"]:
            parts.append("Keywords: " + ", ".join(r["keywords"]))
        if not r["abstract"]:
            row = self.db.execute(
                "SELECT p.text FROM pages p JOIN files f ON f.rec_number = "
                "p.rec_number AND f.path = p.path WHERE p.rec_number = ? "
                "ORDER BY f.pos, p.page LIMIT 2", (r["rec_number"],)).fetchall()
            parts.append(" ".join(x[0] for x in row)[:NO_ABSTRACT_CHARS])
        return " ".join(p for p in parts if p)

    def _sync_ref_vecs(self, say) -> int:
        old = dict(self.db.execute("SELECT rec_number, stamp FROM ref_vecs"))
        todo = []
        for row in self.db.execute("SELECT rec_number, data FROM refs"):
            text = self._ref_embed_text(json.loads(row["data"]))
            stamp = hashlib.sha1(text.encode()).hexdigest()
            if old.get(row["rec_number"]) != stamp:
                todo.append((row["rec_number"], stamp, text))
        if not todo:
            return 0
        say(f"embedding {len(todo)} reference(s)...")
        for i in range(0, len(todo), 512):
            batch = todo[i:i + 512]
            vecs = embed([t for _, _, t in batch])
            self.db.executemany(
                "INSERT OR REPLACE INTO ref_vecs VALUES (?, ?, ?)",
                [(rn, st, v.tobytes()) for (rn, st, _), v in zip(batch, vecs)])
            self.db.commit()
        return len(todo)

    def _sync_passage_vecs(self, say) -> int:
        old = {(r[0], r[1]): r[2] for r in self.db.execute(
            "SELECT rec_number, path, stamp FROM passage_vecs")}
        todo = [(r["rec_number"], r["path"], r["fkey"]) for r in
                self.db.execute("SELECT rec_number, path, fkey FROM files "
                                "WHERE n_pages > 0")
                if old.get((r["rec_number"], r["path"])) != r["fkey"]]
        if not todo:
            return 0
        say(f"embedding passages of {len(todo)} attachment(s)...")
        t0 = time.time()
        for i, (rn, p, fkey) in enumerate(todo, 1):
            spans, texts = [], []
            for row in self.db.execute(
                    "SELECT page, text FROM pages WHERE rec_number = ? AND "
                    "path = ? ORDER BY page", (rn, p)):
                for start, chunk in chunk_text(row["text"]):
                    spans.append((row["page"], start, start + len(chunk)))
                    texts.append(chunk)
            vecs = embed(texts, batch_size=128) if texts else \
                np.zeros((0, 384), np.float32)
            self.db.execute(
                "INSERT OR REPLACE INTO passage_vecs VALUES (?, ?, ?, ?, ?)",
                (rn, p, fkey, json.dumps(spans),
                 vecs.astype(np.float16).tobytes()))
            if i % 25 == 0 or i == len(todo):
                self.db.commit()
                rate = i / (time.time() - t0)
                say(f"  {i}/{len(todo)}  (~{(len(todo) - i) / rate / 60:.0f} "
                    f"min left)")
        self.db.commit()
        return len(todo)

    # -------------------------------------------------------------- lookup
    def ref(self, rn: int) -> dict | None:
        row = self.db.execute("SELECT data FROM refs WHERE rec_number = ?",
                              (rn,)).fetchone()
        return json.loads(row[0]) if row else None

    def attachments(self, rn: int) -> list[dict]:
        return [dict(r) | {"full_path": str(self.paths.pdf_dir / r["path"])}
                for r in self.db.execute(
                    "SELECT path, pos, n_pages, source FROM files WHERE "
                    "rec_number = ? ORDER BY pos", (rn,))]

    def pages(self, rn: int, attachment: int = 1, start: int = 1,
              end: int = 5) -> tuple[dict | None, list[tuple[int, str]], int]:
        atts = self.attachments(rn)
        if not 1 <= attachment <= len(atts):
            return None, [], 0
        a = atts[attachment - 1]
        rows = self.db.execute(
            "SELECT page, text FROM pages WHERE rec_number = ? AND path = ? "
            "AND (page BETWEEN ? AND ? OR page = 0) ORDER BY page",
            (rn, a["path"], start, end)).fetchall()
        total = self.db.execute(
            "SELECT MAX(page) FROM pages WHERE rec_number = ? AND path = ?",
            (rn, a["path"])).fetchone()[0] or 0
        return a, [(r[0], r[1]) for r in rows], total

    def stats(self) -> dict:
        q = lambda s: self.db.execute(s).fetchone()[0]
        return {
            "library": str(self.paths.enl),
            "index": str(self.dir / "index.db"),
            "references": q("SELECT COUNT(*) FROM refs"),
            "with_abstract": q("SELECT COUNT(*) FROM refs WHERE "
                               "json_extract(data, '$.abstract') <> ''"),
            "attachments": q("SELECT COUNT(*) FROM files"),
            "attachments_with_text": q("SELECT COUNT(*) FROM files WHERE n_pages > 0"),
            "attachments_endnote_text_only": q(
                "SELECT COUNT(*) FROM files WHERE source = 'endnote'"),
            "attachments_missing": q("SELECT COUNT(*) FROM files WHERE source = 'missing'"),
            "pages": q("SELECT COUNT(*) FROM pages"),
            "reference_embeddings": q("SELECT COUNT(*) FROM ref_vecs"),
            "passage_embeddings_files": q("SELECT COUNT(*) FROM passage_vecs"),
            "last_sync": self.meta("last_sync"),
        }

    # -------------------------------------------------------------- search
    def _filters(self, year_from, year_to, author, ref_type):
        where, args = [], []
        if year_from:
            where.append("r.year >= ?"); args.append(str(year_from))
        if year_to:
            where.append("r.year <= ?"); args.append(str(year_to))
        if author:
            where.append("r.authors_text LIKE ?"); args.append(f"%{author}%")
        if ref_type:
            where.append("r.ref_type = ? COLLATE NOCASE"); args.append(ref_type)
        return where, args

    def search_references(self, query, year_from=None, year_to=None,
                          author=None, ref_type=None, limit=50) -> list[dict]:
        where, args = self._filters(year_from, year_to, author, ref_type)
        queries = fts_queries(query)
        if not queries:   # filters only: list matches, newest first
            if not where:
                return []
            rows = self.db.execute(
                "SELECT r.rec_number FROM refs r WHERE " + " AND ".join(where)
                + " ORDER BY r.year DESC, r.authors_text LIMIT ?",
                [*args, limit]).fetchall()
            return [self.ref(r[0]) for r in rows]
        for q in queries:
            sql = ("SELECT r.rec_number, bm25(refs_fts, 10, 5, 3, 5, 1, 1) "
                   "AS score FROM refs_fts JOIN refs r ON r.rec_number = "
                   "refs_fts.rowid WHERE refs_fts MATCH ?"
                   + "".join(f" AND {w}" for w in where)
                   + " ORDER BY score LIMIT ?")
            rows = self.db.execute(sql, [q, *args, limit]).fetchall()
            if rows:
                return [self.ref(r[0]) for r in rows]
        return []

    def search_fulltext(self, query, limit=50, snippets=3, year_from=None,
                        year_to=None, author=None, ref_type=None) -> list[dict]:
        where, args = self._filters(year_from, year_to, author, ref_type)
        for q in fts_queries(query, allow_or=False):
            sql = ("SELECT p.rec_number, p.path, p.page, bm25(pages_fts) AS "
                   "score, snippet(pages_fts, 0, '«', '»', ' … ', 24) AS snip "
                   "FROM pages_fts JOIN pages p ON p.id = pages_fts.rowid "
                   "JOIN refs r ON r.rec_number = p.rec_number "
                   "WHERE pages_fts MATCH ?"
                   + "".join(f" AND {w}" for w in where)
                   + " ORDER BY score LIMIT ?")
            rows = self.db.execute(sql, [q, *args, limit * 20]).fetchall()
            if rows:
                break
        else:
            return []
        out: dict[int, dict] = {}
        for r in rows:
            hit = out.get(r["rec_number"])
            if hit is None:
                if len(out) >= limit:
                    continue
                hit = out[r["rec_number"]] = self.ref(r["rec_number"]) | {
                    "snippets": []}
            if len(hit["snippets"]) < snippets:
                hit["snippets"].append({"page": r["page"], "file": r["path"],
                                        "snippet": clean_snippet(r["snip"])})
        return list(out.values())

    def _ref_matrix(self):
        if self._refvecs is None:
            rows = self.db.execute("SELECT rec_number, vec FROM ref_vecs").fetchall()
            ids = np.array([r[0] for r in rows])
            mat = (np.stack([np.frombuffer(r[1], np.float32) for r in rows])
                   if rows else np.zeros((0, 384), np.float32))
            self._refvecs = (ids, mat)
        return self._refvecs

    def search_semantic(self, query: str | None = None, limit: int = 20,
                        vec: np.ndarray | None = None, exclude=(),
                        min_sim: float = 0.1) -> list[dict]:
        ids, mat = self._ref_matrix()
        if not len(ids):
            return []
        v = embed([query])[0] if vec is None else vec
        sims = mat @ v
        out = []
        for i in np.argsort(-sims):
            if sims[i] < min_sim or len(out) >= limit:
                break
            if int(ids[i]) in exclude:
                continue
            out.append(self.ref(int(ids[i])) | {"similarity": float(sims[i])})
        return out

    def find_related(self, rn: int, limit: int = 10) -> list[dict]:
        row = self.db.execute("SELECT vec FROM ref_vecs WHERE rec_number = ?",
                              (rn,)).fetchone()
        if row is None:
            return []
        return self.search_semantic(vec=np.frombuffer(row[0], np.float32),
                                    limit=limit, exclude={rn})

    def _passage_matrix(self):
        if self._passages is None:
            keys, mats = [], []
            for r in self.db.execute(
                    "SELECT rec_number, path, spans, vecs FROM passage_vecs"):
                spans = json.loads(r["spans"])
                if not spans:
                    continue
                mats.append(np.frombuffer(r["vecs"], np.float16).reshape(-1, 384))
                keys.extend((r["rec_number"], r["path"], *s) for s in spans)
            mat = np.concatenate(mats) if mats else np.zeros((0, 384), np.float16)
            self._passages = (keys, mat)
        return self._passages

    def search_passages(self, query: str, limit: int = 10,
                        per_reference: int = 2) -> list[dict]:
        keys, mat = self._passage_matrix()
        if not keys:
            return []
        v = embed([query])[0].astype(np.float16)
        sims = (mat @ v).astype(np.float32)
        top = np.argpartition(-sims, min(len(sims) - 1, limit * 50))[:limit * 50]
        out: dict[int, dict] = {}
        for i in top[np.argsort(-sims[top])]:
            rn, path, page, a, b = keys[i]
            hit = out.get(rn)
            if hit is None:
                if len(out) >= limit:
                    continue
                hit = out[rn] = self.ref(rn) | {"passages": []}
            if len(hit["passages"]) >= per_reference:
                continue
            text = self.db.execute(
                "SELECT text FROM pages WHERE rec_number = ? AND path = ? AND "
                "page = ?", (rn, path, page)).fetchone()[0]
            words = text[a:b].split()
            if a > 0 and not text[a - 1].isspace():
                words = words[1:]          # drop a word cut by the chunk edge
            if b < len(text) and not text[b].isspace():
                words = words[:-1]
            hit["passages"].append({"page": page, "file": path,
                                    "similarity": float(sims[i]),
                                    "text": " ".join(words)})
        return list(out.values())

    def search_library(self, query, year_from=None, year_to=None, author=None,
                       ref_type=None, limit=30) -> list[dict]:
        """Reciprocal-rank fusion of metadata, full-text and semantic search."""
        kw = dict(year_from=year_from, year_to=year_to, author=author,
                  ref_type=ref_type)
        lists = [self.search_references(query, limit=100, **kw),
                 self.search_fulltext(query, limit=100, snippets=2, **kw)]
        allowed = {r["rec_number"] for r in self.search_references(
            "", limit=1_000_000, **kw)} if any(kw.values()) else None
        sem = self.search_semantic(query, limit=100)
        lists.append([r for r in sem if allowed is None
                      or r["rec_number"] in allowed])
        score: dict[int, float] = {}
        best: dict[int, dict] = {}
        for lst in lists:
            for rank, r in enumerate(lst):
                rn = r["rec_number"]
                score[rn] = score.get(rn, 0) + 1 / (60 + rank)
                best[rn] = best.get(rn, {}) | r
        order = sorted(score, key=score.get, reverse=True)[:limit]
        return [best[rn] for rn in order]


# --------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------
def chunk_text(text: str):
    text = re.sub(r"[ \t]+", " ", text)
    if len(text) <= CHUNK_CHARS:
        if text.strip():
            yield 0, text
        return
    for start in range(0, len(text), CHUNK_STEP):
        chunk = text[start:start + CHUNK_CHARS]
        if len(chunk.strip()) > 50:
            yield start, chunk
        if start + CHUNK_CHARS >= len(text):
            break


def clean_snippet(s: str) -> str:
    return " ".join(s.split())


_TOKEN = re.compile(r'"[^"]*"|\S+')


def fts_queries(query: str, allow_or: bool = True) -> list[str]:
    """User text -> FTS5 query strings to try in order.

    Punctuation is safe; unquoted terms of 4+ characters match as prefixes;
    "quoted phrases" match exactly; trailing * and the uppercase operators
    AND, OR, NOT are honoured.
    Without explicit operators, all terms must match; if that finds nothing,
    the reference search retries with any term matching.

    A query with no searchable terms gives no FTS queries; callers handle it.
    """
    parts, explicit = [], False
    for tok in _TOKEN.findall(query or ""):
        if tok in ("AND", "OR", "NOT"):
            parts.append(tok); explicit = True
            continue
        quoted = tok.startswith('"')
        inner = tok.strip('"').rstrip("*")
        if not re.search(r"\w", inner):
            continue
        # Unquoted terms of 4+ characters match as prefixes, so "pinniped"
        # finds Pinnipedia and "endocast" finds endocasts (FTS5 does no
        # stemming). Quote a term to match it exactly.
        star = tok.endswith("*") or (not quoted and len(inner) >= 4)
        parts.append('"' + inner.replace('"', '""') + '"' + ("*" if star else ""))
    while parts and parts[0] in ("AND", "OR", "NOT"):
        parts.pop(0)
    while parts and parts[-1] in ("AND", "OR", "NOT"):
        parts.pop()
    if not parts:
        return []
    first = " ".join(parts)
    if explicit or not allow_or or len(parts) == 1:
        return [first]
    return [first, " OR ".join(parts)]
