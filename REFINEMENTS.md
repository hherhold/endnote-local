# Refinements — considered, not done

Possible improvements to endnote-local, with what was learned about each and
what doing it would take. Nothing here is required: the tool works as it
stands (2026-10-15). Shelved for now because semantic search is little used.

---

## 1. A better embedding model

### Why

Semantic search uses `all-MiniLM-L6-v2`, the model the old endnote-mcp
connector used. It works, but it is an older and weaker model:

- **It reads only 256 tokens.** For a reference, the embedded text is title +
  abstract + keywords, so many abstracts are cut off partway through.
- **Ranking is soft.** Test query: *"olfactory bulb size and social behaviour
  in seals"*. Loza et al. 2023 (pinniped endocasts; olfactory bulb size tied
  to otariid sociality) ranked **6th**, behind cochlea and hearing papers. A
  keyword search for `pinniped endocast` finds it first.

### Candidates

⚠️ **The figures are from recall, not checked against current model cards.**
"Retrieval" is the approximate MTEB retrieval score; published numbers differ
between benchmark versions. Verify before relying on them.

| model | params | dims | max input | retrieval | disk |
|---|---|---|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` (current) | 23M | 384 | 256 tokens | ~42 | ~90 MB |
| `sentence-transformers/all-mpnet-base-v2` | 110M | 768 | 384 | ~44 | ~420 MB |
| **`BAAI/bge-small-en-v1.5`** | 33M | 384 | 512 | ~52 | ~130 MB |
| `BAAI/bge-base-en-v1.5` | 110M | 768 | 512 | ~53 | ~440 MB |
| `BAAI/bge-large-en-v1.5` | 335M | 1024 | 512 | ~54 | ~1.3 GB |

- **The first choice is `bge-small-en-v1.5`.** It is barely larger than MiniLM
  but a newer generation: about +10 on retrieval, against only +2 more for
  going up to bge-large. It has the same 384 dimensions, so the index stays
  about the same size, and query speed is about unchanged.
- **`bge-base` is probably not worth it.** It is roughly 4–5× the compute and
  doubles the vector storage, for about +1 over bge-small.
- **Larger, newer models** (e.g. Qwen3-Embedding-0.6B, ~600M parameters)
  score higher again but would slow both the first build and every query
  noticeably. There are probably newer ones than any listed here.
- **For `find_related` only, consider SPECTER2** (`allenai/specter2_base`).
  It is trained on citation links between papers, which is exactly what
  "related papers" means; it is not meant for free-text queries.

### Cost on this library

Measured with the current model on an Apple Silicon Mac: 2,384 references,
2,330 PDFs, 49,607 pages.

- **Reference embeddings:** under a minute.
- **Passage embeddings:** about 4 minutes; 196 MB of float16 vectors.
- **Index size:** 484 MB in total.
- **Semantic query from the CLI:** about 2.5 s, almost all of it loading the
  model. The MCP server keeps the model loaded, so queries there are faster.

Estimates:

- **bge-small:** about 1.5–2× the embedding time, i.e. about 10 minutes once;
  storage unchanged.
- **bge-base:** about 30–40 minutes once; passage vectors about 400 MB; 1–2 s
  more per CLI query.

### Test before switching

Benchmarks say nothing about carnivoran neuroanatomy, so measure on this
library:

1. Write about 10 queries whose correct answer is known. Phrase them in
   different words from the target's title, since that is the case semantic
   search exists for. Examples:
   - olfactory bulbs and sociality in seals → Loza et al. 2023
   - surface semilandmark placement → Bardua et al. 2019
   - ossified tentorium variation → Selles de Lucas et al. 2018
2. Build a second index with the candidate model, keeping the current one
   (`ENDNOTE_LOCAL_INDEX=<scratch dir>`, with `MODEL_NAME` changed in that
   run).
3. Record the rank of each known answer under both models, for
   `search_semantic` and `search_passages`. Switch only if the new model
   clearly wins.

### How to switch

**Changing `MODEL_NAME` alone is NOT enough, and fails silently.** Stored
vectors are keyed by a hash of the *text* embedded (references) or by the
file's mtime+size (passages), not by the model. A model change therefore
re-embeds nothing, and queries would compare new-model query vectors with
old-model document vectors. The result is meaningless rankings, and nothing
reports an error.

1. `index.py`: set `MODEL_NAME`.
2. **If the dimension changes** (anything other than 384), replace the
   literal `384` in `index.py`. It currently appears at four places: the
   empty-array fallbacks and the passage `reshape`. Better: read it from
   `model().get_sentence_embedding_dimension()`, or from the stored blob
   length.
3. **Invalidate the vectors.** Either bump `SCHEMA_VERSION` (the tool then
   refuses to open the old index and asks for it to be deleted), or run
   `DELETE FROM ref_vecs; DELETE FROM passage_vecs;` on `index.db` and
   re-sync. The robust fix is to record the model name in `meta` and clear
   both vector tables automatically when it differs. Do that as part of the
   switch.
4. **bge query prefix.** bge models expect search queries (not documents) to
   be prefixed with `Represent this sentence for searching relevant passages: `.
   v1.5 is said to work tolerably without it, but test with it. This means
   `embed()` needs a query/document distinction.
5. Run `enlib sync`, then the test queries above.

---

## 2. Smaller items

- **Passage chunks are cut by character count** (900 characters, 200 overlap),
  not at sentence or paragraph boundaries. Display trims the cut words.
  Chunking at sentences would give cleaner passages and slightly better
  embeddings. Changing it needs the passage vectors rebuilt (step 3 above).
- **Prefix matching adds some noise.** Unquoted terms of 4+ characters match
  as prefixes (so `pinniped` finds *Pinnipedia*, since FTS5 does no
  stemming), so `cran` would also match *crane*. Quote a term to match it
  exactly. A Porter-stemming tokenizer is the alternative; it needs the FTS
  tables rebuilt.
- **The citation formatter is approximate, not a CSL engine.** Known gaps:
  - Hyphenated given names give `J-L.` rather than APA's `J.-L.`.
  - Editors and edition are handled only for APA book sections.
  - No sentence-casing of titles.

  For publication-grade references, format them in EndNote itself.
- **Not registered as an MCP server in Claude Code.** Claude Code uses the CLI
  through Bash, which needs no connector. If MCP tools are wanted there too:
  `claude mcp add --scope user endnote-local -- ~/miniconda3/envs/endnote-local/bin/enlib serve`.
- **Not a git repository yet.**

## 3. EndNote data issues found along the way (fix in EndNote, not here)

- **Deformetrica 4** (Bône, Louis, Martin & Durrleman): all four authors are in
  one author field, and the year reads 2025. Frosali et al. 2023 and 2026 both
  cite it as **2018**, in an edited volume (*Shape in Medical Imaging*). The
  tool works around the author field: a single author string with 2+ commas
  is split into names.
- **Some ISBN fields hold ISSNs**, e.g. Loza et al. 2023 shows
  `1064-75541573-7055`. This is harmless, but it shows up in BibTeX as `isbn`.
- **Record numbers are local to one installation.** EndNote renumbers when a
  library is synced down to a new machine, so cite by DOI in any file and use
  `[n]` only within a session.
