# RecSys Papers

Open research hub with paper summaries about recommender systems, Semantic IDs,
generative retrieval, and generative recommendation. A separate conference archive
collects recommender-systems papers from the venues tracked by recsys.info.

GitHub Pages: [https://pechatov.github.io/recsys-papers/](https://pechatov.github.io/recsys-papers/)

The repository is being migrated from a local static HTML archive to a GitHub
Pages site. The current migration keeps all existing `summaries/` pages working
as legacy static content while a new Astro shell is introduced around them.

## Local Development

Requirements:

- Node.js 22 or newer
- npm
- Python 3

```bash
npm install
npm run dev
```

The build mirrors `summaries/` into `public/summaries/` before Astro starts, so
existing HTML pages, images, MathJax formulas, reading-status controls, and
scroll gutters remain available.

## Editing Summaries

Detailed paper summaries are Markdown source files in
`src/content/paper_summaries/`. Astro renders them to the legacy public URLs
under `summaries/paper_summaries/.../*.html`, so old catalog links continue to
work while summaries remain easy to edit.

Figures live in `summaries/assets/`. Keep every figure for one paper under a
single clearly named asset directory, for example `summaries/assets/cost/` or
`summaries/assets/vq4rec/`, and reference them from Markdown with the same
relative paths used by the generated HTML pages.

The separate CIKM 2026 collection is published at `cikm-2026.html`, with review
pages under `cikm-2026/c26-XXX.html`. Its 99 approved papers are recorded in
`docs/cikm-2026-selection.json` (`scope: main`) and
`src/data/cikm-2026-papers.json`. Reviews live in `src/content/cikm_2026/` and
include four questions for the authors. All 99 reviews are based on full texts;
`sourceUrl` identifies the source actually read, and `sourceLabel` distinguishes
publisher PDFs from preprint versions. Optional shortlist entries are excluded.
New conference reviews do not enter the main paper catalog. The conference page
shows all 99 reviews; its source filter is hidden when every review uses full text.
The visit guide groups papers by day and editorial priority, with topic, day,
rating and presentation-format filters. Each paper shows its rationale and three
1–5 scores (interest, evidence, importance). Ratings are in
`src/data/cikm-2026-ratings.json`; the explicit thresholds are in
`src/data/cikm-2026-rating-rubric.json`. They guide visits within this shortlist
and are not objective publication-quality judgments.

`src/data/cikm-2026-schedule.json` records the official appearances, rooms,
tracks and poster boards in `Europe/Rome`. The program lists 44 oral papers and
55 poster-only papers; a poster is not assumed for an oral paper when none is
listed. Oral intervals are session times, never inferred individual talk times.
To refresh from a downloaded official program snapshot, run
`python3 scripts/extract_cikm_2026_schedule.py --program path/to/program.html`.

Run `npm run check:cikm` to verify selection coverage, review identity, source
metadata, author questions, rating thresholds and presentation schedules. After building, run
`python3 scripts/validate_cikm_2026.py --built` to check the conference pages and
their local links. For a build using a path prefix, pass `--base=/recsys-papers/`.

`docs/cikm-2026-fulltext-access.csv` lists the source actually read for each
review. `docs/cikm-2026-missing-fulltexts.csv` is now header-only: all selected
papers have a full-text review. `docs/cikm-2026-browser-acquisition.json` records
the 38 publisher PDFs acquired after the browser fix, including page counts,
identity checks and SHA256 hashes; downloaded PDFs remain outside Git.
The alternative-source search audit in
`docs/cikm-2026-acquisition-audit.json` records per-paper queries, author archives,
API results, and rejected matches. Access errors and missing API records are
recorded separately from the existence of the paper.
The full-text content review is recorded in `docs/cikm-2026-review-audit.json`,
including source versions, checked claims, applied corrections, and review hashes.

## Catalog Metadata

Every paper in `summaries/papers.html` must have two ISO 8601 dates:

- `Дата публикации` is the full publication date (`YYYY-MM-DD`) of the
  canonical version. Prefer proceedings or journal metadata for an accepted
  paper even when the catalog links to arXiv; for a preprint-only paper, use
  the initial arXiv submission date.
- `Дата добавления` is the date of the first repository commit containing the
  paper. Keep this date unchanged when a title, link, or summary is updated;
  for a newly added paper, use the date it is added to the repository.

Run `npm run check:catalog` to validate both fields and reject the legacy
`Год` and `Дата arXiv` fields.

## Updating the Conference Archive

The archive is published at `conferences.html` and covers 17 venues from the
recsys.info tracker for 2024–2026. The small manifest lives in
`src/data/conference-archive.json`; paper metadata is split by venue/year under
`public/data/conference-papers/` and loaded on demand. This keeps search and
pagination responsive: 7,705 selected papers retain provenance for 55,012
source records.

RecSys, CIKM, KDD, SIGIR, ECIR, and UMAP are treated as profile venues and kept
in full. For AAAI, CHI, ECAI, ECML/PKDD, ICDE, ICDM, ICML, IJCAI, NeurIPS, WSDM,
and WWW, the importer keeps papers only when the title or a dedicated track name
contains explicit recommender-systems evidence. The manifest records both the
source count and the selected count for every edition.

ACM RecSys records come from its official Accepted Contributions pages. ICDE,
ICML, IJCAI-ECAI, and SIGIR 2026 use their official accepted-paper catalogs;
published editions for the other venues use DBLP's structured proceedings
tables of contents. The importer preserves each source's section/category split
and links to canonical DOI, publisher, OpenReview, PMLR, or official program
pages. Main and official companion/adjunct volumes are included; standalone
workshop proceedings are not. Missing 2026 lists are shown as pending, and
their `expectedPublicationDate` is a conservative estimate based on the official
conference start date. Incomplete published proceedings are marked partial rather
than supplemented from unverified lists.

To refresh the data after an official list changes:

```bash
python3 -m pip install beautifulsoup4 requests
npm run update:conferences
```

The updater downloads the official accepted-paper exports where configured,
refreshes cached structured proceedings metadata, and attaches abstracts after
an exact DOI or verified title match. It checks official proceedings tables of
contents first, then DOI-indexed metadata. For remaining papers it searches
arXiv in rate-limited batches, requires an exact title plus overlapping author,
and verifies the explicit `Abstract` section in an open HTML/PDF copy when one
is available. The coordinate-aware PDF fallback uses `pdftotext` from Poppler
when installed; without it, the verified structured arXiv abstract remains
available. No paywall is bypassed.

Abstract lookup results are cached under
`.cache/conference-archive/abstracts.json`; the published per-edition JSON files
contain the abstract text. Papers whose abstract or full text has not yet
appeared in a verifiable source retain an explicit `null` value instead of
generated copy. Setting `OPENALEX_API_KEY` is optional but enables faster
batched DOI lookups. `python3 scripts/update_conference_abstracts.py --help`
lists switches for skipping or refreshing the full-text stages.

## Build

```bash
npm run check
npm run build
```

The static output is written to `dist/`.

## Deploy

GitHub Pages deployment is handled by
`.github/workflows/deploy-pages.yml`. In repository settings, configure Pages to
use GitHub Actions as the source. The workflow runs on pushes to `main` and can
also be started manually from the Actions tab.

## Notes

- The full catalog is still maintained as `summaries/papers.html`.
- Missing detailed summaries are marked in the catalog with
  `саммари пропущено`; their takeaway is the paper abstract when it can be
  resolved automatically.
- A project license has not been selected yet.
