# RecSys Papers

Open research hub with paper summaries about recommender systems, Semantic IDs,
generative retrieval, and generative recommendation.

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
