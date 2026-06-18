# RecSys Papers

Open research hub with paper summaries about recommender systems, Semantic IDs,
generative retrieval, and generative recommendation.

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

- The detailed summaries are still HTML in `summaries/`.
- The long-term target is to move source content to Markdown/MDX with structured
  metadata and generate catalog pages from data.
- A project license has not been selected yet.
