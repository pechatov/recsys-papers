# Open-source migration plan

## Goal

Make the paper-summary archive publishable on GitHub Pages without losing the
current reading experience:

- deep links between catalog and summary pages;
- local reading status in `localStorage`;
- left/right scroll gutters;
- MathJax rendering for formulas;
- existing figures and diagrams.

## Technology choice

Recommended stack: **Astro + GitHub Actions + GitHub Pages**.

Why Astro:

- it outputs static files, which fits GitHub Pages and keeps hosting simple;
- it supports Markdown/MDX and content collections for the later migration;
- it does not force the site into a heavy documentation app shape;
- interactive pieces can stay as small client-side scripts;
- the existing static HTML archive can be published unchanged during migration.

Alternatives considered:

- **VitePress**: good for documentation, but the corpus is closer to a research
  database with custom catalogs than to a linear docs tree.
- **Docusaurus**: strong docs product, but heavier than needed for a static
  research hub.
- **Eleventy**: excellent static generator, but Astro gives a smoother path if
  we later want isolated interactive components or MDX.
- **Plain HTML only**: easiest short-term option, but it keeps the current
  duplication and makes generated catalog pages harder to maintain.

## Migration phases

### Phase 1: publishable full catalog

- Add Astro project files.
- Add GitHub Pages workflow.
- Publish the full catalog as the primary entry point.
- Mirror the existing detailed paper-summary pages into the build output.
- Keep legacy catalog and summary-page interactions functionally unchanged.
- Remove local-only links and files from the public surface.

### Phase 2: structured catalog

- Extract paper metadata from `papers.html` and detailed summary pages.
- Create a canonical data file, likely JSON or YAML.
- Generate catalog pages from that data.
- Keep old URLs or redirects for existing summary pages.

### Phase 3: content migration

- Convert detailed summaries from HTML to Markdown/MDX.
- Store metadata in frontmatter.
- Move repeated page chrome, MathJax setup, navigation, and scroll controls into
  reusable Astro layouts.
- Preserve legacy `.html` summary URLs through Astro-generated static routes.

### Phase 4: open-source polish

- Choose a license.
- Add contribution instructions.
- Add link validation and build checks to CI.
- Add a data schema test for paper metadata.
