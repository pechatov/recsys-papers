import { defineConfig } from 'astro/config';

const repository = process.env.GITHUB_REPOSITORY?.split('/')[1] ?? '';
const owner = process.env.GITHUB_REPOSITORY_OWNER ?? '';
const inferredBase =
  process.env.GITHUB_ACTIONS && repository && !repository.endsWith('.github.io')
    ? `/${repository}/`
    : '/';

export default defineConfig({
  site: process.env.SITE_URL ?? (owner ? `https://${owner}.github.io` : 'http://localhost:4321'),
  base: process.env.SITE_BASE ?? inferredBase,
  trailingSlash: 'ignore'
});
