import { defineConfig } from 'astro/config';
import { unified } from '@astrojs/markdown-remark';
import remarkMath from 'remark-math';

const repository = process.env.GITHUB_REPOSITORY?.split('/')[1] ?? '';
const owner = process.env.GITHUB_REPOSITORY_OWNER ?? '';
const inferredBase =
  process.env.GITHUB_ACTIONS && repository && !repository.endsWith('.github.io')
    ? `/${repository}/`
    : '/';

function nodeText(node) {
  if (!node) return '';
  if (node.type === 'text') return node.value ?? '';
  return (node.children ?? []).map(nodeText).join('');
}

function classList(node) {
  const className = node?.properties?.className;
  if (Array.isArray(className)) return className;
  if (typeof className === 'string') return className.split(/\s+/).filter(Boolean);
  return [];
}

function rehypeMathJaxDelimiters() {
  return (tree) => {
    function walk(node) {
      if (!node || !node.children) return;

      for (let childIndex = 0; childIndex < node.children.length; childIndex += 1) {
        const child = node.children[childIndex];
        const classes = classList(child);

        if (child.type === 'element' && child.tagName === 'code' && classes.includes('math-inline')) {
          node.children[childIndex] = {
            type: 'element',
            tagName: 'span',
            properties: { className: ['tex-inline'] },
            children: [{ type: 'text', value: `\\(${nodeText(child)}\\)` }]
          };
          continue;
        }

        if (child.type === 'element' && child.tagName === 'pre') {
          const code = child.children?.find((grandchild) => grandchild.type === 'element' && grandchild.tagName === 'code');
          if (code && classList(code).includes('math-display')) {
            node.children[childIndex] = {
              type: 'element',
              tagName: 'div',
              properties: { className: ['math'] },
              children: [{ type: 'text', value: `\\[\n${nodeText(code)}\n\\]` }]
            };
            continue;
          }
        }

        walk(child);
      }
    }

    walk(tree);
  };
}

export default defineConfig({
  site: process.env.SITE_URL ?? (owner ? `https://${owner}.github.io` : 'http://localhost:4321'),
  base: process.env.SITE_BASE ?? inferredBase,
  trailingSlash: 'ignore',
  markdown: {
    processor: unified({
      remarkPlugins: [remarkMath],
      rehypePlugins: [rehypeMathJaxDelimiters]
    })
  },
  build: {
    format: 'file'
  }
});
