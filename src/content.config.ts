import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const paperSummaries = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/paper_summaries' }),
  schema: z.object({
    title: z.string(),
    category: z.string(),
    slug: z.string(),
    catalogId: z.string(),
    sourceHtml: z.string().optional(),
    generatedFromHtml: z.boolean().optional(),
    paperUrl: z.string().url().optional()
  })
});

export const collections = {
  paperSummaries
};
