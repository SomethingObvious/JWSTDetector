/// <reference types="vitest/config" />
import { readFile, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import { defineConfig, type Plugin } from 'vite';

const runDir = fileURLToPath(new URL('./public/run/', import.meta.url));

// Stands in for the Python server's write endpoint so the writable path can be
// tried against the fixture. It only exists under `npm run dev`.
function devVerdicts(): Plugin {
  return {
    name: 'dev-verdicts',
    apply: 'serve',
    configureServer(server) {
      server.middlewares.use('/run/api/verdict', (req, res) => {
        if (req.method !== 'POST') {
          res.statusCode = 405;
          res.end();
          return;
        }
        let body = '';
        req.on('data', (chunk: Buffer) => (body += chunk.toString('utf8')));
        req.on('end', async () => {
          try {
            const v = JSON.parse(body) as { key: string; verdict: string | null; star: boolean; note: string };
            const file = runDir + 'verdicts.json';
            let all: Record<string, unknown> = {};
            try {
              all = JSON.parse(await readFile(file, 'utf8'));
            } catch {
              all = {};
            }
            const saved = new Date().toISOString().replace(/\.\d+Z$/, 'Z');
            all[v.key] = { verdict: v.verdict, star: v.star, note: v.note, updated: saved };
            await writeFile(file, JSON.stringify(all, null, 1));
            res.setHeader('content-type', 'application/json');
            res.end(JSON.stringify({ saved }));
          } catch (err) {
            res.statusCode = 400;
            res.end(String(err));
          }
        });
      });
    },
  };
}

export default defineConfig({
  base: './',
  plugins: [svelte(), devVerdicts()],
  build: {
    outDir: '../jwstdetector/viewer_static',
    emptyOutDir: true,
    target: 'es2022',
    assetsInlineLimit: 0,
    // public/ only holds the fixture run, which must never ship in the package.
    copyPublicDir: false,
  },
  test: {
    include: ['test/**/*.test.ts'],
  },
});
