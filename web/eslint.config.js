import js from '@eslint/js';
import { defineConfig } from 'eslint/config';
import svelte from 'eslint-plugin-svelte';
import globals from 'globals';
import ts from 'typescript-eslint';

export default defineConfig(
  { ignores: ['node_modules/', 'public/'] },
  js.configs.all,
  ts.configs.strictTypeChecked,
  ts.configs.stylisticTypeChecked,
  svelte.configs.recommended,
  {
    languageOptions: {
      globals: { ...globals.browser },
      parserOptions: {
        // Tests and the Vite config live in tsconfig.node.json, which the
        // service would not find on its own.
        projectService: {
          allowDefaultProject: ['vite.config.ts', 'test/*.ts'],
          defaultProject: 'tsconfig.node.json',
        },
        tsconfigRootDir: import.meta.dirname,
        extraFileExtensions: ['.svelte'],
      },
    },
  },
  {
    files: ['**/*.svelte', '**/*.svelte.ts'],
    languageOptions: {
      parserOptions: { parser: ts.parser },
    },
  },
  {
    rules: {
      // Prettier owns layout and the import order is left as written.
      'sort-imports': 'off',
      'sort-keys': 'off',
      // Image and sky maths is full of plain constants that read better inline.
      'no-magic-numbers': 'off',
      // x, y and i are the clearest names in pixel loops.
      'id-length': 'off',
      // It asks for combined declarations, which fights one per line.
      'one-var': 'off',
      // A ternary is the plain way to pick a value or a class name.
      'no-ternary': 'off',
      // i++ in a for loop is what everyone expects to read.
      'no-plusplus': 'off',
      // Short notes often start with a lower case identifier.
      'capitalized-comments': 'off',
      // The defaults of 10 statements and 50 lines fit nothing that draws a frame.
      'max-statements': ['error', 60],
      'max-lines-per-function': ['error', 200],
      'max-params': ['error', 6],
      'max-lines': ['error', { max: 400, skipBlankLines: true, skipComments: true }],
      // Bundle keys come from the Python side in snake_case.
      camelcase: ['error', { properties: 'never' }],
      // void marks a promise that is left to run on purpose, which
      // no-floating-promises asks for.
      'no-void': ['error', { allowAsStatement: true }],
      // Small helpers are arrows and bigger ones are declarations.
      'func-style': 'off',
      // undefined is how Svelte and the DOM say a value is missing.
      'no-undefined': 'off',
      // Destructuring a single index of a typed array is slower and no clearer.
      'prefer-destructuring': 'off',
      // A let filled in by the branches below it reads fine.
      'init-declarations': 'off',
      // FITS decoding, PNG checksums and the seeded generator need bit operations.
      'no-bitwise': 'off',
      // Skipping a bad pixel early keeps the loop bodies flat.
      'no-continue': 'off',
    },
  },
  {
    files: ['**/*.svelte'],
    rules: {
      // let { a, b = $bindable() } = $props() has to stay let as a whole, and
      // the Svelte version of the rule knows that.
      'prefer-const': 'off',
      'svelte/prefer-const': 'error',
      // A component file carries its markup and styles too, and splitting one
      // changes the scoped CSS.
      'max-lines': 'off',
    },
  },
  {
    files: ['**/*.js', '**/*.mjs'],
    extends: [ts.configs.disableTypeChecked],
    languageOptions: { globals: { ...globals.node } },
  },
  {
    files: ['vite.config.ts', 'test/**/*.ts'],
    languageOptions: { globals: { ...globals.node } },
  },
);
