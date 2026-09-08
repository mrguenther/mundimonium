const path = require('node:path');
const { defineConfig } = require('vite');

// Builds only the renderer (src/renderer/) into dist/renderer/. The main
// and preload processes are plain CommonJS run directly by Electron/Node
// and don't need bundling -- they only use Electron/Node built-ins, never
// an npm-installed browser library the way the renderer uses `three`.
module.exports = defineConfig({
  root: path.resolve(__dirname, 'src/renderer'),
  // Relative asset paths: the built index.html is loaded via `loadFile`
  // (a file:// URL), where absolute (`/...`) paths wouldn't resolve.
  base: './',
  build: {
    outDir: path.resolve(__dirname, 'dist/renderer'),
    emptyOutDir: true,
  },
});
