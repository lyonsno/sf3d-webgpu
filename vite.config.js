import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    port: 5176,
    open: true,
  },
  build: {
    target: 'esnext',
  },
});
