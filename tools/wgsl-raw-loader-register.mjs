import { register } from 'node:module';
import { pathToFileURL } from 'node:url';

register('./wgsl-raw-loader.mjs', pathToFileURL(new URL('.', import.meta.url).pathname));
