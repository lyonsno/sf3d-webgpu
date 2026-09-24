#!/usr/bin/env node
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import appConfig from '../vite.config.js';
import libraryConfig from '../vite.lib.config.js';

const expected = execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim();
for (const [name, config] of [['app', appConfig], ['producer library', libraryConfig]]) {
  assert.equal(JSON.parse(config.define.__COMMIT_HASH__), expected, `${name} must embed the exact checkout HEAD`);
}
console.log('Vite app and producer-library commit identities match exact Git HEAD');
