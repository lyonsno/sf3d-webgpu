import assert from 'node:assert/strict';
import { decodeSf3dPreviewMesh } from '../src/lib/sf3d_producer.js';

assert.equal(typeof decodeSf3dPreviewMesh, 'function');
console.log('SF3D host bundle exposes the coarse preview decoder');
