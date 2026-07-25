/**
 * preprocess_worker.js — Web Worker that runs the expensive image-preprocess
 * math (Lanczos-3 resize + blend + normalize) off the main thread.
 *
 * Receives raw float32 RGBA source pixels (transferred, zero-copy) + params,
 * returns the CHW float32 tensor (transferred back). The main thread stays
 * responsive during the ~700ms that this work would otherwise block it.
 *
 * Uses the SAME preprocess_core math as the main-thread path, so output is
 * byte-identical.
 */
import { resizeBlendNormalize } from './preprocess_core.js';

self.onmessage = (e) => {
  const { srcBuffer, srcW, srcH, size, bg, imageMean, imageStd, id } = e.data;
  try {
    const srcFloat = new Float32Array(srcBuffer);
    const chw = resizeBlendNormalize(srcFloat, srcW, srcH, size, bg, imageMean, imageStd);
    self.postMessage({ id, ok: true, chwBuffer: chw.buffer }, [chw.buffer]);
  } catch (err) {
    self.postMessage({ id, ok: false, error: String(err?.stack || err) });
  }
};
