/**
 * preprocess_worker.js — Web Worker that runs the expensive image-preprocess
 * math (foreground framing, bicubic resize, blend and normalize) off the main thread.
 *
 * Receives raw float32 RGBA source pixels (transferred, zero-copy) + params,
 * returns the CHW tensor and shared framed RGBA pixels (transferred back). The main thread stays
 * responsive during the ~700ms that this work would otherwise block it.
 *
 * Uses the same condition_image math as the main-thread path, so output is
 * byte-identical.
 */
import { prepareConditionImage } from './condition_image.js';

self.onmessage = (e) => {
  const { srcBuffer, srcW, srcH, size, bg, imageMean, imageStd, id } = e.data;
  try {
    const srcFloat = new Float32Array(srcBuffer);
    const { chw, rgba } = prepareConditionImage(srcFloat, srcW, srcH, size, bg, imageMean, imageStd);
    self.postMessage({ id, ok: true, chwBuffer: chw.buffer, rgbaBuffer: rgba.buffer }, [chw.buffer, rgba.buffer]);
  } catch (err) {
    self.postMessage({ id, ok: false, error: String(err?.stack || err) });
  }
};
