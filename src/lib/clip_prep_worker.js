/**
 * clip_prep_worker.js — Web Worker running the CLIP CPU prep (alpha-blend,
 * 512→224 resize + normalize, 32×32 patch embedding) off the main thread.
 *
 * Protocol (request/response by id, driven through worker_call.js):
 *   { type: 'init', id, conv1W, classEmb, posEmb }   ArrayBuffers (copied in
 *       once; the worker keeps them resident)      → { id, ok: true, initialized: true }
 *   { id, rgba, width, height }                     rgba ArrayBuffer (transferred)
 *                                                    → { id, ok: true, embeddings }  (transferred)
 * Uses the SAME clip_prep_core math as the main-thread path, so the
 * [50, 768] embeddings are byte-identical.
 */
import { prepareClipEmbeddings, validateClipPrepWeights } from './clip_prep_core.js';

let prepWeights = null;

self.onmessage = (e) => {
  const d = e.data;
  if (d?.type === 'init') {
    try {
      prepWeights = validateClipPrepWeights({
        conv1W: new Float32Array(d.conv1W),
        classEmb: new Float32Array(d.classEmb),
        posEmb: new Float32Array(d.posEmb),
      });
      self.postMessage({ id: d.id, ok: true, initialized: true });
    } catch (err) {
      self.postMessage({ id: d.id, ok: false, error: String(err?.stack || err) });
    }
    return;
  }
  try {
    if (!prepWeights) throw new Error('clip prep worker used before init');
    const embeddings = prepareClipEmbeddings(new Uint8ClampedArray(d.rgba), d.width, d.height, prepWeights);
    self.postMessage({ id: d.id, ok: true, embeddings: embeddings.buffer }, [embeddings.buffer]);
  } catch (err) {
    self.postMessage({ id: d.id, ok: false, error: String(err?.stack || err) });
  }
};
