import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

const SCHEMA = 'sf3d.parent-phase-journal-event.v0';
const ZERO_HASH = '0'.repeat(64);

function stableJson(value) {
  if (value === null || typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(stableJson).join(',')}]`;
  return `{${Object.keys(value).sort().map(key => (
    `${JSON.stringify(key)}:${stableJson(value[key])}`
  )).join(',')}}`;
}

function eventHash(event) {
  const { eventHash: _ignored, ...body } = event;
  return crypto.createHash('sha256').update(stableJson(body)).digest('hex');
}

export function resolveDurableArtifactPath(journalPath) {
  const resolved = path.resolve(journalPath);
  const volatileRoots = ['/tmp', '/private/tmp'];
  if (volatileRoots.some(root => resolved === root || resolved.startsWith(`${root}/`))) {
    throw new Error(`artifact path is volatile and will not survive host recovery: ${resolved}`);
  }
  const parent = path.dirname(resolved);
  fs.mkdirSync(parent, { recursive: true });
  const realParent = fs.realpathSync(parent);
  if (volatileRoots.some(root => realParent === root || realParent.startsWith(`${root}/`))) {
    throw new Error(`journal parent resolves to volatile storage: ${realParent}`);
  }
  return path.join(realParent, path.basename(resolved));
}

function syncDirectory(directory) {
  const fd = fs.openSync(directory, 'r');
  try { fs.fsyncSync(fd); } finally { fs.closeSync(fd); }
}

export function createParentPhaseJournal({ journalPath, invocationId, requested }) {
  const resolvedPath = resolveDurableArtifactPath(journalPath);
  if (typeof invocationId !== 'string' || !invocationId.trim()) {
    throw new TypeError('invocationId must be a nonempty string');
  }
  if (!requested || typeof requested !== 'object' || Array.isArray(requested)) {
    throw new TypeError('requested must be an object');
  }
  const fd = fs.openSync(resolvedPath, 'wx', 0o600);
  syncDirectory(path.dirname(resolvedPath));
  let sequence = 0;
  let previousHash = ZERO_HASH;
  let closed = false;
  const origin = process.hrtime.bigint();

  const append = (type, payload = {}) => {
    if (closed) throw new Error('journal is closed');
    if (typeof type !== 'string' || !type) throw new TypeError('event type must be nonempty');
    if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
      throw new TypeError('event payload must be an object');
    }
    const safePayload = JSON.parse(JSON.stringify(payload));
    const event = {
      schema: SCHEMA,
      invocationId,
      sequence,
      type,
      writtenAt: new Date().toISOString(),
      monotonicMs: Number(process.hrtime.bigint() - origin) / 1e6,
      previousHash,
      payload: safePayload,
    };
    event.eventHash = eventHash(event);
    fs.writeSync(fd, `${JSON.stringify(event)}\n`, null, 'utf8');
    fs.fsyncSync(fd);
    sequence += 1;
    previousHash = event.eventHash;
    return event;
  };

  append('invocation-requested', requested);
  return {
    path: resolvedPath,
    append,
    close() {
      if (closed) return;
      fs.closeSync(fd);
      closed = true;
    },
  };
}

export function replayParentPhaseJournal(journalPath) {
  const resolvedPath = path.resolve(journalPath);
  const contents = fs.readFileSync(resolvedPath, 'utf8');
  const lines = contents.split('\n');
  const hasPartialTail = !contents.endsWith('\n');
  lines.pop();
  if (lines.some(line => line.length === 0)) throw new Error('journal integrity failure: blank event record');
  const records = lines;
  if (!records.length) throw new Error('journal integrity failure: no complete events');

  let expectedPreviousHash = ZERO_HASH;
  const events = records.map((line, index) => {
    let event;
    try { event = JSON.parse(line); }
    catch (error) { throw new Error(`journal integrity failure at line ${index + 1}: ${error.message}`); }
    if (event.schema !== SCHEMA) throw new Error(`journal integrity failure: schema at ${index}`);
    if (event.sequence !== index) throw new Error(`journal integrity failure: sequence at ${index}`);
    if (event.previousHash !== expectedPreviousHash) throw new Error(`journal integrity failure: previous hash at ${index}`);
    if (event.eventHash !== eventHash(event)) throw new Error(`journal integrity failure: event hash at ${index}`);
    expectedPreviousHash = event.eventHash;
    return event;
  });
  const terminal = [...events].reverse().find(event => event.type === 'terminal') ?? null;
  const entered = [...events].reverse().find(event => event.type === 'phase-entered') ?? null;
  const completed = [...events].reverse().find(event => event.type === 'phase-completed') ?? null;
  return {
    schema: 'sf3d.parent-phase-journal-replay.v0',
    journalPath: resolvedPath,
    invocationId: events[0].invocationId,
    integrityOk: true,
    status: hasPartialTail ? 'interrupted' : (terminal?.payload?.status ?? 'interrupted'),
    terminal: terminal?.payload ?? null,
    lastEnteredPhase: entered?.payload?.phase ?? null,
    lastCompletedPhase: completed?.payload?.phase ?? null,
    eventCount: events.length,
    hasPartialTail,
    events,
  };
}
