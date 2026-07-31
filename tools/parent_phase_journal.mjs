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

export function resolveDurableArtifactPath(journalPath, { createParent = true } = {}) {
  const resolved = path.resolve(journalPath);
  const volatileRoots = ['/tmp', '/private/tmp'];
  if (volatileRoots.some(root => resolved === root || resolved.startsWith(`${root}/`))) {
    throw new Error(`artifact path is volatile and will not survive host recovery: ${resolved}`);
  }
  const parent = path.dirname(resolved);
  if (createParent) fs.mkdirSync(parent, { recursive: true });
  const realParent = fs.realpathSync(parent);
  if (volatileRoots.some(root => realParent === root || realParent.startsWith(`${root}/`))) {
    throw new Error(`journal parent resolves to volatile storage: ${realParent}`);
  }
  return path.join(realParent, path.basename(resolved));
}

function syncDirectory(directory) {
  const directoryFd = fs.openSync(directory, 'r');
  try {
    fs.fsyncSync(directoryFd);
  } finally {
    fs.closeSync(directoryFd);
  }
}

function assertObject(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new TypeError(`${label} must be an object`);
  }
}

export function createParentPhaseJournal({ journalPath, invocationId, requested }) {
  const resolvedPath = resolveDurableArtifactPath(journalPath);
  if (typeof invocationId !== 'string' || !invocationId.trim()) {
    throw new TypeError('invocationId must be a nonempty string');
  }
  assertObject(requested, 'requested');
  let fd;
  try {
    fd = fs.openSync(resolvedPath, 'wx', 0o600);
  } catch (error) {
    if (error?.code === 'EEXIST') {
      throw new Error(`journal already exists and will not be overwritten: ${resolvedPath}`);
    }
    throw error;
  }
  syncDirectory(path.dirname(resolvedPath));

  let sequence = 0;
  let previousHash = ZERO_HASH;
  let closed = false;
  let effectiveIdentityRecorded = false;
  const monotonicOrigin = process.hrtime.bigint();

  const append = (type, payload = {}) => {
    if (closed) throw new Error('journal is closed');
    if (typeof type !== 'string' || !type) throw new TypeError('event type must be nonempty');
    assertObject(payload, 'event payload');
    if (type === 'heavy-work-start' && !effectiveIdentityRecorded) {
      throw new Error('effective identity must be durable before heavy work starts');
    }
    let safePayload;
    try {
      safePayload = JSON.parse(JSON.stringify(payload));
    } catch (error) {
      throw new TypeError(`event payload must be JSON-serializable: ${error.message}`);
    }
    const event = {
      schema: SCHEMA,
      invocationId,
      sequence,
      type,
      writtenAt: new Date().toISOString(),
      monotonicMs: Number(process.hrtime.bigint() - monotonicOrigin) / 1e6,
      previousHash,
      payload: safePayload,
    };
    event.eventHash = eventHash(event);
    fs.writeSync(fd, `${JSON.stringify(event)}\n`, null, 'utf8');
    fs.fsyncSync(fd);
    sequence += 1;
    previousHash = event.eventHash;
    if (type === 'effective-identity') effectiveIdentityRecorded = true;
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

export function writeJsonReportDurable(reportPath, report) {
  const resolvedPath = resolveDurableArtifactPath(reportPath);
  const temporaryPath = `${resolvedPath}.${process.pid}.${crypto.randomUUID()}.tmp`;
  const data = JSON.stringify(report, null, 2);
  let fd;
  try {
    fd = fs.openSync(temporaryPath, 'wx', 0o600);
    fs.writeFileSync(fd, data, 'utf8');
    fs.fsyncSync(fd);
    fs.closeSync(fd);
    fd = undefined;
    fs.renameSync(temporaryPath, resolvedPath);
    syncDirectory(path.dirname(resolvedPath));
  } finally {
    if (fd !== undefined) fs.closeSync(fd);
    fs.rmSync(temporaryPath, { force: true });
  }
  return resolvedPath;
}

export function replayParentPhaseJournal(journalPath) {
  const resolvedPath = path.resolve(journalPath);
  const lines = fs.readFileSync(resolvedPath, 'utf8').split('\n').filter(Boolean);
  if (!lines.length) throw new Error('journal integrity failure: empty journal');

  let expectedPreviousHash = ZERO_HASH;
  const events = lines.map((line, index) => {
    let event;
    try {
      event = JSON.parse(line);
    } catch (error) {
      throw new Error(`journal integrity failure at line ${index + 1}: ${error.message}`);
    }
    if (event.schema !== SCHEMA) throw new Error(`journal integrity failure: schema at ${index}`);
    if (event.sequence !== index) throw new Error(`journal integrity failure: sequence at ${index}`);
    if (event.previousHash !== expectedPreviousHash) {
      throw new Error(`journal integrity failure: previous hash at ${index}`);
    }
    const computedHash = eventHash(event);
    if (event.eventHash !== computedHash) {
      throw new Error(`journal integrity failure: event hash at ${index}`);
    }
    expectedPreviousHash = event.eventHash;
    return event;
  });

  const terminal = [...events].reverse().find(event => event.type === 'terminal') ?? null;
  const trustworthy = [...events].reverse().find(event => (
    event.type === 'phase-checkpoint'
    && event.payload?.trustworthy === true
    && typeof event.payload?.boundary === 'string'
  )) ?? null;

  return {
    schema: 'sf3d.parent-phase-journal-replay.v0',
    journalPath: resolvedPath,
    invocationId: events[0].invocationId,
    integrityOk: true,
    status: terminal?.payload?.status ?? 'interrupted',
    terminal: terminal?.payload ?? null,
    lastTrustworthyBoundary: trustworthy?.payload?.boundary ?? null,
    lastTrustworthyEventSequence: trustworthy?.sequence ?? null,
    eventCount: events.length,
    events,
  };
}
