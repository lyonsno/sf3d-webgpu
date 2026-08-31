import os from 'node:os';
import { execFileSync } from 'node:child_process';

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;

function execute(command, args = []) {
  return execFileSync(command, args, { encoding: 'utf8' });
}

export function parseProcessTable(text) {
  return text.split('\n').filter(Boolean).map(line => {
    const match = line.match(/^\s*(\d+)\s+(\d+)\s+(\d+)\s+(.+)$/);
    if (!match) return null;
    return {
      pid: Number(match[1]),
      parentPid: Number(match[2]),
      rssBytes: Number(match[3]) * 1024,
      command: match[4],
    };
  }).filter(Boolean);
}

function chromeRole(command, isRoot) {
  if (isRoot) return 'browser';
  const type = command.match(/--type=([^\s]+)/)?.[1] ?? null;
  if (type === 'gpu-process') return 'gpu';
  if (type === 'renderer') return 'renderer';
  if (type === 'utility') return 'utility';
  if (type) return type;
  return 'descendant';
}

export function collectChromeProcessCoalition(rootPid, { exec = execute } = {}) {
  if (!Number.isSafeInteger(rootPid) || rootPid <= 0) {
    return { rootPid: null, observable: false, totalRssBytes: null, processes: [] };
  }
  let processes;
  try {
    processes = parseProcessTable(exec('ps', ['-axo', 'pid=,ppid=,rss=,command=']));
  } catch (error) {
    return {
      rootPid,
      observable: false,
      error: error.message,
      totalRssBytes: null,
      processes: [],
    };
  }
  const descendants = new Set([rootPid]);
  let changed = true;
  while (changed) {
    changed = false;
    for (const process of processes) {
      if (descendants.has(process.parentPid) && !descendants.has(process.pid)) {
        descendants.add(process.pid);
        changed = true;
      }
    }
  }
  const coalition = processes
    .filter(process => descendants.has(process.pid))
    .map(process => ({
      ...process,
      role: chromeRole(process.command, process.pid === rootPid),
    }))
    .sort((left, right) => left.pid - right.pid);
  const observable = coalition.some(process => process.pid === rootPid);
  return {
    rootPid,
    observable,
    totalRssBytes: observable
      ? coalition.reduce((sum, process) => sum + process.rssBytes, 0)
      : null,
    processes: coalition,
  };
}

export function parseSwapUsage(text) {
  const match = text.match(/total\s*=\s*([\d.]+)M\s+used\s*=\s*([\d.]+)M\s+free\s*=\s*([\d.]+)M/i);
  if (!match) return null;
  return {
    totalBytes: Number(match[1]) * MIB,
    usedBytes: Number(match[2]) * MIB,
    freeBytes: Number(match[3]) * MIB,
  };
}

export function parseVmStat(text) {
  const pageSize = Number(text.match(/page size of (\d+) bytes/i)?.[1] ?? 0);
  const readPages = label => {
    const match = text.match(new RegExp(`^${label}:\\s+(\\d+)\\.`, 'mi'));
    return match ? Number(match[1]) : null;
  };
  const bytes = pages => pages == null ? null : pages * pageSize;
  if (!pageSize) return null;
  return {
    pageSizeBytes: pageSize,
    freeBytes: bytes(readPages('Pages free')),
    compressedMemoryBytes: bytes(readPages('Pages occupied by compressor')),
    compressedLogicalBytes: bytes(readPages('Pages stored in compressor')),
  };
}

export function parseMemoryPressure(text) {
  const match = text.match(/System-wide memory free percentage:\s*(\d+)%/i);
  return match ? { freePercent: Number(match[1]) } : null;
}

export function observeMacHostPressure({ exec = execute } = {}) {
  let swap = null;
  let virtualMemory = null;
  let memoryPressure = null;
  const errors = [];
  try { swap = parseSwapUsage(exec('sysctl', ['vm.swapusage'])); }
  catch (error) { errors.push({ source: 'sysctl vm.swapusage', error: error.message }); }
  try { virtualMemory = parseVmStat(exec('vm_stat')); }
  catch (error) { errors.push({ source: 'vm_stat', error: error.message }); }
  try { memoryPressure = parseMemoryPressure(exec('memory_pressure', ['-Q'])); }
  catch (error) { errors.push({ source: 'memory_pressure -Q', error: error.message }); }

  const hostFreeMemoryBytes = os.freemem();
  const indicators = [];
  if (hostFreeMemoryBytes < 256 * MIB) indicators.push('host-free-memory-below-256MiB');
  if (swap?.freeBytes != null && swap.freeBytes < 2 * GIB) indicators.push('swap-free-below-2GiB');
  if (memoryPressure?.freePercent != null && memoryPressure.freePercent < 10) {
    indicators.push('memory-pressure-free-below-10-percent');
  }
  return {
    hostFreeMemoryBytes,
    hostTotalMemoryBytes: os.totalmem(),
    hostLoadAverage: os.loadavg(),
    hostCompressedMemoryBytes: virtualMemory?.compressedMemoryBytes ?? null,
    hostCompressedLogicalBytes: virtualMemory?.compressedLogicalBytes ?? null,
    hostSwapTotalBytes: swap?.totalBytes ?? null,
    hostSwapUsedBytes: swap?.usedBytes ?? null,
    hostSwapFreeBytes: swap?.freeBytes ?? null,
    hostMemoryPressureFreePercent: memoryPressure?.freePercent ?? null,
    indicators,
    observerErrors: errors,
  };
}
