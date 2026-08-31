import assert from 'node:assert/strict';
import {
  collectChromeProcessCoalition,
  observeMacHostPressure,
  parseMemoryPressure,
  parseSwapUsage,
  parseVmStat,
} from './m2_pressure_observer.mjs';

const processTable = `
  100 1 200000 /Applications/Google Chrome --remote-debugging-port=1
  101 100 300000 /Applications/Google Chrome Helper --type=renderer
  102 100 400000 /Applications/Google Chrome Helper --type=gpu-process
  103 101 50000 /Applications/Google Chrome Helper --type=utility
  200 1 900000 /Applications/Google Chrome unrelated
`;
const coalition = collectChromeProcessCoalition(100, {
  exec(command, args) {
    assert.equal(command, 'ps');
    assert.deepEqual(args, ['-axo', 'pid=,ppid=,rss=,command=']);
    return processTable;
  },
});
assert.equal(coalition.observable, true);
assert.deepEqual(coalition.processes.map(process => process.pid), [100, 101, 102, 103]);
assert.deepEqual(coalition.processes.map(process => process.role), [
  'browser', 'renderer', 'gpu', 'utility',
]);
assert.equal(coalition.totalRssBytes, (200000 + 300000 + 400000 + 50000) * 1024);

const missingRoot = collectChromeProcessCoalition(999, { exec: () => processTable });
assert.equal(missingRoot.observable, false);
assert.deepEqual(missingRoot.processes, []);

assert.deepEqual(
  parseSwapUsage('vm.swapusage: total = 16384.00M  used = 15360.50M  free = 1023.50M  (encrypted)'),
  {
    totalBytes: 16384 * 1024 * 1024,
    usedBytes: 15360.5 * 1024 * 1024,
    freeBytes: 1023.5 * 1024 * 1024,
  },
);
assert.deepEqual(parseVmStat(`
Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free: 100.
Pages stored in compressor: 300.
Pages occupied by compressor: 20.
`), {
  pageSizeBytes: 16384,
  freeBytes: 100 * 16384,
  compressedMemoryBytes: 20 * 16384,
  compressedLogicalBytes: 300 * 16384,
});
assert.deepEqual(
  parseMemoryPressure('System-wide memory free percentage: 7%'),
  { freePercent: 7 },
);

const pressure = observeMacHostPressure({
  exec(command, args) {
    if (command === 'sysctl') {
      assert.deepEqual(args, ['vm.swapusage']);
      return 'vm.swapusage: total = 16384.00M used = 15360.50M free = 1023.50M (encrypted)';
    }
    if (command === 'vm_stat') return 'Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 100.\nPages stored in compressor: 300.\nPages occupied by compressor: 20.\n';
    if (command === 'memory_pressure') {
      assert.deepEqual(args, ['-Q']);
      return 'System-wide memory free percentage: 7%';
    }
    throw new Error(`unexpected command ${command}`);
  },
});
assert.equal(pressure.hostSwapUsedBytes, 15360.5 * 1024 * 1024);
assert.equal(pressure.hostCompressedMemoryBytes, 20 * 16384);
assert.equal(pressure.hostMemoryPressureFreePercent, 7);
assert.ok(pressure.indicators.includes('swap-free-below-2GiB'));
assert.ok(pressure.indicators.includes('memory-pressure-free-below-10-percent'));

const degraded = observeMacHostPressure({
  exec(command) {
    if (command === 'memory_pressure') throw new Error('unavailable');
    if (command === 'sysctl') return 'not parseable';
    return 'not parseable';
  },
});
assert.equal(degraded.hostSwapUsedBytes, null);
assert.equal(degraded.hostCompressedMemoryBytes, null);
assert.equal(degraded.hostMemoryPressureFreePercent, null);
assert.deepEqual(degraded.observerErrors, [
  { source: 'memory_pressure -Q', error: 'unavailable' },
]);

console.log('M2 pressure observer contract passed');
