// Device refusal/lifetime mechanism adapted from Kaminos models/trellis2/device-memory.js
// at 5c9a2fc2939e0a28ac3560a1e9108e81ba245fa7. CPU reservations cover explicit
// loader allocations, not released-but-uncollected JS backing, metadata, fetch,
// driver-private memory, textures, or total physical RAM. No capacity is inferred.
const authenticated = new WeakSet();
const devices = new WeakSet();
const bytes = (value, name) => {
  if (!Number.isSafeInteger(value) || value < 0) throw new TypeError(`${name} must be nonnegative safe-integer bytes`);
  return value;
};
export const isLoaderMemoryBudget = value => authenticated.has(value);

export function createLoaderMemoryBudget({cpuBytes, gpuBytes} = {}) {
  bytes(cpuBytes, 'cpuBytes'); bytes(gpuBytes, 'gpuBytes');
  const cpu = {liveBytes:0, peakLiveBytes:0, maxBytes:cpuBytes}, gpu = {liveBytes:0, peakLiveBytes:0, maxBytes:gpuBytes};
  const events = [], buffers = new Map();
  let phase = 'new', refusal = null, device, originalCreate, originalDestroy, wrappedCreate, wrappedDestroy, restored = false;
  const emit = (kind, details) => events.push({kind, phase, atMs:performance.now(), ...details});
  const refuse = (ledger, size, label, scope) => {
    bytes(size, 'requestedBytes');
    if (size <= ledger.maxBytes - ledger.liveBytes) return;
    refusal = {scope, phase, label, requestedBytes:size, liveBytes:ledger.liveBytes, maxBytes:ledger.maxBytes};
    emit('allocation-refused', refusal);
    const error = new Error(`SF3D ${scope} budget exceeded before allocation: ${ledger.liveBytes} + ${size} > ${ledger.maxBytes}`);
    error.name = 'SF3DMemoryBudgetError'; error.memoryBudget = refusal; throw error;
  };
  const charge = (ledger, size, label, scope) => {
    ledger.liveBytes += size; ledger.peakLiveBytes = Math.max(ledger.peakLiveBytes, ledger.liveBytes);
    emit('reserved', {scope, label, bytes:size, liveBytes:ledger.liveBytes});
    let released = false;
    return Object.freeze({release() {
      if (released) return; released = true; ledger.liveBytes -= size;
      emit('retirement-requested', {scope, label, bytes:size, liveBytes:ledger.liveBytes});
    }});
  };
  const budget = Object.freeze({
    events,
    setPhase(value) { phase = String(value); },
    reserveCpu(size, label) {
      if (restored) throw new Error('loader budget is restored');
      refuse(cpu, size, label, 'cpu-loader'); return charge(cpu, size, label, 'cpu-loader');
    },
    assertDevice(value) {
      if (!device || restored || value !== device || device.createBuffer !== wrappedCreate || device.destroy !== wrappedDestroy)
        throw new Error('loader budget must be installed on this exact owned device');
    },
    bindOwnedDevice(value) {
      if (device || restored || devices.has(value)) throw new Error('device already has a loader budget');
      if (typeof value?.createBuffer !== 'function' || typeof value.destroy !== 'function') throw new TypeError('owned device createBuffer/destroy required');
      originalCreate = value.createBuffer; originalDestroy = value.destroy;
      wrappedCreate = function(descriptor) {
        if (this !== value) throw new Error('budgeted allocator requires the exact owned device');
        // Pin the authority-bearing size once, including accessor-bearing input.
        const size = bytes(descriptor?.size, 'GPUBuffer.size'), label = descriptor?.label ?? '';
        refuse(gpu, size, label, 'gpu-buffer');
        const buffer = Reflect.apply(originalCreate, value, [{...descriptor, size, label}]);
        if (buffer.size !== size || typeof buffer.destroy !== 'function') {
          buffer.destroy?.(); throw new Error('GPUBuffer size/destroy contract unavailable');
        }
        if (buffers.has(buffer)) return buffer;
        const lease = charge(gpu, size, label, 'gpu-buffer'), destroy = buffer.destroy;
        buffers.set(buffer, {lease, destroy});
        try {
          const wrapped = function(...args) {
            const result = Reflect.apply(destroy, this, args); buffers.get(this)?.lease.release(); return result;
          };
          buffer.destroy = wrapped;
          if (buffer.destroy !== wrapped) throw new Error('GPUBuffer retirement hook installation failed');
        } catch (error) {
          // createBuffer succeeded before instrumentation failed. Retire the
          // actual buffer; if retirement fails, keep its charge until device
          // destruction rather than inventing reclaimed backing.
          try { Reflect.apply(destroy,buffer,[]); lease.release(); buffers.delete(buffer); }
          catch (cleanupError) { throw new AggregateError([error,cleanupError],'GPUBuffer retirement hook and cleanup failed',{cause:error}); }
          throw error;
        }
        return buffer;
      };
      wrappedDestroy = function(...args) {
        const result = Reflect.apply(originalDestroy, this, args);
        if (this === value) for (const row of buffers.values()) row.lease.release(); return result;
      };
      try {
        value.createBuffer = wrappedCreate; value.destroy = wrappedDestroy;
        if (value.createBuffer !== wrappedCreate || value.destroy !== wrappedDestroy) throw new Error('device budget installation failed');
      } catch (error) { value.createBuffer = originalCreate; value.destroy = originalDestroy; throw error; }
      device = value; devices.add(value); return budget;
    },
    snapshot() {
      return {schema:'sf3d.loader-memory-budget.v0', phase, refusal,
        cpu:{...cpu, physicalMemoryMeasured:false, meaning:'explicit loader reservations; retirement is not proven GC'},
        gpu:{...gpu, physicalMemoryMeasured:false, meaning:'API-visible buffers; destruction request is not reclaimed physical backing'},
        excludes:['JS/parse metadata', 'released JS backing before GC', 'network/browser/driver-private backing', 'textures', 'other processes']};
    },
    restore() {
      if (restored) return;
      if (cpu.liveBytes || gpu.liveBytes) throw new Error('retire loader allocations before restoring the budget');
      if (device) {
        device.createBuffer = originalCreate; device.destroy = originalDestroy;
        for (const [buffer, row] of buffers) buffer.destroy = row.destroy;
        devices.delete(device); buffers.clear();
      }
      restored = true;
    },
  });
  authenticated.add(budget); return budget;
}
