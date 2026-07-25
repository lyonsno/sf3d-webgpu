/**
 * WebGPU initialization and device management.
 */

let activeBufferAllocationSink = null;

function recordBufferAllocation(buffer, size, label) {
  if (activeBufferAllocationSink) {
    activeBufferAllocationSink.push({
      buffer,
      size: Math.ceil(size / 4) * 4,
      label: label || buffer.label || '',
    });
  }
  return buffer;
}

/**
 * Capture buffers allocated synchronously through this module while `fn` runs.
 * The caller owns retirement; nested scopes restore the outer sink afterward.
 */
export function captureGpuBufferAllocations(fn) {
  if (typeof fn !== 'function') throw new TypeError('fn must be a function');
  const previous = activeBufferAllocationSink;
  const allocations = [];
  activeBufferAllocationSink = allocations;
  try {
    return { value: fn(), allocations };
  } catch (error) {
    const cleanupErrors = [];
    for (const allocation of allocations) {
      try {
        allocation.buffer.destroy();
      } catch (cleanupError) {
        cleanupErrors.push(cleanupError.message);
      }
    }
    const cleanup = Object.freeze({
      retiredCount: allocations.length - cleanupErrors.length,
      cleanupErrors: Object.freeze(cleanupErrors),
    });
    if ((typeof error === 'object' && error !== null) || typeof error === 'function') {
      try {
        Object.defineProperty(error, 'gpuAllocationCleanup', {
          configurable: true,
          enumerable: true,
          value: cleanup,
        });
      } catch {
        // Cleanup remains complete even when the thrown object is immutable.
      }
    }
    throw error;
  } finally {
    activeBufferAllocationSink = previous;
  }
}

export async function initGPU() {
  if (!navigator.gpu) {
    throw new Error('WebGPU is not supported in this browser. Try Chrome 113+ or Edge 113+.');
  }

  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: 'high-performance',
  });
  if (!adapter) {
    throw new Error('No WebGPU adapter found. Your GPU may not support WebGPU.');
  }

  // Request max limits for large model inference
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
      maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupSizeX: adapter.limits.maxComputeWorkgroupSizeX,
      maxComputeWorkgroupSizeY: adapter.limits.maxComputeWorkgroupSizeY,
    },
  });

  device.lost.then((info) => {
    console.error('WebGPU device lost:', info.message);
    if (info.reason !== 'destroyed') {
      // Could attempt recovery here
    }
  });

  return { adapter, device };
}

/**
 * Create a storage buffer initialized with data.
 */
export function createStorageBuffer(device, data, usage = 0, label = '') {
  if (data.byteLength % 4 !== 0) {
    console.warn(`createStorageBuffer: non-4-aligned size ${data.byteLength} (label: ${label})`);
  }
  const size = Math.ceil(data.byteLength / 4) * 4;
  const buffer = device.createBuffer({
    size, // ensure 4-byte alignment
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | usage,
    mappedAtCreation: true,
    label: label || `storage_${data.byteLength}`,
  });
  new (data.constructor)(buffer.getMappedRange()).set(data);
  buffer.unmap();
  return recordBufferAllocation(buffer, size, label);
}

/**
 * Create an empty storage buffer.
 */
export function createEmptyBuffer(device, size, usage = 0, label = '') {
  if (size % 4 !== 0) {
    console.warn(`createEmptyBuffer: non-4-aligned size ${size} (label: ${label})`);
  }
  const alignedSize = Math.ceil(size / 4) * 4;
  const buffer = device.createBuffer({
    size: alignedSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | usage,
    mappedAtCreation: false,
    label: label || `empty_${size}`,
  });
  return recordBufferAllocation(buffer, alignedSize, label);
}

/**
 * Read back buffer contents to CPU.
 */
export async function readBuffer(device, buffer, size) {
  const staging = device.createBuffer({
    size,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, staging, 0, size);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const result = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return result;
}
