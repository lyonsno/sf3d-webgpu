/**
 * Run explicitly CPU/worker-only work in the foreground service's admitted
 * window. GPU-mixed work must not use this helper: its existing cooperative
 * command-duty boundary owns ordering instead.
 */
export async function runInferenceDuty(options, work) {
  if (typeof work !== 'function') throw new TypeError('inference duty work must be a function');
  return options?.inferenceControl
    ? await options.inferenceControl.runDuty(work)
    : await work();
}

export async function withForegroundScope(options, phase, work) {
  if (typeof work !== 'function') throw new TypeError('foreground scope work must be a function');
  const admittedWork = () => typeof options?.withForeground === 'function'
    ? options.withForeground(phase, work)
    : work();
  return runInferenceDuty(options, admittedWork);
}
