import fs from 'node:fs';

export function writeJsonReportAtomic(reportPath, report) {
  const temporaryPath = `${reportPath}.${process.pid}.tmp`;
  try {
    fs.writeFileSync(temporaryPath, JSON.stringify(report, null, 2));
    fs.renameSync(temporaryPath, reportPath);
  } finally {
    fs.rmSync(temporaryPath, { force: true });
  }
}

export function readJsonReport(reportPath) {
  return JSON.parse(fs.readFileSync(reportPath, 'utf8'));
}

/** Native admission reports own plain data and a complete phase array. Write
 * each phase independently rather than materialize another growing-history
 * JSON string and UTF-8 buffer at every admission/readback. No entry is capped
 * or omitted; the published bytes match JSON.stringify(report, null, 2).
 * Other report consumers keep the original writer and its general semantics. */
export function writeJsonReportAtomicByPhase(reportPath, report) {
  const phases = Object.getOwnPropertyDescriptor(report ?? {}, 'phaseObservations')?.value;
  if (!report || ![Object.prototype, null].includes(Object.getPrototypeOf(report)) ||
      typeof report.toJSON === 'function' || !Array.isArray(phases) || typeof phases.toJSON === 'function')
    throw TypeError('plain native report with its own phase data array required');
  const temporaryPath = `${reportPath}.${process.pid}.tmp`;
  let fd;
  const write = text => {
    const bytes = Buffer.from(text);
    for (let offset = 0; offset < bytes.length;) {
      const count = fs.writeSync(fd, bytes, offset, bytes.length - offset);
      if (!Number.isSafeInteger(count) || count <= 0 || count > bytes.length - offset)
        throw Error('complete report write made invalid progress');
      offset += count;
    }
  };
  try {
    fd = fs.openSync(temporaryPath, 'w');
    write('{\n');
    let first = true;
    for (const key of Object.keys(report)) {
      if (key === 'phaseObservations') {
        if (!first) write(',\n'); first = false;
        write('  "phaseObservations": [');
        for (let index = 0; index < phases.length; index++) {
          // Wrapper preserves array-index toJSON keys; undefined/hole/function
          // entries become null exactly as in the ordinary JSON array encoder.
          const wrapped = JSON.stringify({[index]: phases[index]}, null, 2);
          const prefix = '{\n  ' + JSON.stringify(String(index)) + ': ';
          const value = wrapped === '{}' ? 'null' : wrapped.slice(prefix.length, -2);
          write((index ? ',\n' : '\n') + '    ' + value.replace(/\n/g, '\n  '));
        }
        write(phases.length ? '\n  ]' : ']');
      } else {
        const wrapped = JSON.stringify({[key]: report[key]}, null, 2);
        if (wrapped === '{}') continue;
        if (!first) write(',\n'); first = false;
        write(wrapped.slice(2, -2));
      }
    }
    write('\n}');
    fs.closeSync(fd); fd = undefined;
    fs.renameSync(temporaryPath, reportPath);
  } finally {
    try {if (fd !== undefined) fs.closeSync(fd);}
    finally {fs.rmSync(temporaryPath, {force: true});}
  }
}
