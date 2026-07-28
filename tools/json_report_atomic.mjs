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
