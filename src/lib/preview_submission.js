export async function submitPreviewAndRetire(device, encoder, allocations, consume) {
  try {
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    return await consume();
  } finally {
    try {
      await device.queue.onSubmittedWorkDone();
    } finally {
      for (const { buffer } of allocations) buffer.destroy();
    }
  }
}
