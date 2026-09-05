const { ipcMain } = require('electron');

/**
 * Copies `length` bytes starting at `start` in `sourceBuffer` into a
 * fresh, independent `ArrayBuffer`.
 *
 * `sourceBuffer` (a `Buffer`, i.e. a `Uint8Array` view) generally has a
 * `byteOffset` into its underlying `ArrayBuffer` that isn't a multiple of
 * 4 -- it depends on however much variable-length JSON header preceded it
 * in the frame, plus Node's own Buffer-pooling for small allocations.
 * Typed array constructors like `Float32Array` require their starting
 * offset to be a multiple of their element size, so a view directly over
 * `sourceBuffer`'s shared backing buffer can throw. Copying into a
 * brand-new `ArrayBuffer` (never pooled, always its own zero-offset
 * allocation) sidesteps that alignment requirement entirely.
 *
 * As a side effect, the copy also detaches the result from the Python
 * subprocess's shared read buffer before it crosses into Electron's
 * structured-clone IPC, so nothing downstream can observe that buffer
 * being reused for a later message.
 *
 * @param {Buffer} sourceBuffer
 * @param {number} start
 * @param {number} length
 * @returns {ArrayBuffer}
 */
function copyToAlignedArrayBuffer(sourceBuffer, start, length) {
  const arrayBuffer = new ArrayBuffer(length);
  new Uint8Array(arrayBuffer).set(sourceBuffer.subarray(start, start + length));
  return arrayBuffer;
}

/**
 * Registers every `ipcMain` handler this app uses -- the only file that
 * touches `ipcMain` directly; everything else in the main process goes
 * through `pythonBridge` instead.
 *
 * @param {import('./python-bridge').PythonBridge} pythonBridge
 * @param {() => import('electron').BrowserWindow[]} getWindows - Returns
 *   the windows to broadcast unsolicited status updates to.
 */
function registerIpcHandlers(pythonBridge, getWindows) {
  ipcMain.handle('mesh:get', async (_event, request) => {
    const { header, body } = await pythonBridge.request({
      type: 'get_mesh',
      tessellation: request.tessellation,
      radius: request.radius,
      frequency: request.frequency,
    });

    // The body is [positions bytes][indices bytes], contiguous, per the
    // `mesh` response's byte-length header fields (see
    // mundimonium/rendering/protocol.py and the wire protocol doc in the
    // plan).
    const positions = new Float32Array(copyToAlignedArrayBuffer(
        body, 0, header.positions_byte_length));
    const indices = new Uint32Array(copyToAlignedArrayBuffer(
        body, header.positions_byte_length, header.indices_byte_length));
    return {
      positions, indices,
      adjacency: header.adjacency, vertexFaces: header.vertex_faces,
    };
  });

  ipcMain.handle('mesh:getLod', async (_event, request) => {
    const { header, body } = await pythonBridge.request({
      type: 'get_lod_mesh',
      tessellation: request.tessellation,
      radius: request.radius,
      frequency: request.frequency,
      camera_position: request.cameraPosition,
      auto_subdivide: request.autoSubdivide,
    });

    // Same [positions bytes][indices bytes] body layout as `get_mesh`'s
    // `mesh` response (see mesh_export.py); `lod_mesh` additionally
    // carries each triangle's sector address in the header, not the body.
    const positions = new Float32Array(copyToAlignedArrayBuffer(
        body, 0, header.positions_byte_length));
    const indices = new Uint32Array(copyToAlignedArrayBuffer(
        body, header.positions_byte_length, header.indices_byte_length));
    return { positions, indices, sectors: header.sectors };
  });

  ipcMain.handle('mesh:subdivideSector', async (_event, request) => {
    await pythonBridge.request({
      type: 'subdivide_sector',
      tessellation: request.tessellation,
      radius: request.radius,
      frequency: request.frequency,
      sector: request.sector,
    });
  });

  ipcMain.handle('mesh:getFlat', async (_event, request) => {
    const { header, body } = await pythonBridge.request({
      type: 'get_flat_mesh',
      tessellation: request.tessellation,
      radius: request.radius,
      frequency: request.frequency,
      camera_position: request.cameraPosition,
      auto_subdivide: request.autoSubdivide,
      center: request.center,
      pan_offset: request.panOffset,
      basis: request.basis,
    });

    const positions = new Float32Array(copyToAlignedArrayBuffer(
        body, 0, header.positions_byte_length));
    const indices = new Uint32Array(copyToAlignedArrayBuffer(
        body, header.positions_byte_length, header.indices_byte_length));
    return {
      positions, indices, sectors: header.sectors, center: header.center,
      basis: header.basis,
    };
  });

  ipcMain.handle('items:get', async (_event, request) => {
    const { header } = await pythonBridge.request({
      type: 'get_items',
      tessellation: request.tessellation,
      radius: request.radius,
      frequency: request.frequency,
      camera_position: request.cameraPosition,
    });
    return { items: header.items };
  });

  ipcMain.handle('items:getFlat', async (_event, request) => {
    const { header } = await pythonBridge.request({
      type: 'get_flat_items',
      tessellation: request.tessellation,
      radius: request.radius,
      frequency: request.frequency,
      camera_position: request.cameraPosition,
      center: request.center,
      pan_offset: request.panOffset,
      basis: request.basis,
    });
    return { items: header.items, center: header.center, basis: header.basis };
  });

  pythonBridge.on('python:status', (status) => {
    for (const window of getWindows()) {
      if (!window.isDestroyed()) {
        window.webContents.send('python:status', status);
      }
    }
  });
}

module.exports = { registerIpcHandlers };
