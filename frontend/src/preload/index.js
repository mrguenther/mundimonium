const { contextBridge, ipcRenderer } = require('electron');

// The only file that imports `ipcRenderer` -- the renderer process only
// ever sees the two purpose-built methods below, never raw IPC.
contextBridge.exposeInMainWorld('mundimonium', {
  /**
   * @param {{ tessellation: string, radius?: number, frequency?: number }} request
   * @returns {Promise<{ positions: Float32Array, indices: Uint32Array }>}
   */
  getMesh: (request) => ipcRenderer.invoke('mesh:get', request),

  /**
   * @param {(status: { state: 'starting'|'ready'|'crashed'|'exited', detail?: string }) => void} callback
   * @returns {() => void} Unsubscribe function.
   */
  onPythonStatus: (callback) => {
    const listener = (_event, status) => callback(status);
    ipcRenderer.on('python:status', listener);
    return () => ipcRenderer.removeListener('python:status', listener);
  },
});
