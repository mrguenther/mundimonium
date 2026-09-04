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
   * @param {{ tessellation: string, radius?: number, frequency?: number,
   *   cameraPosition: [number, number, number], autoSubdivide?: boolean }}
   *   request
   * @returns {Promise<{ positions: Float32Array, indices: Uint32Array,
   *   sectors: object[] }>} `sectors[i]` is the address of the sector
   *   `indices`' `i`-th triangle came from, for a later `subdivideSector`
   *   call.
   */
  getLodMesh: (request) => ipcRenderer.invoke('mesh:getLod', request),

  /**
   * Grows one sector's LOD detail. Doesn't return geometry itself -- call
   * `getLodMesh` again afterward to pick up the newly available detail.
   *
   * @param {{ tessellation: string, radius?: number, frequency?: number,
   *   sector: object }} request - `sector` is an address from a prior
   *   `getLodMesh` response's `sectors` array.
   * @returns {Promise<void>}
   */
  subdivideSector: (request) =>
      ipcRenderer.invoke('mesh:subdivideSector', request),

  /**
   * @param {{ tessellation: string, radius?: number, frequency?: number,
   *   cameraPosition: [number, number, number] }} request
   * @returns {Promise<{ items: { kind: string, label?: string, x: number,
   *   y: number, z: number }[] }>}
   */
  getItems: (request) => ipcRenderer.invoke('items:get', request),

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
