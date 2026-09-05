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
   * Like `getLodMesh`, but positions are flattened (a locally flat 2D
   * projection, `z` always `0`) around a center point, rather than in
   * true 3D. `cameraPosition` alone (flat mode's initial entry) derives
   * that center from a real 3D position; once `center` (a prior
   * response's own `center`, echoed back) and `panOffset` (how far the
   * flat camera has since moved from where that center was established)
   * are also given, the center is re-derived exactly via `SphericalTessellation
   * .unflatten_point` instead -- `cameraPosition` is still required
   * either way (see `getFlatItems`).
   *
   * @param {{ tessellation: string, radius?: number, frequency?: number,
   *   cameraPosition: [number, number, number], autoSubdivide?: boolean,
   *   center?: { face: number, b: number, s: number },
   *   panOffset?: [number, number] }} request
   * @returns {Promise<{ positions: Float32Array, indices: Uint32Array,
   *   sectors: object[], center: { face: number, b: number, s: number } }>}
   *   `center` is the resolved center this response was actually built
   *   around -- pass it back as the next call's own `center`.
   */
  getFlatMesh: (request) => ipcRenderer.invoke('mesh:getFlat', request),

  /**
   * @param {{ tessellation: string, radius?: number, frequency?: number,
   *   cameraPosition: [number, number, number] }} request
   * @returns {Promise<{ items: { kind: string, label?: string, x: number,
   *   y: number, z: number }[] }>}
   */
  getItems: (request) => ipcRenderer.invoke('items:get', request),

  /**
   * Like `getItems`, but `x`/`y` are flattened the same way
   * `getFlatMesh`'s positions are, and `z` is always `0`. `cameraPosition`
   * is always required here, even once `center`/`panOffset` are also
   * given -- it's still used to derive item-visibility `scale` (not
   * positioning), which isn't yet tracked through flat-mode zooming (see
   * the plan).
   *
   * @param {{ tessellation: string, radius?: number, frequency?: number,
   *   cameraPosition: [number, number, number],
   *   center?: { face: number, b: number, s: number },
   *   panOffset?: [number, number] }} request
   * @returns {Promise<{ items: { kind: string, label?: string, x: number,
   *   y: number, z: number }[],
   *   center: { face: number, b: number, s: number } }>}
   */
  getFlatItems: (request) => ipcRenderer.invoke('items:getFlat', request),

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
