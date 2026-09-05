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
   * that center from a real 3D position, and its tangent basis from
   * scratch; once `center`/`basis` (a prior response's own `center`/
   * `basis`, echoed back) and `panOffset` (how far the flat camera has
   * since moved from where that center was established) are also given,
   * both are re-derived exactly via `SphericalTessellation
   * .unflatten_point_and_transport_basis` instead -- parallel-
   * transporting `basis` keeps the projection's orientation evolving
   * continuously as the center moves, rather than a small net rotation
   * accumulating relative to the path panned. `cameraPosition` is still
   * required either way (see `getFlatItems`).
   *
   * @param {{ tessellation: string, radius?: number, frequency?: number,
   *   cameraPosition: [number, number, number], autoSubdivide?: boolean,
   *   center?: { face: number, b: number, s: number },
   *   basis?: { e_x: [number, number, number], e_y: [number, number, number] },
   *   panOffset?: [number, number] }} request
   * @returns {Promise<{ positions: Float32Array, indices: Uint32Array,
   *   sectors: object[], center: { face: number, b: number, s: number },
   *   basis: { e_x: [number, number, number], e_y: [number, number, number] } }>}
   *   `center`/`basis` are what this response was actually built around --
   *   pass them back as the next call's own `center`/`basis`.
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
   * is always required here, even once `center`/`basis`/`panOffset` are
   * also given -- it's still used to derive item-visibility `scale` (not
   * positioning), which doesn't yet track flat-mode zooming, so it grows
   * increasingly stale the further the view pans from flat mode's entry
   * point.
   *
   * @param {{ tessellation: string, radius?: number, frequency?: number,
   *   cameraPosition: [number, number, number],
   *   center?: { face: number, b: number, s: number },
   *   basis?: { e_x: [number, number, number], e_y: [number, number, number] },
   *   panOffset?: [number, number] }} request
   * @returns {Promise<{ items: { kind: string, label?: string, x: number,
   *   y: number, z: number }[],
   *   center: { face: number, b: number, s: number },
   *   basis: { e_x: [number, number, number], e_y: [number, number, number] } }>}
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
