const { contextBridge, ipcRenderer } = require('electron');

// The only file that imports `ipcRenderer` -- the renderer process only
// ever sees the two purpose-built methods below, never raw IPC.
contextBridge.exposeInMainWorld('mundimonium', {
  /**
   * @param {{ tessellation: string, radius?: number, frequency?: number }} request
   * @returns {Promise<{ positions: Float32Array, indices: Uint32Array,
   *   adjacency: number[][], vertexFaces: number[][] }>} `adjacency[i]` is
   *   `[neighborB, neighborS, neighborD]` -- the triangle index (into this
   *   same `indices` buffer, not a vertex index) across the edge opposite
   *   triangle `i`'s own b/s/d vertex, or `-1` for a mesh boundary edge.
   *   `vertexFaces[v]` is the triangle indices touching vertex `v`, in
   *   cyclic order around it (or an arbitrary order for a vertex on an
   *   open mesh boundary). Used by `SurfaceCameraController` to walk the
   *   mesh surface without a server round-trip per step.
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
   * For `tessellation: 'generic'`, `orientation` (a 2x2 rotation) plays
   * the same continuity role `basis` plays for `'spherical'`, just with
   * no tangent-basis concept of its own -- see `GenericTessellation
   * .unflatten_point_and_transport_orientation`.
   *
   * `tessellation: 'hyperbolic'` needs neither `cameraPosition` nor
   * `basis`/`orientation` -- this world has no 3D embedding at all (it's
   * never anything but flat) and no orientation-continuity concept of its
   * own. Its response instead carries `stablePanRadius`: the caller
   * should keep the flat camera's pan distance from `center` under this
   * value (e.g. via `FlatMapCameraController.setMaxPanRadius`), since
   * this tessellation's projection is only numerically valid within a
   * bounded radius of wherever it's currently centered.
   *
   * `tessellation: 'hyperbolic'` also supports a second, `overview: true`
   * sub-mode -- a wide, bounded Poincare-disk view (this kind's analogue
   * of the other two kinds' 3D view) rather than the exact local
   * close-up projection `overview: false`/omitted gives. Both sub-modes
   * share the exact same `center`/`panOffset`-based continuation
   * contract described above (each response's own `center`/
   * `stablePanRadius` feeds the next request the same way); they differ
   * only in which server-side projection/frontier renders them, not in
   * the wire shape. `center`/`stablePanRadius` values from one sub-mode
   * are valid `center`s to enter the *other* sub-mode with (e.g.
   * resuming close-up at wherever the view was panned to within the
   * overview) -- but a `stablePanRadius` itself is only meaningful
   * against its own sub-mode's own camera position, never the other's
   * (the two are measured in different, nonlinearly related units).
   *
   * @param {{ tessellation: string, radius?: number, frequency?: number,
   *   cameraPosition?: [number, number, number], autoSubdivide?: boolean,
   *   center?: { face: number, b: number, s: number },
   *   basis?: { e_x: [number, number, number], e_y: [number, number, number] },
   *   orientation?: { cos: number, sin: number },
   *   panOffset?: [number, number], overview?: boolean }} request -
   *   `cameraPosition` is only used (and required) for `tessellation:
   *   'spherical'`'s first call, to resolve an initial `center`.
   *   `'generic'` always supplies `center` directly instead (its
   *   surface-following camera already knows its own exact position), so
   *   `cameraPosition` is never needed there. `'hyperbolic'` needs
   *   neither: its first call (in either sub-mode) omits `center` too,
   *   letting the server start from wherever its own reference point
   *   already is (this tessellation has no 3D position to derive one
   *   from in the first place).
   * @returns {Promise<{ positions: Float32Array, indices: Uint32Array,
   *   sectors: object[], center: { face: number, b: number, s: number },
   *   basis?: { e_x: [number, number, number], e_y: [number, number, number] },
   *   centerPosition?: [number, number, number],
   *   orientation?: { cos: number, sin: number },
   *   stablePanRadius?: number }>}
   *   `center`/`basis`/`orientation` are what this response was actually
   *   built around -- pass them back as the next call's own `center`/
   *   `basis`/`orientation`. `basis`/`centerPosition` are only present
   *   for `tessellation: 'spherical'`; `centerPosition` is `center`'s own
   *   true 3D position, for resuming the 3D orbit camera there if flat
   *   mode is exited. `orientation` is only present for `tessellation:
   *   'generic'`, and only once resolved via a `panOffset` (nothing to
   *   report on the first, `panOffset`-less call). `stablePanRadius` is
   *   only present for `tessellation: 'hyperbolic'` (either sub-mode).
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
   *   orientation?: { cos: number, sin: number },
   *   panOffset?: [number, number] }} request
   * @returns {Promise<{ items: { kind: string, label?: string, x: number,
   *   y: number, z: number }[],
   *   center: { face: number, b: number, s: number },
   *   basis: { e_x: [number, number, number], e_y: [number, number, number] },
   *   orientation?: { cos: number, sin: number } }>}
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
