const { EventEmitter } = require('node:events');
const { spawn } = require('node:child_process');
const path = require('node:path');

const HEADER_LENGTH_BYTES = 4;

/**
 * Incrementally reassembles framed messages out of a stream of
 * arbitrarily-sized chunks. Wire format (mirrors
 * `mundimonium/rendering/protocol.py`): a 4-byte little-endian header
 * length, that many bytes of UTF-8 JSON, a 4-byte little-endian body
 * length, then that many raw bytes.
 *
 * `stdout` delivers chunks with no relationship to message boundaries --
 * never assume one `data` event is one frame.
 */
class FrameReader {
  /** @param {(header: object, body: Buffer) => void} onMessage */
  constructor(onMessage) {
    this._onMessage = onMessage;
    this._buffer = Buffer.alloc(0);
  }

  /** @param {Buffer} chunk */
  push(chunk) {
    this._buffer = Buffer.concat([this._buffer, chunk]);
    while (this._tryConsumeOne()) {
      // Keep draining as long as a full frame is already buffered --
      // one `data` event can easily contain more than one message.
    }
  }

  _tryConsumeOne() {
    if (this._buffer.length < HEADER_LENGTH_BYTES) {
      return false;
    }
    const headerLength = this._buffer.readUInt32LE(0);
    const headerStart = HEADER_LENGTH_BYTES;
    const headerEnd = headerStart + headerLength;
    const bodyLengthEnd = headerEnd + HEADER_LENGTH_BYTES;
    if (this._buffer.length < bodyLengthEnd) {
      return false;
    }
    const bodyLength = this._buffer.readUInt32LE(headerEnd);
    const bodyStart = bodyLengthEnd;
    const bodyEnd = bodyStart + bodyLength;
    if (this._buffer.length < bodyEnd) {
      return false;
    }

    const header = JSON.parse(
        this._buffer.subarray(headerStart, headerEnd).toString('utf-8'));
    const body = this._buffer.subarray(bodyStart, bodyEnd);
    this._buffer = this._buffer.subarray(bodyEnd);

    this._onMessage(header, body);
    return true;
  }
}

/**
 * Serializes one framed message (see `FrameReader`'s docstring for the
 * wire format).
 *
 * @param {object} header
 * @param {Buffer} [body]
 * @returns {Buffer}
 */
function encodeFrame(header, body = Buffer.alloc(0)) {
  const headerBuffer = Buffer.from(JSON.stringify(header), 'utf-8');
  const headerLength = Buffer.alloc(HEADER_LENGTH_BYTES);
  headerLength.writeUInt32LE(headerBuffer.length, 0);
  const bodyLength = Buffer.alloc(HEADER_LENGTH_BYTES);
  bodyLength.writeUInt32LE(body.length, 0);
  return Buffer.concat([headerLength, headerBuffer, bodyLength, body]);
}

// How to launch the bridge process, tried in order until one actually
// spawns. `pipenv` alone (works when it's on PATH, the common case on
// Linux/macOS) is tried first; the second candidate covers Windows, where
// pip's user-installed scripts often *aren't* on PATH by default, but the
// `py` launcher bundled with python.org installs reliably is, and can
// invoke pipenv as a module instead.
//
// `-3.14` mirrors the exact interpreter version the Pipfile requires
// (`[requires] python_version = "3.14"`) -- not an arbitrary pin, just
// matching it explicitly since `py` alone would otherwise pick whatever
// its own default is, which may not be the one pipenv's environment was
// actually created for.
const _PYTHON_SPAWN_CANDIDATES = [
  ['pipenv', ['run', 'python', '-m', 'mundimonium.rendering']],
  ['py', ['-3.14', '-m', 'pipenv', 'run', 'python', '-m', 'mundimonium.rendering']],
];

let nextRequestId = 1;

/**
 * Owns the persistent Python subprocess (`python -m mundimonium.rendering`,
 * via `pipenv run` so the project's own dependencies resolve): spawns it,
 * frames/unframes messages over its stdio, and dispatches responses either
 * to a pending request's Promise (request/response messages, matched by
 * `header.id`) or emits them as events (unsolicited push messages -- not
 * used by any Phase 1 request, but the seam later streaming needs, so it's
 * included now rather than retrofitted under pressure later).
 *
 * Also emits `'python:status'` with `{ state, detail? }`
 * (`state` one of `'starting' | 'ready' | 'crashed' | 'exited'`) for
 * process-lifecycle visibility.
 */
class PythonBridge extends EventEmitter {
  constructor() {
    super();
    this._pending = new Map(); // request id -> { resolve, reject }
    this._process = null;
    this._start();
  }

  _start() {
    this.emit('python:status', { state: 'starting' });
    this._trySpawn(path.resolve(__dirname, '..', '..', '..'), 0);
  }

  /**
   * Tries `_PYTHON_SPAWN_CANDIDATES[candidateIndex]`; on an ENOENT-style
   * failure (the command itself doesn't exist), tries the next one, so
   * platform differences in where `pipenv` lives don't need a manual
   * per-machine fix.
   */
  _trySpawn(repoRoot, candidateIndex) {
    if (candidateIndex >= _PYTHON_SPAWN_CANDIDATES.length) {
      const tried = _PYTHON_SPAWN_CANDIDATES.map(([command]) => command).join(', ');
      this.emit('python:status', {
        state: 'crashed',
        detail: `Could not launch the Python bridge (tried: ${tried}). See `
            + 'python-bridge.js\'s _PYTHON_SPAWN_CANDIDATES.',
      });
      return;
    }

    const [command, args] = _PYTHON_SPAWN_CANDIDATES[candidateIndex];
    const childProcess = spawn(
        command, args, { cwd: repoRoot, stdio: ['pipe', 'pipe', 'pipe'] });
    let spawnFailed = false;

    childProcess.once('error', (error) => {
      if (error.code === 'ENOENT') {
        spawnFailed = true;
        this._trySpawn(repoRoot, candidateIndex + 1);
        return;
      }
      this.emit('python:status', { state: 'crashed', detail: error.message });
    });

    childProcess.once('spawn', () => {
      if (spawnFailed) {
        return; // 'error' already fired and a fallback is being tried.
      }
      this._process = childProcess;
      this._wireProcess(childProcess);
      this.emit('python:status', { state: 'ready' });
    });
  }

  _wireProcess(childProcess) {
    const reader = new FrameReader(
        (header, body) => this._onMessage(header, body));
    childProcess.stdout.on('data', (chunk) => reader.push(chunk));

    // Python tracebacks land here (see server.py's dispatch: it prints
    // them to stderr before turning a failure into an `error` response),
    // so surface them rather than swallowing them silently.
    childProcess.stderr.on('data', (chunk) => {
      this.emit('python:status', {
        state: 'crashed', detail: chunk.toString('utf-8'),
      });
    });

    childProcess.on('exit', (code) => {
      this.emit('python:status', {
        state: 'exited', detail: `exit code ${code}`,
      });
      for (const { reject } of this._pending.values()) {
        reject(new Error('Python process exited before responding.'));
      }
      this._pending.clear();
    });
  }

  _onMessage(header, body) {
    const pending = this._pending.get(header.id);
    if (!pending) {
      // No request is waiting on this id -- an unsolicited push message.
      this.emit(header.type, header, body);
      return;
    }
    this._pending.delete(header.id);
    if (header.type === 'error') {
      pending.reject(new Error(header.message));
    } else {
      pending.resolve({ header, body });
    }
  }

  /**
   * Sends a request and resolves with the matching response.
   *
   * @param {object} header - Must include `type`; `id` is added here.
   * @returns {Promise<{ header: object, body: Buffer }>}
   */
  request(header) {
    const id = String(nextRequestId++);
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this._process.stdin.write(encodeFrame({ ...header, id }));
    });
  }

  stop() {
    if (this._process) {
      this._process.stdin.end();
      this._process = null;
    }
  }
}

module.exports = { PythonBridge, FrameReader, encodeFrame };
