"""Entry point: `python -m mundimonium.rendering`.

Runs the stdio message loop that Electron's main process talks to (see
`server.serve`).
"""
from mundimonium.rendering.server import serve

import sys


if __name__ == '__main__':
  serve(sys.stdin.buffer, sys.stdout.buffer)
