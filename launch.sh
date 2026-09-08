#!/usr/bin/env bash
# Builds and launches the mundimonium renderer (Electron + Three.js).
#
# Runs the frontend's production build, then starts the Electron app.
# Electron's own main process spawns the persistent Python subprocess
# (see frontend/src/main/python-bridge.js) -- there is no separate
# Python step to run here.
#
# Requires `node`/`npm` on PATH.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
frontend_dir="$script_dir/frontend"

# A shell that inherits ELECTRON_RUN_AS_NODE=1 (e.g. VS Code's
# extension-host environment) launches Electron in plain-Node mode
# instead of as a real app. Clearing it here is harmless when it was
# never set.
unset ELECTRON_RUN_AS_NODE

cd "$frontend_dir"

echo "Building renderer bundle..."
npm run build

echo "Launching Electron..."
npm run start
