<#
.SYNOPSIS
  Builds and launches the mundimonium renderer (Electron + Three.js).

.DESCRIPTION
  Runs the frontend's production build, then starts the Electron app.
  Electron's own main process spawns the persistent Python subprocess
  (see frontend/src/main/python-bridge.js) -- there is no separate
  Python step to run here.

  Requires `node`/`npm` on PATH.
#>

$ErrorActionPreference = 'Stop'

$frontendDir = Join-Path $PSScriptRoot 'frontend'

# A shell that inherits ELECTRON_RUN_AS_NODE=1 (e.g. VS Code's
# extension-host environment) launches Electron in plain-Node mode
# instead of as a real app, failing at the same spot every time
# regardless of what changed in the code. Clearing it here is harmless
# when it was never set.
if ($env:ELECTRON_RUN_AS_NODE) {
  Remove-Item Env:\ELECTRON_RUN_AS_NODE
}

Push-Location $frontendDir
try {
  Write-Host 'Building renderer bundle...'
  npm run build
  if ($LASTEXITCODE -ne 0) {
    throw "Vite build failed (exit code $LASTEXITCODE)."
  }

  Write-Host 'Launching Electron...'
  npm run start
} finally {
  Pop-Location
}
