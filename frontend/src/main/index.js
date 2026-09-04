const { app, BrowserWindow } = require('electron');
const path = require('node:path');

const { PythonBridge } = require('./python-bridge');
const { registerIpcHandlers } = require('./ipc-handlers');

let pythonBridge = null;

function createWindow() {
  const window = new BrowserWindow({
    width: 1280,
    height: 800,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      preload: path.join(__dirname, '../preload/index.js'),
    },
  });

  window.loadFile(path.join(__dirname, '../../dist/renderer/index.html'));
  return window;
}

app.whenReady().then(() => {
  pythonBridge = new PythonBridge();
  registerIpcHandlers(pythonBridge, () => BrowserWindow.getAllWindows());
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

function stopPythonBridge() {
  if (pythonBridge) {
    pythonBridge.stop();
    pythonBridge = null;
  }
}

app.on('window-all-closed', () => {
  stopPythonBridge();
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('before-quit', stopPythonBridge);
