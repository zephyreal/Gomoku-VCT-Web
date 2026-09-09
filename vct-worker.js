let modulePromise = null;
let moduleInstance = null;

function getModule() {
  if (!modulePromise) {
    importScripts('./gomoku-engine.js');
    modulePromise = createGomokuModule({
      locateFile: (path) => path.endsWith('.wasm') ? './gomoku-engine.wasm' : path
    }).then((m) => {
      moduleInstance = m;
      return m;
    });
  }
  return modulePromise;
}

self.onmessage = async (event) => {
  const data = event.data || {};

  if (data.type !== 'search') return;

  try {
    const Module = await getModule();
    const history = Array.isArray(data.history) ? data.history : [];
    const depth = Number.isFinite(data.depth) ? data.depth : 20;

    Module._vct_init();

    for (const move of history) {
      const ok = Module._vct_play(move.x | 0, move.y | 0);
      if (!ok) {
        throw new Error(`棋谱同步失败：(${move.x}, ${move.y})`);
      }
    }

    const started = performance.now();
    const score = Module._vct_search(depth | 0);
    const elapsedMs = performance.now() - started;

    self.postMessage({
      type: 'result',
      score,
      x: Module._vct_best_x(),
      y: Module._vct_best_y(),
      nodes: Module._vct_last_nodes(),
      ttHits: Module._vct_last_tt_hits(),
      elapsedMs
    });
  } catch (error) {
    self.postMessage({
      type: 'error',
      message: error && error.message ? error.message : String(error)
    });
  }
};
