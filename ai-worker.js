"use strict";

/*
  ai-worker.js
  ------------------------------------------------------------
  Production candidate:
      UINT8 QOperator Gen1
      ONNX Runtime Web 1.18.0
      WASM single-thread
      Immediate win / immediate block
      Adaptive MCTS 50 -> 100 -> 150

  Important:
  MCTS reuses the SAME tree between 50, 100 and 150.
  Child nodes do NOT store full boards; only one board copy is
  created per simulation. This keeps phone memory much lower.
*/

const SIZE = 15;
const CELLS = SIZE * SIZE;

const HUMAN = 1;
const AI = 2;

const ORT_BASE =
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/";

const MODEL_URL =
  "./gomoku_uint8_qop_u8u8.onnx?v=u8-qop-20260910-1";

const CPUCT = 0.8;

const EARLY_SHARE = 0.60;
const EARLY_RATIO = 2.00;
const MID_RATIO = 1.35;

let session = null;
let sessionPromise = null;

let inputName = null;
let valueOutputName = null;
let policyOutputName = null;


/* =========================================================
   ORT loading
   ========================================================= */

function ensureORT() {
  if (typeof ort !== "undefined") return;

  importScripts(ORT_BASE + "ort.min.js");

  ort.env.wasm.wasmPaths = ORT_BASE;
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
}

async function getSession() {
  if (session) return session;
  if (sessionPromise) return sessionPromise;

  ensureORT();

  sessionPromise = ort.InferenceSession.create(
    MODEL_URL,
    {
      executionProviders: ["wasm"],
      enableCpuMemArena: false,
      enableMemPattern: false
    }
  ).then((s) => {
    session = s;
    inputName =
      s.inputNames.includes("board")
        ? "board"
        : s.inputNames[0];

    detectOutputs(s);
    return s;
  });

  return sessionPromise;
}

function detectOutputs(s) {
  valueOutputName = null;
  policyOutputName = null;

  for (const name of s.outputNames) {
    const low = name.toLowerCase();

    if (low.includes("value")) {
      valueOutputName = name;
    }

    if (low.includes("policy")) {
      policyOutputName = name;
    }
  }

  if (!valueOutputName && s.outputNames.length >= 1) {
    valueOutputName = s.outputNames[0];
  }

  if (!policyOutputName && s.outputNames.length >= 2) {
    policyOutputName = s.outputNames[1];
  }
}


/* =========================================================
   Model input/output
   board representation:
     +1 = current player
     -1 = opponent
      0 = empty
   ========================================================= */

function boardToTensor(relativeBoard) {
  const data = new Float32Array(3 * CELLS);

  for (let i = 0; i < CELLS; i++) {
    const p = relativeBoard[i];

    if (p === 1) {
      data[i] = 1.0;
    } else if (p === -1) {
      data[CELLS + i] = 1.0;
    } else {
      data[2 * CELLS + i] = 1.0;
    }
  }

  return new ort.Tensor(
    "float32",
    data,
    [1, 3, SIZE, SIZE]
  );
}

async function predictRelative(relativeBoard) {
  const s = await getSession();

  const tensor = boardToTensor(relativeBoard);
  const feeds = {};
  feeds[inputName] = tensor;

  const results = await s.run(feeds);

  let valueTensor =
    results[valueOutputName];

  let policyTensor =
    results[policyOutputName];

  if (!valueTensor || !policyTensor) {
    // Defensive fallback by tensor size.
    for (const name of s.outputNames) {
      const t = results[name];
      if (!t || !t.data) continue;

      if (t.data.length === 1) {
        valueTensor = t;
      } else if (t.data.length === CELLS) {
        policyTensor = t;
      }
    }
  }

  if (!valueTensor || !policyTensor) {
    throw new Error(
      "找不到 value / policy 输出：" +
      s.outputNames.join(", ")
    );
  }

  const value = Number(valueTensor.data[0]);
  const raw = policyTensor.data;

  const p = new Float64Array(CELLS);
  let sum = 0;

  /*
    The exported Gen1 model normally outputs probabilities.
    We still sanitize and renormalize over legal moves.
  */
  for (let i = 0; i < CELLS; i++) {
    if (relativeBoard[i] === 0) {
      const q = Number(raw[i]);
      if (Number.isFinite(q) && q > 0) {
        p[i] = q;
        sum += q;
      }
    }
  }

  if (!(sum > 0)) {
    let legalCount = 0;

    for (let i = 0; i < CELLS; i++) {
      if (relativeBoard[i] === 0) {
        legalCount++;
      }
    }

    const q = legalCount > 0 ? 1 / legalCount : 0;

    for (let i = 0; i < CELLS; i++) {
      if (relativeBoard[i] === 0) {
        p[i] = q;
      }
    }

  } else {
    const inv = 1 / sum;

    for (let i = 0; i < CELLS; i++) {
      p[i] *= inv;
    }
  }

  return { value, policy: p };
}


/* =========================================================
   Board utilities
   ========================================================= */

const DIRS = [
  [0, 1],
  [1, 0],
  [1, 1],
  [1, -1]
];

function hasFiveAt(board, idx, stone) {
  const r = Math.floor(idx / SIZE);
  const c = idx % SIZE;

  for (const [dr, dc] of DIRS) {
    let n = 1;

    for (let s = 1; s <= 4; s++) {
      const rr = r + dr * s;
      const cc = c + dc * s;

      if (
        rr >= 0 && rr < SIZE &&
        cc >= 0 && cc < SIZE &&
        board[rr * SIZE + cc] === stone
      ) {
        n++;
      } else {
        break;
      }
    }

    for (let s = 1; s <= 4; s++) {
      const rr = r - dr * s;
      const cc = c - dc * s;

      if (
        rr >= 0 && rr < SIZE &&
        cc >= 0 && cc < SIZE &&
        board[rr * SIZE + cc] === stone
      ) {
        n++;
      } else {
        break;
      }
    }

    if (n >= 5) return true;
  }

  return false;
}

function boardFull(board) {
  for (let i = 0; i < CELLS; i++) {
    if (board[i] === 0) return false;
  }
  return true;
}

function winningMove(board, stone) {
  for (let i = 0; i < CELLS; i++) {
    if (board[i] !== 0) continue;

    board[i] = stone;
    const win = hasFiveAt(board, i, stone);
    board[i] = 0;

    if (win) return i;
  }

  return -1;
}

function immediateRule(relativeBoard) {
  const own = winningMove(relativeBoard, 1);
  if (own >= 0) {
    return { move: own, reason: "一步必赢" };
  }

  const block = winningMove(relativeBoard, -1);
  if (block >= 0) {
    return { move: block, reason: "一步必堵" };
  }

  return null;
}

function absoluteToAIRelative(absBoard) {
  const b = new Int8Array(CELLS);

  for (let i = 0; i < CELLS; i++) {
    if (absBoard[i] === AI) {
      b[i] = 1;
    } else if (absBoard[i] === HUMAN) {
      b[i] = -1;
    } else {
      b[i] = 0;
    }
  }

  return b;
}

function perspectiveBoard(rootBoard, depth) {
  if ((depth & 1) === 0) {
    return rootBoard;
  }

  const b = new Int8Array(CELLS);

  for (let i = 0; i < CELLS; i++) {
    b[i] = -rootBoard[i];
  }

  return b;
}


/* =========================================================
   Memory-light MCTS
   ========================================================= */

class Node {
  constructor(
    move = -1,
    prior = 1.0,
    parent = null,
    depth = 0
  ) {
    this.move = move;
    this.prior = prior;
    this.parent = parent;
    this.depth = depth;

    this.children = null;

    this.N = 0;
    this.W = 0.0;
    this.Q = 0.0;

    this.terminal = null;
  }
}

class SearchTree {
  constructor(modelRootBoard) {
    this.rootBoard = modelRootBoard;
    this.root = new Node();

    this.simulations = 0;
    this.nnEvals = 0;
    this.rootValue = null;
  }

  selectChild(node) {
    const sqrtTotal = Math.sqrt(node.N + 1);

    let best = null;
    let bestScore = -Infinity;

    for (const child of node.children) {
      const u =
        CPUCT *
        child.prior *
        sqrtTotal /
        (1 + child.N);

      // child.Q is from child-to-move perspective.
      const score = u - child.Q;

      if (score > bestScore) {
        bestScore = score;
        best = child;
      }
    }

    return best;
  }

  expand(node, rootPerspectiveBoard, policy) {
    const children = [];

    for (let i = 0; i < CELLS; i++) {
      if (rootPerspectiveBoard[i] === 0) {
        children.push(
          new Node(
            i,
            Number(policy[i]) || 0,
            node,
            node.depth + 1
          )
        );
      }
    }

    node.children = children;
  }

  backup(path, value) {
    let v = value;

    for (let i = path.length - 1; i >= 0; i--) {
      const node = path[i];

      node.N++;
      node.W += v;
      node.Q = node.W / node.N;

      v = -v;
    }
  }

  async runOne() {
    /*
      b is always stored in ROOT player's perspective:
        +1 root player
        -1 root opponent

      At depth d:
        player to move = +1 if d even, -1 if d odd.
    */
    const b = this.rootBoard.slice();

    let node = this.root;
    const path = [node];

    while (node.children && node.children.length > 0) {
      const child = this.selectChild(node);

      const stone =
        (node.depth & 1) === 0
          ? 1
          : -1;

      b[child.move] = stone;

      node = child;
      path.push(node);

      if (node.terminal !== null) {
        this.backup(path, node.terminal);
        this.simulations++;
        return;
      }

      if (hasFiveAt(b, node.move, stone)) {
        /*
          The player who just moved wins.
          Node is now opponent-to-move, so node value = -1.
        */
        node.terminal = -1.0;
        this.backup(path, -1.0);
        this.simulations++;
        return;
      }

      if (boardFull(b)) {
        node.terminal = 0.0;
        this.backup(path, 0.0);
        this.simulations++;
        return;
      }
    }

    const relative =
      perspectiveBoard(b, node.depth);

    const pred =
      await predictRelative(relative);

    this.nnEvals++;

    if (node === this.root && this.rootValue === null) {
      this.rootValue = pred.value;
    }

    this.expand(node, b, pred.policy);
    this.backup(path, pred.value);

    this.simulations++;
  }

  async runUntil(target, searchId) {
    while (this.simulations < target) {
      await this.runOne();

      /*
        Yield to Worker event loop occasionally.
        Keeps long mobile searches cooperative without
        restarting the MCTS tree.
      */
      if ((this.simulations & 15) === 0) {
        await Promise.resolve();
      }
    }

    self.postMessage({
      type: "progress",
      searchId,
      simulations: this.simulations
    });
  }

  rootStats() {
    const children = this.root.children || [];

    if (children.length === 0) {
      return null;
    }

    let first = null;
    let second = null;

    for (const child of children) {
      if (
        first === null ||
        child.N > first.N ||
        (child.N === first.N &&
         child.prior > first.prior)
      ) {
        second = first;
        first = child;

      } else if (
        second === null ||
        child.N > second.N ||
        (child.N === second.N &&
         child.prior > second.prior)
      ) {
        second = child;
      }
    }

    let totalVisits = 0;
    for (const child of children) {
      totalVisits += child.N;
    }

    const n1 = first ? first.N : 0;
    const n2 = second ? second.N : 0;

    return {
      bestMove: first.move,
      n1,
      n2,
      share: n1 / Math.max(totalVisits, 1),
      ratio: (n1 + 1) / (n2 + 1),
      q1: first ? -first.Q : 0,
      q2: second ? -second.Q : 0
    };
  }
}


/* =========================================================
   Adaptive decision
   ========================================================= */

async function adaptiveSearch(relativeBoard, searchId) {
  const started = performance.now();

  const rule = immediateRule(relativeBoard);

  if (rule) {
    return {
      move: rule.move,
      simulations: 0,
      nnEvals: 0,
      rootValue: null,
      elapsedMs: performance.now() - started,
      stopReason: rule.reason,
      source: "rule"
    };
  }

  const tree = new SearchTree(relativeBoard);

  // Stage 1: 50
  await tree.runUntil(50, searchId);

  const s50 = tree.rootStats();

  if (
    s50.share >= EARLY_SHARE &&
    s50.ratio >= EARLY_RATIO
  ) {
    return {
      move: s50.bestMove,
      simulations: 50,
      nnEvals: tree.nnEvals,
      rootValue: tree.rootValue,
      elapsedMs: performance.now() - started,
      stopReason:
        "50次提前停止 · share=" +
        s50.share.toFixed(2) +
        " · ratio=" +
        s50.ratio.toFixed(2),
      source: "adaptive-mcts"
    };
  }

  const move50 = s50.bestMove;

  // Stage 2: continue SAME tree to 100
  await tree.runUntil(100, searchId);

  const s100 = tree.rootStats();
  const stable =
    s100.bestMove === move50;

  if (
    stable &&
    s100.ratio >= MID_RATIO
  ) {
    return {
      move: s100.bestMove,
      simulations: 100,
      nnEvals: tree.nnEvals,
      rootValue: tree.rootValue,
      elapsedMs: performance.now() - started,
      stopReason:
        "100次稳定停止 · ratio=" +
        s100.ratio.toFixed(2),
      source: "adaptive-mcts"
    };
  }

  // Stage 3: continue SAME tree to 150
  await tree.runUntil(150, searchId);

  const s150 = tree.rootStats();

  return {
    move: s150.bestMove,
    simulations: 150,
    nnEvals: tree.nnEvals,
    rootValue: tree.rootValue,
    elapsedMs: performance.now() - started,
    stopReason: "困难局面 · 搜索到150次",
    source: "adaptive-mcts"
  };
}


/* =========================================================
   Worker messages
   ========================================================= */

self.onmessage = async (event) => {
  const data = event.data || {};

  try {
    if (data.type === "init") {
      const s = await getSession();

      self.postMessage({
        type: "ready",
        model: "gomoku_uint8_qop_u8u8.onnx (~147 KB)",
        ortVersion:
          (typeof ort !== "undefined" && ort.env)
            ? "1.18.0"
            : "1.18.0",
        inputNames: s.inputNames,
        outputNames: s.outputNames
      });

      return;
    }

    if (data.type !== "search") {
      return;
    }

    const searchId = data.searchId;

    if (!data.board) {
      throw new Error("search 消息没有棋盘数据");
    }

    const absBoard =
      data.board instanceof Int8Array
        ? data.board
        : new Int8Array(data.board);

    if (absBoard.length !== CELLS) {
      throw new Error(
        "棋盘长度错误：" +
        absBoard.length +
        "，应为 " +
        CELLS
      );
    }

    await getSession();

    const relative =
      absoluteToAIRelative(absBoard);

    const result =
      await adaptiveSearch(relative, searchId);

    const x = result.move % SIZE;
    const y = Math.floor(result.move / SIZE);

    self.postMessage({
      type: "result",
      searchId,
      x,
      y,
      simulations: result.simulations,
      nnEvals: result.nnEvals,
      rootValue: result.rootValue,
      elapsedMs: result.elapsedMs,
      stopReason: result.stopReason,
      source: result.source
    });

  } catch (error) {
    self.postMessage({
      type: "error",
      searchId: data.searchId,
      message:
        error && error.message
          ? error.message
          : String(error)
    });
  }
};
