"use strict";

/*
  ai-worker.js — Four Difficulty Edition
  ------------------------------------------------------------
  Model:
      UINT8 QOperator Gen1 (~147 KB)
      ONNX Runtime Web 1.18.0
      WASM single-thread

  Difficulty:
      rookie   菜鸟  : Policy Top-K stochastic, almost no search
      beginner 新手  : Rule + MCTS20
      expert   专家  : Adaptive MCTS 30 -> 60 -> 100
      master   大师  : Adaptive MCTS 50 -> 100 -> 150

  The MASTER mode is the previously validated strong engine.
*/

const SIZE = 15;
const CELLS = SIZE * SIZE;

const HUMAN = 1;
const AI = 2;

const ORT_BASE =
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/";

const MODEL_URL =
  "./gomoku_uint8_qop_u8u8.onnx?v=u8-qop-difficulty-20260911-1";

const CPUCT = 0.8;

const DIFFICULTIES = {
  rookie: {
    label: "菜鸟",
    type: "policy-random-top2",
    topK: 2,
    blockChance: 0.55
  },

  beginner: {
    label: "新手",
    type: "policy-top1"
  },

  expert: {
    label: "专家",
    type: "fixed",
    simulations: 20,
    blockChance: 1.0
  },

  master: {
    label: "大师",
    type: "adaptive",
    stages: [50, 100, 150],
    earlyShare: 0.60,
    earlyRatio: 2.00,
    midRatio: 1.35,
    blockChance: 1.0
  }
};

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
   Model inference
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

  let valueTensor = results[valueOutputName];
  let policyTensor = results[policyOutputName];

  if (!valueTensor || !policyTensor) {
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

  const policy = new Float64Array(CELLS);
  let sum = 0;

  for (let i = 0; i < CELLS; i++) {
    if (relativeBoard[i] === 0) {
      const q = Number(raw[i]);

      if (Number.isFinite(q) && q > 0) {
        policy[i] = q;
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

    const q = legalCount > 0
      ? 1 / legalCount
      : 0;

    for (let i = 0; i < CELLS; i++) {
      if (relativeBoard[i] === 0) {
        policy[i] = q;
      }
    }

  } else {
    const inv = 1 / sum;

    for (let i = 0; i < CELLS; i++) {
      policy[i] *= inv;
    }
  }

  return { value, policy };
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
    if (board[i] === 0) {
      return false;
    }
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

function immediateRule(relativeBoard, blockChance = 1.0) {
  const own = winningMove(relativeBoard, 1);

  if (own >= 0) {
    return {
      move: own,
      reason: "一步必赢"
    };
  }

  const block = winningMove(relativeBoard, -1);

  if (
    block >= 0 &&
    Math.random() <= blockChance
  ) {
    return {
      move: block,
      reason: "一步必堵"
    };
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
   Rookie: stochastic Policy Top-K
   ========================================================= */

function sampleTopK(policy, board, topK, temperature) {
  const items = [];

  for (let i = 0; i < CELLS; i++) {
    if (board[i] !== 0) continue;

    items.push({
      move: i,
      p: Number(policy[i]) || 0
    });
  }

  items.sort((a, b) => b.p - a.p);

  const candidates =
    items.slice(0, Math.min(topK, items.length));

  if (candidates.length === 0) {
    return -1;
  }

  const invT =
    1 / Math.max(temperature, 0.05);

  let total = 0;

  for (const item of candidates) {
    item.w =
      Math.pow(
        Math.max(item.p, 1e-8),
        invT
      );

    total += item.w;
  }

  let r = Math.random() * total;

  for (const item of candidates) {
    r -= item.w;

    if (r <= 0) {
      return item.move;
    }
  }

  return candidates[candidates.length - 1].move;
}

async function rookieSearch(relativeBoard, cfg) {
  const started = performance.now();

  // 菜鸟：若自己一步可胜，仍会下出来；
  // 对手一步可胜时只有约55%概率会正确堵住。
  const rule =
    immediateRule(
      relativeBoard,
      cfg.blockChance
    );

  if (rule) {
    return {
      move: rule.move,
      simulations: 0,
      nnEvals: 0,
      rootValue: null,
      elapsedMs:
        performance.now() - started,
      stopReason: rule.reason,
      source: "rule"
    };
  }

  const pred =
    await predictRelative(relativeBoard);

  const legal = [];

  for (let i = 0; i < CELLS; i++) {
    if (relativeBoard[i] === 0) {
      legal.push({
        move: i,
        p: Number(pred.policy[i]) || 0
      });
    }
  }

  legal.sort((a, b) => b.p - a.p);

  const top =
    legal.slice(0, Math.min(2, legal.length));

  if (top.length === 0) {
    throw new Error("菜鸟模式没有合法落子");
  }

  const chosen =
    top.length === 1
      ? top[0]
      : top[Math.random() < 0.5 ? 0 : 1];

  return {
    move: chosen.move,
    simulations: 1,
    nnEvals: 1,
    rootValue: pred.value,
    elapsedMs:
      performance.now() - started,
    stopReason: "菜鸟 · Policy 前2名随机",
    source: "policy-rookie"
  };
}


async function beginnerSearch(relativeBoard) {
  const started = performance.now();

  const pred =
    await predictRelative(relativeBoard);

  let bestMove = -1;
  let bestProb = -Infinity;

  for (let i = 0; i < CELLS; i++) {
    if (relativeBoard[i] !== 0) {
      continue;
    }

    const p =
      Number(pred.policy[i]) || 0;

    if (p > bestProb) {
      bestProb = p;
      bestMove = i;
    }
  }

  if (bestMove < 0) {
    throw new Error("新手模式没有合法落子");
  }

  return {
    move: bestMove,
    simulations: 1,
    nnEvals: 1,
    rootValue: pred.value,
    elapsedMs:
      performance.now() - started,
    stopReason: "新手 · Policy 第一名",
    source: "policy-top1"
  };
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
    const sqrtTotal =
      Math.sqrt(node.N + 1);

    let best = null;
    let bestScore = -Infinity;

    for (const child of node.children) {
      const u =
        CPUCT *
        child.prior *
        sqrtTotal /
        (1 + child.N);

      const score =
        u - child.Q;

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
      node.Q =
        node.W / node.N;

      v = -v;
    }
  }

  async runOne() {
    const b =
      this.rootBoard.slice();

    let node =
      this.root;

    const path = [node];

    while (
      node.children &&
      node.children.length > 0
    ) {
      const child =
        this.selectChild(node);

      const stone =
        (node.depth & 1) === 0
          ? 1
          : -1;

      b[child.move] = stone;

      node = child;
      path.push(node);

      if (node.terminal !== null) {
        this.backup(
          path,
          node.terminal
        );

        this.simulations++;
        return;
      }

      if (
        hasFiveAt(
          b,
          node.move,
          stone
        )
      ) {
        node.terminal = -1.0;

        this.backup(
          path,
          -1.0
        );

        this.simulations++;
        return;
      }

      if (boardFull(b)) {
        node.terminal = 0.0;

        this.backup(
          path,
          0.0
        );

        this.simulations++;
        return;
      }
    }

    const relative =
      perspectiveBoard(
        b,
        node.depth
      );

    const pred =
      await predictRelative(relative);

    this.nnEvals++;

    if (
      node === this.root &&
      this.rootValue === null
    ) {
      this.rootValue =
        pred.value;
    }

    this.expand(
      node,
      b,
      pred.policy
    );

    this.backup(
      path,
      pred.value
    );

    this.simulations++;
  }

  async runUntil(target, searchId) {
    while (
      this.simulations < target
    ) {
      await this.runOne();

      if (
        (this.simulations & 15) === 0
      ) {
        await Promise.resolve();
      }
    }

    self.postMessage({
      type: "progress",
      searchId,
      simulations:
        this.simulations
    });
  }

  rootStats() {
    const children =
      this.root.children || [];

    if (children.length === 0) {
      return null;
    }

    let first = null;
    let second = null;

    for (const child of children) {
      if (
        first === null ||
        child.N > first.N ||
        (
          child.N === first.N &&
          child.prior > first.prior
        )
      ) {
        second = first;
        first = child;

      } else if (
        second === null ||
        child.N > second.N ||
        (
          child.N === second.N &&
          child.prior > second.prior
        )
      ) {
        second = child;
      }
    }

    let totalVisits = 0;

    for (const child of children) {
      totalVisits += child.N;
    }

    const n1 =
      first ? first.N : 0;

    const n2 =
      second ? second.N : 0;

    return {
      bestMove: first.move,
      n1,
      n2,
      share:
        n1 /
        Math.max(totalVisits, 1),
      ratio:
        (n1 + 1) /
        (n2 + 1)
    };
  }
}


/* =========================================================
   Beginner / Expert / Master
   ========================================================= */

async function fixedSearch(
  relativeBoard,
  searchId,
  cfg
) {
  const started =
    performance.now();

  const rule =
    immediateRule(
      relativeBoard,
      cfg.blockChance
    );

  if (rule) {
    return {
      move: rule.move,
      simulations: 0,
      nnEvals: 0,
      rootValue: null,
      elapsedMs:
        performance.now() - started,
      stopReason: rule.reason,
      source: "rule"
    };
  }

  const tree =
    new SearchTree(relativeBoard);

  await tree.runUntil(
    cfg.simulations,
    searchId
  );

  const stats =
    tree.rootStats();

  return {
    move: stats.bestMove,
    simulations:
      tree.simulations,
    nnEvals:
      tree.nnEvals,
    rootValue:
      tree.rootValue,
    elapsedMs:
      performance.now() - started,
    stopReason:
      cfg.label +
      " · MCTS" +
      cfg.simulations,
    source: "mcts-fixed"
  };
}

async function adaptiveSearch(
  relativeBoard,
  searchId,
  cfg
) {
  const started =
    performance.now();

  const rule =
    immediateRule(
      relativeBoard,
      cfg.blockChance
    );

  if (rule) {
    return {
      move: rule.move,
      simulations: 0,
      nnEvals: 0,
      rootValue: null,
      elapsedMs:
        performance.now() - started,
      stopReason: rule.reason,
      source: "rule"
    };
  }

  const tree =
    new SearchTree(relativeBoard);

  const s1Target =
    cfg.stages[0];

  const s2Target =
    cfg.stages[1];

  const s3Target =
    cfg.stages[2];

  await tree.runUntil(
    s1Target,
    searchId
  );

  const s1 =
    tree.rootStats();

  if (
    s1.share >= cfg.earlyShare &&
    s1.ratio >= cfg.earlyRatio
  ) {
    return {
      move: s1.bestMove,
      simulations:
        tree.simulations,
      nnEvals:
        tree.nnEvals,
      rootValue:
        tree.rootValue,
      elapsedMs:
        performance.now() - started,
      stopReason:
        cfg.label +
        " · " +
        s1Target +
        "次提前停止",
      source: "adaptive-mcts"
    };
  }

  const move1 =
    s1.bestMove;

  await tree.runUntil(
    s2Target,
    searchId
  );

  const s2 =
    tree.rootStats();

  const stable =
    s2.bestMove === move1;

  if (
    stable &&
    s2.ratio >= cfg.midRatio
  ) {
    return {
      move: s2.bestMove,
      simulations:
        tree.simulations,
      nnEvals:
        tree.nnEvals,
      rootValue:
        tree.rootValue,
      elapsedMs:
        performance.now() - started,
      stopReason:
        cfg.label +
        " · " +
        s2Target +
        "次稳定停止",
      source: "adaptive-mcts"
    };
  }

  await tree.runUntil(
    s3Target,
    searchId
  );

  const s3 =
    tree.rootStats();

  return {
    move: s3.bestMove,
    simulations:
      tree.simulations,
    nnEvals:
      tree.nnEvals,
    rootValue:
      tree.rootValue,
    elapsedMs:
      performance.now() - started,
    stopReason:
      cfg.label +
      " · 困难局面搜索至" +
      s3Target +
      "次",
    source: "adaptive-mcts"
  };
}


/* =========================================================
   Difficulty router
   ========================================================= */

async function searchByDifficulty(
  relativeBoard,
  searchId,
  difficulty
) {
  const cfg =
    DIFFICULTIES[difficulty] ||
    DIFFICULTIES.beginner;

  if (cfg.type === "policy-random-top2") {
    return rookieSearch(
      relativeBoard,
      cfg
    );
  }

  if (cfg.type === "policy-top1") {
    return beginnerSearch(
      relativeBoard
    );
  }

  if (cfg.type === "fixed") {
    return fixedSearch(
      relativeBoard,
      searchId,
      cfg
    );
  }

  return adaptiveSearch(
    relativeBoard,
    searchId,
    cfg
  );
}


/* =========================================================
   Worker messages
   ========================================================= */

self.onmessage = async (event) => {
  const data =
    event.data || {};

  try {
    if (data.type === "init") {
      const s =
        await getSession();

      self.postMessage({
        type: "ready",
        model:
          "gomoku_uint8_qop_u8u8.onnx (~147 KB)",
        ortVersion: "1.18.0",
        inputNames:
          s.inputNames,
        outputNames:
          s.outputNames
      });

      return;
    }

    if (data.type !== "search") {
      return;
    }

    const searchId =
      data.searchId;

    if (!data.board) {
      throw new Error(
        "search 消息没有棋盘数据"
      );
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
      absoluteToAIRelative(
        absBoard
      );

    const difficulty =
      DIFFICULTIES[data.difficulty]
        ? data.difficulty
        : "beginner";

    const cfg =
      DIFFICULTIES[difficulty];

    const result =
      await searchByDifficulty(
        relative,
        searchId,
        difficulty
      );

    const x =
      result.move % SIZE;

    const y =
      Math.floor(
        result.move / SIZE
      );

    self.postMessage({
      type: "result",
      searchId,
      x,
      y,
      simulations:
        result.simulations,
      nnEvals:
        result.nnEvals,
      rootValue:
        result.rootValue,
      elapsedMs:
        result.elapsedMs,
      stopReason:
        result.stopReason,
      source:
        result.source,
      difficulty,
      difficultyLabel:
        cfg.label
    });

  } catch (error) {
    self.postMessage({
      type: "error",
      searchId:
        data.searchId,
      message:
        error && error.message
          ? error.message
          : String(error)
    });
  }
};
