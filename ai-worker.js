"use strict";

/*
  ai-worker.js — Five Realms Edition
  ------------------------------------------------------------
  练气 / 筑基 / 结丹 / 元婴 / 化神

  化神内部：
      一步必赢 / 必堵
      + 有界 VCT/VCF 威胁搜索
      + Adaptive MCTS 50 -> 100 -> 150 回退

  练气 / 筑基 / 结丹 / 元婴均可按需调用“化神指导”。
  玩家界面不显示这些底层策略。
*/

const SIZE = 15;
const CELLS = SIZE * SIZE;

const HUMAN = 1;
const AI = 2;

const ORT_BASE =
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/";

const MODEL_URL =
  "./gomoku_uint8_qop_u8u8.onnx?v=five-realms-vct-20260911-1";

const CPUCT = 0.8;

const DIFFICULTIES = {
  rookie: {
    label: "练气",
    type: "policy-random-topk",
    topK: 3
  },

  beginner: {
    label: "筑基",
    type: "policy-top1"
  },

  expert: {
    label: "结丹",
    type: "fixed",
    simulations: 20,
    blockChance: 1.0
  },

  master: {
    label: "元婴",
    type: "adaptive",
    stages: [50, 100, 150],
    earlyShare: 0.60,
    earlyRatio: 2.00,
    midRatio: 1.35,
    blockChance: 1.0
  },

  deity: {
    label: "化神",
    type: "vct-adaptive",
    stages: [50, 100, 150],
    earlyShare: 0.60,
    earlyRatio: 2.00,
    midRatio: 1.35,
    blockChance: 1.0,
    vctMaxAttacks: 5,
    vctNodeLimit: 5000,
    vctTimeLimitMs: 90,
    vctMaxAttackCandidates: 18,
    vctMaxDefenseCandidates: 18
  }
};

let session = null;
let sessionPromise = null;

let inputName = null;
let valueOutputName = null;
let policyOutputName = null;

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
    inputName = s.inputNames.includes("board") ? "board" : s.inputNames[0];
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
    if (low.includes("value")) valueOutputName = name;
    if (low.includes("policy")) policyOutputName = name;
  }

  if (!valueOutputName && s.outputNames.length >= 1) valueOutputName = s.outputNames[0];
  if (!policyOutputName && s.outputNames.length >= 2) policyOutputName = s.outputNames[1];
}

function boardToTensor(relativeBoard) {
  const data = new Float32Array(3 * CELLS);

  for (let i = 0; i < CELLS; i++) {
    const p = relativeBoard[i];
    if (p === 1) data[i] = 1.0;
    else if (p === -1) data[CELLS + i] = 1.0;
    else data[2 * CELLS + i] = 1.0;
  }

  return new ort.Tensor("float32", data, [1, 3, SIZE, SIZE]);
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
      if (t.data.length === 1) valueTensor = t;
      else if (t.data.length === CELLS) policyTensor = t;
    }
  }

  if (!valueTensor || !policyTensor) {
    throw new Error("找不到 value / policy 输出：" + s.outputNames.join(", "));
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
      if (relativeBoard[i] === 0) legalCount++;
    }
    const q = legalCount > 0 ? 1 / legalCount : 0;
    for (let i = 0; i < CELLS; i++) {
      if (relativeBoard[i] === 0) policy[i] = q;
    }
  } else {
    const inv = 1 / sum;
    for (let i = 0; i < CELLS; i++) policy[i] *= inv;
  }

  return { value, policy };
}

const DIRS = [[0, 1], [1, 0], [1, 1], [1, -1]];

function hasFiveAt(board, idx, stone) {
  const r = Math.floor(idx / SIZE);
  const c = idx % SIZE;

  for (const [dr, dc] of DIRS) {
    let n = 1;

    for (let s = 1; s <= 4; s++) {
      const rr = r + dr * s;
      const cc = c + dc * s;
      if (rr >= 0 && rr < SIZE && cc >= 0 && cc < SIZE && board[rr * SIZE + cc] === stone) n++;
      else break;
    }

    for (let s = 1; s <= 4; s++) {
      const rr = r - dr * s;
      const cc = c - dc * s;
      if (rr >= 0 && rr < SIZE && cc >= 0 && cc < SIZE && board[rr * SIZE + cc] === stone) n++;
      else break;
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

function immediateRule(relativeBoard, blockChance = 1.0) {
  const own = winningMove(relativeBoard, 1);
  if (own >= 0) {
    return { move: own, reason: "一步必赢" };
  }

  const block = winningMove(relativeBoard, -1);
  if (block >= 0 && Math.random() <= blockChance) {
    return { move: block, reason: "一步必堵" };
  }

  return null;
}

function absoluteToAIRelative(absBoard) {
  const b = new Int8Array(CELLS);
  for (let i = 0; i < CELLS; i++) {
    if (absBoard[i] === AI) b[i] = 1;
    else if (absBoard[i] === HUMAN) b[i] = -1;
    else b[i] = 0;
  }
  return b;
}

function absoluteToHumanRelative(absBoard) {
  const b =
    new Int8Array(CELLS);

  for (let i = 0; i < CELLS; i++) {
    if (absBoard[i] === HUMAN) {
      b[i] = 1;
    } else if (absBoard[i] === AI) {
      b[i] = -1;
    } else {
      b[i] = 0;
    }
  }

  return b;
}

function perspectiveBoard(rootBoard, depth) {
  if ((depth & 1) === 0) return rootBoard;

  const b = new Int8Array(CELLS);
  for (let i = 0; i < CELLS; i++) b[i] = -rootBoard[i];
  return b;
}

function pickRandomFromTopK(policy, board, topK) {
  const items = [];
  for (let i = 0; i < CELLS; i++) {
    if (board[i] !== 0) continue;
    items.push({ move: i, p: Number(policy[i]) || 0 });
  }

  items.sort((a, b) => b.p - a.p);
  const candidates = items.slice(0, Math.min(topK, items.length));
  if (candidates.length === 0) return -1;

  const idx = Math.floor(Math.random() * candidates.length);
  return candidates[idx].move;
}

async function topKPolicySearch(relativeBoard, cfg) {
  const started = performance.now();
  const pred = await predictRelative(relativeBoard);
  const move = pickRandomFromTopK(pred.policy, relativeBoard, cfg.topK);

  if (move < 0) throw new Error(cfg.label + " 模式没有合法落子");

  return {
    move,
    simulations: 1,
    nnEvals: 1,
    rootValue: pred.value,
    elapsedMs: performance.now() - started,
    stopReason: cfg.label + "完成落子",
    source: "policy-topk"
  };
}

async function top1PolicySearch(relativeBoard, cfg) {
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
    throw new Error(
      cfg.label + " 模式没有合法落子"
    );
  }

  return {
    move: bestMove,
    simulations: 1,
    nnEvals: 1,
    rootValue: pred.value,
    elapsedMs:
      performance.now() - started,
    stopReason: cfg.label + "完成落子",
    source: "policy-top1"
  };
}


/* =========================================================
   安全有界 VCT / VCF 威胁搜索

   说明：
   - 不调用旧 C++ vct-worker.js。
   - 只在 Web Worker 中运行，不阻塞 UI。
   - 有节点数与时间上限，避免手机端长时间卡住。
   - 强项是连续冲四、双四，以及部分连续活三威胁。
   - 找不到“已证明的强制线路”时，不强行判断，
     直接回退到原来的 Adaptive MCTS。
   ========================================================= */

function listWinningMoves(board, stone, maxCount = 3) {
  const wins = [];
  const candidates =
    nearbyEmptyMoves(board, 1);

  for (const i of candidates) {
    if (board[i] !== 0) {
      continue;
    }

    board[i] = stone;

    if (hasFiveAt(board, i, stone)) {
      wins.push(i);
    }

    board[i] = 0;

    if (wins.length >= maxCount) {
      break;
    }
  }

  return wins;
}

function nearbyEmptyMoves(board, radius = 2) {
  const mark = new Uint8Array(CELLS);
  let stoneCount = 0;

  for (let i = 0; i < CELLS; i++) {
    if (board[i] === 0) {
      continue;
    }

    stoneCount++;

    const r =
      Math.floor(i / SIZE);

    const c =
      i % SIZE;

    for (let dr = -radius; dr <= radius; dr++) {
      for (let dc = -radius; dc <= radius; dc++) {
        const rr = r + dr;
        const cc = c + dc;

        if (
          rr < 0 || rr >= SIZE ||
          cc < 0 || cc >= SIZE
        ) {
          continue;
        }

        const j =
          rr * SIZE + cc;

        if (board[j] === 0) {
          mark[j] = 1;
        }
      }
    }
  }

  if (stoneCount === 0) {
    return [
      Math.floor(SIZE / 2) * SIZE +
      Math.floor(SIZE / 2)
    ];
  }

  const moves = [];

  for (let i = 0; i < CELLS; i++) {
    if (mark[i]) {
      moves.push(i);
    }
  }

  return moves;
}

function localThreatScore(board, move, stone) {
  board[move] = stone;

  if (hasFiveAt(board, move, stone)) {
    board[move] = 0;
    return 1000000;
  }

  const r =
    Math.floor(move / SIZE);

  const c =
    move % SIZE;

  let score = 0;

  for (const [dr, dc] of DIRS) {
    let left = 0;
    let right = 0;
    let openLeft = 0;
    let openRight = 0;

    for (let s = 1; s <= 4; s++) {
      const rr = r - dr * s;
      const cc = c - dc * s;

      if (
        rr < 0 || rr >= SIZE ||
        cc < 0 || cc >= SIZE
      ) {
        break;
      }

      const v =
        board[rr * SIZE + cc];

      if (v === stone) {
        left++;
      } else {
        if (v === 0) {
          openLeft = 1;
        }
        break;
      }
    }

    for (let s = 1; s <= 4; s++) {
      const rr = r + dr * s;
      const cc = c + dc * s;

      if (
        rr < 0 || rr >= SIZE ||
        cc < 0 || cc >= SIZE
      ) {
        break;
      }

      const v =
        board[rr * SIZE + cc];

      if (v === stone) {
        right++;
      } else {
        if (v === 0) {
          openRight = 1;
        }
        break;
      }
    }

    const len =
      1 + left + right;

    const opens =
      openLeft + openRight;

    if (len >= 4) {
      score +=
        10000 + opens * 1000;
    } else if (len === 3) {
      score +=
        1000 + opens * 250;
    } else if (len === 2) {
      score +=
        160 + opens * 40;
    } else {
      score +=
        12 + opens * 3;
    }
  }

  board[move] = 0;

  return score;
}

/*
  返回“这一手参与形成的活三/跳三”的关键防守点。
  只匹配常见 VCT 三类：
      .XXX.
      .X.XX.
      .XX.X.
  其中 X 为攻击方，. 为空位。
*/
function openThreeDefenseMoves(board, move) {
  const result = new Set();
  const mr = Math.floor(move / SIZE);
  const mc = move % SIZE;
  const patterns = [
    "01110",
    "010110",
    "011010"
  ];

  for (const [dr, dc] of DIRS) {
    const vals = [];
    const ids = [];

    for (let k = -4; k <= 4; k++) {
      const r = mr + dr * k;
      const c = mc + dc * k;

      if (
        r < 0 || r >= SIZE ||
        c < 0 || c >= SIZE
      ) {
        vals.push("2");
        ids.push(-1);
        continue;
      }

      const id = r * SIZE + c;
      ids.push(id);

      const v = board[id];
      vals.push(
        v === 1 ? "1" :
        v === 0 ? "0" : "2"
      );
    }

    const line = vals.join("");

    for (const pat of patterns) {
      for (
        let s = 0;
        s + pat.length <= line.length;
        s++
      ) {
        if (
          line.slice(s, s + pat.length) !== pat
        ) {
          continue;
        }

        /* 当前落子在9格窗口中的位置固定为4。 */
        if (!(s <= 4 && 4 < s + pat.length)) {
          continue;
        }

        if (pat[4 - s] !== "1") {
          continue;
        }

        for (let j = 0; j < pat.length; j++) {
          if (pat[j] !== "0") {
            continue;
          }

          const id = ids[s + j];

          if (
            id >= 0 &&
            board[id] === 0
          ) {
            result.add(id);
          }
        }
      }
    }
  }

  return Array.from(result);
}

function vctAttackCandidates(board, ctx) {
  const moves =
    nearbyEmptyMoves(board, 2);

  const items = [];

  for (const move of moves) {
    if (board[move] !== 0) {
      continue;
    }

    const score =
      localThreatScore(
        board,
        move,
        1
      );

    board[move] = 1;

    let kind = 0;

    if (hasFiveAt(board, move, 1)) {
      kind = 4;
    } else {
      const oppWins =
        listWinningMoves(
          board,
          -1,
          1
        );

      if (oppWins.length === 0) {
        const ownWins =
          listWinningMoves(
            board,
            1,
            3
          );

        if (ownWins.length >= 2) {
          kind = 3; // 双四 / 开四
        } else if (ownWins.length === 1) {
          kind = 2; // 冲四
        } else {
          const defenses =
            openThreeDefenseMoves(
              board,
              move
            );

          if (defenses.length >= 2) {
            kind = 1; // 活三 / 跳三
          }
        }
      }
    }

    board[move] = 0;

    if (kind > 0) {
      items.push({
        move,
        kind,
        score
      });
    }
  }

  items.sort((a, b) => {
    if (b.kind !== a.kind) {
      return b.kind - a.kind;
    }

    return b.score - a.score;
  });

  return items.slice(
    0,
    ctx.maxAttackCandidates
  );
}

function vctRelevantDefenseMoves(
  board,
  lastAttackMove,
  ctx
) {
  const defenses =
    openThreeDefenseMoves(
      board,
      lastAttackMove
    );

  const items =
    defenses.map((move) => ({
      move,
      score:
        localThreatScore(
          board,
          move,
          -1
        )
    }));

  items.sort(
    (a, b) => b.score - a.score
  );

  return items
    .slice(
      0,
      ctx.maxDefenseCandidates
    )
    .map((x) => x.move);
}

function vctBudgetExceeded(ctx) {
  if (ctx.nodes >= ctx.nodeLimit) {
    ctx.cutoff = true;
    return true;
  }

  if (
    performance.now() -
      ctx.started >=
    ctx.timeLimitMs
  ) {
    ctx.cutoff = true;
    return true;
  }

  return false;
}

function proveVCTFromAttackerTurn(
  board,
  attacksLeft,
  ctx,
  path
) {
  ctx.nodes++;

  if (vctBudgetExceeded(ctx)) {
    return null;
  }

  /*
    轮到攻击方时，若已有一步胜，直接成立。
  */
  const directWins =
    listWinningMoves(
      board,
      1,
      2
    );

  if (directWins.length > 0) {
    return {
      win: true,
      path:
        path.concat(
          directWins[0]
        )
    };
  }

  /*
    如果对手已经有一步胜，而攻击方自己没有一步胜，
    当前强制攻击不成立。
  */
  const oppDirectWins =
    listWinningMoves(
      board,
      -1,
      1
    );

  if (oppDirectWins.length > 0) {
    return {
      win: false,
      path
    };
  }

  if (attacksLeft <= 0) {
    return {
      win: false,
      path
    };
  }

  const candidates =
    vctAttackCandidates(
      board,
      ctx
    );

  for (const item of candidates) {
    if (vctBudgetExceeded(ctx)) {
      return null;
    }

    const move =
      item.move;

    board[move] = 1;

    if (
      hasFiveAt(
        board,
        move,
        1
      )
    ) {
      board[move] = 0;

      return {
        win: true,
        path:
          path.concat(move)
      };
    }

    /*
      对方如果现在有一步胜，可以直接反杀，
      所以这条攻击线路失败。
    */
    const defenderWins =
      listWinningMoves(
        board,
        -1,
        1
      );

    if (defenderWins.length > 0) {
      board[move] = 0;
      continue;
    }

    const attackerWins =
      listWinningMoves(
        board,
        1,
        3
      );

    /*
      两个或更多一步胜点：
      对方一手无法全部封住，视为已证明强制胜。
    */
    if (attackerWins.length >= 2) {
      board[move] = 0;

      return {
        win: true,
        path:
          path.concat(move)
      };
    }

    /*
      单一冲四：
      防守方只有一个必堵点。
    */
    if (attackerWins.length === 1) {
      const defense =
        attackerWins[0];

      board[defense] = -1;

      let child;

      if (
        hasFiveAt(
          board,
          defense,
          -1
        )
      ) {
        child = {
          win: false,
          path
        };
      } else {
        child =
          proveVCTFromAttackerTurn(
            board,
            attacksLeft - 1,
            ctx,
            path.concat(
              move,
              defense
            )
          );
      }

      board[defense] = 0;
      board[move] = 0;

      if (child === null) {
        return null;
      }

      if (child.win) {
        return child;
      }

      continue;
    }

    /*
      VCT 活三级：
      枚举有限的关键防点。
      只有所有关键防守都挡不住，才认定该攻击成立。
    */
    if (item.kind === 1) {
      const defenses =
        vctRelevantDefenseMoves(
          board,
          move,
          ctx
        );

      if (defenses.length > 0) {
        let allLose = true;
        let bestPath = null;

        for (const defense of defenses) {
          if (vctBudgetExceeded(ctx)) {
            board[move] = 0;
            return null;
          }

          if (board[defense] !== 0) {
            continue;
          }

          board[defense] = -1;

          if (
            hasFiveAt(
              board,
              defense,
              -1
            )
          ) {
            allLose = false;
            board[defense] = 0;
            break;
          }

          const child =
            proveVCTFromAttackerTurn(
              board,
              attacksLeft - 1,
              ctx,
              path.concat(
                move,
                defense
              )
            );

          board[defense] = 0;

          if (child === null) {
            board[move] = 0;
            return null;
          }

          if (!child.win) {
            allLose = false;
            break;
          }

          if (!bestPath) {
            bestPath =
              child.path;
          }
        }

        board[move] = 0;

        if (allLose && bestPath) {
          return {
            win: true,
            path: bestPath
          };
        }

        continue;
      }
    }

    board[move] = 0;
  }

  return {
    win: false,
    path
  };
}

function findVCTMove(relativeBoard, cfg) {
  const started =
    performance.now();

  const ctx = {
    started,
    nodes: 0,
    cutoff: false,
    nodeLimit:
      cfg.vctNodeLimit || 5000,
    timeLimitMs:
      cfg.vctTimeLimitMs || 90,
    maxAttackCandidates:
      cfg.vctMaxAttackCandidates || 18,
    maxDefenseCandidates:
      cfg.vctMaxDefenseCandidates || 18
  };

  const board =
    relativeBoard.slice();

  const result =
    proveVCTFromAttackerTurn(
      board,
      cfg.vctMaxAttacks || 5,
      ctx,
      []
    );

  const elapsedMs =
    performance.now() -
    started;

  if (
    result &&
    result.win &&
    result.path.length > 0
  ) {
    return {
      found: true,
      move:
        result.path[0],
      path:
        result.path,
      nodes:
        ctx.nodes,
      cutoff:
        ctx.cutoff,
      elapsedMs
    };
  }

  return {
    found: false,
    move: -1,
    path: [],
    nodes:
      ctx.nodes,
    cutoff:
      ctx.cutoff,
    elapsedMs
  };
}

async function deitySearch(
  relativeBoard,
  searchId,
  cfg
) {
  const started =
    performance.now();

  /*
    最高优先级仍是一步必赢 / 必堵。
  */
  const rule =
    immediateRule(
      relativeBoard,
      1.0
    );

  if (rule) {
    return {
      move: rule.move,
      simulations: 0,
      nnEvals: 0,
      rootValue: null,
      elapsedMs:
        performance.now() - started,
      stopReason:
        cfg.label + "完成落子",
      source: "rule",
      vctFound: false,
      vctNodes: 0
    };
  }

  /*
    然后尝试有界强制杀搜索。
  */
  const vct =
    findVCTMove(
      relativeBoard,
      cfg
    );

  if (vct.found) {
    return {
      move:
        vct.move,
      simulations:
        vct.nodes,
      nnEvals: 0,
      rootValue: null,
      elapsedMs:
        performance.now() - started,
      stopReason:
        cfg.label + "完成落子",
      source: "vct",
      vctFound: true,
      vctNodes:
        vct.nodes,
      vctPath:
        vct.path
    };
  }

  /*
    没有在预算内找到强制杀，就回退到元婴 MCTS。
  */
  const mcts =
    await adaptiveSearch(
      relativeBoard,
      searchId,
      cfg
    );

  mcts.elapsedMs =
    performance.now() - started;

  mcts.source =
    "vct-fallback-mcts";

  mcts.vctFound =
    false;

  mcts.vctNodes =
    vct.nodes;

  return mcts;
}

class Node {
  constructor(move = -1, prior = 1.0, parent = null, depth = 0) {
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
      const u = CPUCT * child.prior * sqrtTotal / (1 + child.N);
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
        children.push(new Node(i, Number(policy[i]) || 0, node, node.depth + 1));
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
    const b = this.rootBoard.slice();
    let node = this.root;
    const path = [node];

    while (node.children && node.children.length > 0) {
      const child = this.selectChild(node);
      const stone = (node.depth & 1) === 0 ? 1 : -1;
      b[child.move] = stone;
      node = child;
      path.push(node);

      if (node.terminal !== null) {
        this.backup(path, node.terminal);
        this.simulations++;
        return;
      }

      if (hasFiveAt(b, node.move, stone)) {
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

    const relative = perspectiveBoard(b, node.depth);
    const pred = await predictRelative(relative);
    this.nnEvals++;

    if (node === this.root && this.rootValue === null) {
      this.rootValue = pred.value;
    }

    this.expand(node, b, pred.policy);
    this.backup(path, pred.value);
    this.simulations++;
  }

  async runUntil(target, searchId, progressType = "progress") {
    while (this.simulations < target) {
      await this.runOne();

      if ((this.simulations & 15) === 0) {
        await Promise.resolve();
      }
    }

    if (progressType === "hint-progress") {
      self.postMessage({
        type: "hint-progress",
        hintId: searchId,
        simulations: this.simulations
      });
    } else {
      self.postMessage({
        type: "progress",
        searchId,
        simulations: this.simulations
      });
    }
  }

  rootStats() {
    const children = this.root.children || [];
    if (children.length === 0) return null;

    let first = null;
    let second = null;

    for (const child of children) {
      if (first === null || child.N > first.N || (child.N === first.N && child.prior > first.prior)) {
        second = first;
        first = child;
      } else if (second === null || child.N > second.N || (child.N === second.N && child.prior > second.prior)) {
        second = child;
      }
    }

    let totalVisits = 0;
    for (const child of children) totalVisits += child.N;

    const n1 = first ? first.N : 0;
    const n2 = second ? second.N : 0;

    return {
      bestMove: first.move,
      secondMove: second ? second.move : -1,
      n1,
      n2,
      totalVisits,
      share: n1 / Math.max(totalVisits, 1),
      ratio: (n1 + 1) / (n2 + 1)
    };
  }

  topRootChildren(limit = 2, excludeMove = -1) {
    const children =
      (this.root.children || [])
        .filter((ch) => ch.move !== excludeMove)
        .slice();

    children.sort((a, b) => {
      if (b.N !== a.N) {
        return b.N - a.N;
      }
      return b.prior - a.prior;
    });

    return children.slice(0, limit);
  }
}

async function fixedSearch(relativeBoard, searchId, cfg) {
  const started = performance.now();

  const rule = immediateRule(relativeBoard, cfg.blockChance);
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
  await tree.runUntil(cfg.simulations, searchId);
  const stats = tree.rootStats();

  return {
    move: stats.bestMove,
    simulations: tree.simulations,
    nnEvals: tree.nnEvals,
    rootValue: tree.rootValue,
    elapsedMs: performance.now() - started,
    stopReason: cfg.label + "完成落子",
    source: "mcts-fixed"
  };
}

async function adaptiveSearch(relativeBoard, searchId, cfg) {
  const started = performance.now();

  const rule = immediateRule(relativeBoard, cfg.blockChance);
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
  const s1Target = cfg.stages[0];
  const s2Target = cfg.stages[1];
  const s3Target = cfg.stages[2];

  await tree.runUntil(s1Target, searchId);
  const s1 = tree.rootStats();

  if (s1.share >= cfg.earlyShare && s1.ratio >= cfg.earlyRatio) {
    return {
      move: s1.bestMove,
      simulations: tree.simulations,
      nnEvals: tree.nnEvals,
      rootValue: tree.rootValue,
      elapsedMs: performance.now() - started,
      stopReason: cfg.label + "完成落子",
      source: "adaptive-mcts"
    };
  }

  const move1 = s1.bestMove;
  await tree.runUntil(s2Target, searchId);
  const s2 = tree.rootStats();
  const stable = s2.bestMove === move1;

  if (stable && s2.ratio >= cfg.midRatio) {
    return {
      move: s2.bestMove,
      simulations: tree.simulations,
      nnEvals: tree.nnEvals,
      rootValue: tree.rootValue,
      elapsedMs: performance.now() - started,
      stopReason: cfg.label + "完成落子",
      source: "adaptive-mcts"
    };
  }

  await tree.runUntil(s3Target, searchId);
  const s3 = tree.rootStats();

  return {
    move: s3.bestMove,
    simulations: tree.simulations,
    nnEvals: tree.nnEvals,
    rootValue: tree.rootValue,
    elapsedMs: performance.now() - started,
    stopReason: cfg.label + "完成落子",
    source: "adaptive-mcts"
  };
}

function buildHintCandidates(tree, forcedRule = null) {
  const candidates = [];

  if (forcedRule && forcedRule.move >= 0) {
    candidates.push({
      move: forcedRule.move,
      prob: 1.0,
      forced: true,
      reason: forcedRule.reason
    });

    const second =
      tree.topRootChildren(
        1,
        forcedRule.move
      )[0];

    if (second) {
      candidates.push({
        move: second.move,
        prob: 0.0,
        forced: false,
        reason: "参考"
      });
    }

    return candidates;
  }

  const top =
    tree.topRootChildren(2);

  if (top.length === 0) {
    return candidates;
  }

  let denom = 0;

  for (const ch of top) {
    denom += ch.N;
  }

  if (denom <= 0) {
    for (const ch of top) {
      denom += Math.max(ch.prior, 0);
    }

    for (const ch of top) {
      candidates.push({
        move: ch.move,
        prob:
          denom > 0
            ? Math.max(ch.prior, 0) / denom
            : 1 / top.length,
        forced: false,
        reason: ""
      });
    }

  } else {
    for (const ch of top) {
      candidates.push({
        move: ch.move,
        prob: ch.N / denom,
        forced: false,
        reason: ""
      });
    }
  }

  return candidates;
}

async function deityHintSearch(
  relativeBoard,
  hintId
) {
  const started =
    performance.now();

  const deityCfg =
    DIFFICULTIES.deity;

  const masterCfg =
    DIFFICULTIES.master;

  /*
    化神指导：
    1) 一步必赢 / 必堵
    2) 有界 VCT/VCF
    3) 元婴级 Adaptive MCTS 作为第二判断来源
  */
  const forcedRule =
    immediateRule(
      relativeBoard,
      1.0
    );

  let vct = {
    found: false,
    move: -1,
    path: [],
    nodes: 0,
    elapsedMs: 0
  };

  if (!forcedRule) {
    vct =
      findVCTMove(
        relativeBoard,
        deityCfg
      );
  }

  /*
    指导仍运行 MCTS，
    这样即使存在强制点，也可以给出第二候选。
  */
  const tree =
    new SearchTree(
      relativeBoard
    );

  const s1Target =
    masterCfg.stages[0];

  const s2Target =
    masterCfg.stages[1];

  const s3Target =
    masterCfg.stages[2];

  await tree.runUntil(
    s1Target,
    hintId,
    "hint-progress"
  );

  let stats =
    tree.rootStats();

  let finished =
    false;

  if (
    !forcedRule &&
    !vct.found &&
    stats.share >=
      masterCfg.earlyShare &&
    stats.ratio >=
      masterCfg.earlyRatio
  ) {
    finished = true;
  }

  const move50 =
    stats.bestMove;

  if (!finished) {
    await tree.runUntil(
      s2Target,
      hintId,
      "hint-progress"
    );

    stats =
      tree.rootStats();

    const stable =
      stats.bestMove ===
      move50;

    if (
      !forcedRule &&
      !vct.found &&
      stable &&
      stats.ratio >=
        masterCfg.midRatio
    ) {
      finished = true;
    }
  }

  /*
    强制规则 / VCT 命中时，
    为了第二推荐更可靠，直接搜索到150。
    普通局面则保持原来的 Adaptive 行为。
  */
  if (
    !finished &&
    tree.simulations <
      s3Target
  ) {
    await tree.runUntil(
      s3Target,
      hintId,
      "hint-progress"
    );
  }

  let candidates;

  if (
    forcedRule &&
    forcedRule.move >= 0
  ) {
    candidates =
      buildHintCandidates(
        tree,
        forcedRule
      );

  } else if (
    vct.found &&
    vct.move >= 0
  ) {
    candidates = [{
      move:
        vct.move,
      prob: 1.0,
      forced: true,
      reason: "强制"
    }];

    const second =
      tree.topRootChildren(
        1,
        vct.move
      )[0];

    if (second) {
      candidates.push({
        move:
          second.move,
        prob: 0.0,
        forced: false,
        reason: "参考"
      });
    }

  } else {
    candidates =
      buildHintCandidates(
        tree,
        null
      );
  }

  return {
    candidates,
    simulations:
      tree.simulations,
    nnEvals:
      tree.nnEvals,
    rootValue:
      tree.rootValue,
    elapsedMs:
      performance.now() -
      started,
    stopReason:
      "化神指导完成",
    vctFound:
      vct.found,
    vctNodes:
      vct.nodes,
    vctPath:
      vct.path
  };
}

async function searchByDifficulty(
  relativeBoard,
  searchId,
  difficulty
) {
  const cfg =
    DIFFICULTIES[difficulty] ||
    DIFFICULTIES.beginner;

  if (
    cfg.type ===
    "policy-random-topk"
  ) {
    return topKPolicySearch(
      relativeBoard,
      cfg
    );
  }

  if (
    cfg.type ===
    "policy-top1"
  ) {
    return top1PolicySearch(
      relativeBoard,
      cfg
    );
  }

  if (
    cfg.type ===
    "fixed"
  ) {
    return fixedSearch(
      relativeBoard,
      searchId,
      cfg
    );
  }

  if (
    cfg.type ===
    "vct-adaptive"
  ) {
    return deitySearch(
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

self.onmessage = async (event) => {
  const data = event.data || {};

  try {
    if (data.type === "init") {
      const s = await getSession();
      self.postMessage({
        type: "ready",
        model: "gomoku_uint8_qop_u8u8.onnx (~147 KB)",
        ortVersion: "1.18.0",
        inputNames: s.inputNames,
        outputNames: s.outputNames
      });
      return;
    }

    if (data.type === "hint") {
      const hintId =
        data.hintId;

      if (!data.board) {
        throw new Error(
          "hint 消息没有棋盘数据"
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
        absoluteToHumanRelative(
          absBoard
        );

      const result =
        await deityHintSearch(
          relative,
          hintId
        );

      const candidates =
        result.candidates.map(
          (c) => ({
            x: c.move % SIZE,
            y: Math.floor(c.move / SIZE),
            prob: c.prob,
            forced: c.forced,
            reason: c.reason
          })
        );

      self.postMessage({
        type: "hint-result",
        hintId,
        candidates,
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
        vctFound:
          !!result.vctFound,
        vctNodes:
          result.vctNodes || 0
      });

      return;
    }

    if (data.type !== "search") {
      return;
    }

    const searchId = data.searchId;

    if (!data.board) {
      throw new Error(
        "search 消息没有棋盘数据"
      );
    }

    const absBoard = data.board instanceof Int8Array ? data.board : new Int8Array(data.board);
    if (absBoard.length !== CELLS) {
      throw new Error("棋盘长度错误：" + absBoard.length + "，应为 " + CELLS);
    }

    await getSession();

    const relative = absoluteToAIRelative(absBoard);
    const difficulty = DIFFICULTIES[data.difficulty] ? data.difficulty : "beginner";
    const cfg = DIFFICULTIES[difficulty];

    const result = await searchByDifficulty(relative, searchId, difficulty);

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
      source: result.source,
      difficulty,
      difficultyLabel: cfg.label,
      vctFound:
        !!result.vctFound,
      vctNodes:
        result.vctNodes || 0
    });

  } catch (error) {
    if (data.type === "hint") {
      self.postMessage({
        type: "hint-error",
        hintId: data.hintId,
        message:
          error && error.message
            ? error.message
            : String(error)
      });

    } else {
      self.postMessage({
        type: "error",
        searchId: data.searchId,
        message:
          error && error.message
            ? error.message
            : String(error)
      });
    }
  }
};
