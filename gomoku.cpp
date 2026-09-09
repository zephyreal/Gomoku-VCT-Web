#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <limits.h>
#include <stdint.h>

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#define KEEPALIVE EMSCRIPTEN_KEEPALIVE
#else
#define KEEPALIVE
#endif

#define SIZE 15
#define EMPTY 0
#define BLACK 1
#define WHITE 2

#define SCORE_FIVE          1000
#define INF 10000

int board[SIZE][SIZE];
int current_player;
int history_x[225];
int history_y[225];
int history_len;
long long g_nodes = 0;

static uint64_t zobrist[SIZE][SIZE][3];
static uint64_t zobrist_side;
static uint64_t zobrist_color[3];
static uint64_t g_hash = 0;

#define TT_BITS 20
#define TT_SIZE (1u << TT_BITS)
#define TT_EXACT 1
#define TT_LOWER 2
#define TT_UPPER 3
typedef struct {
    uint64_t      key;
    int           score;
    short         depth;
    unsigned char flag;
} TTE;
static TTE tt[TT_SIZE];
static long long g_tt_hits = 0;

typedef struct { uint64_t w[4]; } BB;

static BB g_sh[4][6][3];
static int g_stones[3];

static const int DIRX[4] = { 1, 0, 1,  1 };
static const int DIRY[4] = { 0, 1, 1, -1 };
static const int STEP[4] = { 1, 15, 16, -14 };

static BB g_V[4][6];

static inline void bb_clear(BB *a){ a->w[0]=a->w[1]=a->w[2]=a->w[3]=0; }
static inline int  bb_zero (const BB *a){ return !(a->w[0]|a->w[1]|a->w[2]|a->w[3]); }
static inline void bb_and  (BB *a, const BB *b){ a->w[0]&=b->w[0]; a->w[1]&=b->w[1]; a->w[2]&=b->w[2]; a->w[3]&=b->w[3]; }
static inline void bb_or   (BB *a, const BB *b){ a->w[0]|=b->w[0]; a->w[1]|=b->w[1]; a->w[2]|=b->w[2]; a->w[3]|=b->w[3]; }
static inline void bb_set  (BB *a, int idx){ a->w[idx>>6] |= (1ull<<(idx&63)); }

static inline BB bb_shift(const BB *a, int n)
{
    BB r;
    if (n == 0) { r = *a; return r; }
    if (n > 0) {
        int w = n >> 6, o = n & 63;
        for (int i = 3; i >= 0; i--) {
            uint64_t v = 0; int s = i - w;
            if (s >= 0) {
                v = a->w[s] << o;
                if (o && s >= 1) v |= a->w[s-1] >> (64 - o);
            }
            r.w[i] = v;
        }
    } else {
        int m = -n, w = m >> 6, o = m & 63;
        for (int i = 0; i < 4; i++) {
            uint64_t v = 0; int s = i + w;
            if (s < 4) {
                v = a->w[s] >> o;
                if (o && s + 1 < 4) v |= a->w[s+1] << (64 - o);
            }
            r.w[i] = v;
        }
    }
    return r;
}

static void bb_init_masks(void)
{
    for (int d = 0; d < 4; d++)
        for (int k = 0; k < 6; k++) {
            BB v; bb_clear(&v);
            for (int y = 0; y < SIZE; y++)
                for (int x = 0; x < SIZE; x++) {
                    int ok = 1;
                    for (int j = 1; j <= k; j++) {
                        int xx = x + j*DIRX[d], yy = y + j*DIRY[d];
                        if (xx < 0 || xx >= SIZE || yy < 0 || yy >= SIZE) { ok = 0; break; }
                    }
                    if (ok) bb_set(&v, x + SIZE*y);
                }
            g_V[d][k] = v;
        }
}

static void bb_place(int x, int y, int c)
{
    for (int d = 0; d < 4; d++) {
        int dx = DIRX[d], dy = DIRY[d];
        for (int k = 0; k < 6; k++) {
            int qx = x - k*dx, qy = y - k*dy;
            if ((unsigned)qx < SIZE && (unsigned)qy < SIZE) {
                int qi = qx + SIZE*qy, wd = qi >> 6, b = qi & 63;
                g_sh[d][k][c].w[wd] |=  (1ull << b);
                g_sh[d][k][0].w[wd] &= ~(1ull << b);
            }
        }
    }
    g_stones[c]++;
}
static void bb_remove(int x, int y, int c)
{
    for (int d = 0; d < 4; d++) {
        int dx = DIRX[d], dy = DIRY[d];
        for (int k = 0; k < 6; k++) {
            int qx = x - k*dx, qy = y - k*dy;
            if ((unsigned)qx < SIZE && (unsigned)qy < SIZE) {
                int qi = qx + SIZE*qy, wd = qi >> 6, b = qi & 63;
                g_sh[d][k][c].w[wd] &= ~(1ull << b);
                g_sh[d][k][0].w[wd] |=  (1ull << b);
            }
        }
    }
    g_stones[c]--;
}

static uint64_t rng_state = 0x123456789ABCDEF0ull;
static uint64_t rng_next(void)
{
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

void init_game() {
    memset(board, 0, sizeof(board));
    current_player = BLACK;
    history_len = 0;
    bb_init_masks();
    memset(g_sh, 0, sizeof(g_sh));
    memset(g_stones, 0, sizeof(g_stones));
    for (int d = 0; d < 4; d++)
        for (int k = 0; k < 6; k++)
            g_sh[d][k][0] = g_V[d][k];

    rng_state = 0x123456789ABCDEF0ull;
    for (int x = 0; x < SIZE; x++)
        for (int y = 0; y < SIZE; y++) {
            zobrist[x][y][0] = 0;
            zobrist[x][y][1] = rng_next();
            zobrist[x][y][2] = rng_next();
        }
    zobrist_side      = rng_next();
    zobrist_color[0]  = 0;
    zobrist_color[1]  = rng_next();
    zobrist_color[2]  = rng_next();
    g_hash = 0;
    memset(tt, 0, sizeof(tt));
    g_tt_hits = 0;
}

bool make_move(int x, int y) {
    if (x < 0 || x >= SIZE || y < 0 || y >= SIZE || board[y][x] != EMPTY) return false;
    board[y][x] = current_player;
    g_hash ^= zobrist[x][y][current_player];
    g_hash ^= zobrist_side;
    bb_place(x, y, current_player);
    history_x[history_len] = x;
    history_y[history_len++] = y;
    current_player = (current_player == BLACK) ? WHITE : BLACK;
    return true;
}

void unmake_move(void) {
    if (!history_len) return;
    --history_len;
    int x = history_x[history_len], y = history_y[history_len];
    int c = board[y][x];
    board[y][x] = EMPTY;
    g_hash ^= zobrist[x][y][c];
    g_hash ^= zobrist_side;
    bb_remove(x, y, c);
    current_player = (current_player == BLACK) ? WHITE : BLACK;
}

bool iswin() {
    if (!history_len) return 0;
    int x = history_x[history_len-1], y = history_y[history_len-1], p = board[y][x];
    if (p == EMPTY) return 0;
    #define OK(nx,ny) ((nx)>=0&&(nx)<SIZE&&(ny)>=0&&(ny)<SIZE&&board[ny][nx]==p)
    int d[4][2] = {{1,0},{0,1},{1,1},{1,-1}};
    for (int i = 0; i < 4; i++) {
        int c = 1;
        for (int s = 1; s < 5 && OK(x+d[i][0]*s, y+d[i][1]*s); s++) c++;
        for (int s = 1; s < 5 && OK(x-d[i][0]*s, y-d[i][1]*s); s++) c++;
        if (c >= 5) return 1;
    }
    return 0;
    #undef OK
}

typedef struct { int len; unsigned char p[6]; } Pat;

#define PATTERN_COUNT(a) ((int)(sizeof(a) / sizeof((a)[0])))

static const Pat PAT_OPP3[] = {
    {0, {0,0,0,0,0,0}},
};
static const Pat PAT_FOUR[] = {
    {0, {0,0,0,0,0,0}},
};
static const Pat PAT_ME3[] = {
    {0, {0,0,0,0,0,0}},
};

static const Pat PAT_GEN[] = {
    //己方
    {6, {0,0,1,1,0,0}},//己方活二
    {3, {1,0,1,0,0,0}},//己方活跳二
    {4, {1,0,0,1,0,0}},//己方异活二
    {5, {0,1,1,1,0,0}},//己方活三
    {5, {1,0,1,0,1,0}},//己方断三
    {5, {2,1,1,1,0,0}},//己方冲四
    {5, {0,1,1,1,2,0}},//己方冲四
    {6, {2,1,1,0,0,1}},//己方冲四
    {6, {1,0,0,1,1,2}},//己方冲四
    {6, {2,1,0,0,1,1}},//己方冲四
    {6, {1,1,0,0,1,2}},//己方冲四
    {6, {2,0,0,1,1,1}},//己方冲四
    {6, {1,1,1,0,0,2}},//己方冲四
    {6, {0,1,1,1,1,0}},//己方活四
    //对手
    {5, {2,0,2,0,2,0}},//对方断三
    {6, {1,2,2,2,0,0}},//对方冲四
    {6, {0,0,2,2,2,1}},//对方冲四
    {6, {1,2,2,0,0,2}},//己方冲四
    {6, {2,0,0,2,2,1}},//对方冲四
    {6, {1,2,0,0,2,2}},//对方冲四
    {6, {2,2,0,0,2,1}},//对方冲四
    {6, {1,0,0,2,2,2}},//对方冲四
    {6, {2,2,2,0,0,1}},//对方冲四
    {6, {0,2,2,2,2,0}},//对方活四
};

static void match_pat(const Pat *pp, int d, int me, int opp, BB acc[6])
{
    const unsigned char *pat = pp->p;
    int len = pp->len;
    if (len <= 0 || len > 6) return;
    BB m = g_sh[d][0][(pat[0]==0)?0:((pat[0]==1)?me:opp)];
    for (int k = 1; k < len; k++) {
        int v = pat[k];
        bb_and(&m, &g_sh[d][k][(v==0)?0:((v==1)?me:opp)]);
        if (bb_zero(&m)) return;
    }
    for (int k = 0; k < len; k++)
        if (pat[k] == 0) bb_or(&acc[k], &m);
}

static void flush_acc(const BB acc[6], int d, BB *marks)
{
    int s = STEP[d];
    bb_or(marks, &acc[0]);
    for (int k = 1; k < 6; k++)
        if (!bb_zero(&acc[k])) {
            BB t = bb_shift(&acc[k], k * s);
            bb_or(marks, &t);
        }
}

static void add_five(BB acc[6], int d, int color)
{
    const BB *P  = &g_sh[d][0][color],  *E  = &g_sh[d][0][0];
    const BB *S1 = &g_sh[d][1][color],  *S2 = &g_sh[d][2][color];
    const BB *S3 = &g_sh[d][3][color],  *S4 = &g_sh[d][4][color];
    const BB *E1 = &g_sh[d][1][0], *E2 = &g_sh[d][2][0];
    const BB *E3 = &g_sh[d][3][0], *E4 = &g_sh[d][4][0];
    BB t;

    t = *S1; bb_and(&t,S2); bb_and(&t,S3); bb_and(&t,S4); bb_and(&t,E);
    bb_or(&acc[0], &t);
    t = *S2; bb_and(&t,S3); bb_and(&t,S4); bb_and(&t,P); bb_and(&t,E1);
    bb_or(&acc[1], &t);
    t = *S1; bb_and(&t,S3); bb_and(&t,S4); bb_and(&t,P); bb_and(&t,E2);
    bb_or(&acc[2], &t);
    t = *S1; bb_and(&t,S2); bb_and(&t,S4); bb_and(&t,P); bb_and(&t,E3);
    bb_or(&acc[3], &t);
    t = *S1; bb_and(&t,S2); bb_and(&t,S3); bb_and(&t,P); bb_and(&t,E4);
    bb_or(&acc[4], &t);
}

#if defined(__GNUC__) || defined(__clang__)
#define BB_CTZ64 __builtin_ctzll
#else
static int BB_CTZ64(uint64_t x){ int r=0; while(!(x&1)){x>>=1;r++;} return r; }
#endif

static int bb_extract(const BB *b, int mx[], int my[], int count)
{
    for (int y = 0; y < SIZE; y++) {
        int base = SIZE*y, w = base >> 6, o = base & 63;
        uint64_t bits;
        if (o <= 64 - SIZE) bits = (b->w[w] >> o) & 0x7FFFu;
        else bits = ((b->w[w] >> o) | (b->w[w+1] << (64 - o))) & 0x7FFFu;
        while (bits) {
            mx[count] = BB_CTZ64(bits);
            my[count] = y;
            count++;
            bits &= bits - 1;
        }
    }
    return count;
}

static inline void acc_clear(BB acc[6]) { memset(acc, 0, sizeof(BB) * 6); }

int get_valid_moves(int moves_x[], int moves_y[], int ai_color)
{
    int opp = (ai_color == 1) ? 2 : 1;
    BB marks, acc[6];
    int count;

    if (g_stones[1] + g_stones[2] == 0) {
        moves_x[0] = SIZE / 2;
        moves_y[0] = SIZE / 2;
        return 1;
    }

    bb_clear(&marks);
    if (g_stones[ai_color] >= 4 || g_stones[opp] >= 4) {
        for (int d = 0; d < 4; d++) {
            acc_clear(acc);
            if (g_stones[ai_color] >= 4) add_five(acc, d, ai_color);
            if (g_stones[opp]      >= 4) add_five(acc, d, opp);
            flush_acc(acc, d, &marks);
        }
    }
    if (!bb_zero(&marks))
        return bb_extract(&marks, moves_x, moves_y, 0);

    bb_clear(&marks);
    if (g_stones[opp] >= 3) {
        for (int d = 0; d < 4; d++) {
            acc_clear(acc);
            for (int i = 0; i < PATTERN_COUNT(PAT_OPP3); i++)
                match_pat(&PAT_OPP3[i], d, ai_color, opp, acc);
            flush_acc(acc, d, &marks);
        }
    }
    if (!bb_zero(&marks)) {
        count = bb_extract(&marks, moves_x, moves_y, 0);
        if (g_stones[ai_color] >= 3) {
            for (int d = 0; d < 4; d++) {
                acc_clear(acc);
                for (int i = 0; i < PATTERN_COUNT(PAT_FOUR); i++)
                    match_pat(&PAT_FOUR[i], d, ai_color, opp, acc);
                flush_acc(acc, d, &marks);
            }
        }
        return bb_extract(&marks, moves_x, moves_y, count);
    }

    bb_clear(&marks);
    if (g_stones[ai_color] >= 3) {
        for (int d = 0; d < 4; d++) {
            acc_clear(acc);
            for (int i = 0; i < PATTERN_COUNT(PAT_ME3); i++)
                match_pat(&PAT_ME3[i], d, ai_color, opp, acc);
            flush_acc(acc, d, &marks);
        }
    }
    if (!bb_zero(&marks))
        return bb_extract(&marks, moves_x, moves_y, 0);

    bb_clear(&marks);
    if (g_stones[ai_color] >= 2 || g_stones[opp] >= 2) {
        for (int d = 0; d < 4; d++) {
            acc_clear(acc);
            for (int i = 0; i < PATTERN_COUNT(PAT_GEN); i++)
                match_pat(&PAT_GEN[i], d, ai_color, opp, acc);
            flush_acc(acc, d, &marks);
        }
    }
    return bb_extract(&marks, moves_x, moves_y, 0);
}

int nm_moves_x[100][225], nm_moves_y[100][225];
int nm_move_count[100], nm_current_idx[100], nm_depth = 0;

void GeneratelegalMovesNegaMax(int ai_color) {
    nm_move_count[nm_depth] = get_valid_moves(nm_moves_x[nm_depth], nm_moves_y[nm_depth], ai_color);
    nm_current_idx[nm_depth] = 0;
}

int MovesleftNegaMax() { return nm_move_count[nm_depth]; }

void MakeNextMoveNegaMax() {
    make_move(nm_moves_x[nm_depth][nm_current_idx[nm_depth]],
              nm_moves_y[nm_depth][nm_current_idx[nm_depth]]);
    nm_current_idx[nm_depth]++;
    nm_depth++;
    g_nodes++;
}

void UnmakeMoveNegaMax() { nm_depth--; unmake_move(); }

int NegaMaxeval() {
    if (iswin()) return -INF;
    return 0;
}

int NegaMaxSearch(int depth, int alpha, int beta, int ai_color) {
    if (iswin() || depth <= 0) return NegaMaxeval();
    uint64_t key = g_hash ^ zobrist_color[ai_color];
    TTE *e = &tt[key & (TT_SIZE - 1)];
    if (e->key == key && e->depth >= depth) {
        if (e->flag == TT_EXACT) { g_tt_hits++; return e->score; }
        if (e->flag == TT_LOWER && e->score >= beta) { g_tt_hits++; return e->score; }
        if (e->flag == TT_UPPER && e->score <= alpha) { g_tt_hits++; return e->score; }
    }
    int orig_alpha = alpha;
    GeneratelegalMovesNegaMax(ai_color);
    if (MovesleftNegaMax() == 0) {
        e->key   = key;
        e->score = 0;
        e->depth = 30000;
        e->flag  = TT_EXACT;
        return 0;
    }
    for (int i = 0; i < MovesleftNegaMax(); i++) {
        MakeNextMoveNegaMax();
        int val = -NegaMaxSearch(depth - 1, -beta, -alpha, ai_color);
        UnmakeMoveNegaMax();
        if (val > alpha) alpha = val;
        if (alpha >= beta) break;
    }

    {
        unsigned char flag;
        if (alpha >= beta)            flag = TT_LOWER;
        else if (alpha > orig_alpha)  flag = TT_EXACT;
        else                          flag = TT_UPPER;

        short sdepth = (alpha >= INF) ? 30000 : (short)depth;

        if (e->key != key || e->depth <= sdepth) {
            e->key   = key;
            e->score = alpha;
            e->depth = sdepth;
            e->flag  = flag;
        }
    }

    return alpha;
}

static int g_last_best_x = -1;
static int g_last_best_y = -1;
static int g_last_score = -INF;
static long long g_last_total_nodes = 0;
static long long g_last_total_tt_hits = 0;

int get_best_move(int max_depth, int *best_x, int *best_y, int me) {
    int moves_x[225], moves_y[225];
    int move_count = get_valid_moves(moves_x, moves_y, me);
    if (move_count == 0) { *best_x = -1; *best_y = -1; return -INF; }

    int move_scores[225];
    for (int i = 0; i < move_count; i++) move_scores[i] = -INF;

    *best_x = -1; *best_y = -1;

    int final_best_score = -INF;
    long long total_nodes = 0;
    long long total_tt_hits = 0;

    for (int d = 2; d <= max_depth; d++) {
        if (d > 1) {
            for (int i = 0; i < move_count - 1; i++) {
                for (int j = i + 1; j < move_count; j++) {
                    if (move_scores[i] < move_scores[j]) {
                        int t = move_scores[i]; move_scores[i] = move_scores[j]; move_scores[j] = t;
                        t = moves_x[i]; moves_x[i] = moves_x[j]; moves_x[j] = t;
                        t = moves_y[i]; moves_y[i] = moves_y[j]; moves_y[j] = t;
                    }
                }
            }
        }

        int current_best_score = -INF;
        int bx = -1, by = -1;
        int alpha = -INF, beta = INF;

        g_nodes = 0;
        g_tt_hits = 0;

        for (int i = 0; i < move_count; i++) {
            make_move(moves_x[i], moves_y[i]);
            g_nodes++;
            int score = -NegaMaxSearch(d - 1, -beta, -alpha, me);
            unmake_move();
            move_scores[i] = score;

            if (bx == -1 || score > current_best_score) {
                current_best_score = score;
                bx = moves_x[i];
                by = moves_y[i];
            }
            if (score > alpha) alpha = score;
        }
        total_nodes += g_nodes;
        total_tt_hits += g_tt_hits;

        *best_x = bx;
        *best_y = by;
        final_best_score = current_best_score;

        if (current_best_score >= SCORE_FIVE) break;
    }

    g_last_total_nodes = total_nodes;
    g_last_total_tt_hits = total_tt_hits;
    return final_best_score;
}


extern "C" {

KEEPALIVE int vct_init(void) {
    init_game();
    nm_depth = 0;
    g_last_best_x = -1;
    g_last_best_y = -1;
    g_last_score = -INF;
    g_last_total_nodes = 0;
    g_last_total_tt_hits = 0;
    return 1;
}

KEEPALIVE int vct_play(int x, int y) {
    return make_move(x, y) ? 1 : 0;
}

KEEPALIVE int vct_undo(void) {
    if (history_len <= 0) return 0;
    unmake_move();
    return 1;
}

KEEPALIVE int vct_current_player(void) {
    return current_player;
}

KEEPALIVE int vct_iswin(void) {
    return iswin() ? 1 : 0;
}

KEEPALIVE int vct_get_piece(int x, int y) {
    if (x < 0 || x >= SIZE || y < 0 || y >= SIZE) return -1;
    return board[y][x];
}

KEEPALIVE int vct_search(int max_depth) {
    if (max_depth < 2) max_depth = 2;
    if (max_depth > 30) max_depth = 30;

    g_last_best_x = -1;
    g_last_best_y = -1;
    g_last_total_nodes = 0;
    g_last_total_tt_hits = 0;

    g_last_score = get_best_move(
        max_depth,
        &g_last_best_x,
        &g_last_best_y,
        current_player
    );

    return g_last_score;
}

KEEPALIVE int vct_best_x(void) {
    return g_last_best_x;
}

KEEPALIVE int vct_best_y(void) {
    return g_last_best_y;
}

KEEPALIVE int vct_last_score(void) {
    return g_last_score;
}

KEEPALIVE double vct_last_nodes(void) {
    return (double)g_last_total_nodes;
}

KEEPALIVE double vct_last_tt_hits(void) {
    return (double)g_last_total_tt_hits;
}

} // extern "C"
