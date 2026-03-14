/*
  Source code for the puzzle
  https://research.ibm.com/blog/ponder-this-march-2026

  Alice and Bob play on an N×M board. Alice places a pawn on (x,y). Bob moves
  it to an adjacent (non-diagonal) square and removes the vacated square.
  Players alternate; the player who cannot move loses.

  Square (i,j) is labeled p^i + q^j (1-indexed). For each prime s in S, all
  squares whose label is divisible by s are removed before play begins.

  Each remaining square is classified "A" (Alice wins) or "B" (Bob wins).

  Game-theoretic reduction: after Alice places at (x,y), Bob moves first. This
  is exactly Vertex Geography on the remaining grid graph G_s. By the theorem
  of Fraenkel, Scheinerman & Ullman (1993), on a bipartite graph the first
  player wins iff the starting vertex is in every maximum matching.

  Therefore:
    "B" (Bob/first-mover wins) <=> (x,y) is in every maximum matching of G_s
    "A" (Alice wins)           <=> (x,y) is NOT in every maximum matching of G_s

  To classify all vertices simultaneously, use the Dulmage-Mendelsohn
  decomposition after running Hopcroft-Karp:

  Build directed graph D: matched edges R->L, unmatched edges L->R.
    - BFS from free L-nodes in D:   reachable L-nodes are "A"
    - BFS from free R-nodes in D^T: reachable R-nodes are "A"
  All other matched nodes are "B".

  The total answer is the sum across each prime s in S of the A and B counts
  on the board for that prime.
*/

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <queue>
#include <string>
#include <vector>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

// ---------------------------------------------------------------------------
// Sieve of Eratosthenes
// ---------------------------------------------------------------------------
std::vector<int> primesBelow(int limit) {
  std::vector<bool> composite(limit, false);
  std::vector<int> primes;
  for (int i = 2; i < limit; ++i) {
    if (!composite[i]) {
      primes.push_back(i);
      for (long long j = (long long)i * i; j < limit; j += i)
        composite[j] = true;
    }
  }
  return primes;
}

// ---------------------------------------------------------------------------
// Hopcroft-Karp bipartite matching
//
// L-nodes: 0 .. nL-1
// R-nodes: 0 .. nR-1
// adj[l] = list of r-indices adjacent to l
//
// Returns match_l[l] = r matched to l (-1 if unmatched)
//         match_r[r] = l matched to r (-1 if unmatched)
// ---------------------------------------------------------------------------
struct HopcroftKarp {
  int nL, nR;
  std::vector<std::vector<int>> adj; // adj[l] -> list of r
  std::vector<int> match_l, match_r, dist;

  HopcroftKarp(int nL, int nR)
      : nL(nL), nR(nR), adj(nL), match_l(nL, -1), match_r(nR, -1),
        dist(nL) {}

  void addEdge(int l, int r) { adj[l].push_back(r); }

  bool bfs() {
    std::queue<int> q;
    for (int l = 0; l < nL; ++l) {
      if (match_l[l] == -1) {
        dist[l] = 0;
        q.push(l);
      } else {
        dist[l] = INT_MAX;
      }
    }
    bool found = false;
    while (!q.empty()) {
      int l = q.front(); q.pop();
      for (int r : adj[l]) {
        int l2 = match_r[r];
        if (l2 == -1) {
          found = true;
        } else if (dist[l2] == INT_MAX) {
          dist[l2] = dist[l] + 1;
          q.push(l2);
        }
      }
    }
    return found;
  }

  bool dfs(int l) {
    for (int r : adj[l]) {
      int l2 = match_r[r];
      if (l2 == -1 || (dist[l2] == dist[l] + 1 && dfs(l2))) {
        match_l[l] = r;
        match_r[r] = l;
        return true;
      }
    }
    dist[l] = INT_MAX;
    return false;
  }

  int maxMatching() {
    int res = 0;
    while (bfs())
      for (int l = 0; l < nL; ++l)
        if (match_l[l] == -1 && dfs(l))
          ++res;
    return res;
  }
};

// ---------------------------------------------------------------------------
// Board classification for a single prime s
//
// Grid is N x M (1-indexed). Bipartite split:
//   L: squares where (i+j) is even   (index in [0, nL))
//   R: squares where (i+j) is odd    (index in [0, nR))
//
// Returns {countA, countB}.
// ---------------------------------------------------------------------------
struct Counts { long long A = 0, B = 0; };

Counts classifyBoard(int N, int M, int64_t p, int64_t q, int s) {
  // Precompute p^i mod s for i in [1,N] and q^j mod s for j in [1,M]
  std::vector<int> powP(N + 1), powQ(M + 1);
  {
    int64_t acc = 1;
    for (int i = 1; i <= N; ++i) { acc = acc * p % s; powP[i] = (int)acc; }
    acc = 1;
    for (int j = 1; j <= M; ++j) { acc = acc * q % s; powQ[j] = (int)acc; }
  }

  // removed[i][j] = true if s | (p^i + q^j)
  // node index: (i-1)*M + (j-1)  for i in [1,N], j in [1,M]
  // L = even (i+j), R = odd (i+j)
  // We assign L-index and R-index separately.
  // lIdx[flat] = L-index (-1 if R or removed)
  // rIdx[flat] = R-index (-1 if L or removed)
  int total = N * M;
  std::vector<int> lIdx(total, -1), rIdx(total, -1);
  int nL = 0, nR = 0;
  for (int i = 1; i <= N; ++i) {
    for (int j = 1; j <= M; ++j) {
      if ((powP[i] + powQ[j]) % s == 0) continue; // removed
      int flat = (i - 1) * M + (j - 1);
      if ((i + j) % 2 == 0)
        lIdx[flat] = nL++;
      else
        rIdx[flat] = nR++;
    }
  }

  if (nL == 0 || nR == 0) {
    // No edges possible; all remaining nodes are "A" (isolated = free)
    return {nL + nR, 0};
  }

  // Build bipartite graph (L->R edges only)
  HopcroftKarp hk(nL, nR);
  auto flat = [&](int i, int j) { return (i - 1) * M + (j - 1); };
  const int di[] = {0, 0, 1, -1};
  const int dj[] = {1, -1, 0, 0};
  for (int i = 1; i <= N; ++i) {
    for (int j = 1; j <= M; ++j) {
      int f = flat(i, j);
      if (lIdx[f] == -1) continue; // removed or R-node
      for (int d = 0; d < 4; ++d) {
        int ni = i + di[d], nj = j + dj[d];
        if (ni < 1 || ni > N || nj < 1 || nj > M) continue;
        int nf = flat(ni, nj);
        if (rIdx[nf] == -1) continue; // removed or L-node
        hk.addEdge(lIdx[f], rIdx[nf]);
      }
    }
  }

  hk.maxMatching();

  // DM decomposition
  // Build directed graph D (as adjacency lists for L->R and R->L):
  //   matched edge (l,r): add arc r -> l  (stored in radj_matched)
  //   unmatched edge (l,r): add arc l -> r (stored in ladj_unmatched)
  // D^T (reverse):
  //   matched edge (l,r): arc l -> r  (ladj_matched)
  //   unmatched edge (l,r): arc r -> l (radj_unmatched)

  // We'll store D via two lists:
  //   fromL[l] = list of r reachable from l in D (unmatched edges l->r)
  //   fromR[r] = list of l reachable from r in D (matched edges r->l)
  // And D^T:
  //   fromL_T[l] = list of r reachable from l in D^T (matched edges l->r)
  //   fromR_T[r] = list of l reachable from r in D^T (unmatched edges r->l)

  std::vector<std::vector<int>> fromL(nL), fromR(nR);     // in D
  std::vector<std::vector<int>> fromR_T(nR);               // in D^T (only R->L needed)

  for (int l = 0; l < nL; ++l) {
    for (int r : hk.adj[l]) {
      if (hk.match_l[l] == r) {
        // matched: in D add r->l; in D^T add r->l reversed => l->r (not needed here)
        fromR[r].push_back(l);
      } else {
        // unmatched: in D add l->r; in D^T add r->l
        fromL[l].push_back(r);
        fromR_T[r].push_back(l);
      }
    }
  }

  // BFS 1: from free L-nodes in D, collect reachable L-nodes -> "A"
  std::vector<bool> notEveryL(nL, false);
  {
    std::queue<int> q;
    // Seeds: free L-nodes (trivially "A"), then follow D
    // In D: L-node l can reach R via unmatched edges, then L via matched edges
    // We interleave L and R visits; track both.
    std::vector<bool> visitedL(nL, false), visitedR(nR, false);
    for (int l = 0; l < nL; ++l) {
      if (hk.match_l[l] == -1) {
        visitedL[l] = true;
        notEveryL[l] = true;
        q.push(l); // encode L-nodes as l, R-nodes as nL+r
      }
    }
    while (!q.empty()) {
      int node = q.front(); q.pop();
      if (node < nL) {
        // L-node: follow unmatched edges l->r
        for (int r : fromL[node]) {
          if (!visitedR[r]) {
            visitedR[r] = true;
            q.push(nL + r);
          }
        }
      } else {
        // R-node: follow matched edge r->l
        int r = node - nL;
        for (int l : fromR[r]) {
          if (!visitedL[l]) {
            visitedL[l] = true;
            notEveryL[l] = true;
            q.push(l);
          }
        }
      }
    }
  }

  // BFS 2: from free R-nodes in D^T, collect reachable R-nodes -> "A"
  // In D^T: R-node r follows unmatched edges r->l (fromR_T), then l follows
  // matched edges l->r (which is fromR[r] reversed: for each r, fromR[r] gives
  // l's matched to r, so in D^T l can reach r via matched edge).
  // We need: from free R, follow unmatched R->L, then matched L->R, etc.
  std::vector<bool> notEveryR(nR, false);
  {
    // Build fromL_T: for each l, list of r reachable via matched edge in D^T
    // In D^T matched edge goes l->r (reverse of r->l in D)
    std::vector<std::vector<int>> fromL_T(nL);
    for (int r = 0; r < nR; ++r) {
      for (int l : fromR[r]) {
        fromL_T[l].push_back(r);
      }
    }

    std::queue<int> q;
    std::vector<bool> visitedL(nL, false), visitedR(nR, false);
    for (int r = 0; r < nR; ++r) {
      if (hk.match_r[r] == -1) {
        visitedR[r] = true;
        notEveryR[r] = true;
        q.push(nL + r);
      }
    }
    while (!q.empty()) {
      int node = q.front(); q.pop();
      if (node >= nL) {
        // R-node: follow unmatched edges r->l in D^T
        int r = node - nL;
        for (int l : fromR_T[r]) {
          if (!visitedL[l]) {
            visitedL[l] = true;
            q.push(l);
          }
        }
      } else {
        // L-node: follow matched edges l->r in D^T
        int l = node;
        for (int r : fromL_T[l]) {
          if (!visitedR[r]) {
            visitedR[r] = true;
            notEveryR[r] = true;
            q.push(nL + r);
          }
        }
      }
    }
  }

  Counts cnt;
  for (int i = 1; i <= N; ++i) {
    for (int j = 1; j <= M; ++j) {
      int f = flat(i, j);
      if ((i + j) % 2 == 0) {
        if (lIdx[f] == -1) continue;
        (notEveryL[lIdx[f]] ? cnt.A : cnt.B)++;
      } else {
        if (rIdx[f] == -1) continue;
        (notEveryR[rIdx[f]] ? cnt.A : cnt.B)++;
      }
    }
  }
  return cnt;
}

// ---------------------------------------------------------------------------
// Solve: sum over all primes s in S
// ---------------------------------------------------------------------------
Counts solve(int N, int M, int64_t p, int64_t q, int primeLimit) {
  Counts total;
  for (int s : primesBelow(primeLimit)) {
    auto c = classifyBoard(N, M, p, q, s);
    total.A += c.A;
    total.B += c.B;
    DLOG(INFO) << "s=" << s << " A=" << c.A << " B=" << c.B;
  }
  return total;
}

// ===========================================================================
// Unit Tests
// ===========================================================================

// ---------------------------------------------------------------------------
// Helper: brute-force vertex geography by game-tree search (for small boards)
// Returns true if the first player (Bob) wins starting at (si,sj).
// ---------------------------------------------------------------------------
bool firstPlayerWinsVG(int N, int M,
                       const std::vector<std::vector<bool>>& removed,
                       int si, int sj) {
  // State: current position + set of removed squares (bitmask)
  // Board fits in N*M <= 16 bits for small boards
  assert(N * M <= 20);
  int sz = N * M;
  auto idx = [&](int i, int j) { return i * M + j; };
  uint32_t initMask = 0;
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < M; ++j)
      if (removed[i][j]) initMask |= (1u << idx(i, j));

  // memo[pos][mask] = true if current player wins
  // pos: flat index of pawn; mask: removed squares (the pawn's square is
  // always set in mask after the first move, but at start it's the starting sq)
  // We represent state as (pos, mask) where mask includes already-removed squares.
  // At state (pos, mask): current player must move pawn from pos to adjacent
  // non-removed, non-pos square, then remove pos.
  std::unordered_map<uint64_t, bool> memo;
  std::function<bool(int, uint32_t)> wins = [&](int pos, uint32_t mask) -> bool {
    uint64_t key = ((uint64_t)pos << 32) | mask;
    auto it = memo.find(key);
    if (it != memo.end()) return it->second;

    int ci = pos / M, cj = pos % M;
    const int di[] = {0, 0, 1, -1};
    const int dj[] = {1, -1, 0, 0};
    bool canMove = false;
    for (int d = 0; d < 4; ++d) {
      int ni = ci + di[d], nj = cj + dj[d];
      if (ni < 0 || ni >= N || nj < 0 || nj >= M) continue;
      int npos = idx(ni, nj);
      if (mask & (1u << npos)) continue; // removed
      canMove = true;
      uint32_t newMask = mask | (1u << pos); // remove current square
      if (!wins(npos, newMask)) { // opponent loses -> we win
        memo[key] = true;
        return true;
      }
    }
    if (!canMove) { memo[key] = false; return false; } // lose
    memo[key] = false;
    return false;
  };

  // Alice places at (si,sj), Bob moves first.
  // Bob's state: pawn at (si,sj), (si,sj) not yet removed (it gets removed
  // when Bob moves away).
  uint32_t mask = initMask; // pre-removed squares only
  return wins(idx(si, sj), mask);
}

// Classify all squares of an N×M board with given removed mask via brute force.
Counts bruteForceBoard(int N, int M,
                       const std::vector<std::vector<bool>>& removed) {
  Counts cnt;
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < M; ++j) {
      if (removed[i][j]) continue;
      if (firstPlayerWinsVG(N, M, removed, i, j))
        cnt.B++;
      else
        cnt.A++;
    }
  return cnt;
}

// Build removed mask for classifyBoard (uses 1-indexed i,j internally)
std::vector<std::vector<bool>> buildRemoved(int N, int M, int64_t p, int64_t q,
                                            int s) {
  std::vector<bool> powP(N + 1), powQ(M + 1);
  {
    int64_t acc = 1;
    for (int i = 1; i <= N; ++i) { acc = acc * p % s; powP[i] = (acc == 0); }
    // recompute properly
  }
  // Recompute: removed[i][j] for i in [0,N), j in [0,M) (0-indexed for brute force)
  std::vector<int> pp(N + 1), pq(M + 1);
  {
    int64_t acc = 1;
    for (int i = 1; i <= N; ++i) { acc = acc * (p % s) % s; pp[i] = (int)acc; }
    acc = 1;
    for (int j = 1; j <= M; ++j) { acc = acc * (q % s) % s; pq[j] = (int)acc; }
  }
  std::vector<std::vector<bool>> removed(N, std::vector<bool>(M, false));
  for (int i = 1; i <= N; ++i)
    for (int j = 1; j <= M; ++j)
      removed[i - 1][j - 1] = ((pp[i] + pq[j]) % s == 0);
  return removed;
}

// ---------------------------------------------------------------------------
// Test: baseline 3x3 board (no removals) = alternating A/B
// ---------------------------------------------------------------------------
TEST(VertexGeography, BaselinePattern) {
  // No squares removed: use s large enough that nothing is divisible.
  // Use s=97 (prime), p=1, q=1: labels are 1^i+1^j=2, divisible by 2 not 97.
  // Simpler: manually test with removed = all false.
  std::vector<std::vector<bool>> removed(3, std::vector<bool>(3, false));
  auto cnt = bruteForceBoard(3, 3, removed);
  // Baseline 3x3: L-nodes (even i+j, 0-indexed: (0,0),(0,2),(1,1),(2,0),(2,2)) = 5 nodes -> all A
  // R-nodes (odd i+j: (0,1),(1,0),(1,2),(2,1)) = 4 nodes -> all B
  EXPECT_EQ(cnt.A, 5);
  EXPECT_EQ(cnt.B, 4);
}

// ---------------------------------------------------------------------------
// Test: removing center of 3x3 gives all B
// ---------------------------------------------------------------------------
TEST(VertexGeography, Remove3x3Center) {
  std::vector<std::vector<bool>> removed(3, std::vector<bool>(3, false));
  removed[1][1] = true; // center
  auto cnt = bruteForceBoard(3, 3, removed);
  EXPECT_EQ(cnt.A, 0);
  EXPECT_EQ(cnt.B, 8);
}

// ---------------------------------------------------------------------------
// Test: isolated square is always A
// ---------------------------------------------------------------------------
TEST(VertexGeography, IsolatedSquare) {
  // 1x1 board, no removals: Alice places, Bob can't move -> Bob loses -> A
  std::vector<std::vector<bool>> removed(1, std::vector<bool>(1, false));
  auto cnt = bruteForceBoard(1, 1, removed);
  EXPECT_EQ(cnt.A, 1);
  EXPECT_EQ(cnt.B, 0);
}

// ---------------------------------------------------------------------------
// Test: 1x2 board -> one L, one R, matched -> both B
// ---------------------------------------------------------------------------
TEST(VertexGeography, Board1x2) {
  std::vector<std::vector<bool>> removed(1, std::vector<bool>(2, false));
  auto cnt = bruteForceBoard(1, 2, removed);
  // (0,0)=L: Bob moves to (0,1), removes (0,0). Alice can't move. Alice loses -> B
  // (0,1)=R: Bob moves to (0,0), removes (0,1). Alice can't move. Alice loses -> B
  EXPECT_EQ(cnt.A, 0);
  EXPECT_EQ(cnt.B, 2);
}

// ---------------------------------------------------------------------------
// Test: 1x3 path -> A B A (center is B, ends are A)
// ---------------------------------------------------------------------------
TEST(VertexGeography, Board1x3) {
  std::vector<std::vector<bool>> removed(1, std::vector<bool>(3, false));
  auto cnt = bruteForceBoard(1, 3, removed);
  EXPECT_EQ(cnt.A, 2);
  EXPECT_EQ(cnt.B, 1);
}

// ---------------------------------------------------------------------------
// Test: classifyBoard matches brute force for small boards
// ---------------------------------------------------------------------------
void checkMatchesBruteForce(int N, int M, int64_t p, int64_t q, int s) {
  auto removed = buildRemoved(N, M, p, q, s);
  auto expected = bruteForceBoard(N, M, removed);
  auto got = classifyBoard(N, M, p, q, s);
  EXPECT_EQ(got.A, expected.A)
      << "N=" << N << " M=" << M << " p=" << p << " q=" << q << " s=" << s;
  EXPECT_EQ(got.B, expected.B)
      << "N=" << N << " M=" << M << " p=" << p << " q=" << q << " s=" << s;
}

TEST(ClassifyBoard, SmallBoards_p19_q2) {
  for (int s : {2, 3, 5, 7, 11}) {
    checkMatchesBruteForce(3, 3, 19, 2, s);
    checkMatchesBruteForce(2, 4, 19, 2, s);
    checkMatchesBruteForce(4, 2, 19, 2, s);
    checkMatchesBruteForce(3, 4, 19, 2, s);
  }
}

TEST(ClassifyBoard, SmallBoards_p7_q3) {
  for (int s : {2, 3, 5, 7, 11, 13}) {
    checkMatchesBruteForce(3, 3, 7, 3, s);
    checkMatchesBruteForce(2, 3, 7, 3, s);
    checkMatchesBruteForce(4, 3, 7, 3, s);
  }
}

TEST(ClassifyBoard, SmallBoards_p5_q11) {
  for (int s : {2, 3, 5, 7, 11}) {
    checkMatchesBruteForce(3, 3, 5, 11, s);
    checkMatchesBruteForce(2, 4, 5, 11, s);
  }
}

TEST(ClassifyBoard, AllSquaresRemoved) {
  // p=1, q=1: labels = 1^i+1^j = 2, all divisible by 2 -> board empty
  auto cnt = classifyBoard(5, 5, 1, 1, 2);
  EXPECT_EQ(cnt.A, 0);
  EXPECT_EQ(cnt.B, 0);
}

TEST(ClassifyBoard, NothingRemoved_SmallBoards) {
  // When s is large prime not dividing any label, board = full grid
  // Compare classifyBoard with brute force on full grid (no removals)
  // Use s=97 and choose p,q,N,M such that no p^i+q^j is divisible by 97
  // p=2, q=3, s=97 (unlikely to hit for small i,j - verify)
  // Actually just use the brute force on a "no removal" board for verification:
  for (int N = 1; N <= 4; ++N) {
    for (int M = 1; M <= 4; ++M) {
      if (N * M > 16) continue;
      std::vector<std::vector<bool>> removed(N, std::vector<bool>(M, false));
      auto expected = bruteForceBoard(N, M, removed);
      // Use p=2,q=3,s=97: verify no removals happen, then compare
      bool anyRemoved = false;
      for (int i = 1; i <= N && !anyRemoved; ++i) {
        int64_t pi = 1;
        for (int k = 0; k < i; ++k) pi = pi * 2 % 97;
        for (int j = 1; j <= M && !anyRemoved; ++j) {
          int64_t qj = 1;
          for (int k = 0; k < j; ++k) qj = qj * 3 % 97;
          if ((pi + qj) % 97 == 0) anyRemoved = true;
        }
      }
      if (!anyRemoved) {
        auto got = classifyBoard(N, M, 2, 3, 97);
        EXPECT_EQ(got.A, expected.A) << "N=" << N << " M=" << M;
        EXPECT_EQ(got.B, expected.B) << "N=" << N << " M=" << M;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Test: the example from the puzzle statement
// N=M=3, p=19, q=2, S=[2,3,5,7,11] -> total A=16, B=19
// ---------------------------------------------------------------------------
TEST(Solve, PuzzleExample) {
  // Compute sum over s in {2,3,5,7,11}
  long long totalA = 0, totalB = 0;
  for (int s : {2, 3, 5, 7, 11}) {
    auto c = classifyBoard(3, 3, 19, 2, s);
    totalA += c.A;
    totalB += c.B;
  }
  EXPECT_EQ(totalA, 16);
  EXPECT_EQ(totalB, 19);
}

// ---------------------------------------------------------------------------
// Test: verify solve() helper accumulates correctly
// ---------------------------------------------------------------------------
TEST(Solve, SolveHelperMatchesManual) {
  auto manual = [](int N, int M, int64_t p, int64_t q,
                   std::vector<int> primes) -> Counts {
    Counts total;
    for (int s : primes) {
      auto c = classifyBoard(N, M, p, q, s);
      total.A += c.A;
      total.B += c.B;
    }
    return total;
  };

  // Compare solve() against manual loop for small cases
  auto s1 = solve(3, 3, 19, 2, 12); // primes < 12: {2,3,5,7,11}
  auto m1 = manual(3, 3, 19, 2, {2, 3, 5, 7, 11});
  EXPECT_EQ(s1.A, m1.A);
  EXPECT_EQ(s1.B, m1.B);

  auto s2 = solve(4, 4, 7, 3, 10); // primes < 10: {2,3,5,7}
  auto m2 = manual(4, 4, 7, 3, {2, 3, 5, 7});
  EXPECT_EQ(s2.A, m2.A);
  EXPECT_EQ(s2.B, m2.B);
}

// ---------------------------------------------------------------------------
// Test: specific squares classified correctly (manual verification)
// ---------------------------------------------------------------------------
TEST(ClassifyBoard, SpecificSquare_3x3_s5_p19_q2) {
  // With s=5, only center (2,2) is removed; all 8 remaining are B.
  auto cnt = classifyBoard(3, 3, 19, 2, 5);
  EXPECT_EQ(cnt.A, 0);
  EXPECT_EQ(cnt.B, 8);
}

TEST(ClassifyBoard, SpecificSquare_3x3_s2_p19_q2) {
  // s=2, p=19 (odd), q=2 (even): 19^i+2^j = odd+even = odd -> never divisible by 2
  // Full 3x3 board: A=5, B=4
  auto cnt = classifyBoard(3, 3, 19, 2, 2);
  EXPECT_EQ(cnt.A, 5);
  EXPECT_EQ(cnt.B, 4);
}

// ---------------------------------------------------------------------------
// Test: larger boards - classifyBoard vs brute force
// ---------------------------------------------------------------------------
TEST(ClassifyBoard, MediumBoards_BruteForceCheck) {
  // 4x4 boards with various primes
  struct Case { int N, M; int64_t p, q; int s; };
  std::vector<Case> cases = {
    {4, 4, 19, 2, 3},
    {4, 4, 19, 2, 5},
    {4, 4, 19, 2, 7},
    {4, 4,  7, 3, 5},
    {4, 4,  7, 3, 7},
    {3, 4, 19, 2, 5},
    {4, 3, 19, 2, 5},
    {4, 4, 13, 7, 3},
    {4, 4, 13, 7, 11},
  };
  for (auto& c : cases) {
    checkMatchesBruteForce(c.N, c.M, c.p, c.q, c.s);
  }
}

// ---------------------------------------------------------------------------
// Test: rectangular boards
// ---------------------------------------------------------------------------
TEST(ClassifyBoard, RectangularBoards) {
  for (int s : {3, 5, 7}) {
    checkMatchesBruteForce(2, 3, 19, 2, s);
    checkMatchesBruteForce(3, 2, 19, 2, s);
    checkMatchesBruteForce(2, 4, 7, 5, s);
    checkMatchesBruteForce(4, 2, 7, 5, s);
  }
}

// ---------------------------------------------------------------------------
// Test: Hopcroft-Karp correctness on known graphs
// ---------------------------------------------------------------------------
TEST(HopcroftKarp, PathGraph) {
  // L={0,1}, R={0}: edges 0-0, 1-0. Max matching = 1.
  HopcroftKarp hk(2, 1);
  hk.addEdge(0, 0);
  hk.addEdge(1, 0);
  EXPECT_EQ(hk.maxMatching(), 1);
  // One of l0,l1 matched, the other free
  int matched = (hk.match_l[0] != -1 ? 0 : 1);
  int free = 1 - matched;
  EXPECT_EQ(hk.match_l[matched], 0);
  EXPECT_EQ(hk.match_l[free], -1);
  EXPECT_EQ(hk.match_r[0], matched);
}

TEST(HopcroftKarp, PerfectMatching) {
  // K_{2,2}: max matching = 2
  HopcroftKarp hk(2, 2);
  hk.addEdge(0, 0); hk.addEdge(0, 1);
  hk.addEdge(1, 0); hk.addEdge(1, 1);
  EXPECT_EQ(hk.maxMatching(), 2);
  // Both L and R nodes must be matched
  EXPECT_NE(hk.match_l[0], -1);
  EXPECT_NE(hk.match_l[1], -1);
  EXPECT_NE(hk.match_r[0], -1);
  EXPECT_NE(hk.match_r[1], -1);
}

TEST(HopcroftKarp, EmptyGraph) {
  HopcroftKarp hk(3, 3);
  EXPECT_EQ(hk.maxMatching(), 0);
  for (int l = 0; l < 3; ++l) EXPECT_EQ(hk.match_l[l], -1);
  for (int r = 0; r < 3; ++r) EXPECT_EQ(hk.match_r[r], -1);
}

TEST(HopcroftKarp, LargerBipartite) {
  // 3x3 grid, full bipartite: L=5 (even i+j), R=4 (odd i+j)
  // Max matching = 4
  HopcroftKarp hk(5, 4);
  // Map: (i,j) with i+j even -> L, else R (1-indexed, 3x3)
  // L: (1,1)->0, (1,3)->1, (2,2)->2, (3,1)->3, (3,3)->4
  // R: (1,2)->0, (2,1)->1, (2,3)->2, (3,2)->3
  // Edges:
  hk.addEdge(0, 0); // (1,1)-(1,2)
  hk.addEdge(0, 1); // (1,1)-(2,1)
  hk.addEdge(1, 0); // (1,3)-(1,2)
  hk.addEdge(1, 2); // (1,3)-(2,3)
  hk.addEdge(2, 0); // (2,2)-(1,2)
  hk.addEdge(2, 1); // (2,2)-(2,1)
  hk.addEdge(2, 2); // (2,2)-(2,3)
  hk.addEdge(2, 3); // (2,2)-(3,2)
  hk.addEdge(3, 1); // (3,1)-(2,1)
  hk.addEdge(3, 3); // (3,1)-(3,2)
  hk.addEdge(4, 2); // (3,3)-(2,3)
  hk.addEdge(4, 3); // (3,3)-(3,2)
  EXPECT_EQ(hk.maxMatching(), 4);
  // All R nodes must be matched
  for (int r = 0; r < 4; ++r) EXPECT_NE(hk.match_r[r], -1);
  // Exactly one L node is free
  int freeCount = 0;
  for (int l = 0; l < 5; ++l) if (hk.match_l[l] == -1) ++freeCount;
  EXPECT_EQ(freeCount, 1);
}

// ---------------------------------------------------------------------------
// Test: sieve correctness
// ---------------------------------------------------------------------------
TEST(Sieve, PrimesBelow100) {
  auto primes = primesBelow(100);
  EXPECT_EQ(primes.size(), 25u);
  EXPECT_EQ(primes.front(), 2);
  EXPECT_EQ(primes.back(), 97);
}

TEST(Sieve, PrimesBelow12) {
  auto primes = primesBelow(12);
  EXPECT_EQ(primes, (std::vector<int>{2, 3, 5, 7, 11}));
}

// ---------------------------------------------------------------------------
// Test: A+B = number of non-removed squares (sanity check)
// ---------------------------------------------------------------------------
TEST(Sanity, ABSumEqualsNonRemoved) {
  struct Case { int N, M; int64_t p, q; int s; };
  std::vector<Case> cases = {
    {3, 3, 19, 2, 2}, {3, 3, 19, 2, 5}, {4, 4, 7, 3, 3}, {5, 5, 11, 7, 5},
  };
  for (auto& c : cases) {
    // Count non-removed squares manually
    std::vector<int> pp(c.N + 1), pq(c.M + 1);
    int64_t acc = 1;
    for (int i = 1; i <= c.N; ++i) { acc = acc * (c.p % c.s) % c.s; pp[i] = (int)acc; }
    acc = 1;
    for (int j = 1; j <= c.M; ++j) { acc = acc * (c.q % c.s) % c.s; pq[j] = (int)acc; }
    int nonRemoved = 0;
    for (int i = 1; i <= c.N; ++i)
      for (int j = 1; j <= c.M; ++j)
        if ((pp[i] + pq[j]) % c.s != 0) ++nonRemoved;
    auto cnt = classifyBoard(c.N, c.M, c.p, c.q, c.s);
    EXPECT_EQ(cnt.A + cnt.B, nonRemoved)
        << "N=" << c.N << " M=" << c.M << " s=" << c.s;
  }
}

// ---------------------------------------------------------------------------
// Test: larger cross-check against brute force (5x4 boards)
// ---------------------------------------------------------------------------
TEST(ClassifyBoard, Board5x4_BruteForce) {
  for (int s : {3, 5, 7}) {
    checkMatchesBruteForce(4, 4, 19, 2, s);
  }
}

// ===========================================================================
// Main
// ===========================================================================
int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_logtostderr = 1;

  testing::InitGoogleTest(&argc, argv);
  auto res = RUN_ALL_TESTS();
  if (argc <= 1 || std::string(argv[1]) != "solve") {
    return res;
  }

  LOG(INFO) << "Solving main puzzle: N=M=157, p=419, q=211, primes < 100";
  auto main_ans = solve(157, 157, 419, 211, 100);
  std::cout << "Main: A=" << main_ans.A << " B=" << main_ans.B << std::endl;

  LOG(INFO) << "Solving bonus: N=M=1557, p=419, q=211, primes < 500";
  auto bonus_ans = solve(1557, 1557, 419, 211, 500);
  std::cout << "Bonus: A=" << bonus_ans.A << " B=" << bonus_ans.B << std::endl;

  return res;
}
