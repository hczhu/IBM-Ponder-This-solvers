from typing import Tuple
from typing import Self

class Matrix:
    """
    A matrix of integers of arbitrary length.
    """
    def __init__(self, ma: list[list[int]]|int|Self):
        if isinstance(ma, int):
            self.ma = [[0] * ma for _ in range(ma)]
            for i in range(ma):
                self.ma[i][i] = 1
        elif isinstance(ma, Matrix):
            self.ma = ma.ma
        else:
            self.ma = ma
    
    @staticmethod
    def identity(n: int) -> Self:
        return Matrix(n)

    @staticmethod
    def zero(n: int) -> Self:
        return Matrix([[0] * n for _ in range(n)])
    
    def __mul__(self, other: Self) -> Self:
        if len(self.ma[0]) != len(other.ma):
            raise ValueError("Matrices must have the same number of columns to be multiplied.")
        N, K, M = len(self.ma), len(self.ma[0]), len(other.ma[0])
        res = [[0] * M for _ in range(N)]
        for i in range(N):
            for k in range(K):
                ik = self.ma[i][k]
                for j in range(M):
                    res[i][j] += ik * other.ma[k][j]
        return Matrix(res)

    def __eq__(self, other: Self) -> bool:
        return self.ma == other.ma

    def __str__(self) -> str:
        return "\n" + "\n".join([", ".join(map(str, row)) for row in self.ma])

    def __repr__(self) -> str:
        return str(self)

res = Matrix([[1, 3], [0, 2]]) * Matrix([[2], [1]])
assert res == Matrix([[5], [2]]), res
res = Matrix([[1, 2], [3, 4]]) * Matrix([[5, 6], [7, 8]])
assert res == Matrix([[19, 22], [43, 50]]), res

A = Matrix([[0,0,1,1],[1,0,0,0],[1,1,0,0],[0,0,1,0]])
res = Matrix.identity(4)
for _ in range(480):
    res = res * A
assert all([sum(row) > 10**100 + 1000 for row in res.ma])

def find_first_char(trans_map: dict[str, str], s0: str, n: int) -> str:
    steps = 0
    ch = s0[0]
    seen = {ch: 0}
    assert len(ch) == 1
    while n > 0:
        steps += 1
        n -= 1
        ch = trans_map[ch][0]
        if ch in seen:
            n %= (steps - seen[ch])
        else:
            seen[ch] = steps
    return ch

def solve(trans: Tuple[Tuple[str, str]], s0: str, n: int, p: int) -> str:
    id_map = [0] * 256
    trans_map = {}
    for i, tran in enumerate(trans):
        id_map[ord(tran[0])] = i
        trans_map[tran[0]] = tran[1]
    N = len(trans)
    A = Matrix.zero(N)
    get_id = lambda c: id_map[ord(c)]
    for a, bs in trans:
        id_a = get_id(a)
        for b in bs:
            A.ma[id_a][get_id(b)] = 1
    
    lens = [ [1] * N ]
    ma = Matrix.identity(N)
    for _ in range(n + 1):
        ma = ma * A
        lens.append([sum(row) for row in ma.ma])
        if all([l >= p for l in lens[-1]]):
            break
    print(f"Stopped after {len(lens)} steps")
    if n >= len(lens):
        print(f"Decreasing steps from {n} to {len(lens) - 1}")
        s0 = find_first_char(trans_map, s0, n - len(lens) + 1)
        n = len(lens) - 1
    # print(f"Find char @{p} after {n} steps starting from '{s0}'")
    while n > 0:
        assert p < sum([lens[n][get_id(ch)] for ch in s0]), (s0, p, n, lens[n])
        for ch in s0:
            ch_id = get_id(ch)
            if p < lens[n][ch_id]:
                s0 = trans_map[ch]
                break
            p -= lens[n][ch_id]
        n -= 1
    assert p < len(s0), (s0, p, n, lens[n])
    print(f"Found char '{s0[p]}' after {n} steps: {s0, p, n, lens[n]}")
    return s0[p]


def bruteforce(trans: Tuple[Tuple[str, str]], s0: str, n: int) -> str:
    trans_map = {}
    for a, bs in trans:
        trans_map[a] = bs
    for _ in range(n):
        s0 = "".join([trans_map[ch] for ch in s0])
    print(f"Hit {len(s0)} chars after {n} steps")
    return s0

CAT_TRANS = (
        ("C", "TG"),
        ("A", "C"),
        ("T", "CA"),
        ("G", "T"),
)

assert solve( CAT_TRANS, "CAT", 3, 12) == "G"
assert solve( CAT_TRANS, "CAT", 3, 5) == "C"
assert solve( CAT_TRANS, "CAT", 3, 6) == "A"
assert solve( CAT_TRANS, "CAT", 3, 7) == "T"
assert solve( CAT_TRANS, "CAT", 10**100, 0) == "C"

assert bruteforce( CAT_TRANS, "CAT", 3) == "TGCCACATCATTG"
bt = bruteforce( CAT_TRANS, "CAT", 3)
for p in range(len(bt)):
    assert solve(CAT_TRANS, "CAT", 3, p) == bt[p]

bt = bruteforce(CAT_TRANS, "CAT", 31)
assert len(bt) == 9227465

assert solve(CAT_TRANS, "CAT", 31, 0) == bt[0]
assert solve(CAT_TRANS, "CAT", 31, 10) == bt[10]
assert solve(CAT_TRANS, "CAT", 31, 100) == bt[100]
assert solve(CAT_TRANS, "CAT", 31, 1000) == bt[1000]
assert solve(CAT_TRANS, "CAT", 31, 10000) == bt[10000]
assert solve(CAT_TRANS, "CAT", 31, 10001) == bt[10001]


for p in range(0, 9227465, 2001):
    assert solve(CAT_TRANS, "CAT", 31, p) == bt[p]

RABBIT_TRANS = (
    ("G", "T"),
    ("T", "CA"),
    ("C", "BR"),
    ("A", "I"),
    ("R", "B"),
    ("B", "IS"),
    ("I", "TG"),
    ("S", "C"),
)
bt = bruteforce(RABBIT_TRANS, "RABBITS", 20)
assert len(bt) == 103682
for p in range(0, len(bt), 2001):
    assert solve(RABBIT_TRANS, "RABBITS", 20, p) == bt[p]

n100 = 10**100
cat= "".join(
    [solve(CAT_TRANS, "CAT", n100, n100 + p) for p in range(1000)]
)
rabbit = "".join(
    [solve(RABBIT_TRANS, "RABBITS", n100, n100 + p) for p in range(1000)]
)

print(cat)
print(rabbit)
