"""Solver for IBM Ponder This, July 2026: Return of the Superheroes.

Puzzle: https://research.ibm.com/blog/ponder-this-july-2026

For every hero/villain pair (a, b), iterate

    x[0] = 0
    x[k + 1] = x[k]**2 + a*x[k] + b (mod p)

and let f(a, b) be the first index whose value has appeared before.  The goal
is the maximum, over all perfect matchings, of the minimum matched edge value.

The orbit table uses one polynomial evaluation per step and timestamped visit
markers.  Its worst-case construction cost is O(n^2 p), with typical rho-shaped
orbits taking O(n^2 sqrt(p)); rows are generated in parallel processes for the
full instances and stored as unsigned 16-bit integers.

For one n, threshold feasibility is perfect matching in the graph containing
edges with f(a, b) >= threshold.  Only sorted, distinct values actually present
in the matrix are searched.  Hopcroft-Karp gives matching time
O(n^2 sqrt(n) log d), where d is the number of distinct edge values.

For the bonus, each threshold graph is represented by one Python integer bitset
per row.  Leading k by k graphs are grown incrementally.  Adding one vertex to
each side can increase a maximum matching by at most two, so at most two new
augmenting paths are needed per k.
"""

from __future__ import annotations

import itertools
import os
import sys
import unittest
from array import array
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Callable, Iterable


def _orbit_rows(task: tuple[int, int, int, int]) -> tuple[int, array]:
    """Build rows [start, stop) in a worker process."""
    start, stop, n, p = task
    result = array("H", [0]) * ((stop - start) * n)
    seen = [0] * p
    stamp = 0

    for a_zero_based in range(start, stop):
        a = a_zero_based + 1
        row_offset = (a_zero_based - start) * n
        for b_zero_based in range(n):
            b = b_zero_based + 1
            stamp += 1
            x = 0
            steps = 0
            while seen[x] != stamp:
                seen[x] = stamp
                x = (x * x + a * x + b) % p
                steps += 1
            result[row_offset + b_zero_based] = steps

    return start, result


@dataclass(frozen=True, slots=True)
class OrbitTable:
    n: int
    p: int
    weights: array

    @classmethod
    def build(cls, n: int, p: int, workers: int | None = None) -> OrbitTable:
        if not (0 < n < p < 2**16):
            raise ValueError("expected 0 < n < p < 2**16")

        if workers is None:
            workers = min(n, os.cpu_count() or 1)
        workers = max(1, min(workers, n))

        # Process startup costs more than the tiny test instances themselves.
        if workers == 1 or n < 64:
            _, weights = _orbit_rows((0, n, n, p))
            return cls(n, p, weights)

        weights = array("H", [0]) * (n * n)
        chunk_size = max(8, (n + 4 * workers - 1) // (4 * workers))
        tasks = [
            (start, min(start + chunk_size, n), n, p)
            for start in range(0, n, chunk_size)
        ]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for start, rows in executor.map(_orbit_rows, tasks):
                weights[start * n : start * n + len(rows)] = rows
        return cls(n, p, weights)

    def row(self, left: int) -> memoryview:
        start = left * self.n
        return memoryview(self.weights)[start : start + self.n]

    def value(self, left: int, right: int) -> int:
        return self.weights[left * self.n + right]


def direct_orbit_length(a: int, b: int, p: int) -> int:
    seen = bytearray(p)
    x = 0
    steps = 0
    while not seen[x]:
        seen[x] = 1
        x = (x * x + a * x + b) % p
        steps += 1
    return steps


def sorted_edge_values(table: OrbitTable, n: int) -> list[int]:
    values: set[int] = set()
    for left in range(n):
        values.update(table.row(left)[:n])
    return sorted(values)


def maximum_feasible_value(
    sorted_values: list[int], feasible_at: Callable[[int], bool]
) -> int:
    """Return the largest observed value satisfying a decreasing predicate."""
    feasible = -1
    infeasible = len(sorted_values)
    while infeasible - feasible > 1:
        middle = (feasible + infeasible) // 2
        if feasible_at(sorted_values[middle]):
            feasible = middle
        else:
            infeasible = middle
    if feasible < 0:
        raise AssertionError("the complete graph at the minimum value is feasible")
    return sorted_values[feasible]


def has_perfect_matching(table: OrbitTable, n: int, threshold: int) -> bool:
    """Hopcroft-Karp on the implicit threshold graph."""
    adjacency = [
        [right for right, weight in enumerate(table.row(left)[:n]) if weight >= threshold]
        for left in range(n)
    ]
    match_left = [-1] * n
    match_right = [-1] * n
    distance = [0] * n
    next_edge = [0] * n
    infinity = n + 1
    matching = 0

    def bfs() -> int:
        queue: deque[int] = deque()
        for left in range(n):
            if match_left[left] == -1:
                distance[left] = 0
                queue.append(left)
            else:
                distance[left] = infinity

        nil_distance = infinity
        while queue:
            left = queue.popleft()
            if distance[left] + 1 > nil_distance:
                continue
            for right in adjacency[left]:
                next_left = match_right[right]
                if next_left == -1:
                    nil_distance = distance[left] + 1
                elif distance[next_left] == infinity:
                    distance[next_left] = distance[left] + 1
                    queue.append(next_left)
        return nil_distance

    def dfs(left: int, nil_distance: int) -> bool:
        neighbors = adjacency[left]
        while next_edge[left] < len(neighbors):
            right = neighbors[next_edge[left]]
            next_edge[left] += 1
            next_left = match_right[right]
            if (
                next_left == -1
                and distance[left] + 1 == nil_distance
                or next_left != -1
                and distance[next_left] == distance[left] + 1
                and dfs(next_left, nil_distance)
            ):
                match_left[left] = right
                match_right[right] = left
                return True
        distance[left] = infinity
        return False

    while (nil_distance := bfs()) != infinity:
        next_edge[:] = itertools.repeat(0, n)
        for left in range(n):
            if match_left[left] == -1 and dfs(left, nil_distance):
                matching += 1
        if matching == n:
            return True
    return False


def hero_villain_value(table: OrbitTable, n: int | None = None) -> int:
    if n is None:
        n = table.n
    values = sorted_edge_values(table, n)
    return maximum_feasible_value(
        values, lambda threshold: has_perfect_matching(table, n, threshold)
    )


class PrefixMatcher:
    """Maximum matchings for all leading prefixes at one threshold."""

    def __init__(self, table: OrbitTable, threshold: int) -> None:
        self.n = table.n
        self.adjacency: list[int] = []
        for left in range(self.n):
            bits = 0
            for right, weight in enumerate(table.row(left)):
                if weight >= threshold:
                    bits |= 1 << right
            self.adjacency.append(bits)

        self.match_left = [-1] * self.n
        self.match_right = [-1] * self.n
        self.seen_left = [0] * self.n
        self.parent_right = [-1] * self.n
        self.queue = [0] * self.n
        self.stamp = 0

    def _augment(self, k: int, active_rights: int) -> bool:
        self.stamp += 1
        stamp = self.stamp
        head = 0
        tail = 0
        for left in range(k):
            if self.match_left[left] == -1:
                self.seen_left[left] = stamp
                self.queue[tail] = left
                tail += 1
        if tail == 0:
            return False

        seen_rights = 0
        while head < tail:
            left = self.queue[head]
            head += 1
            candidates = self.adjacency[left] & active_rights & ~seen_rights
            while candidates:
                bit = candidates & -candidates
                candidates ^= bit
                seen_rights |= bit
                right = bit.bit_length() - 1
                self.parent_right[right] = left
                next_left = self.match_right[right]
                if next_left == -1:
                    self._flip_path(right)
                    return True
                if self.seen_left[next_left] != stamp:
                    self.seen_left[next_left] = stamp
                    self.queue[tail] = next_left
                    tail += 1
        return False

    def _flip_path(self, right: int) -> None:
        while right != -1:
            left = self.parent_right[right]
            previous_right = self.match_left[left]
            self.match_left[left] = right
            self.match_right[right] = left
            right = previous_right

    def perfect_prefixes(self, stop_at_first: bool = False) -> list[int]:
        result: list[int] = []
        matching_size = 0
        active_rights = 0

        for k in range(1, self.n + 1):
            active_rights |= 1 << (k - 1)
            for _ in range(2):
                if not self._augment(k, active_rights):
                    break
                matching_size += 1
            if k > 1 and matching_size == k:
                result.append(k)
                if stop_at_first:
                    return result
        return result


@dataclass(frozen=True, slots=True)
class BonusResult:
    value: int
    sizes: list[int]


def best_prefix(table: OrbitTable) -> BonusResult:
    values = sorted_edge_values(table, table.n)

    def feasible_at(threshold: int) -> bool:
        return bool(PrefixMatcher(table, threshold).perfect_prefixes(True))

    value = maximum_feasible_value(values, feasible_at)
    sizes = PrefixMatcher(table, value).perfect_prefixes()
    return BonusResult(value, sizes)


def brute_bottleneck(table: OrbitTable, n: int) -> int:
    return max(
        min(table.value(left, right) for left, right in enumerate(permutation))
        for permutation in itertools.permutations(range(n))
    )


def format_ranges(values: Iterable[int]) -> str:
    values = list(values)
    ranges: list[str] = []
    first = 0
    while first < len(values):
        last = first
        while last + 1 < len(values) and values[last + 1] == values[last] + 1:
            last += 1
        ranges.append(
            str(values[first])
            if first == last
            else f"{values[first]}..{values[last]}"
        )
        first = last + 1
    return ",".join(ranges)


class July2026Test(unittest.TestCase):
    def test_orbit_lengths_from_example(self) -> None:
        self.assertEqual(direct_orbit_length(1, 3, 101), 14)
        self.assertEqual(direct_orbit_length(2, 1, 101), 18)
        self.assertEqual(direct_orbit_length(3, 4, 101), 19)
        self.assertEqual(direct_orbit_length(4, 2, 101), 22)
        self.assertEqual(direct_orbit_length(5, 5, 101), 14)

    def test_given_bottleneck_example(self) -> None:
        table = OrbitTable.build(5, 101)
        self.assertEqual(hero_villain_value(table), 14)

    def test_matches_brute_force(self) -> None:
        table = OrbitTable.build(8, 101)
        for n in range(1, 9):
            with self.subTest(n=n):
                self.assertEqual(hero_villain_value(table, n), brute_bottleneck(table, n))

    def test_prefix_search_matches_independent_answers(self) -> None:
        table = OrbitTable.build(8, 101)
        result = best_prefix(table)
        answers = {n: brute_bottleneck(table, n) for n in range(2, 9)}
        expected_value = max(answers.values())
        expected_sizes = [n for n, value in answers.items() if value == expected_value]
        self.assertEqual(result, BonusResult(expected_value, expected_sizes))


def solve() -> None:
    primary = OrbitTable.build(611, 14411)
    print(
        "Hero-villain value for n=611, p=14411:",
        hero_villain_value(primary),
    )

    bonus = OrbitTable.build(999, 17377)
    result = best_prefix(bonus)
    print(
        "Bonus maximum hero-villain value for p=17377:",
        result.value,
        "at n=" + format_ranges(result.sizes),
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "solve":
        solve()
    else:
        unittest.main()
