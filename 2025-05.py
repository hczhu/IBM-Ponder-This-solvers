"""

Source code for the puzzle
https://research.ibm.com/haifa/ponderthis/challenges/May2025.html

This Python script is written by OpenAI o4-mini with the prompt:

A test has 10 multiple selection questions. Each question has 4 selections. 10 students took the test. Their answers and scores are below. What are the answer to each question?

BABBDDCDCA 6/10
DADBDDAACD 6/10
CADBDBCDCC 7/10
DACBDDCBCB 6/10
DCBBDDCDAB 6/10
DDDBDDCDBD 8/10
DADBDDCCCC 8/10
CADBDDDDAC 7/10
"""
# Let's solve the key using brute force with pruning.
# Student answers and scores:
students = [
    ("BABBDDCDCA", 6),
    ("DADBDDAACD", 6),
    ("CADBDBCDCC", 7),
    ("DACBDDCBCB", 6),
    ("DCBBDDCDAB", 6),
    ("DDDBDDCDBD", 8),
    ("DADBDDCCCC", 8),
    ("CADBDDDDAC", 7),
]

import itertools

# letters:
choices = ['A','B','C','D']

# We can solve via brute force? 4^10 = ~1 million, which is one million combos, feasible.

# But to optimize, we can prune early for prefix partial assignments:
solutions = []

# Let's do depth-first search with pruning.

def dfs(pos, key, partial_matches):
    # key: current prefix key of length pos
    # partial_matches: list of matches counts so far for each student
    if pos == len(students[0][0]):
        # full key, check final
        # partial_matches includes matches count for full key?
        valid = True
        for idx,(ans,s) in enumerate(students):
            if partial_matches[idx] != s:
                valid = False
                break
        if valid:
            solutions.append("".join(key))
        return
    # for each choice
    # prune: for each student, the maximum possible matches = partial_matches + (remaining positions)
    rem = 10 - pos
    for c in choices:
        new_partial = []
        prune = False
        for idx,(ans,s) in enumerate(students):
            match = partial_matches[idx] + (1 if ans[pos] == c else 0)
            # match can't exceed s, and min possible match = match (since zeros)? Actually min possible is match (if all future mismatches) <= s
            # Actually, match <= s, and match + rem >= s
            if match > s or match + (rem-1) < s: # rem-1 after this pos
                prune = True
                break
            new_partial.append(match)
        if prune:
            continue
        dfs(pos+1, key+[c], new_partial)

# initialize partial matches to zeros
dfs(0, [], [0]*len(students))
print(len(solutions), solutions[:5])