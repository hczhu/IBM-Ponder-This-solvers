import sys
from collections import defaultdict, deque

# Cache for memoizing state successor generation
_successors_cache = {}

def normalize_state(positions):
    """
    Normalize state so min position is 0.
    Input: Tuple of positions (e.g., (2, 2, 5, 5, 5))
    Output: Normalized tuple (e.g., (0, 0, 3, 3, 3))
    """
    if not positions:
        return positions
    min_p = min(positions)
    return tuple(sorted(p - min_p for p in positions))

def is_valid_state(positions):
    """
    Check if a configuration has blots.
    A blot is a singleton checker at a location.
    The goal is to AVOID blots.
    """
    counts = defaultdict(int)
    for p in positions:
        counts[p] += 1
    for count in counts.values():
        if count == 1:
            return False
    return True

def get_moves(positions, roll):
    """
    Generate all valid next states from a given position and roll.
    Roll is a tuple of moves to apply, e.g., (1, 2) or (1, 1, 1, 1).
    """
    if not roll:
        if is_valid_state(positions):
            return {normalize_state(positions)}
        return set()

    # Optimization: Memoization could happen here if needed, but rolls are small
    
    die = roll[0]
    remaining_roll = roll[1:]
    
    next_states = set()
    # Identify unique positions to avoid redundant moves
    unique_positions = sorted(list(set(positions)))
    
    for p_val in unique_positions:
        # Move one checker from p_val
        temp_list = list(positions)
        # Find index of first occurrence (doesn't matter which one since sorted/identical)
        idx_to_move = temp_list.index(p_val)
        
        temp_list[idx_to_move] += die
        # We RE-SORT immediately to keep canonical form for the recursive step?
        # Actually, `positions` is sorted. `temp_list` might not be.
        # But `get_moves` expects `positions` to be any tuple?
        # Let's standardize: always pass sorted tuple to recursion.
        new_pos = tuple(sorted(temp_list))
        
        sub_results = get_moves(new_pos, remaining_roll)
        next_states.update(sub_results)
        
    return next_states

def solve_expected_rolls(dice_sides=2, tolerance=1e-8, max_iter=5000):
    start_state = (0, 0, 0, 0, 0)
    
    print(f"Solving for dice_sides={dice_sides}...")
    
    roll_configs = []
    if dice_sides == 2:
        # (1,1) -> 4 moves of 1. Prob 0.25
        roll_configs.append((0.25, tuple([1]*4)))
        # (2,2) -> 4 moves of 2. Prob 0.25
        roll_configs.append((0.25, tuple([2]*4)))
        # (1,2) or (2,1) -> 2 moves: 1 and 2. Prob 0.50
        roll_configs.append((0.50, tuple([1, 2])))
        
    elif dice_sides == 6:
        # Doubles (1/36 each) -> 4 moves
        for d in range(1, 7):
            roll_configs.append((1.0/36.0, tuple([d]*4)))
        # Mixed (2/36 each) -> 2 moves
        import itertools
        for d1, d2 in itertools.combinations(range(1, 7), 2):
            roll_configs.append((2.0/36.0, tuple([d1, d2])))

    # State Discovery
    MAX_GAP = 60
    
    transition_map = {}
    queue = deque([start_state])
    # visited set tracks states that we have decided to EXPLORE (or are exploring)
    visited = {start_state}
    
    states_list = [] # ordered list for iteration
    
    while queue:
        if len(states_list) % 50 == 0:
            print(f"Discovered {len(states_list)} states so far... Queue: {len(queue)}", file=sys.stderr)
        s = queue.popleft()
        states_list.append(s)
        
        gap = s[-1] - s[0]
        if gap > MAX_GAP:
            # We treat this as a boundary state. We do not solve for its value accurately
            # by looking forward, but rather assume a heuristic or leave it as a sink.
            # We won't generate its successors.
            continue
            
        trans_list = []
        possible_moves = True
        
        for prob, dice_vals in roll_configs:
            # Get all valid final configurations for this roll
            succ_set = get_moves(s, dice_vals)
            succ_list = list(succ_set)
            
            trans_list.append((prob, succ_list))
            
            for ns in succ_list:
                if ns not in visited:
                    visited.add(ns)
                    queue.append(ns)
        
        transition_map[s] = trans_list

    print(f"Discovered {len(visited)} states. Computing values...")
    
    # Value Iteration
    # V[s] = Expected number of rolls (including the current one)
    # Initialize V.
    # For boundary states (gap > MAX_GAP) or unexpanded states, we default to 1.0.
    V = {s: 1.0 for s in visited}
    
    # Optimization: iterate only over states in transition_map (inner nodes)
    # Boundary nodes stay fixed at 1.0
    inner_states = [s for s in states_list if s in transition_map]
    
    for i in range(max_iter):
        max_delta = 0.0
        
        for s in inner_states:
            # E[s] = 1 + sum(prob * max(E[next]))
            # If a roll leads to NO valid states (blot unavoidable), that branch adds 0 to the sum.
            
            outcome_sum = 0.0
            
            for prob, successors in transition_map[s]:
                if not successors:
                    # Game ends. Value of next state is 0.
                    pass
                else:
                    # We choose the best move to maximize expected survival/rolls
                    best_val = -1.0
                    for ns in successors:
                        v_ns = V.get(ns, 1.0) # Fallback to 1.0 if missing (shouldn't happen with visited set)
                        if v_ns > best_val:
                            best_val = v_ns
                            
                    outcome_sum += prob * best_val
            
            new_val = 1.0 + outcome_sum
            
            diff = abs(new_val - V[s])
            if diff > max_delta:
                max_delta = diff
            
            V[s] = new_val
            
        if i % 100 == 0:
            print(f"Iter {i}: delta={max_delta:.8f}, V(start)={V[start_state]:.6f}")
            
        if max_delta < tolerance:
            print(f"Converged at iter {i} with delta {max_delta}")
            break
            
    return V[start_state]

if __name__ == "__main__":
    result = solve_expected_rolls(dice_sides=2)
    print(f"Result (1-2 dice): {result:.10f}")
    
    # Uncomment to run bonus
    result_bonus = solve_expected_rolls(dice_sides=6)
    print(f"Result (1-6 dice): {result_bonus:.10f}")
