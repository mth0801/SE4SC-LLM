"""
Baseline state selection strategies for symbolic execution.

Each strategy implements a `select(temp_nodes, context)` method that
returns the index of the selected node.

context dict (built by SymExec.execute each selection step):
    control_flow_graph   : dict  — current partial CFG {source: [target, ...]}
    real_bytecode_len    : int   — len(real_bytecode), used for normalization
    all_jump_jumpi_number: int   — total JUMP+JUMPI count
    STACK_MAX, SUCCESSOR_MAX, TEST_CASE_NUMBER_MAX,
    DEPTH_MAX, ICNT_MAX, SUBPATH_MAX : normalization constants
"""

import random
from collections import deque
import numpy as np


class BaseStrategy:
    """Base class for all state selection strategies."""

    name = "base"

    def select(self, temp_nodes, context=None):
        """Return the index of the selected node in temp_nodes."""
        raise NotImplementedError


class MythrilBFS(BaseStrategy):
    """
    Mythril-style BFS + depth limit.

    Selects the shallowest state (smallest depth).
    States exceeding max_depth are skipped.
    """

    name = "mythril"

    def __init__(self, max_depth=50):
        self.max_depth = max_depth

    def select(self, temp_nodes, context=None):
        best_idx = -1
        best_depth = float("inf")

        for i, node in enumerate(temp_nodes):
            if node.depth > self.max_depth:
                continue
            if node.depth < best_depth:
                best_depth = node.depth
                best_idx = i

        # Fallback: if all exceed max_depth, pick shallowest anyway
        if best_idx == -1:
            for i, node in enumerate(temp_nodes):
                if node.depth < best_depth:
                    best_depth = node.depth
                    best_idx = i

        return best_idx


class ParaDySEStrategy(BaseStrategy):
    """
    ParaDySE-style parameterized heuristic search.

    Core idea (Cha et al., IEEE TSE 2021): define a scoring function
        score(s) = sum( w_i * f_i(s) )
    where f_i are program features and w_i are learned weights.

    We use the 10-dim SEF features already available on each node.
    Weights are optimized via random search on the training set
    (call `optimize_weights` before inference), or can be set manually.

    Feature order (matching SEF):
        [stack_size, successor_number, test_case_number,
         branch_new_instruction, path_new_instruction, depth,
         cpicnt, icnt, covNew, subpath]
    """

    name = "paradyse"

    def __init__(self, weights=None, n_search=200, seed=42):
        """
        Args:
            weights: 10-dim weight vector. If None, uses uniform weights
                     until optimize_weights() is called.
            n_search: number of random weight vectors to try during optimization.
            seed: random seed for reproducibility.
        """
        if weights is not None:
            self.weights = np.array(weights, dtype=float)
        else:
            self.weights = np.ones(10, dtype=float)
        self.n_search = n_search
        self.seed = seed

    def _normalize_features(self, node, context):
        """Extract and normalize the 10-dim SEF from a node."""
        rbl = context["real_bytecode_len"]
        return [
            min(node.stack_size / context["STACK_MAX"], 1),
            min(node.successor_number / context["SUCCESSOR_MAX"], 1),
            min(node.test_case_number / context["TEST_CASE_NUMBER_MAX"], 1),
            min(node.branch_new_instruction / rbl, 1) if rbl > 0 else 0,
            min(node.path_new_instruction / rbl, 1) if rbl > 0 else 0,
            min(node.depth / context["DEPTH_MAX"], 1),
            min(node.cpicnt / rbl, 1) if rbl > 0 else 0,
            min(node.icnt / context["ICNT_MAX"], 1),
            min(node.covNew / rbl, 1) if rbl > 0 else 0,
            min(node.subpath / context["SUBPATH_MAX"], 1),
        ]

    def _score(self, node, context):
        """Compute weighted score for a single node."""
        feats = self._normalize_features(node, context)
        return float(np.dot(self.weights, feats))

    def select(self, temp_nodes, context=None):
        best_idx = 0
        best_score = -float("inf")
        for i, node in enumerate(temp_nodes):
            s = self._score(node, context)
            if s > best_score:
                best_score = s
                best_idx = i
        return best_idx

    def optimize_weights(self, training_fn):
        """
        Optimize weights via random search.

        Args:
            training_fn: callable(weights) -> float
                Given a weight vector, runs symbolic execution on the
                training set and returns the average CFG coverage.
                The caller is responsible for wiring this up.
        """
        rng = np.random.RandomState(self.seed)
        best_coverage = -1
        best_w = self.weights.copy()

        for _ in range(self.n_search):
            # Sample random weights in [-1, 1], then normalize to unit length
            w = rng.uniform(-1, 1, size=10)
            w = w / (np.linalg.norm(w) + 1e-8)

            coverage = training_fn(w)
            if coverage > best_coverage:
                best_coverage = coverage
                best_w = w.copy()

        self.weights = best_w
        return best_coverage, best_w


class SmartExecutorStrategy(BaseStrategy):
    """
    SmartExecutor-style coverage-driven search with state prioritization
    and function-level guidance (Liu et al., ACM TOSEM 2024).

    Two-phase model:
      Phase 1 (warm-up): BFS exploration until the CFG has enough edges
        to identify per-function coverage.
      Phase 2 (targeted): Identify not-fully-covered functions, then
        prioritize states that (a) belong to an under-covered function
        and (b) are closest to uncovered basic blocks within that function.

    Function boundaries are derived from the dispatcher_boundary and
    smartcontract_functions_index_position provided in context.
    """

    name = "smartexecutor"

    def __init__(self, warmup_steps=10):
        self._step = 0
        self._warmup_steps = warmup_steps

    def select(self, temp_nodes, context=None):
        self._step += 1
        cfg = context.get("control_flow_graph", {}) if context else {}

        # Phase 1: warm-up with BFS until we have enough CFG info
        if self._step <= self._warmup_steps or not cfg:
            return self._fallback_bfs(temp_nodes)

        func_positions = context.get("smartcontract_functions_index_position", [])
        dispatcher_boundary = context.get("dispatcher_boundary", -1)

        # Build function ranges from sorted positions
        func_ranges = []
        sorted_pos = sorted(func_positions)
        for i in range(len(sorted_pos) - 1):
            func_ranges.append((sorted_pos[i], sorted_pos[i + 1] - 1))
        if sorted_pos:
            rbl = context.get("real_bytecode_len", 0)
            func_ranges.append((sorted_pos[-1], rbl - 1))

        # Compute per-function coverage
        covered_sources = set(cfg.keys())
        all_targets = set()
        for targets in cfg.values():
            all_targets.update(targets)
        all_covered = covered_sources | all_targets

        func_coverage = []  # (func_start, func_end, covered_count, total_jumps)
        undercovered_funcs = []
        for fstart, fend in func_ranges:
            if fstart <= dispatcher_boundary:
                continue  # skip dispatcher region
            # Count JUMP/JUMPI in this function range from real_bytecode
            real_bytecode = context.get("real_bytecode", [])
            total_jumps = 0
            for idx in range(fstart, min(fend + 1, len(real_bytecode))):
                if real_bytecode[idx] in ("JUMP", "JUMPI"):
                    total_jumps += 1
            if total_jumps == 0:
                continue
            # Count covered jump sources in this function
            covered_in_func = sum(1 for s in covered_sources if fstart <= s <= fend)
            ratio = covered_in_func / total_jumps if total_jumps > 0 else 1.0
            func_coverage.append((fstart, fend, ratio))
            if ratio < 1.0:
                undercovered_funcs.append((fstart, fend, ratio))

        # If all functions are fully covered or no function info, use distance
        if not undercovered_funcs:
            return self._distance_select(temp_nodes, cfg)

        # Sort under-covered functions by coverage ratio (lowest first)
        undercovered_funcs.sort(key=lambda x: x[2])

        # Build uncovered frontier within under-covered functions
        uncovered_in_target = set()
        for fstart, fend, _ in undercovered_funcs:
            for t in all_targets:
                if fstart <= t <= fend and t not in covered_sources:
                    uncovered_in_target.add(t)

        if not uncovered_in_target:
            return self._distance_select(temp_nodes, cfg)

        # Reverse BFS from uncovered nodes to find distance to candidates
        dist = self._reverse_bfs(cfg, uncovered_in_target)

        # Score: prefer candidates in under-covered functions AND close to uncovered
        best_idx = 0
        best_score = (float("inf"), float("inf"))  # (func_coverage, distance)
        for i, node in enumerate(temp_nodes):
            pos = node.bytecode_list_index
            # Find which function this state belongs to
            func_cov = 1.0
            for fstart, fend, ratio in undercovered_funcs:
                if fstart <= pos <= fend:
                    func_cov = ratio
                    break
            d = dist.get(pos, float("inf"))
            score = (func_cov, d)  # lower is better on both dimensions
            if score < best_score:
                best_score = score
                best_idx = i

        return best_idx

    def _distance_select(self, temp_nodes, cfg):
        """Pure distance-based selection as fallback."""
        covered_sources = set(cfg.keys())
        all_targets = set()
        for targets in cfg.values():
            all_targets.update(targets)
        uncovered = all_targets - covered_sources
        if not uncovered:
            return self._fallback_bfs(temp_nodes)
        dist = self._reverse_bfs(cfg, uncovered)
        best_idx = 0
        best_dist = float("inf")
        for i, node in enumerate(temp_nodes):
            d = dist.get(node.bytecode_list_index, float("inf"))
            if d < best_dist:
                best_dist = d
                best_idx = i
        if best_dist == float("inf"):
            return self._fallback_bfs(temp_nodes)
        return best_idx

    def _reverse_bfs(self, cfg, sources):
        """Multi-source BFS on reverse CFG. Returns {node: min_dist}."""
        rev_adj = {}
        for src, tgts in cfg.items():
            if src not in rev_adj:
                rev_adj[src] = []
            for t in tgts:
                if t not in rev_adj:
                    rev_adj[t] = []
                rev_adj[t].append(src)
        dist = {}
        queue = deque()
        for s in sources:
            dist[s] = 0
            queue.append(s)
        while queue:
            node = queue.popleft()
            for neighbor in rev_adj.get(node, []):
                if neighbor not in dist:
                    dist[neighbor] = dist[node] + 1
                    queue.append(neighbor)
        return dist

    def _fallback_bfs(self, temp_nodes):
        best_idx = 0
        best_depth = float("inf")
        for i, node in enumerate(temp_nodes):
            if node.depth < best_depth:
                best_depth = node.depth
                best_idx = i
        return best_idx



class EmpcStrategy(BaseStrategy):
    """
    Empc-style minimum path cover guided search
    (Yao & She, 2025, based on KLEE).

    Core algorithm:
    1. Build DAG from partial CFG by collapsing SCCs (Tarjan's).
    2. Compute Minimum Path Cover (MPC) on the DAG via:
       a. Construct bipartite graph from DAG edges.
       b. Find maximum matching using Hopcroft-Karp algorithm.
       c. MPC size = |V_dag| - |matching|; derive actual paths.
    3. At runtime, for each candidate state, check if it lies on an
       MPC path that contains uncovered nodes. Prefer states on paths
       with the most uncovered nodes.

    The MPC is recomputed when the CFG changes significantly (new edges
    discovered), controlled by _cfg_edge_count cache.
    """

    name = "empc"

    def __init__(self):
        self._cached_mpc_paths = None  # list of paths (each path = list of DAG node ids)
        self._cached_cfg_edge_count = -1
        self._cached_node_to_scc = None
        self._cached_scc_to_nodes = None
        self._cached_dag_adj = None

    def select(self, temp_nodes, context=None):
        cfg = context.get("control_flow_graph", {}) if context else {}

        if not cfg:
            return self._fallback_bfs(temp_nodes)

        # Build adjacency
        adj = {}
        all_nodes_set = set()
        cfg_edge_count = 0
        for src, tgts in cfg.items():
            all_nodes_set.add(src)
            if src not in adj:
                adj[src] = set()
            for t in tgts:
                adj[src].add(t)
                all_nodes_set.add(t)
                if t not in adj:
                    adj[t] = set()
                cfg_edge_count += 1

        # Recompute MPC only when CFG has grown
        if cfg_edge_count != self._cached_cfg_edge_count:
            self._cached_cfg_edge_count = cfg_edge_count
            self._recompute_mpc(all_nodes_set, adj)

        if not self._cached_mpc_paths:
            return self._fallback_bfs(temp_nodes)

        # Identify uncovered nodes
        covered_sources = set(cfg.keys())
        all_targets = set()
        for targets in cfg.values():
            all_targets.update(targets)
        uncovered = all_targets - covered_sources

        if not uncovered:
            return self._fallback_bfs(temp_nodes)

        # Map uncovered to SCC ids
        uncovered_sccs = set()
        for u in uncovered:
            if u in self._cached_node_to_scc:
                uncovered_sccs.add(self._cached_node_to_scc[u])

        # Score each MPC path: count uncovered SCC nodes on it
        path_scores = {}  # scc_id -> max uncovered count on any path containing it
        for path_idx, path in enumerate(self._cached_mpc_paths):
            uncov_on_path = sum(1 for s in path if s in uncovered_sccs)
            for scc_id in path:
                if scc_id not in path_scores or uncov_on_path > path_scores[scc_id]:
                    path_scores[scc_id] = uncov_on_path

        # Also compute reachable uncovered count on DAG for tie-breaking
        reachable_uncov = self._count_reachable_uncovered(
            self._cached_dag_adj, uncovered_sccs
        )

        # Score each candidate state: (mpc_path_uncovered, reachable_uncovered)
        best_idx = 0
        best_score = (-1, -1)
        for i, node in enumerate(temp_nodes):
            pos = node.bytecode_list_index
            if pos in self._cached_node_to_scc:
                scc = self._cached_node_to_scc[pos]
                s1 = path_scores.get(scc, 0)
                s2 = reachable_uncov.get(scc, 0)
            else:
                s1, s2 = 0, 0
            score = (s1, s2)  # higher is better on both
            if score > best_score:
                best_score = score
                best_idx = i

        if best_score <= (0, 0):
            return self._fallback_bfs(temp_nodes)

        return best_idx

    def _recompute_mpc(self, all_nodes_set, adj):
        """Collapse SCCs, build DAG, compute MPC via Hopcroft-Karp."""
        # Step 1: Tarjan's SCC
        n_sccs, dag_adj, node_to_scc = self._build_dag(all_nodes_set, adj)
        self._cached_node_to_scc = node_to_scc

        # Build reverse map: scc_id -> set of original nodes
        scc_to_nodes = {}
        for node, scc_id in node_to_scc.items():
            if scc_id not in scc_to_nodes:
                scc_to_nodes[scc_id] = set()
            scc_to_nodes[scc_id].add(node)
        self._cached_scc_to_nodes = scc_to_nodes

        # Step 2: Compute MPC on DAG
        dag_nodes = list(dag_adj.keys())
        if not dag_nodes:
            self._cached_mpc_paths = []
            return

        self._cached_dag_adj = dag_adj

        # Build bipartite graph: for each edge (u,v) in DAG,
        # add edge (x_u, y_v) in bipartite graph
        # Left nodes: x_0..x_{n-1}, Right nodes: y_0..y_{n-1}
        # Maximum matching -> MPC size = n - |matching|
        matching = self._hopcroft_karp(dag_nodes, dag_adj)

        # Step 3: Derive MPC paths from matching
        # matching: {left_node: right_node} means edge left->right is in matching
        # A path in MPC: follow matching edges: start from unmatched left node,
        # follow matched edge to right, then that right becomes next left, etc.
        self._cached_mpc_paths = self._derive_paths(dag_nodes, dag_adj, matching)

    def _build_dag(self, all_nodes, adj):
        """Collapse SCCs using iterative Tarjan's."""
        index_counter = [0]
        stack = []
        on_stack = set()
        index_map = {}
        lowlink = {}
        sccs = []

        for start in all_nodes:
            if start in index_map:
                continue
            dfs_stack = [(start, iter(adj.get(start, set())), True)]
            index_map[start] = index_counter[0]
            lowlink[start] = index_counter[0]
            index_counter[0] += 1
            stack.append(start)
            on_stack.add(start)

            while dfs_stack:
                v, it, _ = dfs_stack[-1]
                pushed = False
                for w in it:
                    if w not in index_map:
                        index_map[w] = index_counter[0]
                        lowlink[w] = index_counter[0]
                        index_counter[0] += 1
                        stack.append(w)
                        on_stack.add(w)
                        dfs_stack.append((w, iter(adj.get(w, set())), True))
                        pushed = True
                        break
                    elif w in on_stack:
                        lowlink[v] = min(lowlink[v], lowlink[w])

                if not pushed:
                    if lowlink[v] == index_map[v]:
                        scc = []
                        while True:
                            w = stack.pop()
                            on_stack.discard(w)
                            scc.append(w)
                            if w == v:
                                break
                        sccs.append(scc)
                    dfs_stack.pop()
                    if dfs_stack:
                        parent = dfs_stack[-1][0]
                        lowlink[parent] = min(lowlink[parent], lowlink[v])

        node_to_scc = {}
        for i, scc in enumerate(sccs):
            for node in scc:
                node_to_scc[node] = i

        dag_adj = {i: set() for i in range(len(sccs))}
        for src, tgts in adj.items():
            src_scc = node_to_scc.get(src)
            if src_scc is None:
                continue
            for t in tgts:
                t_scc = node_to_scc.get(t)
                if t_scc is not None and src_scc != t_scc:
                    dag_adj[src_scc].add(t_scc)

        return len(sccs), dag_adj, node_to_scc

    def _hopcroft_karp(self, dag_nodes, dag_adj):
        """Hopcroft-Karp maximum bipartite matching on DAG edges.

        Bipartite graph: Left = {x_u for u in dag_nodes},
                         Right = {y_v for v in dag_nodes}.
        Edge (x_u, y_v) exists iff (u, v) is an edge in DAG.

        Returns matching as dict {u: v} meaning x_u matched to y_v.
        """
        # Build adjacency for left side
        left_adj = {u: list(dag_adj.get(u, set())) for u in dag_nodes}

        match_left = {}   # u -> v
        match_right = {}  # v -> u
        INF = float("inf")

        def bfs():
            dist = {}
            queue = deque()
            for u in dag_nodes:
                if u not in match_left:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = INF
            found = False
            while queue:
                u = queue.popleft()
                for v in left_adj.get(u, []):
                    next_u = match_right.get(v)
                    if next_u is None:
                        found = True
                    elif dist.get(next_u, INF) == INF:
                        dist[next_u] = dist[u] + 1
                        queue.append(next_u)
            return found, dist

        def dfs(u, dist):
            for v in left_adj.get(u, []):
                next_u = match_right.get(v)
                if next_u is None or (dist.get(next_u, INF) == dist[u] + 1 and dfs(next_u, dist)):
                    match_left[u] = v
                    match_right[v] = u
                    return True
            dist[u] = INF
            return False

        while True:
            found, dist = bfs()
            if not found:
                break
            for u in dag_nodes:
                if u not in match_left:
                    dfs(u, dist)

        return match_left

    def _derive_paths(self, dag_nodes, dag_adj, matching):
        """Derive MPC paths from the maximum matching.

        Each matched edge u->v means u and v are consecutive on a path.
        Unmatched left nodes are path starts; unmatched right nodes are path ends.
        """
        # matching: {u: v} means on the path, after visiting u we visit v
        # Build successor chain from matching
        successor = {}  # node -> next node on path
        has_predecessor = set()
        for u, v in matching.items():
            successor[u] = v
            has_predecessor.add(v)

        # Path starts: nodes that have no predecessor in matching
        dag_node_set = set(dag_nodes)
        starts = [n for n in dag_node_set if n not in has_predecessor]

        paths = []
        for s in starts:
            path = [s]
            current = s
            while current in successor:
                current = successor[current]
                path.append(current)
            paths.append(path)

        # Also add isolated nodes (not in any matching edge and not a start)
        in_path = set()
        for p in paths:
            in_path.update(p)
        for n in dag_node_set:
            if n not in in_path:
                paths.append([n])

        return paths

    def _count_reachable_uncovered(self, dag_adj, uncovered_sccs):
        """For each DAG node, count how many uncovered SCCs are reachable."""
        cache = {}

        def dfs(node):
            if node in cache:
                return cache[node]
            reachable = set()
            if node in uncovered_sccs:
                reachable.add(node)
            for neighbor in dag_adj.get(node, set()):
                reachable |= dfs(neighbor)
            cache[node] = reachable
            return reachable

        result = {}
        for node in dag_adj:
            result[node] = len(dfs(node))
        return result

    def _fallback_bfs(self, temp_nodes):
        best_idx = 0
        best_depth = float("inf")
        for i, node in enumerate(temp_nodes):
            if node.depth < best_depth:
                best_depth = node.depth
                best_idx = i
        return best_idx
