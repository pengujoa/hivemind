from typing import Optional, Sequence, Tuple

import numpy as np
import scipy.optimize

from hivemind.utils.logging import get_logger

logger = get_logger(__name__)

LOAD_BALANCING_LP_DECIMALS = 9


def load_balance_peers(vector_size, bandwidths: Sequence[Optional[float]], min_size: int = 0) -> Tuple[int, ...]:
    """
    Find an optimal partitioning of weights for butterfly all-reduce given peer bandwidths.
    :param vector_size: total size of the averaged vector (in elements, not bytes)
    :param bandwidths: 1d array of non-negative bandwidths for each peer capable of averaging
      zeros stand for client-only participants, None represents "not specified" (resolved as mean of other pears)
    :param min_size: peers that can aggregate less than this many elements will be assigned nothing
    :returns: an integer array where i-th element is the number of weights assigned to i-th peer
    """
    specified_bandwidth = [item for item in bandwidths if item is not None and item > 0]
    print(f"specified_bandwidth: {specified_bandwidth}")

    bandwidths_2d = np.array([[1,    580,    840,    820],
                         [580,  1,      840,    840],
                         [840,  840,    1,      940],
                         [820,  840,    940,    1]])

    if specified_bandwidth:
        default_bandwidth = np.mean(specified_bandwidth)
        bandwidths = [item if item is not None else default_bandwidth for item in bandwidths]
        # scores = optimize_parts_lp(vector_size, np.asarray(bandwidths), min_size)
        # scores = optimize_parts_lp_with_max_link(100, bandwidths_2d)
        
    else:
        assert not all(item == 0 for item in bandwidths), "Must have at least one nonzero bandwidth"
        scores = np.asarray([1.0 if item is None else 0.0 for item in bandwidths])

    # TODO(jheuristic) we no longer need hagenbach-bishoff with new AllReduceRunner
    # return tuple(hagenbach_bishoff(vector_size, scores))
    return tuple(hagenbach_bishoff(vector_size, bandwidths))


def optimize_parts_lp(vector_size: int, bandwidths: np.ndarray, min_size: int = 0) -> np.ndarray:
    """
    This method solves an optimization problem to minimize the total allreduce time.
    In butterfly all-reduce, each peer acts both as a "client" and as an "aggregator":
    * a "client" splits his local vector into shards and sends each shard to one peer, then downloads the average
    * an "aggregator" receives a certain part of vector components from all peers, aggregates and returns the average

    Peer i network load as a "client" = vector_size * (1 - fraction_assigned_to_peer_i)
    Peer i network load as an "aggregator" = vector_size * (group_size - 1) * fraction_assigned_to_peer_i
    Peer i total communication = vector_size * [1 + (group_size - 2) * fraction_assigned_to_peer_i]
    Total time = max_i (total_communication_for_peer_i / bandwidths[i])

    We solve this optimization problem by reducing it to linear programming with a minimax reduction
    (see lecture notes: https://www.usna.edu/Users/math/dphillip/sa305.s15/phillips/lessons/32/32.pdf )

    :returns: a vector of "scores", i-th score is proportional to the fraction of weights assigned to i-th peer
    """
    assert np.all(bandwidths >= 0) and np.any(bandwidths > 0) # 입력된 모든 대역폭은 음수가 아니고, 적어도 하나 이상의 피어는 양의 대역폭을 가져야 합니다.
    bandwidths = np.asarray(bandwidths, dtype=np.float64) # 입력된 bandwidth 리스트를 numpy 배열로 변환합니다.

    # 대역폭이 큰 피어들을 먼저 처리하여 최적화 과정의 효율을 높입니다.
    permutation = np.argsort(-bandwidths) 
    bandwidths = bandwidths[permutation] # 정렬된 대역폭 배열을 새로운 배열에 할당합니다.

    # 대역폭이 0인 피어들을 제외하고 나머지 피어들을 나타내는 불리언 배열을 생성합니다. (대역폭이 0이 아닌 피어들을 마크합니다.)
    is_nonzero = bandwidths != 0 

    # 최적화 문제에 사용할 변수 개수는 피어 수 + 1개(xi라는 추가 변수가 있음).
    group_size = len(bandwidths) 
    num_variables = group_size + 1  # [w_1, ..., w_N, xi]

    # 최적화 문제에 사용할 변수 배열을 초기화합니다. 
    # 최적화 목표를 xi (통신 시간의 최대값)를 최소화하는 것으로 설정합니다.
    c = np.zeros(num_variables, dtype=np.float64) 
    c[-1] = 1.0  # optimize w.r.t. xi 

    # the constraints below are tuples (A, b) such that Ax <= b
    nonnegative_weights = -np.eye(group_size, num_variables, dtype=c.dtype), np.zeros(group_size, c.dtype) # 할당된 작업 비율은 음수가 될 수 없습니다. 모든 변수가 0 이상이어야 합니다. 
    weights_sum_to_one = c[None, :] - 1.0, np.array([-1.0]) # 모든 피어의 작업 비율의 합은 반드시 1이어야 합니다 

    # 각 피어의 통신 부하를 계산하고, 모든 피어의 통신 부하 중 xi가 최대값이 되어야 하는 제약입니다.
    coeff_per_variable = (group_size - 2.0) / np.maximum(bandwidths, 10**-LOAD_BALANCING_LP_DECIMALS)
    coeff_matrix_minus_xi = np.hstack([np.diag(coeff_per_variable), -np.ones((group_size, 1), c.dtype)])
    xi_is_maximum = coeff_matrix_minus_xi[is_nonzero], -1.0 / bandwidths[is_nonzero]
    force_max_weights = np.eye(group_size, M=num_variables, dtype=c.dtype), is_nonzero.astype(c.dtype)

    A, b = list(map(np.concatenate, zip(nonnegative_weights, weights_sum_to_one, xi_is_maximum, force_max_weights)))

    solution = scipy.optimize.linprog(c, A_ub=A, b_ub=b, method="interior-point")
    if solution.success:
        peer_scores = solution.x[:group_size]
        # if some peers have less than min_size elements, transfer their share to other peers (if any)
        if np.max(peer_scores) >= min_size / float(vector_size):
            peer_scores[peer_scores < min_size / float(vector_size)] = 0.0
        peer_scores = np.round(peer_scores, LOAD_BALANCING_LP_DECIMALS)
    else:
        logger.error(f"Failed to solve load-balancing for bandwidths {bandwidths}")
        peer_scores = np.ones(group_size, c.dtype)

    return peer_scores[np.argsort(permutation)]


def optimize_parts_lp_with_max_link(
    vector_size: int,
    B: np.ndarray,         # shape (N, N)
    min_size: int = 0
) -> np.ndarray:
    """
    Optimize load balancing using LP, considering max link times for send/recv.
    Variables: [w_0..w_{N-1}, xi,
                tau_send_0..tau_send_{N-1},
                tau_recv_0..tau_recv_{N-1}]
    Constraints:
      w_i >= 0, sum_i w_i = 1
      for each i, j != i:
        V*w_j/B[i,j] <= tau_send_i
        V*w_i/B[j,i] <= tau_recv_i
      and for each i:
        tau_send_i <= xi
        tau_recv_i <= xi
    """
    N = B.shape[0]
    num_vars = N + 1 + 2 * N  # w's + xi + tau_send + tau_recv

    # Objective: minimize xi
    c = np.zeros(num_vars, dtype=np.float64)
    c[N] = 1.0

    # 1) w_i >= 0
    A_ub = np.hstack([
        -np.eye(N),                # w
        np.zeros((N, 1)),          # xi
        np.zeros((N, N)),          # tau_send
        np.zeros((N, N))           # tau_recv
    ])
    b_ub = np.zeros(N, dtype=np.float64)

    # 2) sum_i w_i = 1
    A_eq = np.hstack([
        np.ones((1, N)), [[0.0]],  # xi
        np.zeros((1, N)),           # tau_send
        np.zeros((1, N))            # tau_recv
    ])
    b_eq = np.array([1.0], dtype=np.float64)

    rows, rhs = [], []

    # 3a) V*w_j/B[i,j] <= tau_send_i
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            row = np.zeros(num_vars, dtype=np.float64)
            row[j] = vector_size / B[i, j]
            row[N + 1 + i] = -1.0  # tau_send_i
            rows.append(row)
            rhs.append(0.0)

    # 3b) V*w_i/B[j,i] <= tau_recv_i
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            row = np.zeros(num_vars, dtype=np.float64)
            row[i] = vector_size / B[j, i]
            row[N + 1 + N + i] = -1.0  # tau_recv_i
            rows.append(row)
            rhs.append(0.0)

    # 4a) tau_send_i <= xi
    for i in range(N):
        row = np.zeros(num_vars, dtype=np.float64)
        row[N + 1 + i] = 1.0   # tau_send_i
        row[N] = -1.0          # xi
        rows.append(row)
        rhs.append(0.0)

    # 4b) tau_recv_i <= xi
    for i in range(N):
        row = np.zeros(num_vars, dtype=np.float64)
        row[N + 1 + N + i] = 1.0  # tau_recv_i
        row[N] = -1.0             # xi
        rows.append(row)
        rhs.append(0.0)

    # Stack constraints
    A_ub = np.vstack([A_ub, np.array(rows)])
    b_ub = np.concatenate([b_ub, np.array(rhs)])

    # Solve LP
    sol = scipy.optimize.linprog(
        c, A_ub=A_ub, b_ub=b_ub,
        A_eq=A_eq, b_eq=b_eq,
        method="interior-point"
    )
    if not sol.success:
        logger.error("LP failed with max-link model: %s", sol.message)
        return np.ones(N, dtype=np.float64) / N

    w = sol.x[:N]
    # Post-process min_size
    if np.max(w) >= min_size / float(vector_size):
        w[w < min_size / float(vector_size)] = 0.0
    return w


def optimize_parts_lp_hybrid(
    vector_size: int,
    eff_matrix: np.ndarray,
    min_size: int = 0,
) -> np.ndarray:
    """
    Optimize tensor partitioning using per-link fraction-based effective throughput.

    eff[i][j] = w_j_observed / time_observed[i][j]  (fraction per second)

    The LP predicts completion time as:
        time_new[i][j] = w_j_new / eff[i][j] = time_obs[i][j] * (w_j_new / w_j_old)
    This directly uses observed completion times as the baseline, making the
    optimization robust to fixed overhead costs in the communication pipeline.

    Node i's completion time is bottlenecked by the slowest link:
        tau_i = max_{j!=i} { w_j / eff[i][j] }
    We minimize the overall all-reduce time: min max_i { tau_i }.

    :param vector_size: total number of elements (used only for min_size filtering)
    :param eff_matrix: N x N matrix where eff_matrix[i][j] is the fraction-based
        effective throughput (frac/sec). Asymmetric. Diagonal is ignored.
    :param min_size: peers assigned fewer than this many elements get nothing
    :returns: scores vector (fractions), to be passed to hagenbach_bishoff
    """
    N = eff_matrix.shape[0]
    assert eff_matrix.shape == (N, N), f"Expected square matrix, got {eff_matrix.shape}"
    V = float(vector_size)

    num_vars = N + 1 + N

    c = np.zeros(num_vars, dtype=np.float64)
    c[N] = 1.0

    A_eq = np.zeros((1, num_vars), dtype=np.float64)
    A_eq[0, :N] = 1.0
    b_eq = np.array([1.0], dtype=np.float64)

    rows_ub, rhs_ub = [], []

    for i in range(N):
        row = np.zeros(num_vars, dtype=np.float64)
        row[i] = -1.0
        rows_ub.append(row)
        rhs_ub.append(0.0)

    # w_j / eff[i][j] <= tau_i  for all i, j!=i
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            e = eff_matrix[i, j]
            if e <= 0:
                e = 1e-10
            row = np.zeros(num_vars, dtype=np.float64)
            row[j] = 1.0 / e          # w_j / eff[i][j]
            row[N + 1 + i] = -1.0     # -tau_i
            rows_ub.append(row)
            rhs_ub.append(0.0)

    for i in range(N):
        row = np.zeros(num_vars, dtype=np.float64)
        row[N + 1 + i] = 1.0
        row[N] = -1.0
        rows_ub.append(row)
        rhs_ub.append(0.0)

    A_ub = np.array(rows_ub, dtype=np.float64)
    b_ub = np.array(rhs_ub, dtype=np.float64)

    sol = scipy.optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method="highs")
    if sol.success:
        w = sol.x[:N]
        w = np.maximum(w, 0.0)
        if V > 0 and np.max(w) >= min_size / V:
            w[w < min_size / V] = 0.0
        w = np.round(w, LOAD_BALANCING_LP_DECIMALS)
        logger.info(f"LP hybrid solved: w={w}, xi(predicted_max_time)={sol.x[N]:.6f}s")
        return w
    else:
        logger.error(f"LP hybrid failed: {sol.message}. Falling back to uniform.")
        return np.ones(N, dtype=np.float64)


def hagenbach_bishoff(vector_size: int, scores: Sequence[float]) -> Sequence[int]:
    """
    Split a vector between participants based on continuous fractions.
    https://en.wikipedia.org/wiki/Hagenbach-Bischoff_system
    The code is based on https://github.com/crflynn/voting

    :param vector_size: the total number of elements to be split
    :param scores: real-valued vector fractions for each peer
    :returns: integer-valued partitions assigned to every peer
    """
    total_score = sum(scores)
    allocated = [int(vector_size * score_i / total_score) for score_i in scores]
    while sum(allocated) < vector_size:
        quotients = [score / (allocated[idx] + 1) for idx, score in enumerate(scores)]
        idx_max = quotients.index(max(quotients))
        allocated[idx_max] += 1
    return allocated
