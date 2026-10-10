import heapq
import numpy as np
from scipy.ndimage import binary_dilation


def plan(grid, start, goal, inflation=0):
    """0 free, 1 occupied, -1 unknown; unknown is blocked."""
    grid = np.asarray(grid)
    blocked = grid != 0
    if inflation:
        blocked = binary_dilation(blocked, iterations=inflation)

    def valid(p):
        return (
            0 <= p[0] < grid.shape[0] and 0 <= p[1] < grid.shape[1] and not blocked[p]
        )

    if not valid(start) or not valid(goal):
        return None
    queue = [(0, 0, start)]
    cost = {start: 0}
    parent = {}
    while queue:
        _, g, p = heapq.heappop(queue)
        if g != cost[p]:
            continue
        if p == goal:
            path = [p]
            while p in parent:
                p = parent[p]
                path.append(p)
            return path[::-1]
        for d in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            q = (p[0] + d[0], p[1] + d[1])
            ng = g + 1
            if valid(q) and ng < cost.get(q, float("inf")):
                cost[q] = ng
                parent[q] = p
                h = abs(q[0] - goal[0]) + abs(q[1] - goal[1])
                heapq.heappush(queue, (ng + h, ng, q))
    return None
