import search

def h(node):
    state = node.state
    loc_agent = state[-1]
    x, y = loc_agent % 3, loc_agent // 3
    mhd = 0
    sum_dirty = 0
    for idx, s in enumerate(node.state[:-1]):
        if 1 == s:
            sum_dirty = sum_dirty + 1
            dirty_x, dirty_y = idx % 3, idx // 3
            mhd = abs(dirty_x - x) + abs(dirty_y - y) + mhd
    return sum_dirty + mhd


vaccum_problem = search.VacuumClean((1,1,1,0,0,0,0,0,0,4))

node_h, num_node_gen_h = search.astar_search(vaccum_problem, h)
print(node_h.solution())
print("Number of nodes generated using h: {}".format(num_node_gen_h))

node_h2, num_node_gen_h2 = search.astar_search(vaccum_problem)
print(node_h2.solution())
print("Number of nodes generated using h2: {}".format(num_node_gen_h2))
