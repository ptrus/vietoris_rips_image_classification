def fix_labels((V, E)):
    """ Number elements in a list (of lists) E (labels of edges) so that
    they are numbered from 0 to (|V|-1).
    """
    transforms = { orig: new for new, orig in enumerate(V) }
    return map(lambda e: do_one_or_list(e, transforms.__getitem__), E)

def find(u, reps):
    """ Finds representative of node U. """
    if reps[u] != u:
        reps[u] = find(reps[u], reps)
    return reps[u]

def union(u, v, reps):
    """ Unites nodes U and V into the same component. """
    u_rep, v_rep = (find(i, reps) for i in [u, v])
    reps[u_rep] = v_rep
    return reps

def connected_components((V, E)):
    """ Return connected components of graph G = (V, E). """
    E = fix_labels((V, E)) # We must fix labels so they are integers from 0 to n_vertices
    reps = range(len(V))
    for (u, v) in E:
        if find(u, reps) != find(v, reps):
            reps = union(u, v, reps)
    groups = [find(u, reps) for u in reps]
    return fix_labels((list(set(groups)), groups)) # And finally fix indices of groups so they are integers from 0 to n_groups

def n_connected_components((V, E)):
    """ Return number of connected components for graph G=(V, E). """
    return len(set(connected_components((V, E))))


if __name__ == "__main__":
    V = "abcdefghi"
    E = map(list, ["ab", "ac", "bc", "cg", "ef"])
    print connected_components((V, E))
