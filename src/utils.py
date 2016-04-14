def do_all(items, f):
    """ Apply F on either every element of ITEMS, if ITEMS is a list, or just
    on ITEMS as a single element otherwise.
    """
    if type(items) == list:
        return [f(i) for i in items]
    return f(items)

def flatten(lst):
    """ Flatten shallow list LST. """
    return [e for l2 in lst for e in l2]
