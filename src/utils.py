def do_one_or_list(item, f):
    """ Apply F on either every element of ITEM, if ITEM is a list, or just
    on ITEM as a single element otherwise.
    """
    if type(item) == list:
        return map(f, item)
    return f(item)

def flatten(lst):
    """ Flatten shallow list LST. """
    return [e for l2 in lst for e in l2]
