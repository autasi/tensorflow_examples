#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def tuple_list_find(tuple_list, val, idx=0):
    """Finds elements in a list of tuples.
    Args:
        tuple_list: A list of tuples.
        val: The value of the variable to be found.
        idx: An integer representing the index of the variable in the tuple.
    Returns:
        The first tuple that satisfies the criteria, or None otherwise.
    """    
    elem = next((tup for tup in tuple_list if tup[idx] == val), None)
    return elem