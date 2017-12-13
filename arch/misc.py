#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

class ExponentialDecay(object):
    """Iterator class to generate rates with exponential decay.

    Attributes:
        start: A number representing the start rate.
        stop: A number representing the stop rate.
        max_step: A number representing the steps required to decay from start
            to stop rate.
        decay_rate: A number representing the rate of decay required to the
            above variables.
        steps: A number representing the steps an item requested.

    """    
    def __init__(self, start=0.1, stop=0.01, max_steps=100):
        self.start = start
        self.stop = stop
        self.max_steps = max_steps
        self.decay_rate = math.log(self.stop/self.start)/self.max_steps
        self.steps = 0
        
    def __iter__(self):
        # Method required for iterator objects. Returns itself.
        return self
    
    def next(self):
        return self.__next__()
    
    def __next__(self):
        """Returns the next rate item.

        Returns:
            If steps <= max_steps then rate=exp(decay_rate*steps)*start, else
            rate = stop.

        """
        if self.steps >= self.max_steps:
            value = self.stop
        else:
            value = math.exp(self.decay_rate*self.steps)*self.start
        self.steps = self.steps+1
        return value