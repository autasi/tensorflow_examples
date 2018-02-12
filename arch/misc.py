#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from operator import mul
from functools import partial


# Huang et al., Snapshot Ensembles: Train 1, get M for free
# https://arxiv.org/abs/1704.00109
# https://github.com/titu1994/Snapshot-Ensembles
class CyclicCosineAnneal(object):
    def __init__(self, a0, M = 1, max_steps = 100):
        self.a0 = a0
        self.M = M
        self.max_steps = max_steps
        self.steps = 0
        self.TperM = self.max_steps // self.M
        self.a0per2 = self.a0/2.0
        self.value = self.a0per2

    def __iter__(self):
        return self
    
    def next(self):
        return self.__next__()
    
    def __next__(self):
        self.value = self.a0per2 * math.cos(math.pi * (self.steps % self.TperM ) / self.TperM) + 1.0
        
        self.steps = self.steps+1
        return self.value     


class FixValue(object):
    def __init__(self, value = 0.01):
        self.value = value
        
    def __iter__(self):
        return self

    def next(self):
        return self.__next__()
    
    def __next__(self):
        return self.value    


class DivideAt(object):
    def __init__(self, start = 0.1, divide_by = 10, at_steps = [50, 75]):
        self.start = start
        self.divide_by = divide_by
        self.at_steps = at_steps
        self.value = self.start
        self.steps = 0
        
    def __iter__(self):
        return self

    def next(self):
        return self.__next__()
    
    def __next__(self):
        if self.steps in self.at_steps:
            self.value = self.value / self.divide_by
        self.steps = self.steps+1
        return self.value    


class DivideAtRates(object):
    def __init__(self, start = 0.1, divide_by = 10, at = [0.5, 0.75], max_steps = 100):
        self.start = start
        self.divide_by = divide_by
        self.at = at
        self.max_steps = max_steps
        self.at_steps = list(map(int, list(map(partial(mul, self.max_steps), self.at))))
        self.value = self.start
        self.steps = 0
        
    def __iter__(self):
        return self

    def next(self):
        return self.__next__()
    
    def __next__(self):
        if self.steps in self.at_steps:
            self.value = self.value / self.divide_by
        self.steps = self.steps+1
        return self.value    


class DivideAtRatesWithDecay(DivideAtRates):
    def __init__(self, start = 0.1, divide_by = 10, at = [0.5, 0.75], max_steps = 100, decay = 1e-6):
        super(DivideAtRatesWithDecay, self).__init__(start, divide_by, at, max_steps)
        self.decay = decay
        
    def __next__(self):
        if self.steps in self.at_steps:
            self.value = self.value / self.divide_by
        val = self.value*1.0/(1.0+self.decay*self.steps)
        self.steps = self.steps+1
        return val
    

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
    
    
class LinearDecay(object):
    
    def __init__(self, start=0.1, stop=0.01, max_steps=100):
        self.start = start
        self.stop = stop
        self.max_steps = max_steps
        self.decay_rate = (stop-start)/max_steps
        self.steps = 0
        
    def __iter__(self):
        # Method required for iterator objects. Returns itself.
        return self
    
    def next(self):
        return self.__next__()
    
    def __next__(self):

        if self.steps >= self.max_steps:
            value = self.stop
        else:
            value = self.start + self.decay_rate*self.steps
        self.steps = self.steps+1
        return value    
    