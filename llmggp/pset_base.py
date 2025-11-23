#!/usr/bin/env python3
"""Reusable primitive helpers (task-agnostic)."""

def add_basic_primitives(pset):
    """
    Attach basic arithmetic/logic primitives and bool terminals to a deap.gp primitive set.
    """
    pset.addPrimitive(lambda a, b: a + b, [float, float], float, name="add")
    pset.addPrimitive(lambda a, b: a - b, [float, float], float, name="sub")
    pset.addPrimitive(lambda a, b: a * b, [float, float], float, name="mul")
    pset.addPrimitive(lambda a, b: a / b if b != 0 else a, [float, float], float, name="div")

    def ifelse(c: bool, x: float, y: float) -> float:
        return x if c else y

    pset.addPrimitive(ifelse, [bool, float, float], float, name="ifelse")
    pset.addPrimitive(lambda a, b: a < b, [float, float], bool, name="ltf")
    pset.addPrimitive(lambda a, b: a == b, [float, float], bool, name="eqf")
    pset.addPrimitive(lambda a, b: a > b, [float, float], bool, name="gtf")
    pset.addTerminal(True, bool)
    pset.addTerminal(False, bool)
    return pset


__all__ = ["add_basic_primitives"]
