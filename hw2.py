import string
import random

# from logic import WalkSAT, pl_true

# Useful constant Exprs used in examples and code:
from utils import Expr, probability

argmax = max

""" This is my code """


def S_generator(n, m, k, q):
    """
    :param n: the number of distinct propositional symbols in S
    :param m: the number of clauses in S
    :param k: the maximum number of literals in a clause in S
    :param q: 0.4 <= q <= 0.6, q is the probability that a literal in a
    clause in a negative literal
    :return: a random set of clauses
    """
    if q < 0.4 or q > 0.6:
        raise ValueError("q should be between 0.4 and 0.6(inclusive)")
    letters = string.ascii_uppercase
    len_letters = len(letters)
    if n > len_letters:
        raise ValueError(
                "n should be less than or equal to " + str(len_letters))
    unique_symbols = list()
    for i in range(n):
        unique_symbols.append(Expr(letters[i]))
    S = list()
    for i in range(m):
        n_lt = random.randint(1, k)
        clause = None
        for j in range(n_lt):
            idx = random.randint(0, n - 1)
            sym = unique_symbols[idx]
            if probability(q):
                sym = sym.__invert__()
            if clause is None:
                clause = sym
            else:
                clause = clause.__or__(sym)
        S.append(clause)
    print("clauses S: " + str(S) + "\n")
    return S


def get_result(model, flips):
    if model is None:
        print("failure after " + str(
            flips) + " many trials to do \"random walk\" in the WALKSAT "
                     "function\n")
    else:
        print("model: " + str(model))
        print("flips: " + str(flips) + "\n")


""" This is my code """

""" This is partly my code """


def WalkSAT(clauses, p=0.5, max_flips=100):
    """Checks for satisfiability of all clauses by randomly flipping values
    of variables
    True
    """
    # Set of all symbols in all clauses
    symbols = {sym for clause in clauses for sym in prop_symbols(clause)}
    # model is a random assignment of true/false to the symbols in clauses
    model = {s: random.choice([True, False]) for s in symbols}
    for i in range(max_flips):
        satisfied, unsatisfied = [], []
        for clause in clauses:
            (satisfied if pl_true(clause, model) else unsatisfied).append(
                    clause)
        if not unsatisfied:  # if model satisfies all the clauses
            return model, i
        clause = random.choice(unsatisfied)
        if probability(p):
            sym = random.choice(list(prop_symbols(clause)))
        else:
            # Flip the symbol in clause that maximizes number of sat. clauses
            def sat_count(sym):
                # Return the the number of clauses satisfied after flipping
                # the symbol.
                model[sym] = not model[sym]
                count = len([clause for clause in clauses if
                             pl_true(clause, model)])
                model[sym] = not model[sym]
                return count

            sym = argmax(prop_symbols(clause), key=sat_count)
        model[sym] = not model[sym]
    # If no solution is found within the flip limit, we return failure
    return None, max_flips


""" This is partly my code """


def pl_true(exp, model={}):
    """Return True if the propositional logic expression is true in the model,
    and False if it is false. If the model does not specify the value for
    every proposition, this may return None to indicate 'not obvious';
    this may happen even when the expression is tautological.
    True
    """
    if exp in (True, False):
        return exp
    op, args = exp.op, exp.args
    if is_prop_symbol(op):
        return model.get(exp)
    elif op == '~':
        p = pl_true(args[0], model)
        if p is None:
            return None
        else:
            return not p
    elif op == '|':
        result = False
        for arg in args:
            p = pl_true(arg, model)
            if p is True:
                return True
            if p is None:
                result = None
        return result
    elif op == '&':
        result = True
        for arg in args:
            p = pl_true(arg, model)
            if p is False:
                return False
            if p is None:
                result = None
        return result
    p, q = args
    if op == '==>':
        return pl_true(~p | q, model)
    elif op == '<==':
        return pl_true(p | ~q, model)
    pt = pl_true(p, model)
    if pt is None:
        return None
    qt = pl_true(q, model)
    if qt is None:
        return None
    if op == '<=>':
        return pt == qt
    elif op == '^':  # xor or 'not equivalent'
        return pt != qt
    else:
        raise ValueError("illegal operator in logic expression" + str(exp))


def is_prop_symbol(s):
    """A proposition logic symbol is an initial-uppercase string.
    False
    """
    return is_symbol(s) and s[0].isupper()


def is_symbol(s):
    """A string s is a symbol if it starts with an alphabetic char.
    True
    """
    return isinstance(s, str) and s[:1].isalpha()


def prop_symbols(x):
    """Return the set of all propositional symbols in x."""
    if not isinstance(x, Expr):
        return set()
    elif is_prop_symbol(x.op):
        return {x}
    else:
        return {symbol for arg in x.args for symbol in prop_symbols(arg)}


""" This is my code """
if __name__ == '__main__':
    S = S_generator(10, 15, 4, 0.5)
    model, flips = WalkSAT(S)
    get_result(model, flips)
    S = S_generator(6, 20, 3, 0.5)
    model, flips = WalkSAT(S)
    get_result(model, flips)
""" This is my code """
