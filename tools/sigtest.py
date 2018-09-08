#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Adrián Javaloy

"""
It performs an approximate randomization test in order to estimate the significance level of the hypothesis test
where the null hypothesis is that the two predictions come from similar systems. In other words, it estimates
the probability that the two predictions come from similar systems, the lower it is, the more confident one has
about their dissimilarity. A usual value to look for is 0.05.

Reference:
Section 4.1 of
Riezler, S. and Maxwell, J.T., 2005. On some pitfalls in automatic evaluation and significance testing for MT.
In Proceedings of the ACL workshop on intrinsic and extrinsic evaluation measures for machine translation and/or
summarization (pp. 57-64).
"""

import subprocess
import math
import argparse
import os
import re
import numpy as np

parser = argparse.ArgumentParser('significance test script')
parser.add_argument('-seed', type=int, default=None,
                    help='random seed')
parser.add_argument('-file1', type=str, required=True,
                    help='File with the translations of the first model')
parser.add_argument('-file2', type=str, required=True,
                    help='File with the translations of the second model')
parser.add_argument('-trials', type=int, default=1000,
                    help='number of trials')
parser.add_argument('-samples', type=int, default=None,
                    help='Number of samples to take from the files')
parser.add_argument('-tgt', type=str, required=True,
                    help='File with the target outputs')
parser.add_argument('-verbose', action='store_true')
parser.add_argument('-assure_pvalue', type=float, default=None,
                    help='Stop as soon as the required significance level is assured')
parser.add_argument('-command', type=str, nargs='+',
                    default='perl tools/multi-bleu.perl',
                    help="""Command used to evaluate the predictions.
                    It should accept the target filename as argument, 
                    the predictions as standard input, and the first
                    number printed will be taken as the score.""")
args = parser.parse_args()

assert args.assure_pvalue is None or 0 <= args.assure_pvalue <= 1, "pvalue should be between [0,1]"


def _maybe_print(msg, **kwargs):
    if args.verbose:
        print(msg, **kwargs)


def _read_file(filename):
    with open(filename, 'r') as f:
        if args.samples is None:
            return f.readlines()
        else:
            return [x for _,x in zip(range(args.samples), f.readlines())]


def evaluate(preds, tgt):
    with open(preds, 'r') as f:
        out = subprocess.check_output(args.command.split() + [tgt], stdin=f)
        out = out.decode('utf-8')
        _maybe_print(out)
        return float(re.search(r'\d+(?:\.\d+)?', out).group())


def abs_diff(preds1, preds2, tgt):
    return math.fabs(evaluate(preds1, tgt) - evaluate(preds2, tgt))


def shuffle(x, y):
    for i, j in zip(x, y):
        yield [i, j] if np.random.random() < 0.5 else [j, i]


if __name__ == "__main__":
    np.random.seed(args.seed)

    assert not os.path.exists('temp_pred1'), 'the file temp_pred1 already exists'
    assert not os.path.exists('temp_pred2'), 'the file temp_pred2 already exists'

    _maybe_print("Reading {}".format(args.file1))
    preds1 = _read_file(args.file1)
    _maybe_print("Reading {}".format(args.file2))
    preds2 = _read_file(args.file2)

    _maybe_print("Number of lines read: {} {}".format(len(preds1), len(preds2)))
    _maybe_print("Calculating baseline value.")
    baseline = abs_diff(args.file1, args.file2, args.tgt)
    _maybe_print("Baseline value: {}".format(baseline))

    try:
        better, offset = 0, 0
        for i in range(args.trials):
            _maybe_print("Trial {} of {}".format(i+1, args.trials))
            p1, p2 = np.array([[x,y] for x,y, in shuffle(preds1, preds2)]).T
            open('temp_pred1', 'w').writelines(p1)
            open('temp_pred2', 'w').writelines(p2)
            try:
                diff = abs_diff('temp_pred1', 'temp_pred2', args.tgt)
            except subprocess.CalledProcessError:  # In case the subprocess fails for some reason
                diff, offset = 0, offset+1
                print("[{}/{}] Calculation failed!".format(i+1, args.trials), end='\n' if args.verbose else '\r')
                continue
            better += diff >= baseline
            _maybe_print("Difference: {}".format(diff))
            print("[{}/{}] Approximated significance level: {:2.5f}".format(i+1, args.trials, (better+1)/(i+2-offset)),
                  end='\n' if args.verbose else '\r')

            if args.assure_pvalue:
                worst_pvalue = (better+(args.trials-i-1)+1) / (args.trials+1-offset)
#                if better + (args.trials-i-1) <= args.pvalue * (args.trials+1-offset) - 1:
                if worst_pvalue <= args.assure_pvalue:
                    _maybe_print("Stopping early! The pvalue {} will be achieved".format(args.assure_pvalue))
                    _maybe_print("Worst possible pvalue: {}".format(worst_pvalue))

                    if not args.verbose:
                        size = len("[{}/{}] Approximated significance level: {:2.5f}".format(i+1, args.trials,
                                                                                             (better+1)/(i+2-offset)))
                        print(" "*size, end='\r')  # it cleans the line
                    args.trials = i+1
                    break

        print("[{}/{}] Approximated significance level: {:2.5f}".format(args.trials, args.trials,
                                                                        (better+1)/(args.trials+1-offset)))

    finally:
        os.remove('temp_pred1')
        os.remove('temp_pred2')
