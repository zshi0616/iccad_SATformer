'''
source: https://github.com/emreyolcu/sat/blob/master/code/kcolor.py
Author: emreyolcu

Modified by lee-man

`pip install cnfgen` before using this code 
'''
from __future__ import print_function

import argparse
import os
import random
import subprocess
import copy
import numpy as np

import utils.sat_utils as sat_utils
import utils.circuit_utils as circuit_utils

import sys
import external.PyMiniSolvers.minisolvers as minisolvers

def gen_feature(cnf_filename):
    solver = minisolvers.MinisatSolver()
    # Read CNF
    f = open(cnf_filename)
    lines = f.readlines()
    f.close()
    iclauses = []
    for idx, line in enumerate(lines):
        if line[0] == 'c':
            continue
        elif line[0] == 'p':
            arr = line.replace('\n', '').split(' ')
            n_var = int(arr[2])
            n_clause = int(arr[3])
        else:
            arr = line.replace('\n', '').split(' ')
            iclause = []
            for ele in arr[:-1]:
                iclause.append(int(ele))
            iclauses.append(iclause)

    if len(iclauses) == 0:
        return 0, [], False

    # Check SAT
    for i in range(n_var):
        solver.new_var(dvar=True)
    for c in iclauses:
        solver.add_clause(c)
    if solver.solve(): 
        is_sat = True
    else:
        is_sat = False

    return n_var, iclauses, is_sat

def create_problem(args, n, p, k):
    assert args.problem_type in ['kcolor', 'kcover', 'kclique', 'domset']
    if not os.path.exists(args.tmp_dir):
        os.makedirs(args.tmp_dir)
    retry_times = 0

    while True:
        retry_times += 1

        # cnf_path = os.path.join(args.tmp_dir, 'problem.cnf')
        # bench_path = os.path.join(args.tmp_dir, 'problem.bench')
        # bench_abc_path = os.path.join(args.tmp_dir, 'problem_abc.bench')
        # subprocess.call(['cnfgen', '-q', '-o', cnf_path, args.problem_type, str(k), 'gnp', str(n), str(p)])
        # n_var, iclauses, is_sat = gen_feature(cnf_path)
        # circuit_utils.save_bench(n_var, iclauses, bench_path)
        # subprocess.call(['abc', '-c', 
        #     'read {}; strash; balance; rewrite -lz; balance; rewrite -lz; write_cnf {}'.format(
        #         bench_path, bench_abc_path 
        #     )])
        # n_var, iclauses, is_sat = gen_feature(bench_abc_path)
        # os.remove(cnf_path)
        # os.remove(bench_path)
        # os.remove(bench_abc_path)

        cnf_path = os.path.join(args.tmp_dir, 'problem.cnf')
        subprocess.call(['cnfgen', '-q', '-o', cnf_path, args.problem_type, str(k), 'gnp', str(n), str(p)])
        n_var, iclauses, is_sat = gen_feature(cnf_path)
        os.remove(cnf_path)

        if len(iclauses) > 0:
            return n_var, iclauses, is_sat
        if retry_times % 1000 == 0:
            print('[WARNING] Retry generation for {:}k times'.format(int(retry_times/1000)))
                
