import enum
import os
import random
import numpy as np
import subprocess
import time

def save_cnf(iclauses, n_vars, filename): 
    n_clauses = len(iclauses)
    f = open(filename, 'w')
    # head
    f.write('p cnf {:} {:}\n'.format(n_vars, n_clauses))

    # CNF
    for clause in iclauses:
        new_line = ''
        for ele in clause:
            new_line += str(ele) + ' '
        new_line += '0\n'
        f.write(new_line)
    
    f.close()
    
def save_bench(iclauses, n_vars, filename):
    f = open(filename, 'w')
    for pi_idx in range(1, n_vars + 1):
        f.write('INPUT({:})\n'.format(pi_idx))
    f.write('OUTPUT(PO)\n')
    f.write('\n')
    for pi_idx in range(1, n_vars + 1):
        f.write('{:}_INV = NOT({:})\n'.format(pi_idx, pi_idx))
    for clause_idx, clause in enumerate(iclauses):
        newline = 'CLAUSE_{:} = OR('.format(clause_idx)
        for var_idx, var in enumerate(clause):
            if var > 0:
                newline += '{:}'.format(var)
            else:
                newline += '{:}_INV'.format(abs(var))
            if var_idx == len(clause) - 1:
                newline += ')\n'
            else:
                newline += ', '
        f.write(newline)
    newline = 'PO = AND('
    for clause_idx in range(len(iclauses)):
        if clause_idx == len(iclauses) - 1:
            newline += 'CLAUSE_{:})\n'.format(clause_idx)
        else:
            newline += 'CLAUSE_{:}, '.format(clause_idx)
    f.write(newline)
    f.close()
        
def read_cnf(cnf_path):
    f = open(cnf_path, 'r')
    lines = f.readlines()
    f.close()

    n_vars = -1
    n_clauses = -1
    begin_parse_cnf = False
    iclauses = []
    for line in lines:
        if begin_parse_cnf:
            arr = line.replace('\n', '').split(' ')
            clause = []
            for ele in arr:
                if ele.replace('-', '').isdigit() and ele != '0':
                    clause.append(int(ele))
            if len(clause) > 0:
                iclauses.append(clause)
                
        elif line.replace(' ', '')[0] == 'c':
            continue
        elif line.replace(' ', '')[0] == 'p': 
            arr = line.replace('\n', '').split(' ')
            get_cnt = 0
            for ele in arr:
                if ele == 'p':
                    get_cnt += 1
                elif ele == 'cnf':
                    get_cnt += 1
                elif ele != '':
                    if get_cnt == 2:
                        n_vars = int(ele)
                        get_cnt += 1
                    else: 
                        n_clauses = int(ele)
                        break
            assert n_vars != -1
            assert n_clauses != -1
            begin_parse_cnf = True
        
    
    return iclauses, n_vars

def convert_cnf_abc(args, cnf, no_var):
    if not os.path.exists(args.tmp_dir):
        os.mkdir(args.tmp_dir)
    cnf_filename = os.path.join(args.tmp_dir, '{:}_{:}.cnf'.format(int(time.time()), random.randint(0, 100)))
    save_cnf(cnf, no_var, cnf_filename)
    aig_filename = cnf_filename.replace('.cnf', '.aig')
    cnf2aig_cmd = 'cnf2aig {} {}'.format(cnf_filename, aig_filename)
    info = os.popen(cnf2aig_cmd).readlines()
    new_cnf_filename = cnf_filename.replace('.cnf', '_new.cnf')
    subprocess.call(["abc", "-c", "read %s; \
                     balance; rewrite -lz; balance; rewrite -lz; \
                     balance; rewrite -lz; balance; cec; write_cnf %s" \
            % (aig_filename, new_cnf_filename)])

    new_cnf, new_var = read_cnf(new_cnf_filename)
    os.remove(cnf_filename)
    os.remove(aig_filename)
    os.remove(new_cnf_filename)
    
    return new_cnf, new_var
