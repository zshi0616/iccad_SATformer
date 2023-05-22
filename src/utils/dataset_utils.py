import subprocess
import numpy as np
import torch
import os
import math
import random
import copy
from torch_geometric.data import Data
from torch_geometric.transforms import ToUndirected
from torch_geometric.utils import remove_isolated_nodes
from torch_geometric.utils.convert import to_networkx
from torch_geometric.data import Data
from communities.algorithms import louvain_method, girvan_newman
from utils.sat_utils import solve_sat, recursion_generation, one_hot_gate_type, write_dimacs_to
from utils.dag_utils import return_order_info
from utils.circuit_utils import aig_simulation

def one_hot(idx, length):
    if type(idx) is int:
        idx = torch.LongTensor([idx]).unsqueeze(0)
    else:
        idx = torch.LongTensor(idx).unsqueeze(0).t()
    x = torch.zeros((len(idx), length)).scatter_(1, idx, 1)
    return x

# def literal_modularity(n_var, cnf):
#     adj_matrix = np.zeros((n_var, n_var))
#     for clause in cnf:
#         connection = len(clause)
#         connect_length = math.factorial(connection) / (math.factorial(2) * math.factorial(connection - 2))

def community(n_var, cnf):
    connect_list = []
    adj_matrix = np.zeros((len(cnf), len(cnf)))
    for var_idx in range(n_var + 1):
        connect_list.append([])
    for clause_idx, clause in enumerate(cnf):
        for var in clause:
            connect_list[int(abs(var))].append(clause_idx)
    
    # Clause Incidence Graph
    for var_idx in range(1, n_var+1, 1):
        var_connect = connect_list[var_idx]
        if len(var_connect) == 0 or len(var_connect) == 1:
            continue
        connect_length = math.factorial(len(var_connect)) / (math.factorial(2) * math.factorial(len(var_connect) - 2))
        for i in range(len(var_connect)):
            for j in range(i+1, len(var_connect), 1):
                adj_matrix[var_connect[i]][var_connect[j]] += 1.0 / connect_length
                adj_matrix[var_connect[j]][var_connect[i]] += 1.0 / connect_length
                
    # Partition 
    com, partition = louvain_method(adj_matrix)
    return com, partition[-1]['Q']

def reorder_cnf(cnf, com):
    res_cnf = []
    for group in com:
        for clause_idx in group:
            res_cnf.append(cnf[clause_idx])
    return res_cnf
       
class OrderedData(Data):
    def __init__(self, edge_index=None, x=None, forward_level=None, backward_level=None, forward_index=None, backward_index=None, gate_type=None, unsat_core=None):
        super().__init__()
        self.edge_index = edge_index
        self.x = x
        self.forward_level = forward_level
        self.backward_level = backward_level
        self.forward_index = forward_index
        self.backward_index = backward_index
        self.gate_type = gate_type
        self.unsat_core = unsat_core
    
    def __inc__(self, key, value, *args, **kwargs):
        if 'index' in key or 'face' in key:
            return self.num_nodes
        else:
            return 0

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'forward_index' or key == 'backward_index':
            return 0
        elif key == "edge_index":
            return 1
        else:
            return 0

def one_hot_gate_type(gate_type, gate2index):
    res = [0] * len(gate2index.keys())
    for key in gate2index.keys():
        if key == gate_type:
            res[gate2index[key]] = 1
            return res
    print('[ERROR] No gate type: {}'.format(gate_type))
    raise

def cnf_parse_pyg(args, iclauses, label, n_vars, n_clauses):
    x = []
    edge_index = []
    gate_type = []
    for var_idx in range(n_vars):
        x.append(one_hot_gate_type('VAR', args.gate2index))
        gate_type.append(args.gate2index['VAR'])
        # edge_index.append([var_idx, var_idx + n_vars])
        # edge_index.append([var_idx + n_vars, var_idx])
    for negvar_idx in range(n_vars):
        x.append(one_hot_gate_type('NEGVAR', args.gate2index))
        gate_type.append(args.gate2index['NEGVAR'])
    for cla_idex in range(n_clauses):
        x.append(one_hot_gate_type('CLAUSE', args.gate2index))
        gate_type.append(args.gate2index['CLAUSE'])
    po_idx = len(x)
    x.append(one_hot_gate_type('PO', args.gate2index))
    gate_type.append(args.gate2index['PO'])

    # UNSAT Core
    if args.dataset == 'benchmark' or label == 1 or args.disable_core:
        unsat_core = [0] * len(iclauses)
    else:
        unsat_core = get_unsat_core(args, iclauses, n_vars)
    
    # Clause Connection 
    for cnt, clause in enumerate(iclauses):
        clause_idx = cnt + n_vars * 2
        for var in clause:
            if var < 0:
                var_idx = abs(var) + n_vars - 1
            else:
                var_idx = var - 1
            edge_index.append([var_idx, clause_idx])
        edge_index.append([clause_idx, po_idx])
    
    # forward index
    forward_index = []
    for idx in range(len(x)):
        forward_index.append(idx)
    
    x = torch.tensor(x)
    gate_type = torch.tensor(gate_type)
    edge_index = torch.tensor(edge_index)
    forward_index = torch.tensor(forward_index)
    unsat_core = torch.tensor(unsat_core)
    graph = OrderedData(x=x, edge_index=edge_index.t().contiguous(), 
                        forward_index=forward_index, gate_type=gate_type, 
                        unsat_core=unsat_core)
    graph.n_vars = n_vars
    graph.n_clauses = n_clauses
    graph.n_nodes = len(x)
    graph.y = torch.tensor([label])

    # Save cnf
    # max_length = 10
    # for clause in iclauses:
    #     if len(clause) > max_length:
    #         max_length = len(clause)
    # for clause in iclauses:
    #     if len(clause) < max_length:
    #         clause += [0] * (max_length - len(clause))
    # graph.cnf = torch.tensor(iclauses)

    return graph

def cnf_parse_pyg_subinst(args, iclauses, label_window, n_vars, n_clauses):
    x = []
    edge_index = []
    gate_type = []
    for var_idx in range(n_vars):
        x.append(one_hot_gate_type('VAR', args.gate2index))
        gate_type.append(args.gate2index['VAR'])
        # edge_index.append([var_idx, var_idx + n_vars])
        # edge_index.append([var_idx + n_vars, var_idx])
    for negvar_idx in range(n_vars):
        x.append(one_hot_gate_type('NEGVAR', args.gate2index))
        gate_type.append(args.gate2index['NEGVAR'])
    for cla_idex in range(n_clauses):
        x.append(one_hot_gate_type('CLAUSE', args.gate2index))
        gate_type.append(args.gate2index['CLAUSE'])
    po_idx = len(x)
    x.append(one_hot_gate_type('PO', args.gate2index))
    gate_type.append(args.gate2index['PO'])

    # UNSAT Core
    unsat_core = get_unsat_core(args, iclauses, n_vars)
    
    # Clause Connection 
    for cnt, clause in enumerate(iclauses):
        clause_idx = cnt + n_vars * 2
        for var in clause:
            if var < 0:
                var_idx = abs(var) + n_vars - 1
            else:
                var_idx = var - 1
            edge_index.append([var_idx, clause_idx])
        edge_index.append([clause_idx, po_idx])
    
    # forward index
    forward_index = []
    for idx in range(len(x)):
        forward_index.append(idx)
    
    x = torch.tensor(x)
    gate_type = torch.tensor(gate_type)
    edge_index = torch.tensor(edge_index)
    forward_index = torch.tensor(forward_index)
    unsat_core = torch.tensor(unsat_core)
    graph = OrderedData(x=x, edge_index=edge_index.t().contiguous(), 
                        forward_index=forward_index, gate_type=gate_type, 
                        unsat_core=unsat_core)
    graph.n_vars = n_vars
    graph.n_clauses = n_clauses
    graph.n_nodes = len(x)
    graph.y_window = label_window
    graph.y = label_window[-1]

    return graph

def cnf_parse_pyg_includecore(args, iclauses, label, pad_list, n_vars, n_clauses):
    x = []
    edge_index = []
    gate_type = []
    for var_idx in range(n_vars):
        x.append(one_hot_gate_type('VAR', args.gate2index))
        gate_type.append(args.gate2index['VAR'])
        # edge_index.append([var_idx, var_idx + n_vars])
        # edge_index.append([var_idx + n_vars, var_idx])
    for negvar_idx in range(n_vars):
        x.append(one_hot_gate_type('NEGVAR', args.gate2index))
        gate_type.append(args.gate2index['NEGVAR'])
    for cla_idex in range(n_clauses):
        x.append(one_hot_gate_type('CLAUSE', args.gate2index))
        gate_type.append(args.gate2index['CLAUSE'])
    po_idx = len(x)
    x.append(one_hot_gate_type('PO', args.gate2index))
    gate_type.append(args.gate2index['PO'])

    # UNSAT Core
    origin_len = 0
    for idx in range(len(pad_list)):
        if pad_list[idx] == -1:
            origin_len += 1
        else:
            break
    unsat_core = get_unsat_core(args, iclauses[:origin_len], n_vars)
    for idx in range(origin_len, len(pad_list)):
        unsat_core.append(unsat_core[pad_list[idx]])
    
    # Clause Connection 
    for cnt, clause in enumerate(iclauses):
        clause_idx = cnt + n_vars * 2
        for var in clause:
            if var < 0:
                var_idx = abs(var) + n_vars - 1
            else:
                var_idx = var - 1
            edge_index.append([var_idx, clause_idx])
        edge_index.append([clause_idx, po_idx])
    
    # forward index
    forward_index = []
    for idx in range(len(x)):
        forward_index.append(idx)
    
    x = torch.tensor(x)
    gate_type = torch.tensor(gate_type)
    edge_index = torch.tensor(edge_index)
    forward_index = torch.tensor(forward_index)
    unsat_core = torch.tensor(unsat_core)
    graph = OrderedData(x=x, edge_index=edge_index.t().contiguous(), 
                        forward_index=forward_index, gate_type=gate_type, 
                        unsat_core=unsat_core)
    graph.n_vars = n_vars
    graph.n_clauses = n_clauses
    graph.n_nodes = len(x)
    graph.y = label

    # y_window
    graph.y_window = []
    window_size = args.windows_size
    while window_size <= len(iclauses):
        for start_idx in range(0, len(iclauses), window_size):
            has_core = False
            for clause_idx in range(start_idx, start_idx+window_size, 1):
                if unsat_core[clause_idx] == 1:
                    graph.y_window.append(1)
                    has_core = True
                    break
            if not has_core:
                graph.y_window.append(0)
        window_size *= args.windows_size
    graph.y_window = torch.tensor(graph.y_window)

    return graph


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

def get_unsat_core(args, iclauses, n_vars):
    cnf_filename = os.path.join(args.tmp_dir, 'test.cnf')
    save_cnf(iclauses, n_vars, cnf_filename)

    # Get MUS
    mus_cmd = 'picomus ' + cnf_filename
    mus_info = os.popen(mus_cmd).readlines()
    os.remove(cnf_filename)

    # Parse MUS Info
    sat_flag = False
    for line in mus_info:
        if 's SATISFIABLE' in line:
            sat_flag = True
            break
    unsat_core = [0] * len(iclauses)
    if not sat_flag:
        for line in mus_info:
            if 'v ' in line:
                ele = line.replace('v', '').replace(' ', '').replace('\n', '')
                ele = int(ele)
                if ele > 0:
                    unsat_core[ele - 1] = 1
    
    return unsat_core

def circuitsat_parse_pyg(iclauses, n_vars, n_clauses, exp_depth):
    '''
    A function to parse the cnf to circuit to `Pytorch Geometric` Data.
    Input:
        iclause: clauses list
        n_vars: number of variables
        n_clauses: number of clauses
        exp_depth: CNF expansion depth
    Return:
        x: one_hot encoding of [PI, AND, OR, NOT]
        edge_index: edge connection pairs: each pair [x, y] from x to y
    '''
    x = []
    edge_index = []

    # PI and inv_PI
    x.append([])    # 0 is reserved
    inv2idx = {}
    for var_idx in range(1, n_vars+1, 1):
        x.append(one_hot_gate_type('PI'))
    has_inv = [0] * (n_vars+1)
    for clause in iclauses:
        for ele in clause:
            if ele < 0:
                has_inv[abs(ele)] = 1
    for var_idx in range(1, n_vars+1, 1):
        if has_inv[var_idx]:
            inv2idx[var_idx] = len(x)
            x.append(one_hot_gate_type('NOT'))
            edge_index.append([var_idx, inv2idx[var_idx]])

    # PO
    po_idx = len(x)
    x.append(one_hot_gate_type('OR'))
    iclauses_tmp = []
    for clause in iclauses:
        iclauses_tmp.append(clause.copy())
    recursion_generation(iclauses_tmp, po_idx, 0, exp_depth,
                         n_vars, x, edge_index, inv2idx)

    # Remove the reserved 0
    x = x[1: ]
    for edge in edge_index:
        edge[0] -= 1
        edge[1] -= 1
    
    # build the graph
    x = torch.tensor(x, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    y = torch.tensor([1])

    forward_level, forward_index, backward_level, backward_index = return_order_info(edge_index, x.size(0))
    
    graph = OrderedData(x=x, edge_index=edge_index, forward_level=forward_level, forward_index=forward_index, 
    backward_level=backward_level, backward_index=backward_index)

    # graph = Data(x=x, edge_index=edge_index.t().contiguous(), y=y, n_vars=n_vars, n_clauses=n_clauses)
    graph.y, graph.n_vars, graph.n_clauses = y, n_vars, n_clauses

    return graph

def aig_parse_pyg(args, iclauses, n_vars, n_clauses, sat, prob_loss=False):
    '''
    A function to parse the cnf to aig, then to circuit to `Pytorch Geometric` Data.
    Input:
        iclause: clauses list
        n_vars: number of variables
        n_clauses: number of clauses
    Return:
        x: one_hot encoding of [PI, AND, NOT]
        edge_index: edge connection pairs: each pair [x, y] from x to y
    For AIG, the nodes can be categorized as the Literal node, internal AND nodes, internal NOT node. The type values for each kind of nodes are as follows:
        * Literal input node: 0;
        * Internal AND nodes: 1;
        * Internal NOT nodes: 2;
    '''
    tmp_dir = args.tmp_dir
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    if os.getcwd()[-3:] == 'src':
        external_folder = './external/'
    else:
        external_folder = './src/external/'
    
    # step 1: store dimacs format
    dimacs_tmp = os.path.join(tmp_dir, 'sat.dimacs')
    write_dimacs_to(n_vars, iclauses, dimacs_tmp)
    # step 2: dimacs to aig
    aig_tmp = os.path.join(tmp_dir, 'sat.aig')
    subprocess.call([external_folder + "./aiger/cnf2aig/cnf2aig", dimacs_tmp, aig_tmp])
    # step 3: aig to abc opimized aig
    aig_abc_tmp = os.path.join(tmp_dir, 'aig_abc.aig')
    subprocess.call(["abc", "-c", "read %s; balance; \
        balance; rewrite -lz; rewrite -lz; balance; rewrite -lz; balance; cec; write %s" \
            % (aig_tmp, aig_abc_tmp)])
    # step 4: aig to aag
    aag_abc_tmp = os.path.join(tmp_dir, 'aig_abc.aag')
    subprocess.call([external_folder + "./aiger/aiger/aigtoaig", aig_abc_tmp, aag_abc_tmp])
    # step 4: read aag
    with open(aag_abc_tmp, 'r') as f:
        lines = f.readlines()
    header = lines[0].strip().split(" ")
    assert header[0] == 'aag', 'The header of AIG file is wrong.'
    # “M”, “I”, “L”, “O”, “A” separated by spaces.
    n_variables = eval(header[1])
    n_inputs = eval(header[2])
    n_outputs = eval(header[4])
    n_and = eval(header[5])
    if n_outputs != 1 or n_variables != (n_inputs + n_and) or n_variables == n_inputs:
        return [], []
    assert n_outputs == 1, 'The AIG has multiple outputs.'
    assert n_variables == (n_inputs + n_and), 'There are unused AND gates.'
    assert n_variables != n_inputs, '# variable equals to # inputs'
    # Construct AIG graph
    x = []
    edge_index = []
    # node_labels = []
    not_dict = {}
    
    # Add Literal node
    for i in range(n_inputs):
        x += [one_hot(0, 3)]
        # node_labels += [0]

    # Add AND node
    for i in range(n_inputs+1, n_inputs+1+n_and):
        x += [one_hot(1, 3)]
        # node_labels += [1]


    # sanity-check
    for (i, line) in enumerate(lines[1:1+n_inputs]):
        literal = line.strip().split(" ")
        assert len(literal) == 1, 'The literal of input should be single.'
        assert int(literal[0]) == 2 * (i + 1), 'The value of a input literal should be the index of variables mutiplying by two.'

    literal = lines[1+n_inputs].strip().split(" ")[0]
    assert int(literal) == (n_variables * 2) or int(literal) == (n_variables * 2) + 1, 'The value of the output literal shoud be (n_variables * 2)'
    sign_final = int(literal) % 2
    index_final_and = int(literal) // 2 - 1

    for (i, line) in enumerate(lines[2+n_inputs: 2+n_inputs+n_and]):
        literals = line.strip().split(" ")
        assert len(literals) == 3, 'invalidate the definition of two-input AND gate.'
        assert int(literals[0]) == 2 * (i + 1 + n_inputs)
    var_def = lines[2+n_variables].strip().split(" ")[0]

    assert var_def == 'i0', 'The definition of variables is wrong.'
    # finish sanity-check

    # Add edge
    for (i, line) in enumerate(lines[2+n_inputs: 2+n_inputs+n_and]):
        line = line.strip().split(" ")
        # assert len(line) == 3, 'The length of AND lines should be 3.'
        output_idx = int(line[0]) // 2 - 1
        # assert (int(line[0]) % 2) == 0, 'There is inverter sign in output literal.'

        # 1. First edge
        input1_idx = int(line[1]) // 2 - 1
        sign1_idx = int(line[1]) % 2
        # If there's a NOT node
        if sign1_idx == 1:
            if input1_idx in not_dict.keys():
                not_idx = not_dict[input1_idx]
            else:
                x += [one_hot(2, 3)]
                # node_labels += [2]
                not_idx = len(x) - 1
                not_dict[input1_idx] = not_idx
                edge_index += [[input1_idx, not_idx]]
            edge_index += [[not_idx, output_idx]]
        else:
            edge_index += [[input1_idx, output_idx]]


        # 2. Second edge
        input2_idx = int(line[2]) // 2 - 1
        sign2_idx = int(line[2]) % 2
        # If there's a NOT node
        if sign2_idx == 1:
            if input2_idx in not_dict.keys():
                not_idx = not_dict[input2_idx]
            else:
                x += [one_hot(2, 3)]
                # node_labels += [2]
                not_idx = len(x) - 1
                not_dict[input2_idx] = not_idx
                edge_index += [[input2_idx, not_idx]]
            edge_index += [[not_idx, output_idx]]
        else:
            edge_index += [[input2_idx, output_idx]]
    
    
    if sign_final == 1:
        x += [one_hot(2, 3)]
        # node_labels += [2]
        not_idx = len(x) - 1
        edge_index += [[index_final_and, not_idx]]
    
    # simulation
    if prob_loss:
        y_prob = aig_simulation(x, edge_index)
        y_prob = torch.tensor(y_prob, dtype=torch.float)
        graph.y_prob = y_prob


    x = torch.cat(x, dim=0).float()
    edge_index = torch.tensor(edge_index).t().contiguous()
    
    y = torch.tensor([1])

    forward_level, forward_index, backward_level, backward_index = return_order_info(edge_index, x.size(0))
    
    graph = OrderedData(x=x, edge_index=edge_index, forward_level=forward_level, forward_index=forward_index, 
    backward_level=backward_level, backward_index=backward_index)
    
    # Draw graph
    # nx_graph = to_networkx(graph)

    # import matplotlib.pyplot as plt
    # plt.figure(1,figsize=(14,12)) 
    # nx.draw(nx_graph, cmap=plt.get_cmap('Set1'), node_color = np.array(node_labels), node_size=75,linewidths=6)
    # plt.show()

    graph.y, graph.n_vars, graph.n_clauses = y, n_vars, n_clauses

    # CNF
    cnf_edge_index = []
    for idx, iclause in enumerate(iclauses):
        c_idx = n_vars * 2 + idx
        for i in iclause:
            l_idx = (i - 1) if i > 0 else (abs(i) - 1 + n_vars)
            cnf_edge_index.append([l_idx, c_idx])
    cnf_edge_index = torch.tensor(cnf_edge_index, dtype=torch.long)
    cnf_x = torch.zeros(size=(n_vars*2, 1))
    cnf_x = torch.cat([cnf_x, torch.ones(size=(n_clauses, 1))], dim=0)
    graph.cnf_edge_index = cnf_edge_index
    graph.cnf_x = cnf_x
    graph.sat = torch.tensor(sat)

    return graph
