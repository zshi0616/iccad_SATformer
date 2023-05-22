import torch
import numpy as np
import utils.share as share

def stat_attn(unsat_core, gt, attn):
    core_att_core = 0
    core_att_uncore = 0
    uncore_att_core = 0
    uncore_att_uncore = 0

    is_core = unsat_core > 0.18
    for i in range(len(attn)):
        for j in range(len(attn)):
            if i != j:
                if is_core[i] and is_core[j]:
                    core_att_core += attn[i][j]
                elif is_core[i] and not is_core[j]:
                    core_att_uncore += attn[i][j]
                elif not is_core[i] and is_core[j]:
                    uncore_att_core += attn[i][j]
                elif not is_core[i] and not is_core[j]:
                    uncore_att_uncore += attn[i][j]

    tot_core_att = core_att_core + core_att_uncore + 0.0001
    tot_uncore_att = uncore_att_core + uncore_att_uncore + 0.0001

    print('===== SAT: {} ====='.format(share.is_sat))
    print('Core: {:} / {:} = {:.2f}%'.format(is_core.sum(), len(is_core), 100* is_core.sum() / len(is_core)))
    print('Core Correct: {:} / {:} = {:.2f}%'.format(
        ((gt > 0.5) == (is_core)).sum(), len(is_core), ((gt > 0.5) == (is_core)).sum() * 100 / len(is_core)
    ))
    print('CC: {:.4f}, CU: {:.4f}, UC: {:.4f}, UU: {:.4f}'.format(
        core_att_core, core_att_uncore, uncore_att_core, uncore_att_uncore
    ))
    print('Core Attention Rate: {:.4f} / {:.4f} = {:.2f}%'.format(
        core_att_core, tot_core_att, 100* core_att_core / (tot_core_att + tot_uncore_att)
    ))
    print('Correct Core Attention: {:.4f} / {:.4f} = {:.2f}%'.format(
        core_att_core, tot_core_att, 100* core_att_core / tot_core_att
    ))
    print('Correct UnCore Attention: {:.4f} / {:.4f} = {:.2f}%'.format(
        uncore_att_uncore, tot_uncore_att, 100* uncore_att_uncore / tot_uncore_att
    ))
    print()


def get_attn(is_core, attn):
    core_att_core = 0
    core_att_uncore = 0
    uncore_att_core = 0
    uncore_att_uncore = 0
    window, head, x, y = attn.shape
    for w in range(window):
        for h in range(head):
            for i in range(x):
                for j in range(y):
                    if i != j:
                        if is_core[i] and is_core[j]:
                            core_att_core += attn[w][h][i][j]
                        elif is_core[i] and not is_core[j]:
                            core_att_uncore += attn[w][h][i][j]
                        elif not is_core[i] and is_core[j]:
                            uncore_att_core += attn[w][h][i][j]
                        elif not is_core[i] and not is_core[j]:
                            uncore_att_uncore += attn[w][h][i][j]
    
    return core_att_core, core_att_uncore, uncore_att_core, uncore_att_uncore

def stat_attn_window(unsat_core, gt, attn):
    print('===== SAT: {} ====='.format(share.is_sat))
    is_core = unsat_core > unsat_core.mean()
    core_att_core, core_att_uncore, uncore_att_core, uncore_att_uncore = get_attn(is_core, attn)
    tot_att = core_att_core + core_att_uncore + uncore_att_core + uncore_att_uncore
    print('Core: {:} / {:} = {:.2f}%'.format(is_core.sum(), len(is_core), 100* is_core.sum() / len(is_core)))
    print('CC: {:.4f}, CU: {:.4f}, UC: {:.4f}, UU: {:.4f}'.format(
        core_att_core, core_att_uncore, uncore_att_core, uncore_att_uncore
    ))
    print('Core Attention: {:.4f} / {:.4f} = {:.2f}%'.format(
        core_att_core, tot_att, 100 * core_att_core / tot_att
    ))
    # print('----')
    # is_core = gt > 0.5
    # core_att_core, core_att_uncore, uncore_att_core, uncore_att_uncore = get_attn(is_core, attn)
    # tot_att = core_att_core + core_att_uncore + uncore_att_core + uncore_att_uncore
    # print('CC: {:.4f}, CU: {:.4f}, UC: {:.4f}, UU: {:.4f}'.format(
    #     core_att_core, core_att_uncore, uncore_att_core, uncore_att_uncore
    # ))
    # print('Core Attention: {:.4f} / {:.4f} = {:.2f}%'.format(
    #     core_att_core, tot_att, 100 * core_att_core / tot_att
    # ))


    print()