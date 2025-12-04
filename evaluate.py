import gurobipy as grb
from gurobipy import GRB
import os
nthreads = 1
os.environ["OMP_NUM_THREADS"] = str(nthreads)
os.environ["OPENBLAS_NUM_THREADS"] = str(nthreads)
os.environ["MKL_NUM_THREADS"] = str(nthreads)
import pandas as pd
import argparse

import time

from MIPDataset import BipartiteNodeData, compute_mip_representation
from GAT import GATPolicy
import torch
import numpy as np
import pickle


grb.setParam('LogToConsole', 1)

import math
import torch

def binary_positional_features(n_vars: int, to_pm1: bool = False):
    """
    Returns:
      codes: (n_vars, B) binary positional codes (MSB..LSB)
      B:     number of bits used (ceil(log2(n_vars)), at least 1)
    """
    # ensure at least 2 so log2(1) edge-case is handled cleanly
    B = max(1, math.ceil(math.log2(max(2, n_vars))))
    idx = torch.arange(n_vars, dtype=torch.long)[:, None]  # [n,1]
    bit_pos = torch.arange(B, dtype=torch.long)            # [B], 0=LSB
    codes = ((idx >> bit_pos) & 1).to(torch.float32)       # [n,B], LSB..MSB
    codes = torch.flip(codes, dims=[1])                    # MSB..LSB
    if to_pm1:
        codes = codes * 2.0 - 1.0                          # {0,1} -> {-1,+1}
    return codes, B

def mycallback(model, where):

    if where == GRB.Callback.MIPSOL:

        # Access solution values using the custom attribute model._vars
        # sol = model.cbGetSolution(model._vars)
        obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        runtime = time.monotonic() - model._starttime
        log_entry_name = model.Params.LogFile

        log_entry = dict()
        log_entry['primal_bound'] = obj
        log_entry['solving_time'] = runtime
        var_index_to_value = dict()

        gurobi_log[log_entry_name].append(log_entry)

        print("New solution", obj, "Runtime", runtime)

DEVICE = torch.device("cpu")
gurobi_log = dict()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance")
    parser.add_argument("--expname")
    
    args = parser.parse_args()

    # k_0 = 600
    # delta = 10

    # os.makedirs(f"results/{args.expname}/propel_{k_0}_{delta}/", exist_ok=True)

    # A, v_map, v_nodes, c_nodes, b_vars, i_vars, _ = compute_mip_representation(args.instance)
    # X_var = torch.as_tensor(v_nodes, dtype=torch.float32, device=DEVICE)
    # X_con = torch.as_tensor(c_nodes, dtype=torch.float32, device=DEVICE)
    # A = A.coalesce().to(torch.float32).to(DEVICE)

    # # same aggregator you used in training
    # from propel import build_var_agg
    # agg = build_var_agg(A, X_con, use_mean=True, use_max=True, use_min=False, weighted=False)
    # X_full = torch.cat([X_var, agg], dim=1)  # [n_var, d_in]
    # d_in = X_full.size(1)

    # # compact integer IDs (from the training dataset)
    # int_orig_ids = i_vars.long().tolist()
    # orig_from_compact = int_orig_ids[:]     # compact id k -> original id int_orig_ids[k]
    # n_int = len(int_orig_ids)
    # var_ids_compact = torch.arange(len(orig_from_compact), dtype=torch.long, device=DEVICE)
    # X_int = X_full[torch.tensor(orig_from_compact, dtype=torch.long, device=DEVICE)]  # [N_INT, D_IN]

    # # 2) Predict probs ONLY for integer variables
    # saved_dict = torch.load("pretrain/MMCN_very_hard_BI_propel/model_best.pth", map_location=torch.device('cpu'))
    # from propel import PerVariableEnsembleINT
    # model = PerVariableEnsembleINT(n_int, d_in, hidden=64).to(DEVICE)
        
    # model.load_state_dict(saved_dict["model_state_dict"])
    # model.eval()
    # logits = model(var_ids_compact, X_int)           # [N_INT]
    # probs_int = logits.sigmoid().cpu()               # [N_INT]

    # # 3) Build score table like your previous code (index, name, prob, fix, type)
    # all_varname = list(v_map.keys())
    # items = []  # (orig_id, var_name, prob, fix=-1, type='INTEGER')
    # for k, orig_id in enumerate(orig_from_compact):
    #     items.append([orig_id, all_varname[orig_id], probs_int[k].item(), -1, 'INTEGER'])

    # # 4) Pick the lowest k0 probs among integers and set fix=0
    # items.sort(key=lambda x: x[2])                   # ascending by prob
    # fixer = 0
    # for t in range(min(k_0, len(items))):
    #     items[t][3] = 0
    #     fixer += 1

    # # 5) Build and solve the Gurobi model with L1 budget (sum alpha <= delta)
    # m = grb.read(args.instance)
    # m.Params.TimeLimit = 1000
    # m.Params.Threads = 1
    # m.Params.MIPFocus = 1
    # m.Params.LogFile = 'gat'
    # gurobi_log[m.Params.LogFile] = []

    # # Map var names -> Gurobi Var
    # gvars = m.getVars()
    # gvars.sort(key=lambda v: v.VarName)
    # varmap = {v.VarName: v for v in gvars}

    # alphas = []
    # for idx, name, p, x_star, _tp in items:
    #     if x_star < 0:   # skip if not chosen for fixing
    #         continue
    #     v = varmap[name]
    #     a = m.addVar(name=f"alpha_{name}", vtype=GRB.CONTINUOUS)
    #     alphas.append(a)
    #     # |v - x_star| <= a  -> two linear constraints
    #     m.addConstr(a >= v - x_star, name=f"alpha_up_{name}")
    #     m.addConstr(a >= x_star - v, name=f"alpha_dn_{name}")

    # if alphas:
    #     m.addConstr(grb.quicksum(alphas) <= delta, name="sum_alpha")

    # m._starttime = time.monotonic()
    # m.optimize(mycallback)

    # name = args.instance.split("/")[-1]
    
    # with open(f"results/{args.expname}/propel_{k_0}_{delta}/{name}", "wb") as fp:
    #     pickle.dump(gurobi_log[m.Params.LogFile], fp)

    METHODS = ["IM","IM_id"]
    # k_0_list = [50,100,150,200,250,300]
    k_0_list = [70000]
    delta_list = [10]

    # METHODS = [["IM", 200,10], ["IM_id_v1", 200, 10], ["IM_id_v2", 200,10],["IM", 250,10], ["IM_id_v1", 250, 10], ["IM_id_v2", 250,10]]

    # DEFAULT_MODEL_BY_METHOD = {
    #     "IM": "pretrain/MMCN_hard_BI_IM/model_best.pth",
    #     "CL": "pretrain/MMCN_hard_BI_CL_v3/model_best.pth",
    #     "HY": "pretrain/MMCN_hard_BI_HY_v3/model_best.pth",
    #     "IM_id_v1": "pretrain/MMCN_hard_BI_IM_id_v1/model_best.pth",
    #     "IM_id_v2": "pretrain/MMCN_hard_BI_IM_id_v2/model_best.pth",
    # }

    DEFAULT_MODEL_BY_METHOD = {
        "IM": "pretrain/slap_hard_IM/model_best.pth",
        "IM_id": "pretrain/slap_hard_IM_id/model_best.pth"
    }

    for k_0 in k_0_list:
        for method in METHODS:
            for delta in delta_list:
                
                # delta = int(k_0 / delta)
                
                os.makedirs(f"results/{args.expname}/gat{method}_{k_0}_{delta}/", exist_ok=True)
                saved_dict = torch.load(DEFAULT_MODEL_BY_METHOD[method], map_location=torch.device('cpu'))
        
                #get bipartite graph as input
                A, v_map, v_nodes, c_nodes, b_vars, i_vars,objective_coefficients=compute_mip_representation(args.instance)
                constraint_features = c_nodes.cpu()
                # constraint_features[np.isnan(constraint_features)] = 1 #remove nan value
                variable_features = v_nodes
                edge_indices = A._indices()
                edge_features = A._values().unsqueeze(1)
                edge_features=torch.ones(edge_features.shape)
        
                # if "v1" in method:
                #     edge_idx = A._indices()                 # [2, E] -> row=constraints, col=variables
                #     edge_val = A._values().to(torch.float32)  # [E] ensure float32
                #     n_con, n_var = A.size()
                    
                #     # Column degree (# nonzeros per variable)
                #     var_deg = torch.bincount(edge_idx[1], minlength=n_var).to(torch.float32).unsqueeze(1)
                    
                #     # Absolute coeffs per edge
                #     abs_ev = edge_val.abs()
                    
                #     # L1 sum per column
                #     var_l1 = torch.zeros(n_var, dtype=torch.float32)
                #     var_l1.index_add_(0, edge_idx[1], abs_ev)
                #     var_l1 = var_l1.unsqueeze(1)
                    
                #     # L2 norm per column
                #     var_l2_sq = torch.zeros(n_var, dtype=torch.float32)
                #     var_l2_sq.index_add_(0, edge_idx[1], edge_val * edge_val)
                #     var_l2 = var_l2_sq.clamp_min(0).sqrt().unsqueeze(1)
                    
                #     # Min / Max / Mean |a_ij| per column
                #     cols = edge_idx[1]
                #     if hasattr(torch, "scatter_reduce"):  # PyTorch >= 2.0
                #         var_min = torch.full((n_var,), float('inf'), dtype=torch.float32)
                #         var_min = var_min.scatter_reduce(0, cols, abs_ev, reduce='amin', include_self=True)
                #         var_max = torch.zeros(n_var, dtype=torch.float32)
                #         var_max = var_max.scatter_reduce(0, cols, abs_ev, reduce='amax', include_self=True)
                #     else:
                #         # Fallback: sort by column and reduce per unique column (loop over ≤ n_var keys)
                #         idx_sorted = torch.argsort(cols)
                #         cols_s = cols[idx_sorted]
                #         vals_s = abs_ev[idx_sorted]
                    
                #         var_min = torch.full((n_var,), float('inf'), dtype=torch.float32)
                #         var_max = torch.zeros(n_var, dtype=torch.float32)
                    
                #         if vals_s.numel() > 0:
                #             # boundaries of runs
                #             change = torch.ones_like(cols_s, dtype=torch.bool)
                #             change[1:] = cols_s[1:] != cols_s[:-1]
                #             starts = torch.nonzero(change, as_tuple=False).squeeze(-1)
                #             ends = torch.cat([starts[1:], torch.tensor([cols_s.numel()])])
                #             uniq_cols = cols_s[starts]
                    
                #             # small loop over unique columns
                #             for s, e, c in zip(starts.tolist(), ends.tolist(), uniq_cols.tolist()):
                #                 v = vals_s[s:e]
                #                 var_min[c] = v.min()
                #                 var_max[c] = v.max()
                    
                #     # Mean |a_ij| = L1 / degree (avoid divide-by-zero)
                #     var_mean = (var_l1.squeeze(1) / var_deg.squeeze(1).clamp_min(1.0)).unsqueeze(1)
                    
                #     # Clean +inf mins (columns with no nnz, rare)
                #     var_min = torch.where(torch.isinf(var_min), torch.zeros_like(var_min), var_min).unsqueeze(1)
                #     var_max = var_max.unsqueeze(1)
                    
                #     # Objective rank percentile in [0,1]
                #     obj = torch.tensor(objective_coefficients, dtype=torch.float32)  # [n_var]
                #     ranks = torch.argsort(torch.argsort(obj))
                #     obj_rank_pct = (ranks.to(torch.float32) / max(1, n_var - 1)).unsqueeze(1)
                    
                #     # Positional scalar based on the shared global order
                #     if n_var > 1:
                #         var_pos = (torch.arange(n_var, dtype=torch.float32) / (n_var - 1)).unsqueeze(1)
                #     else:
                #         var_pos = torch.zeros(n_var, 1, dtype=torch.float32)
                    
                #     # Concatenate new features to the right of existing 15-D vector
                #     extra_var_feats = torch.cat([var_deg, var_l1, var_l2, var_min, var_max, var_mean, obj_rank_pct, var_pos], dim=1)
                    
                #     # Ensure variable_features is float32
                #     variable_features = variable_features.to(dtype=torch.float32)
                #     variable_features = torch.cat([variable_features, extra_var_feats], dim=1)
                
                #     model = GATPolicy(var_nfeats = 23)
                if "id" in method:
                    n_vars = variable_features.size(0)
                    bin_feats, B = binary_positional_features(n_vars)
                    variable_features = torch.cat([variable_features, bin_feats], dim=1)
                    model = GATPolicy(var_nfeats = 32)
                else:
                    model = GATPolicy()
                
                model.load_state_dict(saved_dict)
                model.eval()

                num_total_params = sum(p.numel() for p in model.parameters())
                print("Total parameters:", num_total_params)

            
                BD = model(
                    constraint_features.to(DEVICE),
                    edge_indices.to(DEVICE),
                    edge_features.to(DEVICE),
                    variable_features.to(DEVICE),
                ).sigmoid().cpu().squeeze()

                print("finish inference")
            
                #align the variable name betweend the output and the solver
                all_varname=[]
                for name in v_map:
                    all_varname.append(name)
                integer_name=[all_varname[i] for i in i_vars]
                scores=[]#get a list of (index, VariableName, Prob, -1, type)
                for i in range(len(v_map)):
                    type="C"
                    if all_varname[i] in integer_name:
                        type='INTEGER'
                    scores.append([i, all_varname[i], BD[i].item(), -1, type])
                
                scores.sort(key=lambda x:x[2],reverse=True)
                
                scores=[x for x in scores if x[4]=='INTEGER']#get binary
                
                fixer=0
                #fixing variable picked by confidence scores
                # count1=0
                # for i in range(len(scores)):
                #     if count1<k_1:
                #         scores[i][3] = 1
                #         count1+=1
                #         fixer += 1
                scores.sort(key=lambda x: x[2], reverse=False)
                count0 = 0
                for i in range(len(scores)):
                    if count0 < k_0:
                        scores[i][3] = 0
                        count0 += 1
                        fixer += 1
                
                m = grb.read(args.instance)
                m.Params.TimeLimit = 1000
                m.Params.Threads = 1
                m.Params.MIPFocus = 1
                m.Params.LogFile = 'gat'
                # m.Params.Heuristics = 0
                gurobi_log[m.Params.LogFile] = []
                
                instance_variabels = m.getVars()
                instance_variabels.sort(key=lambda v: v.VarName)
                variabels_map = {}
                for v in instance_variabels:  # get a dict (variable map), varname:var clasee
                    variabels_map[v.VarName] = v
                alphas = []
                for i in range(len(scores)):
                    tar_var = variabels_map[scores[i][1]]  # target variable <-- variable map
                    x_star = scores[i][3]  # 1,0,-1, decide whether need to fix
                    if x_star < 0:
                        continue
                    tmp_var = m.addVar(name=f'alp_{tar_var}', vtype=GRB.CONTINUOUS)
                    alphas.append(tmp_var)
                    m.addConstr(tmp_var >= tar_var - x_star, name=f'alpha_up_{i}')
                    m.addConstr(tmp_var >= x_star - tar_var, name=f'alpha_dowm_{i}')
                all_tmp = 0
                for tmp in alphas:
                    all_tmp += tmp
                m.addConstr(all_tmp <= delta, name="sum_alpha")
                
                m._starttime = time.monotonic()
                m.optimize(mycallback)
        
                name = args.instance.split("/")[-1]
                
                with open(f"results/{args.expname}/gat{method}_{k_0}_{delta}/{name}", "wb") as fp:
                    pickle.dump(gurobi_log[m.Params.LogFile], fp)

    # os.makedirs('results/' + args.expname + '/gurobi/', exist_ok=True) 
    # m = grb.read(args.instance)
    # m.Params.TimeLimit = 1000
    # m.Params.Threads = 1
    # m.Params.MIPFocus = 1
    # m.Params.LogFile = 'gurobi'
    # # m.Params.Heuristics = 0
    # gurobi_log['gurobi'] = []
    # m._starttime = time.monotonic()
    # m.optimize(mycallback)
    # with open('results/' + args.expname + '/gurobi/' + args.instance.split("/")[-1], "wb") as fp:
    #     pickle.dump(gurobi_log[m.Params.LogFile], fp)