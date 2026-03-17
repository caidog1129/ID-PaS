import argparse
import math
import os
import pickle
import time
from typing import Dict, List, Sequence, Tuple

import gurobipy as grb
from gurobipy import GRB
import torch

from GAT import GATPolicy
from MIPDataset import compute_mip_representation


nthreads = 1
os.environ["OMP_NUM_THREADS"] = str(nthreads)
os.environ["OPENBLAS_NUM_THREADS"] = str(nthreads)
os.environ["MKL_NUM_THREADS"] = str(nthreads)

grb.setParam("LogToConsole", 1)
DEVICE = torch.device("cpu")
gurobi_log: Dict[str, List[dict]] = {}


def binary_positional_features(n_vars: int, to_pm1: bool = False):
    bits = max(1, math.ceil(math.log2(max(2, n_vars))))
    idx = torch.arange(n_vars, dtype=torch.long)[:, None]
    bit_pos = torch.arange(bits, dtype=torch.long)
    codes = ((idx >> bit_pos) & 1).to(torch.float32)
    codes = torch.flip(codes, dims=[1])
    if to_pm1:
        codes = codes * 2.0 - 1.0
    return codes, bits


def parse_percent_list(raw: str) -> List[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def mycallback(model, where):
    if where == GRB.Callback.MIPSOL:
        obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        runtime = time.monotonic() - model._starttime
        log_entry_name = model.Params.LogFile
        log_entry = {"primal_bound": obj, "solving_time": runtime}
        gurobi_log[log_entry_name].append(log_entry)
        print("New solution", obj, "Runtime", runtime)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ID-PAS+ search on a single instance.")
    parser.add_argument("--instance", required=True)
    parser.add_argument("--expname", required=True)
    parser.add_argument("--methods", default="IM,IM_id")
    parser.add_argument("--kb0-pcts", default="0,10,30,50,70,90")
    parser.add_argument("--ki0-pcts", default="0,10,30,50,70,90")
    parser.add_argument("--delta-pcts", default="1,3,5")
    parser.add_argument("--time-limit", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--mip-focus", type=int, default=1)
    parser.add_argument("--im-model", default="pretrain/slap_hard_IM/model_best.pth")
    parser.add_argument("--im-id-model", default="pretrain/slap_hard_IM_id/model_best.pth")
    return parser.parse_args()


def build_model_inputs(method: str, variable_features: torch.Tensor) -> Tuple[torch.Tensor, GATPolicy]:
    if "id" in method:
        n_vars = variable_features.size(0)
        bin_feats, _ = binary_positional_features(n_vars)
        variable_features = torch.cat([variable_features, bin_feats], dim=1)
        model = GATPolicy(var_nfeats=32)
    else:
        model = GATPolicy()
    return variable_features, model


def get_type_scores(
    v_map: Dict[str, int],
    probs: torch.Tensor,
    b_vars: torch.Tensor,
    i_vars: torch.Tensor,
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    all_var_names = list(v_map.keys())
    binary_scores = [(all_var_names[idx], probs[idx].item()) for idx in b_vars.tolist()]
    integer_scores = [(all_var_names[idx], probs[idx].item()) for idx in i_vars.tolist()]
    binary_scores.sort(key=lambda item: item[1])
    integer_scores.sort(key=lambda item: item[1])
    return binary_scores, integer_scores


def select_zero_variables(
    binary_scores: Sequence[Tuple[str, float]],
    integer_scores: Sequence[Tuple[str, float]],
    kb0_pct: int,
    ki0_pct: int,
) -> List[str]:
    n_bin = int(len(binary_scores) * kb0_pct / 100.0)
    n_int = int(len(integer_scores) * ki0_pct / 100.0)
    selected = [name for name, _ in binary_scores[:n_bin]]
    selected.extend(name for name, _ in integer_scores[:n_int])
    return selected


def add_nonzero_indicator(model: grb.Model, var: grb.Var, suffix: str) -> grb.Var:
    if var.VType == GRB.BINARY:
        indicator = model.addVar(vtype=GRB.BINARY, name=f"nz_{suffix}")
        model.addConstr(indicator == var, name=f"link_binary_{suffix}")
        return indicator

    indicator = model.addVar(vtype=GRB.BINARY, name=f"nz_{suffix}")

    if var.LB >= 0:
        model.addGenConstrIndicator(indicator, 0, var, GRB.EQUAL, 0, name=f"force_zero_{suffix}")
        model.addGenConstrIndicator(indicator, 1, var, GRB.GREATER_EQUAL, 1, name=f"force_pos_{suffix}")
        return indicator

    if var.UB <= 0:
        model.addGenConstrIndicator(indicator, 0, var, GRB.EQUAL, 0, name=f"force_zero_{suffix}")
        model.addGenConstrIndicator(indicator, 1, var, GRB.LESS_EQUAL, -1, name=f"force_neg_{suffix}")
        return indicator

    pos_indicator = model.addVar(vtype=GRB.BINARY, name=f"nz_pos_{suffix}")
    neg_indicator = model.addVar(vtype=GRB.BINARY, name=f"nz_neg_{suffix}")
    model.addConstr(indicator == pos_indicator + neg_indicator, name=f"nz_split_{suffix}")
    model.addConstr(pos_indicator + neg_indicator <= 1, name=f"nz_split_cap_{suffix}")
    model.addGenConstrIndicator(indicator, 0, var, GRB.EQUAL, 0, name=f"force_zero_{suffix}")
    model.addGenConstrIndicator(pos_indicator, 1, var, GRB.GREATER_EQUAL, 1, name=f"force_pos_{suffix}")
    model.addGenConstrIndicator(neg_indicator, 1, var, GRB.LESS_EQUAL, -1, name=f"force_neg_{suffix}")
    return indicator


def solve_with_search(
    instance_path: str,
    expname: str,
    method: str,
    selected_var_names: Sequence[str],
    delta_pct: int,
    time_limit: int,
    threads: int,
    mip_focus: int,
) -> None:
    run_name = f"gat{method}_kbki_{len(selected_var_names)}_d{delta_pct}"
    output_dir = os.path.join("results", expname, run_name)
    os.makedirs(output_dir, exist_ok=True)

    model = grb.read(instance_path)
    model.Params.TimeLimit = time_limit
    model.Params.Threads = threads
    model.Params.MIPFocus = mip_focus
    model.Params.LogFile = run_name
    gurobi_log[run_name] = []

    model_vars = model.getVars()
    model_vars.sort(key=lambda var: var.VarName)
    var_map = {var.VarName: var for var in model_vars}

    indicators = []
    for idx, var_name in enumerate(selected_var_names):
        indicators.append(add_nonzero_indicator(model, var_map[var_name], f"{idx}_{var_name}"))

    delta_budget = int(round(len(selected_var_names) * delta_pct / 100.0))
    if indicators:
        model.addConstr(grb.quicksum(indicators) <= delta_budget, name="sum_nonzero")

    model._starttime = time.monotonic()
    model.optimize(mycallback)

    instance_name = os.path.basename(instance_path)
    with open(os.path.join(output_dir, instance_name), "wb") as handle:
        pickle.dump(gurobi_log[run_name], handle)


def main() -> None:
    args = parse_args()
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    kb0_pcts = parse_percent_list(args.kb0_pcts)
    ki0_pcts = parse_percent_list(args.ki0_pcts)
    delta_pcts = parse_percent_list(args.delta_pcts)

    default_model_by_method = {
        "IM": args.im_model,
        "IM_id": args.im_id_model,
    }

    A, v_map, v_nodes, c_nodes, b_vars, i_vars, _ = compute_mip_representation(args.instance)
    constraint_features = c_nodes.cpu()
    base_variable_features = v_nodes
    edge_indices = A._indices()
    edge_features = torch.ones(A._values().unsqueeze(1).shape)

    for method in methods:
        variable_features, model = build_model_inputs(method, base_variable_features.clone())
        saved_dict = torch.load(default_model_by_method[method], map_location=torch.device("cpu"))
        model.load_state_dict(saved_dict)
        model.eval()

        probs = model(
            constraint_features.to(DEVICE),
            edge_indices.to(DEVICE),
            edge_features.to(DEVICE),
            variable_features.to(DEVICE),
        ).sigmoid().cpu().squeeze()

        binary_scores, integer_scores = get_type_scores(v_map, probs, b_vars, i_vars)

        for kb0_pct in kb0_pcts:
            for ki0_pct in ki0_pcts:
                selected_var_names = select_zero_variables(binary_scores, integer_scores, kb0_pct, ki0_pct)
                for delta_pct in delta_pcts:
                    print(
                        f"Running {method} with kb0={kb0_pct}% ki0={ki0_pct}% "
                        f"delta={delta_pct}% over |X0|={len(selected_var_names)}"
                    )
                    solve_with_search(
                        instance_path=args.instance,
                        expname=args.expname,
                        method=f"{method}_kb{kb0_pct}_ki{ki0_pct}",
                        selected_var_names=selected_var_names,
                        delta_pct=delta_pct,
                        time_limit=args.time_limit,
                        threads=args.threads,
                        mip_focus=args.mip_focus,
                    )


if __name__ == "__main__":
    main()
