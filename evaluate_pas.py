import argparse
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
    parser = argparse.ArgumentParser(description="Evaluate the original binary-only PaS search on a single instance.")
    parser.add_argument("--instance", required=True)
    parser.add_argument("--expname", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--k0-pcts", default="0,10,20,30,40,50")
    parser.add_argument("--k1-pcts", default="0,10,20,30,40,50")
    parser.add_argument("--delta-pcts", default="1,3,5")
    parser.add_argument("--time-limit", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--mip-focus", type=int, default=1)
    return parser.parse_args()


def select_binary_sets(
    binary_scores: Sequence[Tuple[str, float]],
    k0_pct: int,
    k1_pct: int,
) -> Tuple[List[str], List[str]]:
    n_zero = int(len(binary_scores) * k0_pct / 100.0)
    n_one = int(len(binary_scores) * k1_pct / 100.0)

    zero_set = [name for name, _ in binary_scores[:n_zero]]
    zero_names = set(zero_set)

    one_set = []
    for name, _ in reversed(binary_scores):
        if name in zero_names:
            continue
        one_set.append(name)
        if len(one_set) >= n_one:
            break

    return zero_set, one_set


def solve_with_pas(
    instance_path: str,
    expname: str,
    zero_var_names: Sequence[str],
    one_var_names: Sequence[str],
    delta_pct: int,
    time_limit: int,
    threads: int,
    mip_focus: int,
    run_tag: str,
) -> None:
    run_name = f"pas_{run_tag}_d{delta_pct}"
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

    flip_terms = []
    for var_name in zero_var_names:
        flip_terms.append(var_map[var_name])
    for var_name in one_var_names:
        flip_terms.append(1 - var_map[var_name])

    target_size = len(zero_var_names) + len(one_var_names)
    delta_budget = int(round(target_size * delta_pct / 100.0))
    if flip_terms:
        model.addConstr(grb.quicksum(flip_terms) <= delta_budget, name="pas_neighborhood")

    model._starttime = time.monotonic()
    model.optimize(mycallback)

    instance_name = os.path.basename(instance_path)
    with open(os.path.join(output_dir, instance_name), "wb") as handle:
        pickle.dump(gurobi_log[run_name], handle)


def main() -> None:
    args = parse_args()
    k0_pcts = parse_percent_list(args.k0_pcts)
    k1_pcts = parse_percent_list(args.k1_pcts)
    delta_pcts = parse_percent_list(args.delta_pcts)

    A, v_map, v_nodes, c_nodes, b_vars, _i_vars, _ = compute_mip_representation(args.instance)
    constraint_features = c_nodes.cpu()
    variable_features = v_nodes
    edge_indices = A._indices()
    edge_features = torch.ones(A._values().unsqueeze(1).shape)

    model = GATPolicy()
    saved_dict = torch.load(args.model, map_location=torch.device("cpu"))
    model.load_state_dict(saved_dict)
    model.eval()

    probs = model(
        constraint_features.to(DEVICE),
        edge_indices.to(DEVICE),
        edge_features.to(DEVICE),
        variable_features.to(DEVICE),
    ).sigmoid().cpu().squeeze()

    all_var_names = list(v_map.keys())
    binary_scores = [(all_var_names[idx], probs[idx].item()) for idx in b_vars.tolist()]
    binary_scores.sort(key=lambda item: item[1])

    for k0_pct in k0_pcts:
        for k1_pct in k1_pcts:
            zero_var_names, one_var_names = select_binary_sets(binary_scores, k0_pct, k1_pct)
            run_tag = f"k0{k0_pct}_k1{k1_pct}"
            for delta_pct in delta_pcts:
                print(
                    f"Running PaS with k0={k0_pct}% k1={k1_pct}% "
                    f"delta={delta_pct}% over {len(zero_var_names) + len(one_var_names)} binaries"
                )
                solve_with_pas(
                    instance_path=args.instance,
                    expname=args.expname,
                    zero_var_names=zero_var_names,
                    one_var_names=one_var_names,
                    delta_pct=delta_pct,
                    time_limit=args.time_limit,
                    threads=args.threads,
                    mip_focus=args.mip_focus,
                    run_tag=run_tag,
                )


if __name__ == "__main__":
    main()
