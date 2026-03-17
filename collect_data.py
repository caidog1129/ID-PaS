import argparse
import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import gurobipy as gp
import numpy as np


def solve_grb(filepath, log_dir, settings, savelog=True):
    print("solving instance", filepath)
    if savelog:
        gp.setParam("LogToConsole", 0)
    model = gp.read(filepath)

    model.Params.PoolSolutions = settings.get("maxsol", 520)
    model.Params.PoolSearchMode = settings.get("mode", 2)
    model.Params.Heuristics = 0
    model.Params.TimeLimit = settings["maxtime"]
    print(model.Params.PoolSolutions, model.Params.PoolSearchMode, model.Params.TimeLimit)

    log_path = os.path.join(log_dir, os.path.basename(filepath) + ".log")
    with open(log_path, "w", encoding="utf-8"):
        pass
    if savelog:
        model.Params.LogFile = log_path

    model.optimize()

    sols = []
    objs = []
    solc = int(model.getAttr("SolCount"))
    pd_gap = model.MIPGap
    model_vars = model.getVars()

    for sn in range(min(solc, settings["maxsol"])):
        model.Params.SolutionNumber = sn
        sols.append(np.array(model.Xn, dtype=np.float32))
        objs.append(float(model.PoolObjVal))

    sols = np.array(sols, dtype=np.float32)
    objs = np.array(objs, dtype=np.float32)

    discrete_indices = [i for i, var in enumerate(model_vars) if var.VType != gp.GRB.CONTINUOUS]
    unique_sols = []
    unique_objs = []
    seen_discrete_assignments = set()

    for sol, obj in zip(sols, objs):
        discrete_vals = tuple(int(sol[i]) for i in discrete_indices)
        if discrete_vals not in seen_discrete_assignments:
            seen_discrete_assignments.add(discrete_vals)
            unique_sols.append(sol)
            unique_objs.append(obj)

    sols = np.array(unique_sols, dtype=np.float32)
    objs = np.array(unique_objs, dtype=np.float32)

    sol_data = {
        "inst_name": filepath,
        "var_names": [var.varName for var in model_vars],
        "sols": sols,
        "objs": objs,
        "pd_gap": pd_gap,
    }
    print(f"Filtered {len(sols)} unique discrete solutions (out of {solc} found)")
    return sol_data


def collect_instance(ins_dir, filename, sol_dir, log_dir, settings, savelog=True):
    filepath = os.path.join(ins_dir, filename)
    out_path = os.path.join(sol_dir, filename + ".sol")
    if not os.path.exists(out_path):
        sol_data = solve_grb(filepath, log_dir, settings, savelog=savelog)
        if savelog:
            with open(out_path, "wb") as handle:
                pickle.dump(sol_data, handle)
    else:
        print("solution data already exists")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="IP")
    parser.add_argument("--instance-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="datasets")
    parser.add_argument("--nWorkers", type=int, default=1)
    parser.add_argument("--maxTime", type=int, default=3600)
    parser.add_argument("--maxStoredSol", type=int, default=500)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    sizes = [args.task]

    for size in sizes:
        ins_dir = args.instance_dir or f"../order_fulfillment/{size}"

        os.makedirs(f"{args.output_dir}/{size}/solution", exist_ok=True)
        os.makedirs(f"{args.output_dir}/{size}/logs", exist_ok=True)

        sol_dir = f"{args.output_dir}/{size}/solution"
        log_dir = f"{args.output_dir}/{size}/logs"

        settings = {
            "maxtime": args.maxTime,
            "mode": 2,
            "maxsol": args.maxStoredSol,
            "threads": args.threads,
        }

        filenames = [x for x in os.listdir(ins_dir) if "mps" in x or "lp" in x]
        filenames.sort()
        print("collecting data for %d instances" % len(filenames))

        if args.nWorkers <= 1:
            for filename in filenames:
                collect_instance(ins_dir, filename, sol_dir, log_dir, settings)
        else:
            with ProcessPoolExecutor(max_workers=args.nWorkers) as executor:
                futures = [
                    executor.submit(collect_instance, ins_dir, filename, sol_dir, log_dir, settings)
                    for filename in filenames
                ]
                for future in futures:
                    future.result()

        print("done")


if __name__ == "__main__":
    main()
