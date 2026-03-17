import argparse
import os
import time
from dataclasses import dataclass

import gurobipy as gp
from gurobipy import GRB


@dataclass
class ModelStats:
    num_vars: int
    num_bin: int
    num_int: int
    num_cont: int
    num_constrs: int
    num_eq: int
    num_leq: int
    num_geq: int
    num_nz: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Solve a single Gurobi-readable instance and write a text report with "
            "basic model statistics and the best solution found."
        )
    )
    parser.add_argument("instance_file", help="Path to the instance file to solve.")
    parser.add_argument(
        "--output-dir",
        default="solution_new",
        help="Directory where the solution report will be written.",
    )
    parser.add_argument(
        "--log-dir",
        default="new",
        help="Directory where the Gurobi log file will be written.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of Gurobi threads to use.",
    )
    parser.add_argument(
        "--write-vars",
        action="store_true",
        help="Include variable assignments when a feasible solution is available.",
    )
    return parser.parse_args()


def collect_model_stats(model: gp.Model) -> ModelStats:
    variables = model.getVars()
    constraints = model.getConstrs()

    return ModelStats(
        num_vars=model.numVars,
        num_bin=sum(1 for var in variables if var.VType == GRB.BINARY),
        num_int=sum(1 for var in variables if var.VType == GRB.INTEGER),
        num_cont=sum(1 for var in variables if var.VType == GRB.CONTINUOUS),
        num_constrs=model.numConstrs,
        num_eq=sum(1 for constr in constraints if constr.Sense == "="),
        num_leq=sum(1 for constr in constraints if constr.Sense == "<"),
        num_geq=sum(1 for constr in constraints if constr.Sense == ">"),
        num_nz=model.numNZs,
    )


def status_name(status_code: int) -> str:
    for name in dir(GRB.Status):
        if name.startswith("_"):
            continue
        if getattr(GRB.Status, name) == status_code:
            return name
    return f"UNKNOWN_STATUS_{status_code}"


def objective_sense_name(model: gp.Model) -> str:
    if model.ModelSense == GRB.MINIMIZE:
        return "minimize"
    return "maximize"


def write_report(
    report_path: str,
    model: gp.Model,
    stats: ModelStats,
    runtime: float,
    write_vars: bool,
) -> None:
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(f"Runtime (seconds): {runtime:.2f}\n")
        handle.write(f"Status: {status_name(model.Status)}\n")
        handle.write(f"Objective sense: {objective_sense_name(model)}\n")
        handle.write(f"Solutions found: {model.SolCount}\n")

        if model.SolCount > 0:
            handle.write(f"Best objective value: {model.ObjVal}\n")

        if model.Status in {GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL} and hasattr(model, "ObjBound"):
            handle.write(f"Best bound: {model.ObjBound}\n")

        if model.SolCount > 0 and hasattr(model, "MIPGap"):
            handle.write(f"MIP gap: {model.MIPGap}\n")

        handle.write("\n")
        handle.write(f"# Variables: {stats.num_vars}\n")
        handle.write(f"  - Binary: {stats.num_bin}\n")
        handle.write(f"  - Integer: {stats.num_int}\n")
        handle.write(f"  - Continuous: {stats.num_cont}\n")

        handle.write(f"# Constraints: {stats.num_constrs}\n")
        handle.write(f"  - Equality (=): {stats.num_eq}\n")
        handle.write(f"  - Less-equal (<=): {stats.num_leq}\n")
        handle.write(f"  - Greater-equal (>=): {stats.num_geq}\n")

        handle.write(f"# Non-zeros in A: {stats.num_nz}\n")

        if write_vars and model.SolCount > 0:
            handle.write("\nBest solution variable values:\n")
            for var in model.getVars():
                handle.write(f"{var.VarName} = {var.X}\n")


def main() -> int:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    instance_file = args.instance_file
    model_name = os.path.splitext(os.path.basename(instance_file))[0]
    report_path = os.path.join(args.output_dir, model_name)
    log_path = os.path.join(args.log_dir, model_name)

    try:
        start_time = time.time()
        model = gp.read(instance_file)
        stats = collect_model_stats(model)

        model.Params.Threads = args.threads
        model.Params.LogFile = log_path
        model.optimize()
        runtime = time.time() - start_time

        write_report(report_path, model, stats, runtime, args.write_vars)
        return 0
    except Exception as exc:
        error_path = os.path.join(args.output_dir, f"{model_name}_error.txt")
        with open(error_path, "w", encoding="utf-8") as handle:
            handle.write(f"Error processing {instance_file}: {exc}\n")
        print(f"Error processing {instance_file}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
