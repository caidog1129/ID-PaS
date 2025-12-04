import sys
import gurobipy as gp
from gurobipy import GRB
import os
import time

"""
solve_instance.py
-----------------
Loads a model file readable by Gurobi, solves it on a single thread, and writes a
summary report (plus full decision‑variable values if optimal) to the
``solutions_very_hard`` folder.

New statistics added (replacing the old matrix‑shape line):
  • Total variables, and the breakdown by binary, integer, and continuous
  • Total constraints, and the breakdown by sense (=, ≤, ≥)
  • Total non‑zeros in the constraint matrix
  • Runtime in seconds
"""

# === Check CLI arguments ===
if len(sys.argv) != 2:
    print("Usage: python solve_instance.py <instance_file>")
    sys.exit(1)

instance_file = sys.argv[1]
model_name = os.path.splitext(os.path.basename(instance_file))[0]
output_folder = "solution_new"
os.makedirs(output_folder, exist_ok=True)
log_folder = "new"
os.makedirs(log_folder, exist_ok=True)

try:
    # === Load model and collect pre‑solve statistics ===
    start_time = time.time()
    model = gp.read(instance_file)

    num_vars = model.numVars
    num_bin  = sum(1 for v in model.getVars() if v.VType == GRB.BINARY)
    num_int  = sum(1 for v in model.getVars() if v.VType == GRB.INTEGER)
    num_cont = sum(1 for v in model.getVars() if v.VType == GRB.CONTINUOUS)

    num_constrs = model.numConstrs
    num_eq   = sum(1 for c in model.getConstrs() if c.Sense == '=')
    num_leq  = sum(1 for c in model.getConstrs() if c.Sense == '<')
    num_geq  = sum(1 for c in model.getConstrs() if c.Sense == '>')

    num_nz = model.numNZs  # non‑zeros in the constraint matrix

    # === Solve ===
    model.Params.Threads = 1
    log_path = os.path.join(log_folder, model_name)
    model.Params.LogFile = log_path
    model.optimize()
    runtime = time.time() - start_time

    # === Write results ===
    report_path = os.path.join(output_folder, model_name)
    with open(report_path, "w") as f:
        # General run info
        f.write(f"Runtime (seconds): {runtime:.2f}\n")

        # Variable statistics
        f.write(f"# Variables: {num_vars}\n")
        f.write(f"  • Binary     : {num_bin}\n")
        f.write(f"  • Integer    : {num_int}\n")
        f.write(f"  • Continuous : {num_cont}\n")

        # Constraint statistics
        f.write(f"# Constraints: {num_constrs}\n")
        f.write(f"  • Equality ( = ) : {num_eq}\n")
        f.write(f"  • Less‑eq  ( ≤ ) : {num_leq}\n")
        f.write(f"  • Greater‑eq( ≥ ) : {num_geq}\n")

        # Matrix sparsity
        f.write(f"# Non‑zeros in A: {num_nz}\n\n")

        # Solution values if solved to optimality
        if model.status == GRB.OPTIMAL:
            f.write(f"Optimal Objective Value: {model.ObjVal}\n\n")
            f.write("Optimal Variable Values:\n")
            for v in model.getVars():
                f.write(f"{v.varName} = {v.x}\n")
        else:
            f.write("No optimal solution found.\n")

except Exception as e:
    print(f"Error processing {instance_file}: {str(e)}")
    with open(f"{output_folder}/{model_name}_error.txt", "w") as f:
        f.write(f"Error processing {instance_file}: {str(e)}\n")
