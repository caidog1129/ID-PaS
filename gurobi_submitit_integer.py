import os.path
import pickle
from multiprocessing import Process, Queue
import gurobipy as gp
import numpy as np
import argparse
from IPython import embed
from helper import get_a_new2
import submitit
import random
import time
import copy
from helper import compute_mip_representation 

def solve_grb(filepath, log_dir, settings, savelog=True):
    print("solving instance", filepath)
    if savelog:
        gp.setParam('LogToConsole', 0)
    m = gp.read(filepath)

    # Configure solution pool
    m.Params.PoolSolutions = settings.get('maxsol', 520)
    m.Params.PoolSearchMode = settings.get('mode', 2)
    m.Params.Heuristics = 0

    m.Params.TimeLimit = settings['maxtime']
    print(m.Params.PoolSolutions, m.Params.PoolSearchMode, m.Params.TimeLimit)

    # Prepare log file
    log_path = os.path.join(log_dir, os.path.basename(filepath) + '.log')
    with open(log_path, 'w'):
        pass
    if savelog:
        m.Params.LogFile = log_path

    # Optimize to populate solution pool
    m.optimize()

    # Retrieve raw pool solutions
    sols = []
    objs = []
    solc = int(m.getAttr('SolCount'))
    pd_gap = m.MIPGap
    mvars = m.getVars()

    # Extract each solution vector and objective
    for sn in range(min(solc, settings['maxsol'])):
        m.Params.SolutionNumber = sn
        sols.append(np.array(m.Xn, dtype=np.float32))
        objs.append(float(m.PoolObjVal))

    sols = np.array(sols, dtype=np.float32)
    objs = np.array(objs, dtype=np.float32)

    # Identify integer variable indices
    int_indices = [i for i, var in enumerate(mvars)
                   if var.VType in {gp.GRB.INTEGER}]

    # Filter to unique integer-value solutions
    unique_sols = []
    unique_objs = []
    seen_int_assignments = set()

    for sol, obj in zip(sols, objs):
        # Create a tuple of integer values for hashing
        int_vals = tuple(int(sol[i]) for i in int_indices)
        if int_vals not in seen_int_assignments:
            seen_int_assignments.add(int_vals)
            unique_sols.append(sol)
            unique_objs.append(obj)

    sols = np.array(unique_sols, dtype=np.float32)
    objs = np.array(unique_objs, dtype=np.float32)

    # Prepare solution data
    sol_data = {
        'inst_name': filepath,
        'var_names': [var.varName for var in mvars],
        'sols': sols,
        'objs': objs,
        'pd_gap': pd_gap,
    }
    print(f"Filtered {len(sols)} unique integer solutions (out of {solc} found)")
    return sol_data


def get_negative_examples_via_inverse_local_branching(filepath,settings,ref_sol,iLB_bound):
    m = gp.read(filepath)
    m.Params.PoolSolutions = 20#1000000#settings['maxsol']
    m.Params.PoolSearchMode = 2#settings['mode']

    m.Params.TimeLimit = 300
    #m.Params.Threads = settings['threads']
    print(m.Params.PoolSolutions, m.Params.PoolSearchMode, m.Params.TimeLimit)

    sols = []
    objs = []
    
    mvars = m.getVars()
    gp.setParam('LogToConsole', 1)

    objective_sense = m.getAttr("ModelSense")
    
    if objective_sense == gp.GRB.MAXIMIZE:
        m.setAttr("ModelSense", gp.GRB.MINIMIZE)
    else:
        m.setAttr("ModelSense", gp.GRB.MAXIMIZE)
    

    sol_value = {}

    # for i, v in enumerate(mvars):
    #     if v.vtype == gp.GRB.INTEGER:
    #         sol_value[v] = ref_sol[i]
    #     # elif v.vtype == gp.GRB.CONTINUOUS:
    #     #     sol_value[v] = 10000
    #     #     #v.Obj *= -1
    #     # else:
    #     #     assert False, "we don't support general integer"
    # #embed()
    # m.addConstr(gp.quicksum(v for v in mvars if v.vtype == gp.GRB.INTEGER and abs(sol_value[v])<=1e-9) <= iLB_bound,"iLB")
    chg_from_zero = []
    # chg_to_zero = []
    
    for i, v in enumerate(mvars):
        if v.vtype == gp.GRB.INTEGER and abs(ref_sol[i]) <= 1e-9:
            z = m.addVar(vtype=gp.GRB.BINARY, name=f"chg0_{v.VarName}")
            chg_from_zero.append(z)

            # z = 0  ⇒  v = 0
            m.addGenConstrIndicator(z, 0, v, gp.GRB.EQUAL, 0)

            # # z = 1  ⇒  v ≥ 1
            # m.addGenConstrIndicator(z, 1, v, gp.GRB.GREATER_EQUAL, 1)

        # if v.VType == gp.GRB.INTEGER and abs(ref_sol[i]) > 1e-9:
        #     z = m.addVar(vtype=gp.GRB.BINARY, name=f"to0_{v.VarName}")
        #     chg_to_zero.append(z)
    
        #     # z = 0  ⇒  v ≥ 1   (so if v = 0, z can't be 0 → must be 1)
        #     m.addGenConstrIndicator(z, 0, v, gp.GRB.GREATER_EQUAL, 1)
    
        #     # z = 1  ⇒  v = 0   (tightens: don't let z=1 while v>0)
        #     m.addGenConstrIndicator(z, 1, v, gp.GRB.EQUAL, 0)

    m.addConstr(gp.quicksum(chg_from_zero) <= iLB_bound, name="iLB_zero")

    m.optimize()
    
    solc = m.getAttr('SolCount')

    for sn in range(min(solc,20)):
        m.Params.SolutionNumber = sn
        sols.append(np.array(m.Xn))
        objs.append(m.PoolObjVal)

    #embed()
    sols = np.array(sols,dtype=np.float32)
    objs = np.array(objs,dtype=np.float32)
    return sols, objs


def get_negative_examples_via_perturbation(
    filepath,
    settings,
    ref_sol,
    num_perturb,
    ref_obj=None,          # pass this if you have it; else we’ll estimate for linear objs
    degrade_frac=0.05,     # 20% worse
    max_keep=5,
    max_tries=100,
):
    sols, count, tried = [], 0, 0
    objs = []

    while count < max_keep and tried < max_tries:
        tried += 1

        m = gp.read(filepath)
        m.Params.TimeLimit = 5
        m.setParam('LogToConsole', 0)

        mvars = m.getVars()

        # randomly nudge some integer vars that were 0 in the reference to be nonzero
        ivars = [i for i, v in enumerate(mvars) if v.VType == gp.GRB.INTEGER]
        k = min(int(num_perturb), len(ivars))
        if k > 0:
            for i in random.sample(ivars, k):
                if ref_sol[i] <= 1e-8:                # only those previously zero
                    m.addConstr(mvars[i] >= 1, name=f"move_{i}")

        m.optimize()
        if m.SolCount == 0:
            print(f"try #{tried}: no feasible solution")
            continue

        cur_obj = m.ObjVal

        denom = max(1e-9, abs(ref_obj))
        if m.ModelSense == gp.GRB.MINIMIZE:          # 1
            rel_worse = (cur_obj - ref_obj) / denom
        else:                                        # MAXIMIZE (-1)
            rel_worse = (ref_obj - cur_obj) / denom

        if rel_worse >= degrade_frac:
            sols.append([v.X for v in mvars])
            objs.append(cur_obj)
            count += 1
            print(f"accepted try #{tried}: {rel_worse*100:.1f}% worse "
                  f"(obj {cur_obj:.6g} vs {ref_obj:.6g}) [{count}/{max_keep}]")
        else:
            print(f"rejected try #{tried}: only {rel_worse*100:.1f}% worse "
                  f"(obj {cur_obj:.6g} vs {ref_obj:.6g})")

    return np.array(sols, dtype=np.float32), np.array(objs,dtype=np.float32)
    

def collect(ins_dir,filename,sol_dir,log_dir,bg_dir,settings,savelog=True):

    filepath = os.path.join(ins_dir,filename)        
    if not os.path.exists(os.path.join(sol_dir, filename+'.sol')):
        sol_data = solve_grb(filepath,log_dir,settings,savelog=savelog)
        #get bipartite graph , binary variables' indices
        # A2,v_map2,v_nodes2,c_nodes2,b_vars2,i_vars2=get_a_new2(filepath)
        # BG_data=[A2,v_map2,v_nodes2,c_nodes2,b_vars2,i_vars2]
        
        # save data
        if savelog:
            pickle.dump(sol_data, open(os.path.join(sol_dir, filename+'.sol'), 'wb'))
            # pickle.dump(BG_data, open(os.path.join(bg_dir, filename+'.bg'), 'wb'))
    else:
        print("sol data already exists")
        with open(os.path.join(sol_dir, filename+'.sol'), "rb") as f:
            sol_data = pickle.load(f)
        with open(os.path.join(bg_dir, filename+'.bg'), "rb") as f:
            BG_data =  pickle.load(f)

    # sols = sol_data['sols']
    # objs = sol_data['objs']
    # neg_examples_iLB = []
    # neg_examples_perturb = []
    # num_ivars = len(BG_data[-1])


    # for rat, num_pvar in enumerate([num_ivars * 0.05, num_ivars * 0.1]):
        
    #     for i in range(min(sols.shape[0],50)):
    #         print("collect neg examples for %d and solution %d"%(rat, i))
    #         neg_ex, neg_ex_obj = get_negative_examples_via_inverse_local_branching(filepath, settings, sols[i], int(num_pvar))
    #         neg_examples_iLB.append((neg_ex, neg_ex_obj))
    #     sol_data["neg_examples_iLB_%d"%(rat)] = neg_examples_iLB

    #     # for i in range(min(sols.shape[0],50)):
    #     #     print("collect neg examples for %d and solution %d"%(rat, i))
    #     #     neg_ex, neg_ex_obj = get_negative_examples_via_perturbation(filepath, settings, sols[i], int(num_pvar), objs[i])
    #     #     neg_examples_iLB.append((neg_ex, neg_ex_obj))
    #     # sol_data["neg_examples_perturb_%d"%(rat)] = neg_examples_perturb
    #     #embed()
    # if savelog:
    #     pickle.dump(sol_data, open(os.path.join(sol_dir, filename+'.sol'), 'wb'))





if __name__ == '__main__':
    #sizes=['small','large']
    #sizes=["IP","WA","IS","CA","NNV"]
    #sizes=["IP"]
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='IP')
    parser.add_argument('--dataDir', type=str, default='./')
    parser.add_argument('--nWorkers', type=int, default=20)
    parser.add_argument('--maxTime', type=int, default=3600)
    parser.add_argument('--maxStoredSol', type=int, default=500)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--gatfeat', type=int, default=0)
    args = parser.parse_args()

    #args.nWorkers = int(os.environ['SLURM_CPUS_PER_TASK'])

    sizes = [args.task]

    for size in sizes:
    

        dataDir = args.dataDir

        INS_DIR = f'../order_fulfillment/{size}'

        if not os.path.isdir(f'datasets/{size}'):
            os.mkdir(f'datasets/{size}')
        if not os.path.isdir(f'datasets/{size}/solution'):
            os.mkdir(f'datasets/{size}/solution')
        if not os.path.isdir(f'datasets/{size}/NBP'):
            os.mkdir(f'datasets/{size}/NBP')
        if not os.path.isdir(f'datasets/{size}/logs'):
            os.mkdir(f'datasets/{size}/logs')
        if not os.path.isdir(f'datasets/{size}/BG'):
            os.mkdir(f'datasets/{size}/BG')

        SOL_DIR =f'datasets/{size}/solution'
        LOG_DIR =f'datasets/{size}/logs'
        BG_DIR =f'datasets/{size}/BG'
        os.makedirs(SOL_DIR, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)

        os.makedirs(BG_DIR, exist_ok=True)

        N_WORKERS = args.nWorkers

        # gurobi settings
        SETTINGS = {
            'maxtime': args.maxTime,
            'mode': 2,
            'maxsol': args.maxStoredSol,
            'threads': args.threads,

        }

        filenames = os.listdir(INS_DIR)
        filenames = [x for x in os.listdir(INS_DIR) if "mps" in x or "lp" in x]
        filenames.sort()
        # filenames = filenames[0:600] 
        #embed()

        print("collecting data for %d instances"%(len(filenames)))

        executor = submitit.AutoExecutor(folder="/project2/dilkina_438/caijunya/slurm")
        executor.update_parameters(tasks_per_node=1)
        #executor.update_parameters(gpus_per_node=1)
        executor.update_parameters(cpus_per_task=8)
        executor.update_parameters(slurm_account="dilkina_438")
        executor.update_parameters(slurm_mem_per_cpu="16G")
        executor.update_parameters(name="datacollection", timeout_min=10*60, slurm_partition="main")
        jobs = []
        print("submitting jobs to clusters")

        n_tasks = len(filenames)
        INS_DIR = [INS_DIR] * n_tasks
        SOL_DIR = [SOL_DIR] * n_tasks
        LOG_DIR = [LOG_DIR] * n_tasks
        BG_DIR = [BG_DIR] * n_tasks
        SETTINGS = [SETTINGS] * n_tasks

        jobs = executor.map_array(collect, INS_DIR,filenames,SOL_DIR,LOG_DIR,BG_DIR,SETTINGS)
        outputs = [job.result() for job in jobs]
        print(outputs)
        print("done")
        exit()

        q = Queue()
        # add ins
        for filename in filenames:
            if not os.path.exists(os.path.join(BG_DIR,filename+'.bg')):
                q.put(filename)
        print("Queue size", q.qsize())
        # add stop signal
        for i in range(N_WORKERS):
            q.put(None)

        ps = []
        for i in range(N_WORKERS):
            p = Process(target=collect,args=(INS_DIR,q,SOL_DIR,LOG_DIR,BG_DIR,SETTINGS))
            p.start()
            ps.append(p)
        for p in ps:
            p.join()

        print('done')


