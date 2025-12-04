import os
import torch
import torch_geometric
import random
import time
import argparse
import copy
from tqdm import tqdm
from IPython import embed
import pickle
from pytorch_metric_learning import losses
from torchmetrics.functional.classification import binary_accuracy, auroc as tm_auroc
#from torchmetrics.classification import BinaryAccuracy
from torcheval.metrics import BinaryAccuracy
from pytorch_metric_learning.distances import DotProductSimilarity
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
import numpy as np

from MIPDataset import GraphDataset, GraphDataset_id_v1, GraphDataset_id_v2
from GAT import GATPolicy
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
#this file is to train a predict model. given a instance's bipartite graph as input, the model predict the binary distribution.

def filter_data_file(sample_files):
    print("starting to filter bad data files")
    bad_list = ["MVC_barabasi_albert_297.lp","MVC_barabasi_albert_295.lp","MVC_barabasi_albert_287.lp","MVC_barabasi_albert_298.lp","MVC_barabasi_albert_28.lp","MVC_barabasi_albert_292.lp","MVC_barabasi_albert_280.lp"]+\
        ["CA_2000_4000_519.lp"]
    corrupted_files = 0
    valid_files = []
    has_iLB = 0
    for i,file in enumerate(sample_files):
        BGFilepath, solFilePath = file
        corrupted = False
        for bad_file in bad_list:
            if bad_file in solFilePath:
                corrupted = True
        # with open(BGFilepath, "rb") as f:
        #     try:
        #         bgData = pickle.load(f)
        #     except:
        #         corrupted = True
        with open(solFilePath, "rb") as f:
            try:
                solData = pickle.load(f)
            except:
                corrupted = True
        if corrupted:
            corrupted_files += 1
        else:
            if "neg_examples_iLB_0" in solData:
                has_iLB += 1
                #print(file)
                valid_files.append(file)
            else:
                print(file,"has no iLB")
        if i%50==0: print("processed %d files"%(i+1))
            
    print("filted out %d corrupted files"%(corrupted_files), ";", has_iLB, "has iLB")
    #exit()
    return valid_files

def EnergyWeightNorm(task):
    if task=="IP":
        return 1
    elif task=="WA":
        return 100
    elif task=="WA_90_1500":
        return 100
    elif task == "IS":
        return -100
    elif task == "CA":
        return -1000
    elif task == "CA_2000_4000":
        return -10000
    elif task == "CA_3000_6000":
        return -10000
    elif task == "CA_400_2000":
        return -1000
    elif task == "INDSET_BA4_3000":
        return -200
    elif task == "INDSET_BA5_6000":
        return -200
    elif task == "MVC_BA5_3000":
        return 100
    elif task == "MVC_BA5_6000":
        return 200

parser = argparse.ArgumentParser()
parser.add_argument('--taskName', type=str, default="IP")
parser.add_argument('--pretrainModel', type=str, default=None)
parser.add_argument('--temp',type=float, default=0.07)
parser.add_argument('--perturb',type=float, default=0.05)
parser.add_argument('--loss',type=str, default='logloss') # "infoNCEloss"
parser.add_argument('--fracdata',type=float, default=1)
parser.add_argument('--weight', type=bool, default=False)
parser.add_argument('--freeze', type=int, default = 0)
parser.add_argument('--negex', type=str,default="iLB") # can be "iLB" or "perturb"
parser.add_argument('--instance_dir', type=str)
parser.add_argument('--result_dir', type=str)
parser.add_argument('--var_nfeats', type=int, default = 15)

args = parser.parse_args()

loss_function = args.loss

#set folder
train_task=args.taskName
if not os.path.isdir(f'./train_logs'):
    os.mkdir(f'./train_logs')
if not os.path.isdir(f'./train_logs/{train_task}'):
    os.mkdir(f'./train_logs/{train_task}')
if not os.path.isdir(f'./pretrain'):
    os.mkdir(f'./pretrain')
if not os.path.isdir(f'./pretrain/{train_task}'):
    os.mkdir(f'./pretrain/{train_task}')
model_save_path = f'./pretrain/{train_task}/'
log_save_path = f"train_logs/{train_task}/"
log_file = open(f'{log_save_path}{train_task}_train.log', 'wb')
writer = SummaryWriter(f'{log_save_path}')

#set params
LEARNING_RATE = 0.00005
NB_EPOCHS =9999
BATCH_SIZE = 8
NUM_WORKERS = 0
WEIGHT_NORM=100

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

infoNCE_loss_function = losses.NTXentLoss(temperature=args.temp, distance=DotProductSimilarity()).to(DEVICE)
bce_loss = torch.nn.BCELoss().to(DEVICE)

sample_names = os.listdir(args.result_dir)
sample_files = [(os.path.join(args.instance_dir,name).replace('.sol',''), os.path.join(args.result_dir,name)) for name in sample_names]
sample_files = sorted(sample_files)
# sample_files = filter_data_file(sample_files)
    
random.seed(67)
random.shuffle(sample_files)

sample_files = sample_files[:int(len(sample_files)*args.fracdata)]

train_files = sample_files[int(0.2 * len(sample_files)):]
valid_files = sample_files[:int(0.20 * len(sample_files))]

print("Training on", int(0.80 * len(sample_files)), "instances")
print("Validating on", int(0.2 * len(sample_files)), "instances")

if args.var_nfeats == 15:
    train_data = GraphDataset(train_files, args)
    valid_data = GraphDataset(valid_files, args)
elif args.var_nfeats == 23:
    train_data = GraphDataset_id_v1(train_files, args)
    valid_data = GraphDataset_id_v1(valid_files, args)
elif args.var_nfeats == 27:
    train_data = GraphDataset_id_v2(train_files, args)
    valid_data = GraphDataset_id_v2(valid_files, args)
elif args.var_nfeats == 30:
    train_data = GraphDataset_id_v2(train_files, args)
    valid_data = GraphDataset_id_v2(valid_files, args)
elif args.var_nfeats == 31:
    train_data = GraphDataset_id_v2(train_files, args)
    valid_data = GraphDataset_id_v2(valid_files, args)
elif args.var_nfeats == 32:
    train_data = GraphDataset_id_v2(train_files, args)
    valid_data = GraphDataset_id_v2(valid_files, args)

train_loader = torch_geometric.loader.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
valid_loader = torch_geometric.loader.DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

PredictModel = GATPolicy(var_nfeats = args.var_nfeats).to(DEVICE)

if not args.pretrainModel is None:
    PredictModel.load_state_dict(torch.load(args.pretrainModel), strict=False)
    print(f"loaded model from {args.pretrainModel}")
    if args.freeze == 1:
        for name, param in PredictModel.named_parameters():
            if 'conv_v_to_c' in name or 'conv_c_to_v' in name or 'embedding' in name:
                param.requires_grad = False

def train(epoch, predict, data_loader, optimizer=None,weight_norm=100,loss_function = "logloss"):
    """
    This function will process a whole epoch of training or validation, depending on whether an optimizer is provided.
    """

    if optimizer:
        predict.train()
    else:
        predict.eval()
    mean_loss = 0
    n_samples_processed = 0
    
    with torch.set_grad_enabled(optimizer is not None):
        for step, batch in enumerate(data_loader):
            batch = batch.to(DEVICE)
            solInd = batch.nsols
            if loss_function in ['infoNCEloss', 'hybrid']: 
                negInd = batch.nnegsamples
            target_sols = []
            target_vals = []
            target_negs = []
            instances = []
            solEndInd = 0
            valEndInd = 0
            negSampEndInd = 0

            for i in range(solInd.shape[0]):#for in batch
                nvar = len(batch.varInds[i][0][0])
                solStartInd = solEndInd
                solEndInd = solInd[i] * nvar + solStartInd
                valStartInd = valEndInd
                valEndInd = valEndInd + solInd[i]
                
                sols = batch.solutions[solStartInd:solEndInd].reshape(-1, nvar)
                vals = batch.objVals[valStartInd:valEndInd]

                target_sols.append(sols)
                target_vals.append(vals)

                if loss_function in ['infoNCEloss', 'hybrid']: 
                    negSampStartInd = negSampEndInd 
                    negSampEndInd = negInd[i] * nvar + negSampStartInd
                    negs = batch.negsamples[negSampStartInd:negSampEndInd].reshape(-1, nvar)
                    target_negs.append(negs)
                else:
                    target_negs.append(None)

            # Compute the logits (i.e. pre-softmax activations) according to the policy on the concatenated graphs
            batch.constraint_features[torch.isinf(batch.constraint_features)] = 10 #remove nan value
            BD = predict(
                batch.constraint_features,
                batch.edge_index,
                batch.edge_attr,
                batch.variable_features,
            )
            BD = BD.sigmoid()

            # calculate weights
            index_arrow = 0
            embeddings = None

            total_samples = 0
            anchor_positive = []
            positive_idx = []
            anchor_negative = []
            negative_idx = []

            loss = 0
            batch_bce_loss = 0

            count = 0
            
            for ind,(sols,vals,negs) in enumerate(zip(target_sols,target_vals, target_negs)):  
                #compute weight
                n_vals = vals
                exp_weight = torch.exp(-n_vals/weight_norm)
                # print("min", min(exp_weight), "max", max(exp_weight))
                weight = exp_weight/exp_weight.sum()
                
                # get a integer mask
                varInds = batch.varInds[ind]
                varname_map=varInds[0][0]
                i_vars=varInds[1][0].long()

                #get integer variables
                sols = sols[:,varname_map][:,i_vars]

                # cross-entropy
                n_var = batch.ntvars[ind]
                pre_sols = BD[index_arrow:index_arrow + n_var].squeeze()[i_vars]
                index_arrow = index_arrow + n_var
                pos_loss = -(pre_sols+ 1e-8).log()[None,:]*(sols==1).float()
                neg_loss = -(1-pre_sols + 1e-8).log()[None,:]*(sols==0).float()
                sum_loss = pos_loss + neg_loss

                sample_loss = sum_loss
                sample_loss = sum_loss*weight[:,None]
                count += 1
                loss += sample_loss.sum()

                if loss_function in ['infoNCEloss', 'hybrid']: 
                    negs = negs[:,varname_map][:,i_vars]

                    # positive_sample_weights = n_vals/weight_norm #w1
                    # cur_embeddings = torch.cat([pre_sols.reshape(1,-1), sols * positive_sample_weights[:,None], negs * positive_sample_weights[:,None]])
                    cur_embeddings = torch.cat([pre_sols.reshape(1,-1), sols, negs])
    
                    anchor_positive = anchor_positive + [total_samples] * sols.shape[0]
                    anchor_negative = anchor_negative + [total_samples] * negs.shape[0]
                    positive_idx = positive_idx + list(range(total_samples + 1, total_samples + 1 + sols.shape[0]))
                    negative_idx = negative_idx + list(range(total_samples + 1 + sols.shape[0], total_samples + 1 + sols.shape[0] + negs.shape[0]))
                    total_samples += 1 + sols.shape[0] + negs.shape[0]
    
                    if embeddings is None:
                        embeddings = cur_embeddings
                    else:
                        embeddings = torch.cat([embeddings,cur_embeddings])

                # if loss_function in ['hybrid']:
                    # temp_samples = torch.cat((sols, negs), dim = 0)
                    # temp_samples = np.array(temp_samples.cpu())
                    # same_ids = list(np.where(np.all(temp_samples == temp_samples[0, :], axis=0))[0])
                    # batch_bce_loss += bce_loss(pre_sols[same_ids].to(DEVICE), torch.tensor(np.squeeze(np.array(temp_samples[0, :]))[same_ids]).float().to(DEVICE))

            loss = loss / count

            if loss_function in ['infoNCEloss', 'hybrid']: 
                triplets = (torch.tensor(anchor_positive).to(DEVICE), torch.tensor(positive_idx).to(DEVICE), torch.tensor(anchor_negative).to(DEVICE), torch.tensor(negative_idx).to(DEVICE))
                cl_loss = infoNCE_loss_function(embeddings, indices_tuple = triplets)

            if loss_function in ['hybrid']:
                print(loss, cl_loss)
                loss = loss / 120 + cl_loss
            if loss_function in ['infoNCEloss']:
                loss = cl_loss
                
            if optimizer is not None:
                writer.add_scalar('train_loss', loss, epoch)
            else:
                writer.add_scalar('valid_loss', loss, epoch)

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            mean_loss += loss
            n_samples_processed += len(batch)
    mean_loss /= n_samples_processed
    return mean_loss


# def train(epoch, predict, data_loader, optimizer=None,weight_norm=100,loss_function = "logloss"):
#     """
#     This function will process a whole epoch of training or validation, depending on whether an optimizer is provided.
#     """

#     if optimizer:
#         predict.train()
#     else:
#         predict.eval()
#     mean_loss = 0
#     n_samples_processed = 0
    
#     with torch.set_grad_enabled(optimizer is not None):
#         for step, batch in enumerate(data_loader):
#             batch = batch.to(DEVICE)
#             # get target solutions in list format
#             solInd = batch.nsols
#             if loss_function in ['infoNCEloss', 'hybrid']: 
#                 negInd = batch.nnegsamples
#             target_sols = []
#             target_vals = []
#             target_negs = []
#             instances = []
#             solEndInd = 0
#             valEndInd = 0
#             negSampEndInd = 0

#             for i in range(solInd.shape[0]):#for in batch
#                 nvar = len(batch.varInds[i][0][0])
#                 solStartInd = solEndInd
#                 solEndInd = solInd[i] * nvar + solStartInd
#                 valStartInd = valEndInd
#                 valEndInd = valEndInd + solInd[i]
                
#                 sols = batch.solutions[solStartInd:solEndInd].reshape(-1, nvar)
#                 vals = batch.objVals[valStartInd:valEndInd]

#                 target_sols.append(sols)
#                 target_vals.append(vals)

#                 if loss_function in ['infoNCEloss', 'hybrid']: 
#                     negSampStartInd = negSampEndInd 
#                     negSampEndInd = negInd[i] * nvar + negSampStartInd
#                     negs = batch.negsamples[negSampStartInd:negSampEndInd].reshape(-1, nvar)
#                     target_negs.append(negs)
#                 else:
#                     target_negs.append(None)

#             # Compute the logits (i.e. pre-softmax activations) according to the policy on the concatenated graphs
#             batch.constraint_features[torch.isinf(batch.constraint_features)] = 10 #remove nan value
#             #predict the binary distribution, BD
#             BD = predict(
#                 batch.constraint_features,
#                 batch.edge_index,
#                 batch.edge_attr,
#                 batch.variable_features,
#             )
#             BD = BD.sigmoid()
    
#             # calculate weights
#             index_arrow = 0
#             embeddings = None

#             total_samples = 0
#             anchor_positive = []
#             positive_idx = []
#             anchor_negative = []
#             negative_idx = []

#             loss = 0
            
#             for ind,(sols,vals,negs) in enumerate(zip(target_sols,target_vals, target_negs)):  
#                 #compute weight
#                 n_vals = vals
#                 exp_weight = torch.exp(-n_vals/weight_norm)
#                 # print("min", min(exp_weight), "max", max(exp_weight))
#                 weight = exp_weight/exp_weight.sum()
                
#                 # get a integer mask
#                 varInds = batch.varInds[ind]
#                 varname_map=varInds[0][0]
#                 i_vars=varInds[1][0].long()

#                 #get integer variables
#                 sols = sols[:,varname_map][:,i_vars]

#                 # cross-entropy
#                 n_var = batch.ntvars[ind]
#                 pre_sols = BD[index_arrow:index_arrow + n_var].squeeze()[i_vars]
#                 index_arrow = index_arrow + n_var
#                 pos_loss = -(pre_sols+ 1e-8).log()[None,:]*(sols==1).float()
#                 neg_loss = -(1-pre_sols + 1e-8).log()[None,:]*(sols==0).float()
#                 sum_loss = pos_loss + neg_loss

#                 sample_loss = sum_loss
#                 sample_loss = sum_loss*weight[:,None]
#                 loss += sample_loss.sum()

#                 if loss_function in ['infoNCEloss', 'hybrid']: 
#                     negs = negs[:,varname_map][:,i_vars]

#                     positive_sample_weights = n_vals/weight_norm #w1
#                     cur_embeddings = torch.cat([pre_sols.reshape(1,-1), sols * positive_sample_weights[:,None], negs])
#                     # cur_embeddings = torch.cat([pre_sols.reshape(1,-1), sols, negs])
    
#                     anchor_positive = anchor_positive + [total_samples] * sols.shape[0]
#                     anchor_negative = anchor_negative + [total_samples] * negs.shape[0]
#                     positive_idx = positive_idx + list(range(total_samples + 1, total_samples + 1 + sols.shape[0]))
#                     negative_idx = negative_idx + list(range(total_samples + 1 + sols.shape[0], total_samples + 1 + sols.shape[0] + negs.shape[0]))
#                     total_samples += 1 + sols.shape[0] + negs.shape[0]
    
#                     if embeddings is None:
#                         embeddings = cur_embeddings
#                     else:
#                         embeddings = torch.cat([embeddings,cur_embeddings])

#             if loss_function in ['infoNCEloss', 'hybrid']: 
#                 triplets = (torch.tensor(anchor_positive).to(DEVICE), torch.tensor(positive_idx).to(DEVICE), torch.tensor(anchor_negative).to(DEVICE), torch.tensor(negative_idx).to(DEVICE))
#                 cl_loss = infoNCE_loss_function(embeddings, indices_tuple = triplets)

#             if loss_function in ['hybrid']:
#                 loss = loss + cl_loss
#             if loss_function in ['infoNCEloss']:
#                 loss = cl_loss
                
#             if optimizer is not None:
#                 writer.add_scalar('train_loss', loss, epoch)
#             else:
#                 writer.add_scalar('valid_loss', loss, epoch)

#             if optimizer is not None:
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
            
#             mean_loss += loss
#             n_samples_processed += len(batch)
#     mean_loss /= n_samples_processed
#     return mean_loss

# def train(predict, data_loader, optimizer=None,weight_norm=1, loss_function = "logloss"):
#     """
#     This function will process a whole epoch of training or validation, depending on whether an optimizer is provided.
#     """

#     if optimizer:
#         predict.train()
#     else:
#         predict.eval()
#     mean_loss = 0
#     mean_cl_loss = 0
#     mean_auc = 0
#     mean_auc0 = 0
#     mean_final_loss = 0
#     mean_acc = 0
#     n_samples_processed = 0
#     with torch.set_grad_enabled(optimizer is not None):
#         for step, batch in enumerate(data_loader):
#             batch = batch.to(DEVICE)
#             # get target solutions in list format
#             solInd = batch.nsols
#             if loss_function in ['infoNCEloss', 'hybrid']: 
#                 negInd = batch.nnegsamples
#             target_sols = []
#             target_vals = []
#             target_negs = []
#             instances = []
#             solEndInd = 0
#             valEndInd = 0
#             negSampEndInd = 0
#             #embed()
#             for i in range(solInd.shape[0]):#for in batch
#                 nvar = len(batch.varInds[i][0][0])
#                 solStartInd = solEndInd
#                 solEndInd = solInd[i] * nvar + solStartInd
#                 valStartInd = valEndInd
#                 valEndInd = valEndInd + solInd[i]
                
#                 sols = batch.solutions[solStartInd:solEndInd].reshape(-1, nvar)
#                 vals = batch.objVals[valStartInd:valEndInd]

#                 target_sols.append(sols)
#                 target_vals.append(vals)
                
#                 if loss_function in ['infoNCEloss', 'hybrid']:
#                     negSampStartInd = negSampEndInd 
#                     negSampEndInd = negInd[i] * nvar + negSampStartInd
#                     negs = batch.negsamples[negSampStartInd:negSampEndInd].reshape(-1, nvar)
#                     target_negs.append(negs)
#                 else:
#                     target_negs.append(None)


#             # Compute the logits (i.e. pre-softmax activations) according to the policy on the concatenated graphs
#             batch.constraint_features[torch.isinf(batch.constraint_features)] = 10 #remove nan value
#             #predict the binary distribution, BD
#             BD = predict(
#                 batch.constraint_features,
#                 batch.edge_index,
#                 batch.edge_attr,
#                 batch.variable_features,
#             )
#             BD = BD.sigmoid()
    
#             # compute loss
#             loss = 0
#             # calculate weights
#             index_arrow = 0
#             embeddings = None
#             # print("start calculate loss  :")
#             total_samples = 0
#             anchor_positive = []
#             positive_idx = []
#             anchor_negative = []
#             negative_idx = []
            
#             tmp_CL_loss_sum = 0
#             auc_loss = 0
#             auc_count = 0
#             auc_loss0 = 0
#             binary_acc = 0
#             for ind,(sols,vals,negs) in enumerate(zip(target_sols,target_vals, target_negs)):

#                 #compute weight
#                 n_vals = vals
#                 exp_weight = torch.exp(-n_vals/weight_norm)
#                 #print("min", min(exp_weight), "max", max(exp_weight))
#                 weight = exp_weight/exp_weight.sum()
                
#                 # get a binary mask
#                 varInds = batch.varInds[ind]
#                 varname_map=varInds[0][0]
#                 b_vars=varInds[1][0].long()

#                 #get binary variables
#                 sols = sols[:,varname_map][:,b_vars]

                

#                 # cross-entropy
#                 n_var = batch.ntvars[ind]
#                 pre_sols = BD[index_arrow:index_arrow + n_var].squeeze()[b_vars]
#                 index_arrow = index_arrow + n_var
#                 pos_loss = -(pre_sols+ 1e-8).log()[None,:]*(sols==1).float()
#                 neg_loss = -(1-pre_sols + 1e-8).log()[None,:]*(sols==0).float()
#                 sum_loss = pos_loss + neg_loss

#                 sample_loss = sum_loss*weight[:,None]
#                 loss += sample_loss.sum()

#                 #embed()
#                 #print(sols.shape)
#                 # metric = BinaryAccuracy(threshold = 0.5)
#                 if sols.shape[1] == 0: continue
#                 for l in range(sols.shape[0]):
#                     # #embed() 
#                     # #print(l)
#                     # metric.update(pre_sols, sols[l].int())
#                     # binary_acc += metric.compute().item() / sols.shape[0]
#                     # auc_loss += auroc(pre_sols, sols[l].int(), pos_label = 1).item() / sols.shape[0]
#                     # auc_loss0 += auroc(pre_sols, sols[l].int(), pos_label = 0).item() / sols.shape[0]
#                     y = sols[l].int().to(pre_sols.device)   # shape: [n_vars]
#                     p = pre_sols.float()                    # shape: [n_vars]
            
#                     # accuracy (threshold=0.5 by default)
#                     binary_acc += binary_accuracy(p, y).item() / sols.shape[0]
            
#                     # AUC is undefined if only one class present; skip those rows
#                     if (y.min() == y.max()):
#                         continue
            
#                     # AUC with positive = 1
#                     auc_pos = tm_auroc(p, y, task="binary").item()
#                     auc_loss += auc_pos / sols.shape[0]
            
#                     # AUC with positive = 0 -> flip probs (or flip labels)
#                     # mathematically similar to 1 - auc_pos (ties aside)
#                     auc_neg = tm_auroc(1.0 - p, y, task="binary").item()
#                     # alternatively: tm_auroc(p, 1 - y, task="binary")
#                     auc_loss0 += auc_neg / sols.shape[0]
#                 #auc_loss /= sols.shape[0]
#                 #auc_loss0 /= sols.shape[0]
#                 #binary_acc /= sols.shape[0]
#                 auc_count += 1

#                 #embed()

#                 #negs = torch.randint(2, negs.shape).to(DEVICE)
#                 if loss_function in ['infoNCEloss', 'hybrid']:
#                     negs = negs[:,varname_map][:,b_vars]
#                     #embed()
#                     #cur_embeddings = torch.cat([pre_sols.reshape(1,-1), sols, negs])
                    
#                     if args.weight == True:
#                         positive_sample_weights = n_vals/weight_norm #w1
#                         #positive_sample_weights = weight + 1 #w2 doesn't work
#                         cur_embeddings = torch.cat([pre_sols.reshape(1,-1), sols * positive_sample_weights[:,None], negs])
#                     else:
#                         cur_embeddings = torch.cat([pre_sols.reshape(1,-1), sols, negs])

#                     anchor_positive = anchor_positive + [total_samples] * sols.shape[0]
#                     anchor_negative = anchor_negative + [total_samples] * negs.shape[0]
#                     positive_idx = positive_idx + list(range(total_samples + 1, total_samples + 1 + sols.shape[0]))
#                     negative_idx = negative_idx + list(range(total_samples + 1 + sols.shape[0], total_samples + 1 + sols.shape[0] + negs.shape[0]))
#                     total_samples += 1 + sols.shape[0] + negs.shape[0]
#                     #embed();exit()
#                     '''
#                     tmp_anchor_positive = [0] * sols.shape[0]
#                     tmp_anchor_negative = [0] * negs.shape[0]
#                     tmp_positive_idx = list(range(1, 1 + sols.shape[0]))
#                     tmp_negative_idx = list(range(1 + sols.shape[0], 1 + sols.shape[0] + negs.shape[0]))
#                     tmp_triplets = (torch.tensor(tmp_anchor_positive).to(DEVICE), torch.tensor(tmp_positive_idx).to(DEVICE), torch.tensor(tmp_anchor_negative).to(DEVICE), torch.tensor(tmp_negative_idx).to(DEVICE))
#                     #embed()
#                     tmp_CL_loss = infoNCE_loss_function(cur_embeddings, indices_tuple = tmp_triplets)
#                     tmp_CL_loss_sum += tmp_CL_loss.item()
#                     '''
#                     #embed()

#                     if embeddings is None:
#                         embeddings = cur_embeddings
#                     else:
#                         embeddings = torch.cat([embeddings,cur_embeddings])
            
#             if loss_function in ['infoNCEloss', 'hybrid']:
#                 triplets = (torch.tensor(anchor_positive).to(DEVICE), torch.tensor(positive_idx).to(DEVICE), torch.tensor(anchor_negative).to(DEVICE), torch.tensor(negative_idx).to(DEVICE))
#                 #embed()
#                 CL_loss = infoNCE_loss_function(embeddings, indices_tuple = triplets)
#                 #embed()


#             if optimizer is not None:
#                 optimizer.zero_grad()
#                 #embed()
#                 if loss_function == "logloss":
#                     loss.backward()
#                 elif loss_function == "infoNCEloss":
#                     CL_loss.backward()
#                 elif loss_function == 'hybrid':
#                     final_loss = loss/50000. + CL_loss
#                     final_loss.backward()
#                 optimizer.step()
            
#             mean_auc += auc_loss 
#             mean_auc0 += auc_loss0
#             mean_acc += binary_acc

#             mean_loss += loss.item()
#             if loss_function in ['infoNCEloss', 'hybrid']:
#                 mean_cl_loss += CL_loss.item()

#             if loss_function == "logloss":
#                 mean_final_loss = mean_loss
#             elif loss_function == "infoNCEloss":
#                 mean_final_loss = mean_cl_loss
#             elif loss_function == 'hybrid':
#                 final_loss = loss/5000. + CL_loss
#                 mean_final_loss += final_loss
#             n_samples_processed += batch.num_graphs
#     mean_loss /= n_samples_processed
#     mean_auc /= n_samples_processed
#     mean_auc0 /= n_samples_processed
#     mean_cl_loss /= n_samples_processed
#     mean_final_loss /= n_samples_processed
#     mean_acc /= n_samples_processed

#     print("mean_auc", mean_auc, "mean_auc0", mean_auc0, "bin acc", mean_acc)

#     return mean_final_loss, mean_loss, mean_cl_loss, mean_auc

optimizer = torch.optim.Adam(PredictModel.parameters(), lr=LEARNING_RATE)
best_val_loss = 99999
weight_norm = 100000

# if not args.pretrainModel is None:
#     valid_loss = train(epoch, PredictModel, valid_loader, None)
#     print(f"Epoch {epoch} Valid loss: {valid_loss:0.3f}")
#     best_val_loss = valid_loss

    
# if args.freeze == 1:
#     print("freezing some layers")
#     for param in PredictModel.parameters():
#         param.requires_grad = False
#     for param in PredictModel.output_module.parameters():
#         param.requires_grad = True

for epoch in range(NB_EPOCHS):
    print(f"Start epoch {epoch}")
    begin=time.time()
    train_loss = train(epoch, PredictModel, train_loader, optimizer, weight_norm, loss_function)
    print(f"Epoch {epoch} Train loss: {train_loss:0.3f}")
    valid_loss = train(epoch, PredictModel, valid_loader, None, weight_norm, loss_function)
    print(f"Epoch {epoch} Valid loss: {valid_loss:0.3f}")
        
    if valid_loss<best_val_loss:
        best_val_loss = valid_loss
        torch.save(PredictModel.state_dict(),model_save_path+'model_best.pth')
    torch.save(PredictModel.state_dict(), model_save_path+'model_last.pth')
    st = f'@epoch{epoch}   Train loss:{train_loss}  Valid loss:{valid_loss} TIME:{time.time()-begin}\n'
    print(st,"\n===================")
    log_file.write(st.encode())
    log_file.flush()
writer.close()
print('done')

# for epoch in range(NB_EPOCHS):
#     print(f"Start epoch {epoch}")
#     begin=time.time()
#     main_train_loss, train_log_loss, train_cl_loss, train_auc = train(PredictModel, train_loader, optimizer,weight_norm, loss_function=loss_function)
#     print(f"Epoch {epoch} Train loss: {main_train_loss:0.3f} Train log loss: {train_log_loss:0.3f} CL loss: {train_cl_loss:0.3f} Train AUC: {train_auc:0.3f}")
#     main_valid_loss, valid_log_loss, valid_cl_loss, valid_auc = train(PredictModel, valid_loader, None,weight_norm, loss_function=loss_function)
#     print(f"Epoch {epoch} Valid loss: {main_valid_loss:0.3f} Valid log loss: {valid_log_loss:0.3f} CL loss: {valid_cl_loss:0.3f}  Valid AUC: {valid_auc:0.3f}")
#     #main_valid_loss= valid_cl_loss if loss_function == "infoNCEloss" else valid_loss
        
#     if main_valid_loss<best_val_loss:
#         best_val_loss = main_valid_loss
#         torch.save(PredictModel.state_dict(),model_save_path+'model_best.pth')
#     torch.save(PredictModel.state_dict(), model_save_path+'model_last.pth')
#     st = f'@epoch{epoch}   Train log loss:{train_log_loss} Train CL loss:{train_cl_loss}  Valid log loss:{valid_log_loss} Valid CL loss:{valid_cl_loss}    TIME:{time.time()-begin}\n'
#     print(st,"\n===================")
#     log_file.write(st.encode())
#     log_file.flush()
# print('done')


