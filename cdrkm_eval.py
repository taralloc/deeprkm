import json
from definitions import *
from cdrkm_model import CDRKM
from kernels import kernel_factory
import argparse
from pathlib import Path
import pandas
import torch
from utils import save_altairplot, load_dataset, merge_two_dicts
import numpy as np

def eval_training(filepath: Path):
    sd_mdl = torch.load(filepath, map_location=torch.device('cpu'))
    args = sd_mdl['args']
    _, xtrain, _ = load_dataset(args.dataset, args.Nd, [], seed=args.seed)
    kernels = [kernel_factory(*x) for x in zip(args.kernel, args.kernelparam)]
    model = CDRKM(kernels, args.s, xtrain.shape[0], gamma=args.gamma, layerwisein=args.layerwisein, xtrain=xtrain,
                  ortoin=(args.train_algo == 2)).to(device)
    model.load_state_dict(sd_mdl['cdrkm_state_dict'])

    train_table_datatypes = sd_mdl['train_table_datatypes']
    train_table = pandas.read_json(sd_mdl['train_table'], orient='records', dtype=json.loads(train_table_datatypes))
    final_i = train_table['i'].iat[-1]  # total number of iterations
    final_j = train_table['j'].iat[-1] # final objective value
    final_orto = train_table['orto'].iat[-1]  # final feasibility
    final_outer_i = train_table['outer_i'].iat[-1] #final number of outer iterations

    final_X = np.array(train_table['X'].iat[-1]) if 'X' in train_table else model.h # final H
    if 'X' in train_table:
        model.h[:] = torch.tensor(final_X)
    model.h.requires_grad_(True)
    h = model.h
    loss = model(xtrain)[0]
    loss.backward()
    u, s, v = torch.svd(h.cpu() - h.grad.cpu())
    out = torch.norm(h - torch.mm(u.to(device), v.to(device).t()))
    final_XUVT = float(out)  # final ||X-U*V’||

    elapsed_seconds = sd_mdl.get('elapsed_time', -1)
    if type(elapsed_seconds) != int:
        elapsed_seconds = elapsed_seconds.seconds

    eval_dict = {'final_j': final_j,
                 'final_orto': final_orto,
                 'final_X': final_X,
                 'final_XUVT': final_XUVT,
                 'final_i': final_i,
                 'final_outer_i': final_outer_i,
                 'elapsed_seconds': elapsed_seconds}
    return eval_dict

def distance_matrix(x, y):
    if type(x[0]) == float:
        diffs = [abs(a-b) for a in x for b in y]
    else:
        diffs = [torch.pow(torch.norm(a - b), 1) for a in x for b in y]
    diffs = torch.tensor(diffs).view(len(x), len(y)).numpy()
    return diffs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', type=str, nargs="+")
    args = parser.parse_args()
    filenames = [OUT_DIR.joinpath('%s/model.pt' % filename) for filename in args.labels]

    sd_mdls = [torch.load(str(filename), map_location=torch.device('cpu')) for filename in filenames]

    # Evaluate each model
    df = []
    for sd_mdl, filename in zip(sd_mdls, filenames):
        label = sd_mdl.get('label', filename.parent.stem)
        print(f"Evaluating {label}")
        print(sd_mdl['args'])
        eval_dict = eval_training(filename, plotting=False)
        df.append(merge_two_dicts(eval_dict, vars(sd_mdl['args'])))
        if eval_dict['plot'] is not None:
            save_altairplot(eval_dict['plot'], filename.parent.joinpath("train_table_plot_%s.pdf" % label))
        print("\n".join("{}\t{}".format(k, str(v)) for k, v in eval_dict.items() if k not in ["final_X", "plot"]))
        print("-------------------------")
    df = pandas.DataFrame(df)

    # Compare solutions
    algos = df['train_algo'].unique()
    algos_names = {2: 'Cayley ADAM', 3: 'Projected Gradient', 5: 'Augmented Lagrangian'}
    algos_names = [algos_names[algo] for algo in algos]
    hs_by_algo = {algo: [torch.from_numpy(h[0]) if len(h[0].shape) == 2 else torch.from_numpy(h) for h in df[(df['train_algo'] == algo) & (df['seed'] == 0)]['final_X'].to_list()]
                  for algo in algos}
    cost_by_algo = {algo: [cost for cost in df[(df['train_algo'] == algo) & (df['seed'] == 0)]['final_j'].to_list()]
                  for algo in algos}
    hs_diffs = [distance_matrix(hs_by_algo[algo1], hs_by_algo[algo2]) for algo1 in algos for algo2 in algos]
    cost_diffs = [distance_matrix(cost_by_algo[algo1], cost_by_algo[algo2]) for algo1 in algos for algo2 in algos]
    mean_hs_diffs = np.array([np.mean(d) for d in hs_diffs]).reshape((len(algos), len(algos)))
    mean_cost_diffs = np.array([np.mean(d) for d in cost_diffs]).reshape((len(algos), len(algos)))
    std_hs_diffs = np.array([np.std(d) for d in hs_diffs]).reshape((len(algos), len(algos)))
    std_cost_diffs = np.array([np.std(d) for d in cost_diffs]).reshape((len(algos), len(algos)))
    mean_hs_crosstable = pandas.DataFrame(mean_hs_diffs, columns=algos_names, index=algos_names)
    mean_cost_crosstable = pandas.DataFrame(mean_cost_diffs, columns=algos_names, index=algos_names)
    std_hs_crosstable = pandas.DataFrame(std_hs_diffs, columns=algos_names, index=algos_names)
    std_cost_crosstable = pandas.DataFrame(std_cost_diffs, columns=algos_names, index=algos_names)
    hs_crosstable = mean_hs_crosstable.applymap(lambda x: "{:.3f}".format(x)) + ' ± ' + std_hs_crosstable.applymap(lambda x: "{:.3f}".format(x))
    cost_crosstable = mean_cost_crosstable.applymap(lambda x: "{:.5f}".format(x)) + ' ± ' + std_cost_crosstable.applymap(lambda x: "{:.5f}".format(x))
    with pandas.option_context('display.max_rows', None, 'display.max_columns', None, 'expand_frame_repr', False):
        print("Distance matrix with mean and std is")
        print(hs_crosstable)
        print("Distance matrix of cost with mean and std is")
        print(cost_crosstable)
