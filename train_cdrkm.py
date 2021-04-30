import json
from opt_algorithms import st_optimizers
import logging
import argparse
import time
from datetime import datetime
import numpy as np
import pandas
import utils
from cayley_adam import stiefel_optimizer
from cdrkm_eval import eval_training
from utils import load_dataset
from kernels import kernel_factory
from cdrkm_model import CDRKM, orto
from definitions import *
from opt_algorithms.st_optimizers import OptimizerMethod
import sys


def train_cayleyadam(model, param_h, param_k, xtrain, maxiterations, maxinnertime=60, lr=0.1, epsilon=5e-5) -> pandas.DataFrame:
    param = param_h[0]
    prev_param = torch.clone(param)

    dict_m = {'params': param_h, 'lr': lr, 'stiefel': True}
    dict_nm = {'params': param_k, 'lr': lr, 'stiefel': False}
    optimizer = stiefel_optimizer.AdamG([dict_m, dict_nm])
    t = 1
    terminating_condition = True
    elapsed_minutes = 0
    start_timestamp = time.time()
    train_table = pandas.DataFrame()

    while terminating_condition and t < maxiterations and elapsed_minutes < maxinnertime:  # inner loop
        loss, _, ortos, interorto2, _, _ = model(xtrain)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        rel_delta = float(torch.max(torch.abs(param - prev_param))) / lr
        terminating_condition = rel_delta > epsilon
        prev_param = torch.clone(param)
        grad_q = max([float(torch.max(torch.abs(h.grad))) for h in param_h])
        log_dict = {'outer_i': 1, 'inner_i': t, 'i': t, 'j': float(loss.detach().cpu()),
                    'grad': float(grad_q), 'orto': float(interorto2.detach().cpu()),
                    'X': None if t % 100 != 0 and t != 1 else [torch.clone(param).detach().cpu().numpy()]}
        train_table = train_table.append(pandas.DataFrame(log_dict, index=[t]))
        logging.info((train_table.iloc[len(train_table) - 1:len(train_table)]).
                     to_string(header=(t == 1), index=False, justify='right', col_space=15,
                               float_format=utils.float_format, formatters={'mu': lambda x: "%.2f" % x},
                               columns=train_table.columns.drop(['X', 'outer_i', 'inner_i'])))
        t += 1
        elapsed_minutes = (time.time() - start_timestamp) / 60

    # Add last X to train_table
    train_table.at[t - 1, 'X'] = [torch.clone(param).detach().cpu().numpy()]

    return train_table


def train_projectedgradient(model, param_h, xtrain, maxiterations, maxinnertime=60, lr=0.1, epsilon=5e-5) -> pandas.DataFrame:
    param = param_h[0]
    param_temp = torch.clone(param)
    prev_param = torch.clone(param)
    optimizer = st_optimizers.ProjectedGradient(param_h, lr=lr)
    t = 1
    alpha = [0.0]  # or gamma, it is the step size
    terminating_condition = True
    elapsed_minutes = 0
    start_timestamp = time.time()
    train_table = pandas.DataFrame()

    while terminating_condition and t < maxiterations and elapsed_minutes < maxinnertime:  # inner loop
        def closure(x):
            # Save current x
            param_temp[:] = param.data[:]
            # Set x to module
            param.data[:] = x
            # Compute loss
            loss, _, ortos, interorto2, _, _ = model(xtrain)
            # Restore x
            param.data[:] = param_temp[:]
            return loss

        loss, _, ortos, interorto2, _, _ = model(xtrain)
        optimizer.zero_grad()
        loss.backward()
        alpha.append(optimizer.step(closure)[1])

        terminating_condition = float(torch.max(torch.abs(param - prev_param))) / alpha[-1] > epsilon
        prev_param = torch.clone(param)
        grad_q = max([float(torch.max(torch.abs(h.grad))) for h in param_h])
        log_dict = {'outer_i': 1, 'inner_i': t, 'i': t, 'j': float(loss.detach().cpu()),
                    'grad': grad_q, 'orto': float(interorto2.detach().cpu()), 'alpha': alpha[t],
                    'X': None if t % 100 != 0 and t != 1 else [torch.clone(param).detach().cpu().numpy()]}
        train_table = train_table.append(pandas.DataFrame(log_dict, index=[t]))
        logging.info((train_table.iloc[len(train_table) - 1:len(train_table)]).
                     to_string(header=(t == 1), index=False, justify='right', col_space=15,
                               float_format=utils.float_format, formatters={'mu': lambda x: "%.2f" % x},
                               columns=train_table.columns.drop(['X', 'outer_i', 'inner_i'])))
        t += 1
        elapsed_minutes = (time.time() - start_timestamp) / 60

    # Add last X to train_table
    train_table.at[t - 1, 'X'] = [torch.clone(param).detach().cpu().numpy()]

    return train_table

def train_penalty(model, param_h, param_k, xtrain, maxouteriterations, maxiterations, maxinnertime=60, lr=0.1, tau_min=1e-3, p=4, beta=0.5, algo='lbfgs') -> pandas.DataFrame:
    tt = 1
    i = 1
    mu = [0.0, 1.0]
    tau = [0.0, 0.5]
    patience = np.inf
    train_table = pandas.DataFrame()
    while tt < maxouteriterations:  # and orto(param_h[0].t())[0] > 1e-10:  # outer loop
        if algo == 'adam':
            optimizer = torch.optim.Adam(param_h + param_k, lr=lr, weight_decay=0)
        elif algo == 'lbfgs':
            optimizer = st_optimizers.LBFGS(param_h, lr=lr, max_iter=1, line_search_fn='wolfe')
        t = 1
        grad_q = np.inf  # Initialize cost
        elapsed_minutes = 0
        start_timestamp = time.time()
        improved = True
        improved_thresh = lambda old, new, threshold: np.abs((old - new)) > threshold or True

        while grad_q > tau[tt] and t < maxiterations and elapsed_minutes < maxinnertime and improved:  # inner loop
            def closure():
                loss, _, _, interorto2, interorto, _ = model(xtrain)
                q = loss + 0.5 * mu[tt] * interorto2
                return q

            loss, _, _, interorto2, interorto, _ = model(xtrain)
            q = loss + 0.5 * mu[tt] * interorto2
            optimizer.zero_grad()
            q.backward()
            optimizer.step(closure)

            grad_q = sum([torch.norm(h.grad) for h in param_h])
            log_dict = {'outer_i': tt, 'inner_i': t, 'i': i, 'j': float(loss.detach().cpu()), 'q': float(q.detach().cpu()),
                        'grad_q': float(grad_q.detach().cpu()), 'orto': float(interorto2.detach().cpu()), 'mu': mu[tt]}
            train_table = train_table.append(pandas.DataFrame(log_dict, index=[0]))
            logging.info((train_table.iloc[len(train_table) - 1:len(train_table)]).to_string(header=(tt == 1 and t == 1), index=False, justify='right', col_space=15, float_format=utils.float_format, formatters={'mu': lambda x: "%.2f" % x}))
            if t > patience:
                improved = improved_thresh(train_table['j'].iloc[i - patience], float(loss.detach().cpu()), 1e-6)
            t += 1
            i += 1
            elapsed_minutes = (time.time() - start_timestamp) / 60

        # Update tau
        tau.append(tau[tt] * beta)
        if tau[tt + 1] < tau_min:
            tau[tt + 1] = tau_min
        # Update mu
        mu.append(p * mu[tt])

        tt += 1
    return train_table


def train_alagrange(model, param_h, xtrain, maxouteriterations, maxiterations, maxinnertime=60, lr=0.1, no_y=False, beta=0.2, min_epsilon=1e-6, delta=1e3, algo='lbfgs') -> pandas.DataFrame:
    param = param_h[0]
    tt = 1
    i = 1
    sigma_max = 1e7
    theta = 0.002
    epsilon = [0.0, 0.9]
    alpha = [0.0]  # or gamma, it is the step size
    Y = [0.0, 0.0 * torch.eye(*(param.shape[0], param.shape[0]), device=param.device, requires_grad=False)]
    E = [0.0, orto(param.t().detach(), full=True)[2]]
    S = [0.0, max(1e-4, min(1e4, 20 * max(1, abs(model(xtrain)[0].detach())) / max(1, 0.5 * orto(param)[0].detach()))) * torch.ones((param.shape[0], param.shape[0]), device=param.device, requires_grad=False)]
    train_table = pandas.DataFrame(dtype=object)
    while tt < maxouteriterations and orto(param.t())[0] > 1e-9:  # outer loop
        if algo == "adam":
            optimizer = torch.optim.Adam(param_h, lr=lr)
        elif algo == "lbfgs":
            optimizer = st_optimizers.LBFGS(param_h, lr=lr, max_iter=1, line_search_fn='wolfe')
        t = 1
        grad_q = np.inf  # Initialize cost
        elapsed_minutes = 0
        start_timestamp = time.time()
        terminating_condition = True

        while grad_q > epsilon[tt] and t < maxiterations and elapsed_minutes < maxinnertime and terminating_condition:  # inner loop
            def closure():
                loss, _, _, interorto2, interorto, _ = model(xtrain)
                o = orto(param.t(), full=True)[2]
                p = utils.dot_mm(Y[tt], o)
                phi = loss + 0.5 * utils.skewed_norm2(o, S[tt]) + p
                return phi, loss, interorto2, p

            phi, loss, interorto2, p = closure()
            optimizer.zero_grad()
            phi.backward()
            optimizer.step(lambda: closure()[0])
            alpha.append(lr)

            grad_q = max([float(torch.max(torch.abs(h.grad))) for h in param_h])
            log_dict = {'outer_i': tt, 'inner_i': t, 'i': i, 'j': float(loss.detach().cpu()), 'phi': float(phi.detach().cpu()),
                        'grad': grad_q, 'orto': float(interorto2.detach().cpu()),
                        '||S||': float(torch.norm(S[tt])), '||Y||': float(torch.norm(Y[tt])),
                        'X': None if t > 1 else [torch.clone(param).detach().cpu().numpy()],
                        'S': None if t > 1 else [torch.clone(S[tt]).cpu().numpy()],
                        'Y': None if t > 1 else [torch.clone(Y[tt]).cpu().numpy()],
                        'alpha': alpha[i],
                        }
            train_table = train_table.append(pandas.DataFrame(log_dict, index=[i]))
            logging.info((train_table.iloc[len(train_table) - 1:len(train_table)]).
                         to_string(header=(tt == 1 and t == 1), index=False, justify='right', col_space=15,
                                   float_format=utils.float_format, formatters={'mu': lambda x: "%.2f" % x},
                                   columns=train_table.columns.drop(['i', 'X', 'S', 'Y'])))
            t += 1
            i += 1
            elapsed_minutes = (time.time() - start_timestamp) / 60

        # Add last X to train_table
        train_table.at[i - 1, 'X'] = [torch.clone(param).detach().cpu().numpy()]
        # Update E
        o = orto(param.t(), full=True)[2].detach()
        E.append(o)
        # Update S
        cond = torch.abs(E[tt + 1]) > theta * torch.abs(E[tt])
        temp = delta / torch.max(torch.abs(E[tt + 1])) * torch.abs(E[tt + 1])
        temp[temp < 1.] = 1.
        temp = torch.mul(temp, S[tt])
        temp[temp > sigma_max] = sigma_max
        S.append(torch.clone(S[tt]))
        S[tt + 1][cond] = temp[cond]
        S[tt + 1] = S[tt + 1].detach()
        # Update Y
        temp = torch.mul(S[tt], o) if not no_y else 0.0
        Y.append((torch.clone(Y[tt]) + temp).detach())
        # Update epsilon
        epsilon1 = max(beta * epsilon[tt], min_epsilon)
        epsilon.append(epsilon1)

        tt += 1
    return train_table


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", "--Nd", type=int, help="number of training samples", default=50)
    parser.add_argument("-a", "--train_algo", help="training algorithm: (0) penalty, (5) augmented Lagrangian, (2) Cayley ADAM, (3) Projected Gradient, (7) quadratic penalty based on AM",
                        type=int, choices=[0, 5, 2, 3, 7], default=0)
    parser.add_argument("-s", "--s", nargs="+", default=[10, 5], type=int, help="number of components in each layer")
    parser.add_argument("-k", "--kernel", nargs="+", help="kernel of each layer: (0) RBF, (1) poly, (2) Laplace or (3) sigmoid",
                        type=int, choices=[0, 1, 2, 3], default=[0, 0])
    parser.add_argument("-kp", "--kernelparam", nargs="+", help="kernel parameter of each level: RBF or Laplace bandwidth or poly degree",
                        type=float, default=[50, 50])
    parser.add_argument("-lwi", "--layerwisein", help="layer-wise initialization", action="store_true")
    parser.add_argument("-d", "--dataset", type=str, help="name of the dataset", default="mnist2")
    parser.add_argument("-mi", "--maxiterations", type=int, help="maximum number of training inner iterations", default=10)
    parser.add_argument("-mit", "--maxinnertime", type=int, help="maximum number of minutes for inner loop", default=-1)
    parser.add_argument("-moi", "--maxouteriterations", type=int, help="maximum number of training outer iterations", default=5)
    parser.add_argument("-ok", "--optimizekernel", help="optimizes kernel parameters", action="store_true")
    parser.add_argument("-gamma", "--gamma", help="gamma for all levels", default=1.0, type=float)
    parser.add_argument("-epsilon", "--epsilon", help="epsilon for terminating condition of optimization", default=5e-5, type=float)
    parser.add_argument("-rs", "--seed", type=int, help="random seed", default=0)
    parser.add_argument("-lr", "--lr", help="learning rate", default=1, type=float)
    parser.add_argument("--tau_min", help="tau min for algorithm 0 or epsilon min for algorithm 7", default=1e-3, type=float)
    parser.add_argument("--p", help="p for algorithm 0 or delta for algorithm 7", default=4, type=float)
    parser.add_argument("--beta", help="beta for algorithm 0 and 7", default=0.5, type=float)
    parser.add_argument("-ia", "--inneralgorithm", type=str, help="inner training algorithm", default="lbfgs", choices=["lbfgs", "adam"])
    args = parser.parse_args()
    assert len(args.s) == len(args.kernel) == len(args.kernelparam)
    if args.maxiterations == -1:
        vars(args)['maxiterations'] = np.inf
    if args.maxinnertime == -1:
        vars(args)['maxinnertime'] = np.inf
    # ==================================================================================================================
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # ==================================================================================================================
    # Set up logging and directories
    label = None
    list_of_experiments = list(OUT_DIR.glob('offline*'))
    n = -1
    if len(list_of_experiments) != 0:
        n = max([k for k in map(lambda path: int(str(path.name)[len("offline"):]), list_of_experiments)])
    label = "offline%04d" % (n + 1)
    model_dir = OUT_DIR.joinpath(label)
    model_dir.mkdir()
    ct = time.strftime("%Y%m%d-%H%M")
    # noinspection PyArgumentList
    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        handlers=[logging.FileHandler(model_dir.joinpath('{}_Trained_cdrkm_algo{}_{}.log'.format(args.dataset, args.train_algo, ct))),
                                  logging.StreamHandler(sys.stdout)])
    # ==================================================================================================================
    # Load Training Data
    _, xtrain, _ = load_dataset(args.dataset, args.Nd, [], seed=args.seed)
    # ==================================================================================================================
    # Define model
    kernels = list(map(lambda x: kernel_factory(*x), zip(args.kernel, args.kernelparam)))
    cdrkm = CDRKM(kernels, args.s, xtrain.shape[0], gamma=args.gamma, layerwisein=args.layerwisein, xtrain=xtrain,
                  ortoin=False).to(device)
    logging.info(cdrkm)
    logging.info(args)
    # ==================================================================================================================
    # Divide differentiable parameters in 2 groups: 1. Manifold parameters 2. Kernel parameters
    param_h, param_k = [param[1] for param in cdrkm.named_parameters() if "h" in param[0]], \
                       [param[1] for param in cdrkm.named_parameters() if "kernel" in param[0]]
    for param in param_h:
        param.requires_grad = True
    for param in param_k:
        if args.optimizekernel:
            param.requires_grad = True
        else:
            param.requires_grad = False
    # Train =========================================================================================================
    start = datetime.now()
    if args.train_algo == OptimizerMethod.QP:
        train_table = train_penalty(cdrkm, param_h, param_k if args.optimizekernel else [], xtrain,
                                    maxiterations=args.maxiterations + 1,
                                    maxinnertime=args.maxinnertime,
                                    maxouteriterations=args.maxouteriterations + 1,
                                    lr=args.lr,
                                    tau_min=args.tau_min,
                                    p=args.p,
                                    beta=args.beta,
                                    algo=args.inneralgorithm)
    elif args.train_algo == OptimizerMethod.CAYLEY:
        train_table = train_cayleyadam(cdrkm, param_h, param_k if args.optimizekernel else [],
                                       xtrain, args.maxiterations + 1, maxinnertime=args.maxinnertime, lr=args.lr, epsilon=args.epsilon)
    elif args.train_algo == OptimizerMethod.PG:
        train_table = train_projectedgradient(cdrkm, param_h,
                                              xtrain, args.maxiterations + 1, maxinnertime=args.maxinnertime, lr=args.lr, epsilon=args.epsilon)
    elif args.train_algo == OptimizerMethod.AM2:
        train_table = train_alagrange(cdrkm, param_h,
                                      xtrain, maxiterations=args.maxiterations + 1,
                                      maxinnertime=args.maxinnertime,
                                      maxouteriterations=args.maxouteriterations + 1,
                                      lr=args.lr)
    elif args.train_algo == OptimizerMethod.QP2:
        train_table = train_alagrange(cdrkm, param_h,
                                      xtrain, maxiterations=args.maxiterations + 1,
                                      maxinnertime=args.maxinnertime,
                                      maxouteriterations=args.maxouteriterations + 1,
                                      lr=args.lr,
                                      no_y=True,
                                      beta=args.beta,
                                      min_epsilon=args.tau_min,
                                      delta=args.p,
                                      algo=args.inneralgorithm)

    time.sleep(1)
    elapsed_time = datetime.now() - start
    logging.info("\nTraining complete in: " + str(elapsed_time))

    # Save Model ======================================================================================
    torch.save({'cdrkm_state_dict': cdrkm.state_dict(),
                'args': args,
                'train_table': train_table.to_json(orient='records'),
                'train_table_datatypes': json.dumps(train_table.dtypes.apply(lambda x: x.name).to_dict()),
                'elapsed_time': elapsed_time,
                'label': label[:-3]
                }, str(model_dir.joinpath("model.pt")))
    logging.info('Saved Label: %s' % label)

    # Evaluate Model ======================================================================================
    eval_dict = eval_training(model_dir.joinpath("model.pt"))
    eval_dict.pop('final_X')
    logging.info("\n".join("{}\t{}".format(k, str(v)) for k, v in eval_dict.items()))
