import torch
from torch.optim import Optimizer
from enum import IntEnum
from functools import reduce
import numpy as np

class OptimizerMethod(IntEnum):
    QP = 0
    CAYLEY = 2
    PG = 3
    AM2 = 5
    QP2 = 7

class ProjectedGradient(Optimizer):
    def __init__(self, params, lr=1.0, beta=0.5):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if beta <= 0.0 or beta >= 1.0:
            raise ValueError("Invalid beta: {}".format(lr))
        defaults = dict(lr=lr,beta=beta)
        super(ProjectedGradient, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("ProjectedGradient doesn't support per-parameter options "
                             "(parameter groups)")
        if len(self.param_groups[0]['params']) != 1:
            raise ValueError("ProjectedGradient doesn't support multiple parameters")

        _param = self.param_groups[0]['params'][0]
        assert _param.shape[0] < _param.shape[1] # it's s x N


    @torch.no_grad()
    def step(self, closure):
        """Performs a single optimization step.

        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """
        group = self.param_groups[0]
        p = group['params'][0].t() #use H as N x s
        N, s = p.shape
        g = group['params'][0].grad.t() #use H as N x s
        lr = group['lr']
        beta = group['beta']
        loss = float(closure(p.t()))

        # Backtracking line-search
        p1 = None
        u, s, v = torch.empty((s,s), device=p.device), torch.empty((s,), device=p.device), torch.empty((N,s), device=p.device)
        i = 0
        while i < 1000:
            try:
                p1 = self.st_project(p - lr * g, (u, s, v))
            except:
                break
            loss1 = float(closure(p1.t()))
            condition = loss1 <= loss + torch.trace(torch.mm(g.t(), (p1-p))) + torch.trace(torch.mm(p1-p, (p1-p).t())) / (2*lr)
            if condition:
                break
            else:
                lr *= beta
            i += 1
        group['params'][0][:] = p1.t()[:]

        return loss, lr

    @staticmethod
    def st_project(x, out=None):
        u, s, v = torch.svd(x, out=out)
        return torch.mm(u, v.t())

#Modified from PyTorch's LBFGS optimizer
#LICENSE https://raw.githubusercontent.com/pytorch/pytorch/master/LICENSE
class LBFGS(Optimizer):
    """Implements L-BFGS algorithm, from
    https://pytorch.org/docs/stable/_modules/torch/optim/lbfgs.html#LBFGS
    heavily inspired by `minFunc
    <https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html>`.

    .. warning::
        This optimizer doesn't support per-parameter options and parameter
        groups (there can be only one).

    .. warning::
        Right now all parameters have to be on a single device. This will be
        improved in the future.

    .. note::
        This is a very memory intensive optimizer (it requires additional
        ``param_bytes * (history_size + 1)`` bytes). If it doesn't fit in memory
        try reducing the history size, or use a different algorithm.

    Arguments:
        lr (float): learning rate (default: 1)
        max_iter (int): maximal number of iterations per optimization step
            (default: 20)
        max_eval (int): maximal number of function evaluations per optimization
            step (default: max_iter * 1.25).
        tolerance_grad (float): termination tolerance on first order optimality
            (default: 1e-5).
        tolerance_change (float): termination tolerance on function
            value/parameter changes (default: 1e-9).
        history_size (int): update history size (default: 100).
        line_search_fn (str): either 'strong_wolfe' or None (default: None).
    """

    def __init__(self,
                 params,
                 lr=1,
                 max_iter=20,
                 max_eval=None,
                 tolerance_grad=1e-7,
                 tolerance_change=1e-9,
                 history_size=100,
                 line_search_fn=None):
        if max_eval is None:
            max_eval = max_iter * 5 // 4
        defaults = dict(
            lr=lr,
            max_iter=max_iter,
            max_eval=max_eval,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            history_size=history_size,
            line_search_fn=line_search_fn)
        super(LBFGS, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("LBFGS doesn't support per-parameter options "
                             "(parameter groups)")

        self._params = self.param_groups[0]['params']
        self._numel_cache = None

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.flatten()
            views.append(view)
        return torch.cat(views, 0)

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.data.add_(step_size, update[offset:offset + numel].view_as(p.data))
            offset += numel
        assert offset == self._numel()

    def _clone_param(self):
        return [p.clone() for p in self._params]

    def _set_param(self, params_data):
        for p, pdata in zip(self._params, params_data):
            p.data.copy_(pdata)

    def _directional_evaluate(self, closure, x, t, d):
        self._add_grad(t, d)
        self.zero_grad()
        loss = closure()
        loss.backward()
        flat_grad = self._gather_flat_grad()
        self._set_param(x)
        return float(loss), flat_grad

    def step(self, closure):
        """Performs a single optimization step.

        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """
        assert len(self.param_groups) == 1

        group = self.param_groups[0]
        lr = group['lr']
        max_iter = group['max_iter']
        max_eval = group['max_eval']
        tolerance_grad = group['tolerance_grad']
        tolerance_change = group['tolerance_change']
        line_search_fn = group['line_search_fn']
        history_size = group['history_size']

        # NOTE: LBFGS has only global state, but we register it as state for
        # the first param, because this helps with casting in load_state_dict
        state = self.state[self._params[0]]
        state.setdefault('func_evals', 0)
        state.setdefault('n_iter', 0)

        # evaluate initial f(x) and df/dx
        orig_loss = closure()
        loss = float(orig_loss)
        current_evals = 1
        state['func_evals'] += 1

        flat_grad = self._gather_flat_grad()
        opt_cond = flat_grad.abs().max() <= tolerance_grad

        # optimal condition
        if opt_cond:
            return orig_loss

        # tensors cached in state (for tracing)
        d = state.get('d')
        t = state.get('t')
        old_dirs = state.get('old_dirs')
        old_stps = state.get('old_stps')
        ro = state.get('ro')
        H_diag = state.get('H_diag')
        prev_flat_grad = state.get('prev_flat_grad')
        prev_loss = state.get('prev_loss')

        n_iter = 0
        # optimize for a max of max_iter iterations
        while n_iter < max_iter:
            # keep track of nb of iterations
            n_iter += 1
            state['n_iter'] += 1

            ############################################################
            # compute gradient descent direction
            ############################################################
            if state['n_iter'] == 1:
                d = flat_grad.neg()
                old_dirs = []
                old_stps = []
                ro = []
                H_diag = 1
            else:
                # do lbfgs update (update memory)
                y = flat_grad.sub(prev_flat_grad)
                s = d.mul(t)
                ys = y.dot(s)  # y*s
                if ys > 1e-10:
                    # updating memory
                    if len(old_dirs) == history_size:
                        # shift history by one (limited-memory)
                        old_dirs.pop(0)
                        old_stps.pop(0)
                        ro.pop(0)

                    # store new direction/step
                    old_dirs.append(y)
                    old_stps.append(s)
                    ro.append(1. / ys)

                    # update scale of initial Hessian approximation
                    H_diag = ys / y.dot(y)  # (y*y)

                # compute the approximate (L-BFGS) inverse Hessian
                # multiplied by the gradient
                num_old = len(old_dirs)

                if 'al' not in state:
                    state['al'] = [None] * history_size
                al = state['al']

                # iteration in L-BFGS loop collapsed to use just one buffer
                q = flat_grad.neg()
                for i in range(num_old - 1, -1, -1):
                    al[i] = old_stps[i].dot(q) * ro[i]
                    q.add_(-al[i], old_dirs[i])

                # multiply by initial Hessian
                # r/d is the final direction
                d = r = torch.mul(q, H_diag)
                for i in range(num_old):
                    be_i = old_dirs[i].dot(r) * ro[i]
                    r.add_(al[i] - be_i, old_stps[i])

            if prev_flat_grad is None:
                prev_flat_grad = flat_grad.clone()
            else:
                prev_flat_grad.copy_(flat_grad)
            prev_loss = loss

            ############################################################
            # compute step length
            ############################################################
            # reset initial guess for step size
            if state['n_iter'] == 1:
                t = min(1., 1. / flat_grad.abs().sum()) * lr
            else:
                t = lr

            # directional derivative
            gtd = flat_grad.dot(d)  # g * d

            # directional derivative is below tolerance
            if gtd > -tolerance_change:
                break

            # optional line search: user function
            ls_func_evals = 0
            if line_search_fn is not None:
                # perform line search, using user function
                if line_search_fn != "wolfe":
                    raise RuntimeError("only 'wolfe' is supported")
                else:
                    x_init = self._clone_param()

                    def obj_func(x, t, d):
                        return self._directional_evaluate(closure, x, t, d)

                    t = self._my_wolfe(obj_func, x_init, d, loss, flat_grad)
                self._add_grad(t, d)
                loss = float(closure())
                flat_grad = self._gather_flat_grad()
                ls_func_evals = 1
                opt_cond = flat_grad.abs().max() <= tolerance_grad
            else:
                # no line search, simply move with fixed-step
                self._add_grad(t, d)
                if n_iter != max_iter:
                    # re-evaluate function only if not in last iteration
                    # the reason we do this: in a stochastic setting,
                    # no use to re-evaluate that function here
                    loss = float(closure())
                    flat_grad = self._gather_flat_grad()
                    opt_cond = flat_grad.abs().max() <= tolerance_grad
                    ls_func_evals = 1

            # update func eval
            current_evals += ls_func_evals
            state['func_evals'] += ls_func_evals

            ############################################################
            # check conditions
            ############################################################
            if n_iter == max_iter:
                break

            if current_evals >= max_eval:
                break

            # optimal condition
            if opt_cond:
                break

            # lack of progress
            if d.mul(t).abs().max() <= tolerance_change:
                break

            if abs(loss - prev_loss) < tolerance_change:
                break

        state['d'] = d
        state['t'] = t
        state['old_dirs'] = old_dirs
        state['old_stps'] = old_stps
        state['ro'] = ro
        state['H_diag'] = H_diag
        state['prev_flat_grad'] = prev_flat_grad
        state['prev_loss'] = prev_loss

        return orig_loss, t

    @staticmethod
    def _my_wolfe(obj_func, x, d, loss, g, sigma=1e-4, eta=0.9):
        assert 0 < sigma < eta < 1
        alpha, alpha_min, alpha_max = 1., 0., float('inf')
        max_iter = 10
        gd = torch.dot(g.flatten(), d)

        armijo = False
        curvature = False

        i = 0
        while (not armijo or not curvature) and i < max_iter:
            # Armijo condition
            loss_new, g_new = obj_func(x, alpha, d)
            while loss_new > loss + sigma * alpha * gd:
                alpha_max = alpha
                alpha = 0.5 * (alpha_min + alpha_max)
                loss_new, g_new = obj_func(x, alpha, d)
            armijo = True

            # Curvature condition
            if torch.dot(g_new, d) < eta * gd:
                alpha_min = alpha
                alpha = 2 * alpha_min if np.isinf(alpha_max) else 0.5 * (alpha_min + alpha_max)
                curvature = False
            else:
                curvature = True
            i += 1
        return max(alpha, 2e-4)