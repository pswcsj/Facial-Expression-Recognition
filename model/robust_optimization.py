import torch


class RobustOptimizer(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, eps=0.05, **kwargs):
        defaults = dict(rho=eps, **kwargs)
        super(RobustOptimizer, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        param = []
        param_vector = []
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
                param.append(p)
                param_vector.append(p.grad)
        print(f'first_step|grad_nrom:{grad_norm}|param:{param[:5]}|param_grad:{param_vector}')
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        param = []
        param_vector = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])
                param.append(p)
                param_vector.append(p.grad)

        self.base_optimizer.step()
        print(f'first_step|grad_nrom:{grad_norm}|param:{param[:5]}|param_grad:{param_vector}')
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        closure = torch.enable_grad()(closure)

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm
