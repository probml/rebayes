# https://pastebin.com/VzJTcsuv
# https://github.com/pytorch/pytorch/issues/50688

import torch
 
def bench_triton(f, name=None, iters=1000, warmup=5, display=True, profile=False):
    import time
    from triton.testing import do_bench
 
    for _ in range(warmup):
        f()
    if profile:
        with torch.profiler.profile() as prof:
            f()
        prof.export_chrome_trace(f"{name if name is not None else 'trace'}.json")
 
 
    us_per_iter = do_bench(lambda: f())[0]*iters
 
    if name is None:
        res = us_per_iter
    else:
        res= f"{name}: {us_per_iter:.3f}us"
 
    if display:
        print(res)
    return res

def bench(f, name='', iters=100, warmup=5, display=True, profile=False):
    import time
    for i in range(warmup):
        f()
    start = time.time()
    for i in range(iters):
        f()
    end = time.time()
    res = end-start
    print('time for {name}={res:0.2f}'.format(name=name, res=res))
    return res

def scan(alphas, betas, h):
    """Loop over a simple RNN.
 
    Args:
        alphas (torch.tensor): shape [B, T, C]
        betas (torch.tensor): shape [B, T, C]
        h (torch.tensor): shape [B, C]
    """
    T = betas.shape[-2]
    hs = torch.zeros_like(betas)
    exp_alphas = torch.exp(alphas)
    for t in range(T):
        h = exp_alphas[:, t] * h + betas[:, t]
        hs[:, t] = h
    return hs
 
T = 128
B = 64
C = 256
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

 
h = torch.randn(B, C, device=device)
alphas = torch.randn(B, T, C, device=device)
betas = torch.randn(B, T, C, device=device)
 
bench(lambda: scan(alphas, betas, h), name='vanilla')
opt_scan = torch.compile(scan)
bench(lambda: opt_scan(alphas, betas, h), name='torch.compile')
scan_jit = torch.jit.trace(scan, (alphas, betas, h))
bench(lambda: scan_jit(alphas, betas, h), name='torch.jit.trace')
