"""Software unit tests. Uses SYNTHETIC data for checks only; no result in the paper comes from here."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.optimize import check_grad
from src.fusion import ReliabilityGate
rng = np.random.default_rng(0); n = 500
pa, pb = rng.uniform(.01, .99, n), rng.uniform(.01, .99, n); R = rng.normal(size=(n, 4)); y = rng.integers(0, 2, n).astype(float)
g = ReliabilityGate().fit(pa, pb, R, y)
D = g._design(R)
def f(w):  # recreate objective
    a = 1/(1+np.exp(-(D@w))); p = np.clip(a*pa+(1-a)*pb, 1e-6, 1-1e-6)
    return -(y*np.log(p)+(1-y)*np.log(1-p)).mean() + g.l2*np.sum(w[1:]**2)
def gr(w):
    a = 1/(1+np.exp(-(D@w))); p = np.clip(a*pa+(1-a)*pb, 1e-6, 1-1e-6)
    gp = y/p-(1-y)/(1-p); return -(D.T@(gp*(pa-pb)*a*(1-a)))/n + np.r_[0, 2*g.l2*w[1:]]
err = check_grad(f, gr, rng.normal(size=D.shape[1])); print("grad check err", err); assert err < 1e-5
# gate must recover a known routing rule: y follows pa when R0>0 else pb
yy = np.where(R[:, 0] > 0, rng.uniform(size=n) < pa, rng.uniform(size=n) < pb).astype(float)
g2 = ReliabilityGate(l2=1e-4).fit(pa, pb, R, yy); a = g2.alpha(R)
print("alpha R0>0 %.2f  R0<0 %.2f" % (a[R[:, 0] > 0].mean(), a[R[:, 0] < 0].mean())); assert a[R[:, 0] > 0].mean() > a[R[:, 0] < 0].mean()
print("ALL UNIT TESTS PASSED")
