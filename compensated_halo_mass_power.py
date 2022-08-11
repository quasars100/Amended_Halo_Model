import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import sici
from scipy.optimize import curve_fit
from lenstools.simulations import Gadget2Snapshot
import emcee
from astropy import units
from pylab import *


def read_data_by_column(name, lines):
    return np.loadtxt(name, skiprows=lines, unpack=True)

def read_data(fname):
    f1 = open(fname, 'r')
    info = f1.readlines()
    f1.seek(0)
    f1.close()
    return info

boxlength = 120.0
volume = boxlength**3
om_8 = 0.9
H = 72.0
om_m = 0.29
om_l = 1.0-om_m
rho_m = om_m*2.77536e11 
rho_crit = (3.0*H**2)/(8.0*np.pi*3.0273e-3)
dt_factor = (5.0*om_m/2.0)*(om_m**(4.0/7.0) - om_l + (1+0.5*om_m)*(1+om_l/H))**(-1)

m = np.logspace(10.0, 15.5, 500)
r = (3.0*m/(4.0*np.pi*rho_m))**(1.0/3.0)


def halo_properties(pos, r, x, y, z, point_mass, pts):
    n = 0
    bins = np.linspace(0.0, r, pts)
    bins = np.append([0.0], bins)
    bin_f = []
    error = []
    density = []
    for i in range(1, len(bins)):
        all_N = 0
        for point in pos:
            rc = np.sqrt((x-point[0])**2 + (y-point[1])**2 + (z-point[2])**2)
            if rc <= bins[i]:
                all_N += 1
        n_curr = all_N - n
        n = all_N
        bin_f.append((bins[i]+bins[i-1])/2.0)
        print(bin_f[i-1])
        print(n_curr, all_N, n)
        V_curr = (4./3.)*np.pi*((bins[i]**3)-(bins[i-1]**3))
        density.append(n_curr/V_curr)
        error.append(particle_mass*np.sqrt(n_curr)/V_curr)
    density = np.array(density)
    error = np.array(error)
    bin_f = np.array(bin_f)/r
    return bin_f, density, error

def f(x, R):
    om_m0 = 0.29
    h = 0.72
    H = 100.0*h
    ns = 0.99
    q = x/(0.073*om_m0*h)
    brackets = (1.0 + 0.284*q + (1.18*q)**2 + (0.399*q)**3 + (0.49*q)**4)**(-1./4.)
    T_f = (np.log(1+0.171*q)/(0.171*q))*brackets
    P_f = (T_f**2)*(q**ns)/(h**(ns+3.0))
    w_k = (np.sin(x*R)/(x*R) - np.cos(x*R))
    sigma_k = (9.0/(np.pi*(R**4)*(x**4)))*(x**2)*P_f*(w_k**2)
    return sigma_k

def fSigma(sig):
    A = 0.2315
    a = 1.727
    b = 1.744
    c = 1.5085
    return (A*(abs(sig/b)**(-a) + 1.0)*np.exp(-c/(sig**2)))

def lnSigma(mass, lnSig):
    ln_sig_d = [1.0]
    ln_sig_n = [1.0]
    for i3 in range(len(lnSig)):
        ln_sig_d.append(abs((lnSig[i3] - lnSig[i3 -1])/(m[i3]-m[i3-1])))
        ln_sig_n.append(abs(lnSig[i3] - lnSig[i3 -1]))
    ln_d = np.array(ln_sig_d)
    ln_n = np.array(ln_sig_n)
    return ln_d, ln_n

def sigma(func, R):
    sub = abs(quad(func, 1.0e-4, 400.0, args=(R,))[0])
    return sub

def bias(sig, del_c):
    over_d = 200.0/0.29 
    y = np.log10(over_d)
    A = 1.0 + 0.24*y*np.exp(-(4.0/y)**4.0)
    a = 0.44*y - 0.88
    B = 0.183
    b = 1.5
    C = 0.019 + 0.107*y + 0.19*np.exp(-(4.0/y)**4.0)
    c = 2.4
    nu = del_c/sig
    bias_1 = 1.0 + ((nu**2 - 1.0)/del_c)
    bias_2 = 1.0 - A*(nu**a)/(nu**a + del_c**a) + B*(nu**b) + C*(nu**c) 
    return bias_2

def u(ki, r_s, rho_s, m_i, c):
    Si_c = sici(ki*r_s*(1.0+c))[0]
    Ci_c = sici(ki*r_s*(1.0+c))[1]
    Si = sici(ki*r_s)[0]
    Ci = sici(ki*r_s)[1]
    #cs = np.sin(ki*r_s)*(np.pi/2.0 - Si) + np.cos(ki*r_s)*(-Ci)
    cs = np.sin(ki*r_s)*(Si_c-Si)-np.sin(c*ki*r_s)/(ki*r_s*(1+c))+np.cos(ki*r_s)*(Ci_c-Ci)
    u_cs = (4.0*np.pi*rho_s*(r_s**3)/m_i)*cs
    return u_cs


Sigma = np.zeros(len(r))
d_ln_sig = np.zeros(len(m)-1)
df = sigma(f, 8.0)
norm = np.sqrt((om_8**2)/df)

for l in range(len(r)):
    Sigma[l] = norm*np.sqrt(sigma(f, r[l]))

f_sig0 = fSigma(Sigma)
ln_sig0 = np.log(abs(Sigma)**(-1))
bias_f0 = bias(Sigma, 1.686)

dm = np.zeros(len(m)-1)
f_sig = np.zeros(len(m)-1)
ln_sig = np.zeros(len(m)-1)
bias_f = np.zeros(len(m)-1)
m2 = np.zeros(len(m)-1)

for a in range(len(m)-1):
    f_sig[a] = 0.5*(f_sig0[a] + f_sig0[a+1])
    ln_sig[a] = 0.5*(ln_sig0[a] + ln_sig0[a+1])
    bias_f[a] = 0.5*(bias_f0[a] + bias_f0[a+1])
    m2[a] = 0.5*(m[a] + m[a+1])
    if abs(m[a+1]-m[a]) > 0.0:
        dm[a] = (m[a+1]-m[a])
        d = (ln_sig0[a+1]-ln_sig0[a])/dm[a]
    else:
        dm[a] = 0.0
        d = 0.0
    d_ln_sig[a] = abs(d)


overd = 200.0/om_m
r2 = (3.0*m2/(4.0*np.pi*rho_m))**(1.0/3.0)
rvir = (3.0*m2/(4.0*np.pi*rho_m*overd))**(1.0/3.0)
m_star = 3.6e13

sigma2 = np.zeros(len(r2))
for l2 in range(len(r2)):
    sigma2[l2] = norm*np.sqrt(sigma(f, r2[l2]))
nu = (1.686/sigma2)**2
y_param = (0.42 + 0.2*nu**(-1.23) + 0.083*(nu**(-0.6)))/0.982
c = 10.0**(0.78*np.log(y_param) + 1.09)   #(9.0)*((m2/m_star)**(-0.13))

rs = rvir/c
rho_scale = (m2*(c**3.0))/(4.0*np.pi*(rvir**3.0)*(np.log(c+1.0) - (c/(c+1.0))))

n_m = f_sig*rho_m*d_ln_sig/m2

def f_kr(kr, c1, c2, c3, c5):
    return (c1*kr**2 + c2*kr**3 + c5*kr**4)/(c3*kr**3 + 1.0 + c5*kr**4)


data_takahashi = np.loadtxt('pk_wmap1.txt', unpack=True)
k_ext = data_takahashi[0]
pk_ext = data_takahashi[1]
pk_simerr = data_takahashi[2]

new_dat = np.loadtxt('k_wmap1.txt', unpack=True)
k0 = new_dat[0]
plin0 = np.interp(k_ext, k0, new_dat[1])
pknl0 = np.interp(k_ext, k0, new_dat[2])

k_small = np.logspace(-5.0, 2.0, 1000)
dat_ext = np.loadtxt('camb_wmap1.txt', unpack=True)
pl_ext = np.interp(k_small, dat_ext[0], dat_ext[1])
pnl_ext = np.interp(k_small, dat_ext[0], dat_ext[2])


def mcmc_MH(func, theta, k_i, pk_i):
    num = 5000
    accept = 0
    params = np.array([3.2, 320.0, 290.0, 0.034])
    count_params = np.zeros([num, len(params)])
    count_params[0] = params
    for i in range(num-1):
        step1 = np.random.normal(0.0, 1.0)
        step2 = np.random.normal(0.0, 1.0)
        step3 = np.random.normal(0.0, 1.0)
        step4 = np.random.normal(0.0, 1.0)
        step5 = np.random.normal(0.0, 1.0)
        param_curr = params + np.array([step1, step2, step3, step4, step5])
        val1 = func(params, k_i, pk_i, theta)
        val2 = func(param_curr, k_i, pk_i, theta)
        #diff = np.exp(val2 - val1)
        count_params[i+1] = param_curr
        if val2 < val1:
            accept += 1.0
            params = param_curr
            print(params, val2)
    return params, float(accept/num), count_params.transpose()

def ln_prior(params):
    a, b, c, d, e = params
    if a==0.0 or c==0.0 or e==0.0:
        return -np.inf
    return 0

def ln_likelihood(theta, params, k, pk_real):
    pl, uf, fkr, mi, rhoscale, r_s, nf, bf, dm, rhom = theta
    q1, q2, q3, q4, q5 = params
    stn = np.std(pk_real[:120])
    pk_amd = np.zeros(len(k[:120]))
    for i in range(len(k[:120])):
        u_n = uf(k[i], r_s, rhoscale, mi, 1.e50)
        fk = fkr(k[i]*r_s, q1, q2, q3, q4, q5)
        p1 = np.sum(dm*nf*(fk*u_n*mi)**2)/(rhom**2)
        p2 = (1.0 + np.sum(dm*mi*nf*bf*fk*u_n)/rhom)**2
        pk_amd[i] = p1 + pl[i]*p2
    diff = np.sum( ((pk_real[:120]-pk_amd[:120])**2)/pk_real[:120]**2)
    return diff

def lnprob(params, x, y, theta):
    lnp = ln_prior(params)
    if np.isfinite(lnp)==False:
        return -np.inf
    ln_prob = lnp + ln_likelihood(theta, params, x, y)
    return ln_prob

theta = (plin0, u, f_kr, m2, rho_scale, rs, n_m, bias_f, dm, rho_m)

pk_st = np.zeros(len(k_ext))
pk_am = np.zeros(len(k_ext))
pk_am_i = np.zeros(len(k_ext))


rate = mcmc_MH(lnprob, theta, k_ext, pk_ext)[1]
test_params = mcmc_MH(lnprob, theta, k_ext, pk_ext)[2]
b1, b2, b3, b4 = test_params
for q2 in range(len(k_ext)):
    u_0 = u(k_ext[q2], rs, rho_scale, m2, c)
    p01 = np.sum(dm*n_m*(u_0*m2)**2)/(rho_m**2)
    p02 = (np.sum(dm*m2*n_m*bias_f*u_0)/rho_m)**2
    pk_st[q2] = p01 + plin0[q2]*(p02 + 0.6)
    
    u_n = u(k_ext[q2], rs, rho_scale, m2, 1.e50)
    fk = f_kr(k_ext[q2]*rs, b1, b2, b3, b4)
    p1 = np.sum(dm*n_m*(fk*u_n*m2)**2)/(rho_m**2)
    p2 = (1.0 + np.sum(dm*m2*n_m*bias_f*fk*u_n)/rho_m)**2
    pk_am[q2] = p1 + plin0[q2]*p2
    
    if q2==0:
        print(p02, p2)


pk_small_am = np.zeros(len(k_small))
pk_small_st = np.zeros(len(k_small))
for q3 in range(len(k_small)):
    u_0 = u(k_small[q3], rs, rho_scale, m2, c)
    p01 = np.sum(dm*n_m*(u_0*m2)**2)/(rho_m**2)
    p02 = (np.sum(dm*m2*n_m*bias_f*u_0)/rho_m)**2
    pk_small_st[q3] = p01 + pl_ext[q3]*(p02 + 0.6)
    
    u_n = u(k_small[q3], rs, rho_scale, m2, 1.e50)
    fk = f_kr(k_small[q3]*rs, b1, b2, b3, b4)
    p1 = np.sum(dm*n_m*(fk*u_n*m2)**2)/(rho_m**2)
    p2 = (1.0 + np.sum(dm*m2*n_m*bias_f*fk*u_n)/rho_m)**2
    pk_small_am[q3] = p1 + pl_ext[q3]*p2
    
    if q3==0:
        print(p02, p2)


diff_am = np.sum((pk_am[:120] - pk_ext[:120])**2/(pk_ext[:120]**2))
diff_st = np.sum((pk_st[:120] - pk_ext[:120])**2/(pk_ext[:120]**2))
diff_HL = np.sum((pknl0[:120] - pk_ext[:120])**2/(pk_ext[:120]**2))

chi_am = np.sum((pk_am[:120] - pk_ext[:120])**2/(pk_simerr[:120]**2))
chi_HL = np.sum((pknl0[:120] - pk_ext[:120])**2/(pk_simerr[:120]**2))

print('\n Nishimichi simuations: ')
print('Amended: ', diff_am)
print('Standard: ', diff_st)
print('HALOFIT: ', diff_HL)
print('Chi amended: ', chi_am)
print('Chi halofit: ', chi_HL)

eq = np.argwhere(np.round(pl_ext)==np.round(pk_small_st))
print(eq)
plt.figure()
plt.errorbar(k_ext[:120],pk_ext[:120],yerr=pk_simerr[:120],fmt='o',label='Takahashi et al.',c='dimgrey',zorder=0)
plt.plot(k_small, pk_small_am, c='purple', linestyle='--')
plt.plot(k_ext[:120], pk_am[:120], label='amended', c='purple')
plt.plot(k_ext[:120], pknl0[:120], c='orange', label='HALOFIT')
plt.plot(k_ext, pk_st, label='standard', c='green')
plt.plot(k_small, pk_small_st, c='green', label='standard halo model')
plt.plot(k_small, pl_ext, label='linear', c='blue')
plt.fill_between(k_small[:210],pl_ext[:210],pk_small_st[:210],color='red',label='overprediction')
plt.yscale('log')
plt.xscale('log')
plt.ylabel('P(k) ($Mpc^3/h^3$)')
plt.xlabel('k (h/Mpc)')
plt.legend(loc='best')
plt.show()

res_am = (pk_am/pk_ext)[:120]
res_st = (pk_st/pk_ext)[:120]
res_hl = (pknl0/pk_ext)[:120]
plt.figure()
plt.title('Theory/Simulation Ratios')
plt.plot(k_ext[:120], abs(res_am), label='amended/simulation', c='purple')
plt.plot(k_ext[:120], abs(res_st), label='standard/simulation', c='green')
plt.plot(k_ext[:120], res_hl, label='HALOFIT/simulation', c='orange')
plt.plot(k_ext[:120], np.full(120, 1.0), c='black', linestyle='--')
plt.ylabel('Pk_sim/Pk_model')
plt.xlabel('k')
plt.xscale(value='log')
plt.legend(loc='lower left')
plt.savefig('residuals_mcmc_wmap1.pdf')
plt.show()










'''
n_dim = 5
n_walk = 10
sampler = emcee.EnsembleSampler(n_walk, n_dim, lnprob, args=(k_ext, pk_ext, theta))
step = [params + 0.1*np.random.randn(n_dim) for t in range(n_walk)]
sampler.run_mcmc(step, 100)
samples = sampler.chain[:, :, :]
print(samples)
samp_param = np.mean(samples, axis=1)

diff_n = 1.0
for o in samp_param:
    a1, a2, a3, a4, a5 = o
    for q in range(len(k_ext)):
        u_n = u(k_ext[q], rs, rho_scale, m2, 1.e50)
        fk = f_kr(k_ext[q]*rs, a1, a2, a3, a4, a5)
        #manually ~ 2900.0, -380.0, 6300.0, 42.0, 19000.0
        #preliminary mcmc ~ 3012.0, -365.46, 5994.2428, 34.0, 16993.64
        p1 = np.sum(dm*n_m*(fk*u_n*m2)**2)/(rho_m**2)
        p2 = (1.0 + np.sum(dm*m2*n_m*bias_f*fk*u_n)/rho_m)**2
        pk_am_i[q] = p1 + plin0[q]*p2
    
    diff_curr = np.sum((pk_am_i[:150] - pk_ext[:150])**2/(pk_ext[:150]**2))
    if diff_curr < diff_n:
        diff_am = diff_curr
        diff_n = diff_am
        b1, b2, b3, b4, b5 = o
'''

