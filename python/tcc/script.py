import numpy as np
sample_rate = 20e6
num_samples = 1e6
f_carrier = 5e9


def gauss_pulse(t, t_samples_shift, A0, sigma):
    # t_samples_shift: delta time (drive - ressonator) in number of samples

    # Convert number of samples to time
    T1 = t[1]-t[0]
    t_shift = t_samples_shift * T1

    gauss = A0*np.exp(-0.5*((t-t_shift)/sigma)**2)
    return gauss

t = np.arange(-num_samples/2, num_samples/2+1) / sample_rate

sig = gauss_pulse(t=t, t_samples_shift=0, A0=1, sigma=400e-9)
carr = np.exp(1j*2*np.pi*f_carrier)

send_sig = sig*carr

