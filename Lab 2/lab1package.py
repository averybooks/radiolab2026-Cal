import numpy as np

def load_array(filename):
    """
    Load arr_0 from an NPZ file and return it.

    Parameters
    ----------
    filename : str
        Path to the .npz file (default: "test_data.npz")

    Returns
    -------
    numpy.ndarray
        Array stored under key 'arr_0'
    """
    with np.load(filename) as data:
        arr = data["arr_0"]
        
    print(filename, "\n", arr)
    
    return arr

def alias_freq(f0,fs):
    return abs(((f0 + fs/2) % fs) - fs/2)

def voltage_spectrum_fft(x, fs):
    x = np.asarray(x).flatten()
    N = len(x)
    x0 = x - np.mean(x) # remove DC
    X = np.fft.fft(x0)
    f = np.fft.fftfreq(N, d=1/fs)
    Xsh = np.fft.fftshift(X)
    fsh = np.fft.fftshift(f)
    
    # voltage magnitude spectrum
    Vsh = np.abs(Xsh)
    V_real = np.real(Xsh)
    V_imag = np.imag(Xsh)
    
    return fsh, Vsh, V_real, V_imag

def power_spectrum_ifft(x, fs):
    x = np.asarray(x).flatten()
    N = len(x)
    x0 = x - np.mean(x) # remove DC
    X = np.fft.fft(x0)
    f = np.fft.fftfreq(N, d=1/fs)
    Xsh = np.fft.fftshift(X)
    fsh = np.fft.fftshift(f)
    Psh = np.abs(Xsh)**2
    return fsh, Psh, Xsh

def nyquist_zone(f0, fs):
    return int(np.floor(f0 / (fs/2)))

def leakage_metric_db(x, fs, f0, main_lobe_bins=3, dc_exclude_hz=100.0):
    """
    Compute leakage ratio in dB:
        L = 10*log10(P_leak / P_main)
    where P_main is power near the predicted alias frequency,
    and P_leak is power elsewhere (excluding DC region).

    Parameters
    ----------
    x : array-like
        time series (1D)
    fs : float
        sampling frequency [Hz]
    f0 : float
        input tone frequency [Hz]
    main_lobe_bins : int
        half-width in FFT bins for main lobe window around f_alias
        (e.g., 2-5 is typical)
    dc_exclude_hz : float
        exclude |f| < dc_exclude_hz from leakage computation

    Returns
    -------
    L_db : float
    f_alias : float
    df_bin : float
    """
    x = np.asarray(x).flatten()
    N = len(x)
    df_bin = fs / N

    # remove DC offset
    x0 = x - np.mean(x)

    # power spectrum (use rfft: only nonnegative freqs)
    X = np.fft.rfft(x0)
    P = np.abs(X)**2
    f = np.fft.rfftfreq(N, d=1/fs)

    # predicted alias location in [0, fs/2]
    f_alias = alias_freq(f0, fs)

    # main lobe window around predicted alias (± main_lobe_bins*df)
    bw = main_lobe_bins * df_bin
    main_mask = (f >= (f_alias - bw)) & (f <= (f_alias + bw))

    # leakage region = everything except main lobe and except DC neighborhood
    dc_mask = (f <= dc_exclude_hz)  # includes f=0
    leak_mask = (~main_mask) & (~dc_mask)

    P_main = np.sum(P[main_mask])
    P_leak = np.sum(P[leak_mask])

    # guard against divide-by-zero (rare but safe)
    eps = 1e-30
    L_db = 10.0 * np.log10((P_leak + eps) / (P_main + eps))

    return L_db, f_alias, df_bin

# FROM UGRADIO (just couldn't use it on my laptop, so importing the functions I need this way

def _compute_dft(in_x,in_y,out_x,inverse=False):
    if not inverse:
        in_y = np.fft.fftshift(in_y) 
        j = -1j
    else:
        in_y = in_y*(1.0/len(in_x))
        j = 1j

    N = len(in_x)
    out_y = np.zeros(len(out_x),dtype=np.complex128)
    for k,f in enumerate(out_x):
        out_y[k] = np.sum(in_y*np.exp(2*j*np.pi*f*in_x))

    return out_y

def dft(xt,t=[],f=[],vsamp=1):
    """
    Input 
    -----
    xt    : complex array, input time domain signal
    t     : (opt.) real array, input sample times. 
    f     : (opt.) real array, output sample frequencies
    vsamp : (opt.) float, sampling frequency
            default: 1
    Output
    ------
    f     : The same frequencies input
    Fx    : The discrete fourier transform of the input array

    """
    N = len(xt)
    if (len(t)):
        assert(len(t) == N), "Samples and sample times do not match!"
    else:
        t = np.linspace(-N/(2.0*vsamp),N/(2.0*vsamp),num=N,endpoint=False)

    if not (len(f)):
        #vsamp = N/float(np.ceil(t.max() - t.min()))
        f = np.linspace(-vsamp/2.,vsamp/2.,num=N,endpoint=False)
    
    Fx = _compute_dft(t,xt,f)

    return f,Fx

def idft(Fx,f=[],t=[],vsamp=1):
    """
    Input
    -----
    Fx    : complex array, input frequency domain signal
    f     : (opt.) real array, input sample frequencies
    t     : (opt.) real array, output sample times
    
    Output
    ------
    xt: The time domain signal of the input array

    """
    N = len(Fx)
    if (len(f)):
        assert(len(f) == N), "Samples and sample frequencies do not match!"
    else:
        f = np.linspace(-vsamp/2.,vsamp/2.,num=N,endpoint=False)

    if not (len(t)):
        #T = N/float(np.ceil(f.max()) - f.min())
        t = np.linspace(-N/(2.0*vsamp),N/(2.0*vsamp),num=N,endpoint=False)
    
    xt = _compute_dft(f,Fx,t,inverse=True)

    return t,xt