import numpy as np
from shared_state import total_acc, AVG_WINDOW, LO, SR

LO_freq     = LO  # GHz
sample_rate = SR   # GHz
i = 0

def ballin(data_array, filename="/home/radiopi/RTFM_Lab_3/data/test"):
    global i
    i += 1

    power_spec    = data_array['power_spec']
    start_time    = data_array['timestamp']
    end_time      = start_time + AVG_WINDOW
    n_acc         = data_array['n_acc']
    n_chan        = len(power_spec)

    freq_axis     = LO_freq + np.linspace(0, sample_rate / 2, n_chan)  # GHz
    peak_chan      = np.argmax(power_spec)
    peak_freq_ghz  = freq_axis[peak_chan]
    peak_power     = power_spec[peak_chan]
    total_power    = np.trapz(power_spec, freq_axis)

    total_acc.append(n_acc)

    np.savez(
        f"{filename}_{i}",
        power_spec    = power_spec,
        freq_axis     = freq_axis,
        start_time    = start_time,
        end_time      = end_time,
        n_acc         = n_acc,
        total_acc     = np.array(total_acc),
        peak_freq_ghz = peak_freq_ghz,
        peak_power    = peak_power,
        total_power   = total_power,
    )
    print(f"[WRITE] Saved window {i} | peak={peak_freq_ghz:.4f} GHz | power={peak_power:.4e}")







    
    
