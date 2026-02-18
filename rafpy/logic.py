def id_signal_candidates(fig_num, observed, current_fs, filter_range=None, show_graph=True):
    #identifies potential true frequencies and highlights the one within the analog filter range
    #filter_range: tuple (f_min, f_max)
    
    candidates =[]
    f_nyquist = currentfs /2 

    #logic for finding frequency within filter range
    match = None
    nyquist_zone = None
    if filter_range:
        f_min, f_max = filter_range
        num_candidates = int(np.ceil(f_max / current_fs))
    else:
        num_candidates = 5
        
        
                
    for N in range(num_candidates+1):
        f_possible_1 = N*current_fs + observed
        f_possible_2 = N*current_fs - observed
        if f_possible_1 > 0: candidates.append(f_possible_1)
        if f_possible_2 > 0: candidates.append(f_possible_2)  
        #plt.figure(figsize=(12, 6))
    unique_candidates = sorted(list(set(candidates)))

    if filter_range:
        for f in unique_candidates:
            if f_min <= f <= f_max:
                match = f
                nyquist_zone = int(np.ceil(f/(current_fs/2)))
                break
        
    if show_graph:
        fig, ax = plt.subplots(figsize=(12, 5))
        plt.subplots_adjust(bottom=0.3)

        #plots all candidates
        markerline, stemlines, baseline = ax.stem(unique_candidates, np.ones(len(unique_candidates)))
        plt.setp(markerline, color='red', marker='D', markersize=6, alpha=0.5)
        plt.setp(stemlines, color='red', linestyle='--', alpha = 0.3)

        #highlight filter range
        if filter_range:
            ax.axvspan(filter_range[0], filter_range[1], color='orange', alpha=0.15, label='Filter Passband') 
        #highlight frequency
        if match:
            ax.stem([match], [1], linefmt='r-', markerfmt='rD', basefmt=' ')
            ax.annotate(f'MATCH: {match} Hz\nZone {nyquist_zone}', xy=(match, 1), xytext=(match, 1.4), arrowprops=dict(facecolor='green', shrink=0.05), ha='center', fontweight='bold', color='green')
        if observed:
            ax.stem([observed], [1], linefmt='r-', markerfmt='rD', basefmt=' ')
            ax.annotate(f'OBSERVED: {observed} Hz', xy=(observed, 1), xytext=(observed, 1.3), arrowprops=dict(facecolor='blue', shrink=0.05), ha='center', fontweight='bold', color='blue')
        #ax.annotate('Observed\n(Alias)', xy=(observed, 1), xytext=(observed, 1.3),
                    #arrowprops=dict(facecolor='black', shrink=0.05), ha='center')
        ax.set_yticks([]) 
        ax.set_title(f"Fig {fig_num}. Signal Identification (Observed: {observed} Hz | $f_s$: {current_fs} Hz | Bandpass {np.min(filter_range)}-{np.max(filter_range)} Hz)", fontweight='bold')
        ax.set_xlabel("Frequency (Hz)", fontweight='bold')
        ax.set_ylim(0, 1.6)
        desc = (
        f"""Fig {fig_num}. You saw a signal at {observed} Hz with a sampling rate of {current_fs} Hz. If you used a bandpass filter then the highlighted frequency is the original frequency considering any aliasing observed and which Nyquist zone it resides in. If you did not use a bandpass filter, then this lists the possible frequencies associated with the sampled signal, which can be filtered manually using known physics of your source."""
        )
        wrapped_desc = textwrap.fill(desc, width=90)
        plt.figtext(0.5, 0.05, wrapped_desc, ha='center', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", fc="#f0f0f0", ec="black", alpha=0.5))
        plt.legend(loc='upper right')
        print("-" * 30)
        print(f"ANALYSIS FOR {observed} Hz (Fs = {current_fs} Hz)")
        print(f"Potential Frequencies: {unique_candidates}")
        if match:
            print(f"IDENTIFIED SIGNAL: {match} Hz (Nyquist Zone {nyquist_zone})")
        print("-" * 30)
        plt.show()
    return {"all_candidates": unique_candidates, "identified_f": match, "nyquist_zone":nyquist_zone}, match