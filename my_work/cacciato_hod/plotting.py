import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams["font.size"] = 16
mpl.rcParams["text.usetex"] = "True"
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, MultipleLocator, AutoMinorLocator

def apply_custom_ticks(ax, top=False, bottom=False, left=False, right=False):
    ax.tick_params(
        which="major",
        direction="in",
        length=8,
        width=1.2,
        labelsize=14,
        top=top,
        right=right,
        bottom=bottom,
        left=left,
    )
    ax.tick_params(
        which="minor",
        direction="in",
        length=4,
        width=1.0,
        top=top,
        right=right,
        bottom=bottom,
        left=left,
    )
    ax.grid(True, which="both", axis="both", ls="--", alpha=0.3)
    return ax

def setup_grid_ticks(axes):
    for ax in axes.flat:
        ax.minorticks_on()
        # figure out this axis position (row, col) within the axes grid
        row, col = tuple(np.argwhere(axes == ax)[0].astype(int))
        # enable top/bottom ticks only for the second row (row index = 1)
        top_on = row == axes.shape[0] - 1
        # disable right ticks for the last column
        right_on = col != axes.shape[1] - 1
        ax = apply_custom_ticks(
                ax, 
                top=top_on, 
                bottom=True,
                left=True,
                right=right_on
        )
    return axes

def store_png(fig, outname):
    if ".pdf" in outname:
        fpng = outname.split(".pdf")[0] + ".png"
        fig.savefig(fpng, dpi=200)
        print(f"saved {fpng}")

def plot_hod_single(hod, outname='Cacciato_HOD.pdf', save_png=False):
    # CCL's internal mass grid
    M = np.geomspace(1e8, 1e16, 128) # Msun 
    M_h = M * hod.hval # Msun/h
    Nc = hod._Nc(M)
    Ns = hod._Ns(M)
    fig, ax = plt.subplots(figsize=(5,4))
    ax = apply_custom_ticks(ax, True, True, True, True)
    ax.loglog(M_h, Nc, label='Centrals')
    ax.loglog(M_h, Ns, label='Satellites')
    ax.loglog(M_h, Nc+Ns, '--', label='Total')
    ax.set_ylim(1e-4, 1e3)
    ax.set_xlabel(r'$M\/[M_\odot/h]$')
    ax.set_ylabel(r'$\langle N_i|M \rangle$')
    ax.legend(loc="upper left", )
    fig.tight_layout()
    fig.savefig(outname, dpi=200)
    print(f"saved {outname}")
    if save_png:
        store_png(fig, outname)
    plt.close(fig)


def plot_hod_grid(hod_constructor, sampleinfo, magnitude_to_luminosity, outname='Cacciato_HOD_Nc_Ns_vs_M_all_bins.pdf', save_png=False):
    fig, axes = plt.subplots(2,3, figsize=(14,8), sharex=True, sharey=True) #
    fig.subplots_adjust(hspace=0.0, wspace=0.0)
    axes = setup_grid_ticks(axes)
    axes = axes.flatten()

    for ii, (M2, M1) in enumerate(sampleinfo['mag_bins']):
        log_L2 = np.log10(magnitude_to_luminosity(M2)) #Lsun/h^2
        log_L1 = np.log10(magnitude_to_luminosity(M1)) #Lsun/h^2
        hod = hod_constructor(log_L1=log_L1, log_L2=log_L2)
        logM = np.linspace(8,16,128) #Msun -- use this array for CCL HMF integration
        logM_h = logM + np.log10(hod.hval) # Msun/h
        Nc = hod._Nc(10**logM)
        Ns = hod._Ns(10**logM)
        # plot
        #info = rf"$M_{{r}}^{{0.1}}-5\log h \in ({M2:.2f},{M1:.2f}]$" 
        ax = axes[ii]
        (c,) = ax.plot(logM_h, np.log10(Nc), lw=2)
        (s,) = ax.plot(logM_h, np.log10(Ns), lw=2)
        (t,) = ax.plot(logM_h, np.log10(Nc + Ns), "--", c="k", lw=2)
        if ii == 0:
            ax.plot([], [], color=c.get_color(), lw=2, label="Centrals")
            ax.plot([], [], color=s.get_color(), lw=2, label="Satellites")
            ax.plot([], [], "--", color=t.get_color(), lw=2, label="Total")
            info = rf"$M_{{r}}^{{0.1}}-5\log h \in ({M2:.2f},{M1:.2f}]$" 
            ax.legend(loc="upper left", title=info)
        else:
            info = rf"$({M2:.2f},{M1:.2f}]$" 
            ax.legend(loc="upper left", title=info)
        if ii in [3, 4, 5]:
            ax.set_xlabel(r"$\log_{10} \left( M[M_\odot/h] \right)$")
        if ii in [0, 3]:
            ax.set_ylabel(r"$\langle N_i|M \rangle$")
        # remove the first and last major x-tick for the 2nd and 3rd columns
        # (columns indexed 1 and 2)
        if ii not in [0, 3]:
            ax.xaxis.set_major_locator(FixedLocator([11, 12, 13, 14, 15]))
            ax.yaxis.set_major_locator(FixedLocator([-3, -2, -1, 0, 1, 2]))

        ax.set_xlim(10,16)
        ax.set_ylim(-4,3)
    fig.savefig(outname, dpi=200)
    print(f"saved {outname}")
    if save_png:
        store_png(fig, outname)
    plt.close(fig)


def plot_phi_L_vs_L(log_lum_bin_centers, phi_L_cen, phi_L_sat, fsat, outname='Cacciato_HOD_phi_L_cen_sat_vs_L.pdf', save_png=False):
    fig, axes = plt.subplots(2,1, sharex=True, gridspec_kw={'height_ratios':[1,0.5], "hspace": 0.0})
    fig.subplots_adjust(hspace=0.02)
    ax = axes[0]
    ax = apply_custom_ticks(ax, True, True, True, True)
    ax.loglog(10**log_lum_bin_centers, phi_L_cen, 'o--', ms=5, label='Centrals')
    ax.loglog(10**log_lum_bin_centers, phi_L_sat, 's--', ms=5, label='Satellites')
    ax.loglog(10**log_lum_bin_centers, phi_L_cen + phi_L_sat, 'd--', ms=5, label='Total')
    ax.set_ylabel(r'$\Phi(L)$ [$h^3$ Mpc$^{-3}$]')
    ax.legend()

    ax = axes[1]
    ax = apply_custom_ticks(ax, True, True, True, True)
    ax.plot(10**log_lum_bin_centers, fsat, 'o--')
    ax.set_xscale('log')
    ax.set_xlabel(r'$L\; \left[h^{-2}L_\odot\right]$')
    ax.set_ylabel(r'$f_{\rm sat}$')
    fig.tight_layout()
    fig.savefig(outname, dpi=200)
    print(f"saved {outname}")
    if save_png:
        store_png(fig, outname)
    plt.close(fig)


def plot_phi_L_vs_M(mag_bin_centers, phi_L_cen, phi_L_sat, fsat, outname='Cacciato_HOD_phi_L_cen_sat_vs_M.pdf', save_png=False):
    fig, axes = plt.subplots(2,1, sharex=True, gridspec_kw={'height_ratios':[1,0.5], "hspace": 0.02})
    ax = axes[0]
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.2))
    ax.xaxis.set_inverted(True)
    ax = apply_custom_ticks(ax, True, True, True, True)
    ax.plot(mag_bin_centers, np.log10(phi_L_cen), 'o--', ms=5, label='Centrals')
    ax.plot(mag_bin_centers, np.log10(phi_L_sat), 's--', ms=5, label='Satellites')
    ax.plot(mag_bin_centers, np.log10(phi_L_cen + phi_L_sat), 'd--', ms=5, label='Total')
    ax.set_ylabel(r"$\log \left( \Phi \, [h^3\, {\rm Mpc}^{-3}\, {\rm mag}^{-1}] \right)$")
    ax.legend()

    ax = axes[1]
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.xaxis.set_inverted(True)
    ax = apply_custom_ticks(ax, True, True, True, True)
    ax.plot(mag_bin_centers, fsat, 'o--')
    ax.set_xlabel(r"$M_{r}^{0.1}-5\log h$")
    ax.set_ylabel(r'$f_{\rm sat}$')
    fig.tight_layout()
    fig.savefig(outname, dpi=200)
    print(f"saved {outname}")
    if save_png:
        store_png(fig, outname)
    plt.close(fig)

def plot_binned_LF(mag_bin_centers, binned_cen_hod, binned_sat_hod, fsat, outname="binned_LF_cacciato2013.pdf", save_png=False):
    fig, axes = plt.subplots(2,1, sharex=True, gridspec_kw={'height_ratios':[1,0.5], "hspace": 0.02})
    ax = axes[0]
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.2))
    ax.xaxis.set_inverted(True)
    ax = apply_custom_ticks(ax, True, True, True, True)
    ax.set_ylim(-9,-1)
    ax.plot(
        mag_bin_centers, np.log10(binned_cen_hod),
        "o--",
        ms=5,
        label="Centrals",
    )
    ax.plot(
        mag_bin_centers, np.log10(binned_sat_hod),
        "s--",
        ms=5,
        label="Satellites",
    )
    ax.plot(
        mag_bin_centers, np.log10(binned_cen_hod+binned_sat_hod),
        ".",
        color="magenta",
        ms=5,
        label="Total",
    )
    ax.set_ylabel(r"$\log \left( \Phi \, [h^3\, {\rm Mpc}^{-3}\, {\rm mag}^{-1}] \right)$")
    ax.legend(title="Cacciato+2013")

    ax = axes[1]
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.xaxis.set_inverted(True)
    ax = apply_custom_ticks(ax, True, True, True, True)
    ax.plot(
        mag_bin_centers, fsat,
        "o--",
        ms=5,
        label="Satellite Fraction",
    )
    ax.set_xlabel(r"$M_{r}^{0.1}-5\log h$")
    ax.set_ylabel(r'$f_{\rm sat}$')
    #fig.tight_layout()
    fig.savefig(outname, dpi=200)
    print(f"saved {outname}")
    if save_png:
        store_png(fig, outname)
    plt.close(fig)

