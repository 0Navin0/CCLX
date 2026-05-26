from scipy.integrate import simpson
from scipy.special import erf, gammaincc, gamma

@np.vectorize(excluded=["a"])
def safe_upper_gamma(a, x):
    """
    Computes the upper incomplete gamma function Γ(a, x) for any real 'a'.
    Uses the recurrence relation to handle negative 'a' values.

    Parameters:
        a (float): The shape parameter, can be any real number.
        x (float): The upper/lower limit of lower/upper-gamma function integration, must be non-negative.
    Returns:
        float: The value of the upper incomplete gamma function Γ(a, x).
    """
    # Base case: x is 0 and a <= 0, the integral diverges
    if x <= 0 and a <= 0:
        return np.inf

    # Base case: a is positive, use standard scipy
    if a > 0:
        return gammaincc(a, x) * gamma(a)

    # Recurrence relation: Γ(a, x) = (Γ(a+1, x) - x^a * e^-x) / a
    # This shifts 'a' upward until it hits the positive base case
    return (safe_upper_gamma(a + 1, x) - x**a * np.exp(-x)) / a

def magnitude_to_luminosity(M, M_ref=4.76):
    """
    Converts absolute AB magnitude M (K+E corrected to z=0.1) to luminosity in
    units of L_sun/h^2.

    We use this function when the lens subsamples are defined on absolute
    magnitudes rather than luminosities. But the HOD model is defined in terms
    of luminosity, so we need to convert the magnitude cuts to luminosity cuts
    before computing the occupation numbers.

    $$M - 5\\log_{10}(h) = M_{\\rm ref} - 2.5\\log_{10}(L / L_{\\rm ref})$$

    Rearranging for $L$ yields the scaling factor $h^2$:
    $$L = 10^{0.4(M_{\rm ref} - M)} \\cdot h^2$$

    Parameters
    ----------
    mag : float or array_like
        Absolute AB magnitude of the source, scaled as$`M - 5\\log_{10}(h)$.
    m_ref : float, optional
        Absolute AB magnitude of the reference source (e.g., the Sun) in the
        target band. Default is 4.76 (SDSS $r$-band value from Blanton et al. 2003).

    Returns
    -------
    float or array_like
        Luminosity of the source in units of $L_{\rm ref}/h^2$ (where
        $L_{\rm ref}$ is the solar luminosity in the specified band).

    Notes
    -----
    The value of m_ref must be adjusted if working with photometric bands
    other than the SDSS $r$-band or if utilizing alternative $K+E$ correction
    baselines or no correction at all.
    """
    return 10 ** (0.4 * (M_ref - M))
