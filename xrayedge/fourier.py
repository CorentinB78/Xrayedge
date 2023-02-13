import numpy as np
from numpy import fft


def _next_regular(target):
    """
    Find the next regular number greater than or equal to target.
    Regular numbers are composites of the prime factors 2, 3, and 5.
    Also known as 5-smooth numbers or Hamming numbers, these are the optimal
    size for inputs to FFTPACK.
    Target must be a positive integer.
    """
    if target <= 6:
        return target

    # Quickly check if it's already a power of 2
    if not (target & (target - 1)):
        return target

    match = float("inf")  # Anything found will be smaller
    p5 = 1
    while p5 < target:
        p35 = p5
        while p35 < target:
            # Ceiling integer division, avoiding conversion to float
            # (quotient = ceil(target / p35))
            quotient = -(-target // p35)

            # Quickly find next power of 2 >= quotient
            try:
                p2 = 2 ** ((quotient - 1).bit_length())
            except AttributeError:
                # Fallback for Python <2.7
                p2 = 2 ** (len(bin(quotient - 1)) - 2)

            N = p2 * p35
            if N == target:
                return N
            elif N < match:
                match = N
            p35 *= 3
            if p35 == target:
                return p35
        if p35 < match:
            match = p35
        p5 *= 5
        if p5 == target:
            return p5
    if p5 < match:
        match = p5
    return match


def fourier_transform(t, ft, n="auto", axis=-1):
    r"""
    $f(\omega) = \int{dt} f(t) e^{-i\omega t}$
    times is assumed sorted and regularly spaced
    """

    if len(t) != ft.shape[axis]:
        raise ValueError(
            "coordinates should have the same length as values array on specified `axis`."
        )
    if n is None:
        n = len(t)
    elif n == "auto":
        n = _next_regular(len(t))

    dt = float(t[1] - t[0])

    ft = np.swapaxes(ft, -1, axis)

    w = fft.fftshift(fft.fftfreq(n, dt))
    fw = fft.fftshift(fft.fft(ft, n=n, axis=-1), axes=-1)

    w = -2 * np.pi * w[::-1]
    fw = fw[..., ::-1]
    fw[..., :] *= dt * np.exp(1j * w * float(t[0]))

    return w, np.swapaxes(fw, -1, axis)


def inv_fourier_transform(w, fw, n="auto", axis=-1):
    r"""
    $f(t) = \int\frac{d\omega}{2\pi} f(\omega) e^{i \omega t}$
    `w` is assumed sorted and regularly spaced
    """
    t, ft = fourier_transform(w, fw, n=n, axis=axis)
    t = -t[::-1]
    ft = np.moveaxis(ft, axis, -1)
    ft = ft[..., ::-1] / (2 * np.pi)

    return t, np.moveaxis(ft, -1, axis)
