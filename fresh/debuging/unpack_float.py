import numpy as np

def unpack_float_acphy(format_code, nfft, H):
    if format_code == 0:
        nbits = 10
        autoscale = False
        shft = 0
        fmt = 1
        nman = 9
        nexp = 5
    elif format_code == 1:
        nbits = 10
        autoscale = False
        shft = 0
        fmt = 1
        nman = 12
        nexp = 6
    else:
        raise ValueError("format can only be 0 or 1")

    iq_mask = (1 << (nman - 1)) - 1
    e_mask = (1 << nexp) - 1
    e_p = (1 << (nexp - 1))
    sgnr_mask = (1 << (nexp + 2 * nman - 1))
    sgni_mask = (sgnr_mask >> nman)
    e_zero = -nman

    He = np.zeros(nfft, dtype=np.int8)
    Hout = np.zeros(nfft * 2, dtype=np.int32)

    maxbit = -e_p
    for i in range(nfft):
        vi = (H[i] >> (nexp + nman)) & iq_mask
        vq = (H[i] >> nexp) & iq_mask
        e = int(H[i] & e_mask)
        if e >= e_p:
            e -= (e_p << 1)

        He[i] = int(e)
        x = (vi | (vq << nman))
        if autoscale and x:
            m = 0xffff0000
            b = 0xffff
            s = 16
            while s > 0:
                if x & m:
                    e += s
                    x >>= s
                s >>= 1
                m = (m >> s) & b
                b >>= s

        if H[i] & sgnr_mask:
            vi |= 1 << (nbits - 1)
        if H[i] & sgni_mask:
            vq |= 1 << (nbits - 1)

        Hout[i * 2] = vi
        Hout[(i * 2) + 1] = vq

    shft = nbits - maxbit
    for i in range(nfft * 2):
        e = He[i >> 1] + shft
        vi = Hout[i]
        sgn = 1
        if vi & (1 << (nbits - 1)):
            sgn = -1
            vi &= ~(1 << (nbits - 1))
        if e < e_zero:
            vi = 0
        elif e < 0:
            e = -e
            vi = (vi >> e)
        else:
            vi = (vi << e)

        Hout[i] = sgn * vi

    return Hout[:nfft * 2]
