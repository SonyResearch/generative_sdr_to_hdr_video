import numpy as np


class PU21Encoder:
    """
    Transform absolute linear luminance values to/from the perceptually
    uniform (PU21) space.

    This follows the MATLAB implementation by Mantiuk & Azimi (PU21).
    Luminance Y is in cd/m^2 (nits), in [L_min, L_max].
    Encoded values V are roughly in [0, ~600], with ~100 nits -> ~256.
    """

    _TYPE_PARAMS = {
        "banding": np.array([
            1.070275272,
            0.4088273932,
            0.153224308,
            0.2520326168,
            1.063512885,
            1.14115047,
            521.4527484,
        ]),
        "banding_glare": np.array([
            0.353487901,
            0.3734658629,
            8.277049286e-05,
            0.9062562627,
            0.09150303166,
            0.9099517204,
            596.3148142,
        ]),
        "peaks": np.array([
            1.043882782,
            0.6459495343,
            0.3194584211,
            0.374025247,
            1.114783422,
            1.095360363,
            384.9217577,
        ]),
        "peaks_glare": np.array([
            816.885024,
            1479.463946,
            0.001253215609,
            0.9329636822,
            0.06746643971,
            1.573435413,
            419.6006374,
        ]),
    }

    def __init__(self, type: str = "banding_glare",
                 L_min: float = 0.005,
                 L_max: float = 10000.0):
        if type not in self._TYPE_PARAMS:
            raise ValueError(f"Unknown PU21 type: {type}")
        self.par = self._TYPE_PARAMS[type].astype(np.float64)
        self.L_min = float(L_min)
        self.L_max = float(L_max)

    def encode(self, Y):
        """
        Convert from linear (optical) luminance Y to encoded values V.

        Y: array-like, luminance in cd/m^2, expected in [L_min, L_max].
        V: array-like, encoded PU21 values (roughly [0, ~600]).
        """
        Y = np.asarray(Y, dtype=np.float64)

        # Clamp Y to valid range
        Y_clamped = np.clip(Y, self.L_min, self.L_max)

        p = self.par
        # V = max( p7 * (((p1 + p2*Y^p4)/(1 + p3*Y^p4))^p5 - p6), 0 )
        Y_pow = np.power(Y_clamped, p[3])
        num = p[0] + p[1] * Y_pow
        den = 1.0 + p[2] * Y_pow
        frac = num / den
        inner = np.power(frac, p[4]) - p[5]
        V = p[6] * inner

        return np.maximum(V, 0.0)

    def decode(self, V):
        """
        Convert from encoded values V to linear luminance Y.

        V: array-like, encoded PU21 values.
        Y: array-like, luminance in cd/m^2, approx in [L_min, L_max].
        """
        V = np.asarray(V, dtype=np.float64)
        p = self.par

        # V_p = max(V/p7 + p6, 0)^(1/p5)
        V_p = np.maximum(V / p[6] + p[5], 0.0)
        V_p = np.power(V_p, 1.0 / p[4])

        # Y = ( max(V_p - p1, 0) / (p2 - p3*V_p) )^(1/p4)
        num = np.maximum(V_p - p[0], 0.0)
        den = p[1] - p[2] * V_p
        frac = num / den
        Y = np.power(frac, 1.0 / p[3])

        # Clamp to nominal range
        return np.clip(Y, self.L_min, self.L_max)
