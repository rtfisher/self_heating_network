import numba
import numpy as np
from scipy import constants
from numba.experimental import jitclass

from pynucastro.rates import TableIndex, TableInterpolator, TabularRate, Tfactors
from pynucastro.screening import PlasmaState, ScreenFactors

jn = 0
jp = 1
jhe4 = 2
jc12 = 3
jn13 = 4
jn14 = 5
jo16 = 6
jne20 = 7
jne21 = 8
jna23 = 9
jmg23 = 10
jmg24 = 11
jal27 = 12
jsi27 = 13
jsi28 = 14
nnuc = 15

A = np.zeros((nnuc), dtype=np.int32)

A[jn] = 1
A[jp] = 1
A[jhe4] = 4
A[jc12] = 12
A[jn13] = 13
A[jn14] = 14
A[jo16] = 16
A[jne20] = 20
A[jne21] = 21
A[jna23] = 23
A[jmg23] = 23
A[jmg24] = 24
A[jal27] = 27
A[jsi27] = 27
A[jsi28] = 28

Z = np.zeros((nnuc), dtype=np.int32)

Z[jn] = 0
Z[jp] = 1
Z[jhe4] = 2
Z[jc12] = 6
Z[jn13] = 7
Z[jn14] = 7
Z[jo16] = 8
Z[jne20] = 10
Z[jne21] = 10
Z[jna23] = 11
Z[jmg23] = 12
Z[jmg24] = 12
Z[jal27] = 13
Z[jsi27] = 14
Z[jsi28] = 14

# masses in ergs
mass = np.zeros((nnuc), dtype=np.float64)

mass[jn] = 0.0015053497659156634
mass[jp] = 0.0015040963030260536
mass[jhe4] = 0.0059735574925878256
mass[jc12] = 0.017909017027273523
mass[jn13] = 0.01940999951603316
mass[jn14] = 0.020898440903103103
mass[jo16] = 0.023871099858982767
mass[jne20] = 0.02983707929641827
mass[jne21] = 0.03133159647374143
mass[jna23] = 0.034310347465945384
mass[jmg23] = 0.03431684618276469
mass[jmg24] = 0.03579570996619953
mass[jal27] = 0.04026773584000819
mass[jsi27] = 0.040275446154841646
mass[jsi28] = 0.041753271135012315

names = []
names.append("n")
names.append("H1")
names.append("He4")
names.append("C12")
names.append("N13")
names.append("N14")
names.append("O16")
names.append("Ne20")
names.append("Ne21")
names.append("Na23")
names.append("Mg23")
names.append("Mg24")
names.append("Al27")
names.append("Si27")
names.append("Si28")

def to_composition(Y):
    """Convert an array of molar fractions to a Composition object."""
    from pynucastro import Composition, Nucleus
    nuclei = [Nucleus.from_cache(name) for name in names]
    comp = Composition(nuclei)
    for i, nuc in enumerate(nuclei):
        comp.X[nuc] = Y[i] * A[i]
    return comp


def energy_release(dY):
    """return the energy release in erg/g (/s if dY is actually dY/dt)"""
    enuc = 0.0
    for i, y in enumerate(dY):
        enuc += y * mass[i]
    enuc *= -1*constants.Avogadro
    return enuc

@jitclass([
    ("n__p__weak__wc12", numba.float64),
    ("Mg23__Na23__weak__wc12", numba.float64),
    ("Si27__Al27__weak__wc12", numba.float64),
    ("N13__p_C12", numba.float64),
    ("N14__n_N13", numba.float64),
    ("O16__He4_C12", numba.float64),
    ("Ne20__He4_O16", numba.float64),
    ("Ne21__n_Ne20", numba.float64),
    ("Mg24__n_Mg23", numba.float64),
    ("Mg24__p_Na23", numba.float64),
    ("Mg24__He4_Ne20", numba.float64),
    ("Al27__He4_Na23", numba.float64),
    ("Si27__He4_Mg23", numba.float64),
    ("Si28__n_Si27", numba.float64),
    ("Si28__p_Al27", numba.float64),
    ("Si28__He4_Mg24", numba.float64),
    ("C12__He4_He4_He4", numba.float64),
    ("p_C12__N13", numba.float64),
    ("He4_C12__O16", numba.float64),
    ("n_N13__N14", numba.float64),
    ("He4_O16__Ne20", numba.float64),
    ("n_Ne20__Ne21", numba.float64),
    ("He4_Ne20__Mg24", numba.float64),
    ("p_Na23__Mg24", numba.float64),
    ("He4_Na23__Al27", numba.float64),
    ("n_Mg23__Mg24", numba.float64),
    ("He4_Mg23__Si27", numba.float64),
    ("He4_Mg24__Si28", numba.float64),
    ("p_Al27__Si28", numba.float64),
    ("n_Si27__Si28", numba.float64),
    ("C12_C12__n_Mg23", numba.float64),
    ("C12_C12__p_Na23", numba.float64),
    ("C12_C12__He4_Ne20", numba.float64),
    ("He4_N13__p_O16", numba.float64),
    ("p_O16__He4_N13", numba.float64),
    ("C12_O16__n_Si27", numba.float64),
    ("C12_O16__p_Al27", numba.float64),
    ("C12_O16__He4_Mg24", numba.float64),
    ("O16_O16__He4_Si28", numba.float64),
    ("He4_Ne20__n_Mg23", numba.float64),
    ("He4_Ne20__p_Na23", numba.float64),
    ("He4_Ne20__C12_C12", numba.float64),
    ("C12_Ne20__He4_Si28", numba.float64),
    ("He4_Ne21__n_Mg24", numba.float64),
    ("p_Na23__n_Mg23", numba.float64),
    ("p_Na23__He4_Ne20", numba.float64),
    ("p_Na23__C12_C12", numba.float64),
    ("n_Mg23__p_Na23", numba.float64),
    ("n_Mg23__He4_Ne20", numba.float64),
    ("n_Mg23__C12_C12", numba.float64),
    ("n_Mg24__He4_Ne21", numba.float64),
    ("He4_Mg24__n_Si27", numba.float64),
    ("He4_Mg24__p_Al27", numba.float64),
    ("He4_Mg24__C12_O16", numba.float64),
    ("p_Al27__n_Si27", numba.float64),
    ("p_Al27__He4_Mg24", numba.float64),
    ("p_Al27__C12_O16", numba.float64),
    ("n_Si27__p_Al27", numba.float64),
    ("n_Si27__He4_Mg24", numba.float64),
    ("n_Si27__C12_O16", numba.float64),
    ("He4_Si28__C12_Ne20", numba.float64),
    ("He4_Si28__O16_O16", numba.float64),
    ("He4_He4_He4__C12", numba.float64),
])
class RateEval:
    def __init__(self):
        self.n__p__weak__wc12 = np.nan
        self.Mg23__Na23__weak__wc12 = np.nan
        self.Si27__Al27__weak__wc12 = np.nan
        self.N13__p_C12 = np.nan
        self.N14__n_N13 = np.nan
        self.O16__He4_C12 = np.nan
        self.Ne20__He4_O16 = np.nan
        self.Ne21__n_Ne20 = np.nan
        self.Mg24__n_Mg23 = np.nan
        self.Mg24__p_Na23 = np.nan
        self.Mg24__He4_Ne20 = np.nan
        self.Al27__He4_Na23 = np.nan
        self.Si27__He4_Mg23 = np.nan
        self.Si28__n_Si27 = np.nan
        self.Si28__p_Al27 = np.nan
        self.Si28__He4_Mg24 = np.nan
        self.C12__He4_He4_He4 = np.nan
        self.p_C12__N13 = np.nan
        self.He4_C12__O16 = np.nan
        self.n_N13__N14 = np.nan
        self.He4_O16__Ne20 = np.nan
        self.n_Ne20__Ne21 = np.nan
        self.He4_Ne20__Mg24 = np.nan
        self.p_Na23__Mg24 = np.nan
        self.He4_Na23__Al27 = np.nan
        self.n_Mg23__Mg24 = np.nan
        self.He4_Mg23__Si27 = np.nan
        self.He4_Mg24__Si28 = np.nan
        self.p_Al27__Si28 = np.nan
        self.n_Si27__Si28 = np.nan
        self.C12_C12__n_Mg23 = np.nan
        self.C12_C12__p_Na23 = np.nan
        self.C12_C12__He4_Ne20 = np.nan
        self.He4_N13__p_O16 = np.nan
        self.p_O16__He4_N13 = np.nan
        self.C12_O16__n_Si27 = np.nan
        self.C12_O16__p_Al27 = np.nan
        self.C12_O16__He4_Mg24 = np.nan
        self.O16_O16__He4_Si28 = np.nan
        self.He4_Ne20__n_Mg23 = np.nan
        self.He4_Ne20__p_Na23 = np.nan
        self.He4_Ne20__C12_C12 = np.nan
        self.C12_Ne20__He4_Si28 = np.nan
        self.He4_Ne21__n_Mg24 = np.nan
        self.p_Na23__n_Mg23 = np.nan
        self.p_Na23__He4_Ne20 = np.nan
        self.p_Na23__C12_C12 = np.nan
        self.n_Mg23__p_Na23 = np.nan
        self.n_Mg23__He4_Ne20 = np.nan
        self.n_Mg23__C12_C12 = np.nan
        self.n_Mg24__He4_Ne21 = np.nan
        self.He4_Mg24__n_Si27 = np.nan
        self.He4_Mg24__p_Al27 = np.nan
        self.He4_Mg24__C12_O16 = np.nan
        self.p_Al27__n_Si27 = np.nan
        self.p_Al27__He4_Mg24 = np.nan
        self.p_Al27__C12_O16 = np.nan
        self.n_Si27__p_Al27 = np.nan
        self.n_Si27__He4_Mg24 = np.nan
        self.n_Si27__C12_O16 = np.nan
        self.He4_Si28__C12_Ne20 = np.nan
        self.He4_Si28__O16_O16 = np.nan
        self.He4_He4_He4__C12 = np.nan

@numba.njit()
def ye(Y):
    return np.sum(Z * Y)/np.sum(A * Y)

@numba.njit()
def n__p__weak__wc12(rate_eval, tf):
    # n --> p
    rate = 0.0

    # wc12w
    rate += np.exp(  -6.78161)

    rate_eval.n__p__weak__wc12 = rate

@numba.njit()
def Mg23__Na23__weak__wc12(rate_eval, tf):
    # Mg23 --> Na23
    rate = 0.0

    # wc12w
    rate += np.exp(  -2.79132)

    rate_eval.Mg23__Na23__weak__wc12 = rate

@numba.njit()
def Si27__Al27__weak__wc12(rate_eval, tf):
    # Si27 --> Al27
    rate = 0.0

    # wc12w
    rate += np.exp(  -1.78962)

    rate_eval.Si27__Al27__weak__wc12 = rate

@numba.njit()
def N13__p_C12(rate_eval, tf):
    # N13 --> p + C12
    rate = 0.0

    # ls09r
    rate += np.exp(  40.4354 + -26.326*tf.T9i + -5.10735*tf.T913i + -2.24111*tf.T913
                  + 0.148883*tf.T9)
    # ls09n
    rate += np.exp(  40.0408 + -22.5475*tf.T9i + -13.692*tf.T913i + -0.230881*tf.T913
                  + 4.44362*tf.T9 + -3.15898*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.N13__p_C12 = rate

@numba.njit()
def N14__n_N13(rate_eval, tf):
    # N14 --> n + N13
    rate = 0.0

    # wiesr
    rate += np.exp(  19.5584 + -125.474*tf.T9i + 9.44873e-10*tf.T913i + -2.33713e-09*tf.T913
                  + 1.97507e-10*tf.T9 + -1.49747e-11*tf.T953)
    # wiesn
    rate += np.exp(  37.1268 + -122.484*tf.T9i + 1.72241e-10*tf.T913i + -5.62522e-10*tf.T913
                  + 5.59212e-11*tf.T9 + -4.6549e-12*tf.T953 + 1.5*tf.lnT9)

    rate_eval.N14__n_N13 = rate

@numba.njit()
def O16__He4_C12(rate_eval, tf):
    # O16 --> He4 + C12
    rate = 0.0

    # nac2 
    rate += np.exp(  279.295 + -84.9515*tf.T9i + 103.411*tf.T913i + -420.567*tf.T913
                  + 64.0874*tf.T9 + -12.4624*tf.T953 + 138.803*tf.lnT9)
    # nac2 
    rate += np.exp(  94.3131 + -84.503*tf.T9i + 58.9128*tf.T913i + -148.273*tf.T913
                  + 9.08324*tf.T9 + -0.541041*tf.T953 + 71.8554*tf.lnT9)

    rate_eval.O16__He4_C12 = rate

@numba.njit()
def Ne20__He4_O16(rate_eval, tf):
    # Ne20 --> He4 + O16
    rate = 0.0

    # co10r
    rate += np.exp(  34.2658 + -67.6518*tf.T9i + -3.65925*tf.T913
                  + 0.714224*tf.T9 + -0.00107508*tf.T953)
    # co10r
    rate += np.exp(  28.6431 + -65.246*tf.T9i)
    # co10n
    rate += np.exp(  48.6604 + -54.8875*tf.T9i + -39.7262*tf.T913i + -0.210799*tf.T913
                  + 0.442879*tf.T9 + -0.0797753*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.Ne20__He4_O16 = rate

@numba.njit()
def Ne21__n_Ne20(rate_eval, tf):
    # Ne21 --> n + Ne20
    rate = 0.0

    # ka02r
    rate += np.exp(  34.9807 + -80.162*tf.T9i)
    # ka02n
    rate += np.exp(  30.8228 + -78.458*tf.T9i
                  + 1.5*tf.lnT9)

    rate_eval.Ne21__n_Ne20 = rate

@numba.njit()
def Mg24__n_Mg23(rate_eval, tf):
    # Mg24 --> n + Mg23
    rate = 0.0

    # ths8r
    rate += np.exp(  32.0344 + -191.835*tf.T9i + 2.66964*tf.T913
                  + -0.448904*tf.T9 + 0.0326505*tf.T953 + 1.5*tf.lnT9)

    rate_eval.Mg24__n_Mg23 = rate

@numba.njit()
def Mg24__p_Na23(rate_eval, tf):
    # Mg24 --> p + Na23
    rate = 0.0

    # il10r
    rate += np.exp(  34.0876 + -138.968*tf.T9i + -0.360588*tf.T913
                  + 1.4187*tf.T9 + -0.184061*tf.T953)
    # il10r
    rate += np.exp(  20.0024 + -137.3*tf.T9i)
    # il10n
    rate += np.exp(  43.9357 + -135.688*tf.T9i + -20.6428*tf.T913i + 1.52954*tf.T913
                  + 2.7487*tf.T9 + -1.0*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.Mg24__p_Na23 = rate

@numba.njit()
def Mg24__He4_Ne20(rate_eval, tf):
    # Mg24 --> He4 + Ne20
    rate = 0.0

    # il10n
    rate += np.exp(  49.3244 + -108.114*tf.T9i + -46.2525*tf.T913i + 5.58901*tf.T913
                  + 7.61843*tf.T9 + -3.683*tf.T953 + 0.833333*tf.lnT9)
    # il10r
    rate += np.exp(  16.0203 + -120.895*tf.T9i + 16.9229*tf.T913
                  + -2.57325*tf.T9 + 0.208997*tf.T953)
    # il10r
    rate += np.exp(  26.8017 + -117.334*tf.T9i)
    # il10r
    rate += np.exp(  -13.8869 + -110.62*tf.T9i)

    rate_eval.Mg24__He4_Ne20 = rate

@numba.njit()
def Al27__He4_Na23(rate_eval, tf):
    # Al27 --> He4 + Na23
    rate = 0.0

    # ths8r
    rate += np.exp(  69.2185 + -117.109*tf.T9i + -50.2042*tf.T913i + -1.64239*tf.T913
                  + -1.59995*tf.T9 + 0.184933*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.Al27__He4_Na23 = rate

@numba.njit()
def Si27__He4_Mg23(rate_eval, tf):
    # Si27 --> He4 + Mg23
    rate = 0.0

    # ths8r
    rate += np.exp(  68.1469 + -108.333*tf.T9i + -53.203*tf.T913i + -4.6318*tf.T913
                  + -0.130951*tf.T9 + 0.014691*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.Si27__He4_Mg23 = rate

@numba.njit()
def Si28__n_Si27(rate_eval, tf):
    # Si28 --> n + Si27
    rate = 0.0

    # ths8r
    rate += np.exp(  34.5835 + -199.363*tf.T9i + 0.559804*tf.T913
                  + -0.0233808*tf.T9 + -0.00121313*tf.T953 + 1.5*tf.lnT9)

    rate_eval.Si28__n_Si27 = rate

@numba.njit()
def Si28__p_Al27(rate_eval, tf):
    # Si28 --> p + Al27
    rate = 0.0

    # il10r
    rate += np.exp(  11.7765 + -136.349*tf.T9i + 23.8634*tf.T913
                  + -3.70135*tf.T9 + 0.28964*tf.T953)
    # il10n
    rate += np.exp(  46.5494 + -134.445*tf.T9i + -23.2205*tf.T913i
                  + -2.0*tf.T953 + 0.833333*tf.lnT9)
    # il10r
    rate += np.exp(  111.466 + -134.832*tf.T9i + -26.8327*tf.T913i + -116.137*tf.T913
                  + 0.00950567*tf.T9 + 0.00999755*tf.T953)

    rate_eval.Si28__p_Al27 = rate

@numba.njit()
def Si28__He4_Mg24(rate_eval, tf):
    # Si28 --> He4 + Mg24
    rate = 0.0

    # st08r
    rate += np.exp(  32.9006 + -131.488*tf.T9i)
    # st08r
    rate += np.exp(  -25.6886 + -128.693*tf.T9i + 21.3721*tf.T913i + 37.7649*tf.T913
                  + -4.10635*tf.T9 + 0.249618*tf.T953)

    rate_eval.Si28__He4_Mg24 = rate

@numba.njit()
def C12__He4_He4_He4(rate_eval, tf):
    # C12 --> He4 + He4 + He4
    rate = 0.0

    # fy05n
    rate += np.exp(  45.7734 + -84.4227*tf.T9i + -37.06*tf.T913i + 29.3493*tf.T913
                  + -115.507*tf.T9 + -10.0*tf.T953 + 1.66667*tf.lnT9)
    # fy05r
    rate += np.exp(  22.394 + -88.5493*tf.T9i + -13.49*tf.T913i + 21.4259*tf.T913
                  + -1.34769*tf.T9 + 0.0879816*tf.T953 + -10.1653*tf.lnT9)
    # fy05r
    rate += np.exp(  34.9561 + -85.4472*tf.T9i + -23.57*tf.T913i + 20.4886*tf.T913
                  + -12.9882*tf.T9 + -20.0*tf.T953 + 0.83333*tf.lnT9)

    rate_eval.C12__He4_He4_He4 = rate

@numba.njit()
def p_C12__N13(rate_eval, tf):
    # C12 + p --> N13
    rate = 0.0

    # ls09n
    rate += np.exp(  17.1482 + -13.692*tf.T913i + -0.230881*tf.T913
                  + 4.44362*tf.T9 + -3.15898*tf.T953 + -0.666667*tf.lnT9)
    # ls09r
    rate += np.exp(  17.5428 + -3.77849*tf.T9i + -5.10735*tf.T913i + -2.24111*tf.T913
                  + 0.148883*tf.T9 + -1.5*tf.lnT9)

    rate_eval.p_C12__N13 = rate

@numba.njit()
def He4_C12__O16(rate_eval, tf):
    # C12 + He4 --> O16
    rate = 0.0

    # nac2 
    rate += np.exp(  254.634 + -1.84097*tf.T9i + 103.411*tf.T913i + -420.567*tf.T913
                  + 64.0874*tf.T9 + -12.4624*tf.T953 + 137.303*tf.lnT9)
    # nac2 
    rate += np.exp(  69.6526 + -1.39254*tf.T9i + 58.9128*tf.T913i + -148.273*tf.T913
                  + 9.08324*tf.T9 + -0.541041*tf.T953 + 70.3554*tf.lnT9)

    rate_eval.He4_C12__O16 = rate

@numba.njit()
def n_N13__N14(rate_eval, tf):
    # N13 + n --> N14
    rate = 0.0

    # wiesr
    rate += np.exp(  -3.63074 + -2.99547*tf.T9i + 9.44873e-10*tf.T913i + -2.33713e-09*tf.T913
                  + 1.97507e-10*tf.T9 + -1.49747e-11*tf.T953 + -1.5*tf.lnT9)
    # wiesn
    rate += np.exp(  13.9377 + -0.0054652*tf.T9i + 1.72241e-10*tf.T913i + -5.62522e-10*tf.T913
                  + 5.59212e-11*tf.T9 + -4.6549e-12*tf.T953)

    rate_eval.n_N13__N14 = rate

@numba.njit()
def He4_O16__Ne20(rate_eval, tf):
    # O16 + He4 --> Ne20
    rate = 0.0

    # co10r
    rate += np.exp(  9.50848 + -12.7643*tf.T9i + -3.65925*tf.T913
                  + 0.714224*tf.T9 + -0.00107508*tf.T953 + -1.5*tf.lnT9)
    # co10r
    rate += np.exp(  3.88571 + -10.3585*tf.T9i
                  + -1.5*tf.lnT9)
    # co10n
    rate += np.exp(  23.903 + -39.7262*tf.T913i + -0.210799*tf.T913
                  + 0.442879*tf.T9 + -0.0797753*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He4_O16__Ne20 = rate

@numba.njit()
def n_Ne20__Ne21(rate_eval, tf):
    # Ne20 + n --> Ne21
    rate = 0.0

    # ka02r
    rate += np.exp(  12.7344 + -1.70393*tf.T9i
                  + -1.5*tf.lnT9)
    # ka02n
    rate += np.exp(  8.57651)

    rate_eval.n_Ne20__Ne21 = rate

@numba.njit()
def He4_Ne20__Mg24(rate_eval, tf):
    # Ne20 + He4 --> Mg24
    rate = 0.0

    # il10r
    rate += np.exp(  -38.7055 + -2.50605*tf.T9i
                  + -1.5*tf.lnT9)
    # il10n
    rate += np.exp(  24.5058 + -46.2525*tf.T913i + 5.58901*tf.T913
                  + 7.61843*tf.T9 + -3.683*tf.T953 + -0.666667*tf.lnT9)
    # il10r
    rate += np.exp(  -8.79827 + -12.7809*tf.T9i + 16.9229*tf.T913
                  + -2.57325*tf.T9 + 0.208997*tf.T953 + -1.5*tf.lnT9)
    # il10r
    rate += np.exp(  1.98307 + -9.22026*tf.T9i
                  + -1.5*tf.lnT9)

    rate_eval.He4_Ne20__Mg24 = rate

@numba.njit()
def p_Na23__Mg24(rate_eval, tf):
    # Na23 + p --> Mg24
    rate = 0.0

    # il10n
    rate += np.exp(  18.9075 + -20.6428*tf.T913i + 1.52954*tf.T913
                  + 2.7487*tf.T9 + -1.0*tf.T953 + -0.666667*tf.lnT9)
    # il10r
    rate += np.exp(  9.0594 + -3.28029*tf.T9i + -0.360588*tf.T913
                  + 1.4187*tf.T9 + -0.184061*tf.T953 + -1.5*tf.lnT9)
    # il10r
    rate += np.exp(  -5.02585 + -1.61219*tf.T9i
                  + -1.5*tf.lnT9)

    rate_eval.p_Na23__Mg24 = rate

@numba.njit()
def He4_Na23__Al27(rate_eval, tf):
    # Na23 + He4 --> Al27
    rate = 0.0

    # ths8r
    rate += np.exp(  44.7724 + -50.2042*tf.T913i + -1.64239*tf.T913
                  + -1.59995*tf.T9 + 0.184933*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He4_Na23__Al27 = rate

@numba.njit()
def n_Mg23__Mg24(rate_eval, tf):
    # Mg23 + n --> Mg24
    rate = 0.0

    # ths8r
    rate += np.exp(  7.00613 + 2.66964*tf.T913
                  + -0.448904*tf.T9 + 0.0326505*tf.T953)

    rate_eval.n_Mg23__Mg24 = rate

@numba.njit()
def He4_Mg23__Si27(rate_eval, tf):
    # Mg23 + He4 --> Si27
    rate = 0.0

    # ths8r
    rate += np.exp(  43.7008 + -53.203*tf.T913i + -4.6318*tf.T913
                  + -0.130951*tf.T9 + 0.014691*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He4_Mg23__Si27 = rate

@numba.njit()
def He4_Mg24__Si28(rate_eval, tf):
    # Mg24 + He4 --> Si28
    rate = 0.0

    # st08r
    rate += np.exp(  -50.5494 + -12.8332*tf.T9i + 21.3721*tf.T913i + 37.7649*tf.T913
                  + -4.10635*tf.T9 + 0.249618*tf.T953 + -1.5*tf.lnT9)
    # st08r
    rate += np.exp(  8.03977 + -15.629*tf.T9i
                  + -1.5*tf.lnT9)

    rate_eval.He4_Mg24__Si28 = rate

@numba.njit()
def p_Al27__Si28(rate_eval, tf):
    # Al27 + p --> Si28
    rate = 0.0

    # il10r
    rate += np.exp(  -13.6664 + -1.90396*tf.T9i + 23.8634*tf.T913
                  + -3.70135*tf.T9 + 0.28964*tf.T953 + -1.5*tf.lnT9)
    # il10r
    rate += np.exp(  86.0234 + -0.387313*tf.T9i + -26.8327*tf.T913i + -116.137*tf.T913
                  + 0.00950567*tf.T9 + 0.00999755*tf.T953 + -1.5*tf.lnT9)
    # il10n
    rate += np.exp(  21.1065 + -23.2205*tf.T913i
                  + -2.0*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_Al27__Si28 = rate

@numba.njit()
def n_Si27__Si28(rate_eval, tf):
    # Si27 + n --> Si28
    rate = 0.0

    # ths8r
    rate += np.exp(  9.14054 + 0.559804*tf.T913
                  + -0.0233808*tf.T9 + -0.00121313*tf.T953)

    rate_eval.n_Si27__Si28 = rate

@numba.njit()
def C12_C12__n_Mg23(rate_eval, tf):
    # C12 + C12 --> n + Mg23
    rate = 0.0

    # cf88r
    rate += np.exp(  -12.8056 + -30.1498*tf.T9i + 11.4826*tf.T913
                  + 1.82849*tf.T9 + -0.34844*tf.T953)

    rate_eval.C12_C12__n_Mg23 = rate

@numba.njit()
def C12_C12__p_Na23(rate_eval, tf):
    # C12 + C12 --> p + Na23
    rate = 0.0

    # cf88r
    rate += np.exp(  60.9649 + -84.165*tf.T913i + -1.4191*tf.T913
                  + -0.114619*tf.T9 + -0.070307*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.C12_C12__p_Na23 = rate

@numba.njit()
def C12_C12__He4_Ne20(rate_eval, tf):
    # C12 + C12 --> He4 + Ne20
    rate = 0.0

    # cf88r
    rate += np.exp(  61.2863 + -84.165*tf.T913i + -1.56627*tf.T913
                  + -0.0736084*tf.T9 + -0.072797*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.C12_C12__He4_Ne20 = rate

@numba.njit()
def He4_N13__p_O16(rate_eval, tf):
    # N13 + He4 --> p + O16
    rate = 0.0

    # cf88n
    rate += np.exp(  40.4644 + -35.829*tf.T913i + -0.530275*tf.T913
                  + -0.982462*tf.T9 + 0.0808059*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He4_N13__p_O16 = rate

@numba.njit()
def p_O16__He4_N13(rate_eval, tf):
    # O16 + p --> He4 + N13
    rate = 0.0

    # cf88n
    rate += np.exp(  42.2324 + -60.5523*tf.T9i + -35.829*tf.T913i + -0.530275*tf.T913
                  + -0.982462*tf.T9 + 0.0808059*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_O16__He4_N13 = rate

@numba.njit()
def C12_O16__n_Si27(rate_eval, tf):
    # O16 + C12 --> n + Si27
    rate = 0.0

    # cf88r
    rate += np.exp(  -132.213 + -1.46479*tf.T9i + -293.089*tf.T913i + 414.404*tf.T913
                  + -28.0562*tf.T9 + 1.61807*tf.T953 + -178.28*tf.lnT9)

    rate_eval.C12_O16__n_Si27 = rate

@numba.njit()
def C12_O16__p_Al27(rate_eval, tf):
    # O16 + C12 --> p + Al27
    rate = 0.0

    # cf88r
    rate += np.exp(  68.5253 + 0.205134*tf.T9i + -119.242*tf.T913i + 13.3667*tf.T913
                  + 0.295425*tf.T9 + -0.267288*tf.T953 + -9.91729*tf.lnT9)

    rate_eval.C12_O16__p_Al27 = rate

@numba.njit()
def C12_O16__He4_Mg24(rate_eval, tf):
    # O16 + C12 --> He4 + Mg24
    rate = 0.0

    # cf88r
    rate += np.exp(  48.5341 + 0.37204*tf.T9i + -133.413*tf.T913i + 50.1572*tf.T913
                  + -3.15987*tf.T9 + 0.0178251*tf.T953 + -23.7027*tf.lnT9)

    rate_eval.C12_O16__He4_Mg24 = rate

@numba.njit()
def O16_O16__He4_Si28(rate_eval, tf):
    # O16 + O16 --> He4 + Si28
    rate = 0.0

    # cf88r
    rate += np.exp(  97.2435 + -0.268514*tf.T9i + -119.324*tf.T913i + -32.2497*tf.T913
                  + 1.46214*tf.T9 + -0.200893*tf.T953 + 13.2148*tf.lnT9)

    rate_eval.O16_O16__He4_Si28 = rate

@numba.njit()
def He4_Ne20__n_Mg23(rate_eval, tf):
    # Ne20 + He4 --> n + Mg23
    rate = 0.0

    # ths8r
    rate += np.exp(  17.9544 + -83.7215*tf.T9i + 1.83199*tf.T913
                  + -0.290485*tf.T9 + 0.0242929*tf.T953)

    rate_eval.He4_Ne20__n_Mg23 = rate

@numba.njit()
def He4_Ne20__p_Na23(rate_eval, tf):
    # Ne20 + He4 --> p + Na23
    rate = 0.0

    # il10r
    rate += np.exp(  0.227472 + -29.4348*tf.T9i
                  + -1.5*tf.lnT9)
    # il10n
    rate += np.exp(  19.1852 + -27.5738*tf.T9i + -20.0024*tf.T913i + 11.5988*tf.T913
                  + -1.37398*tf.T9 + -1.0*tf.T953 + -0.666667*tf.lnT9)
    # il10r
    rate += np.exp(  -6.37772 + -29.8896*tf.T9i + 19.7297*tf.T913
                  + -2.20987*tf.T9 + 0.153374*tf.T953 + -1.5*tf.lnT9)

    rate_eval.He4_Ne20__p_Na23 = rate

@numba.njit()
def He4_Ne20__C12_C12(rate_eval, tf):
    # Ne20 + He4 --> C12 + C12
    rate = 0.0

    # cf88r
    rate += np.exp(  61.4748 + -53.6267*tf.T9i + -84.165*tf.T913i + -1.56627*tf.T913
                  + -0.0736084*tf.T9 + -0.072797*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He4_Ne20__C12_C12 = rate

@numba.njit()
def C12_Ne20__He4_Si28(rate_eval, tf):
    # Ne20 + C12 --> He4 + Si28
    rate = 0.0

    # rolfr
    rate += np.exp(  -308.905 + -47.2175*tf.T9i + 514.197*tf.T913i + -200.896*tf.T913
                  + -6.42713*tf.T9 + 0.758256*tf.T953 + 236.359*tf.lnT9)

    rate_eval.C12_Ne20__He4_Si28 = rate

@numba.njit()
def He4_Ne21__n_Mg24(rate_eval, tf):
    # Ne21 + He4 --> n + Mg24
    rate = 0.0

    # nacrn
    rate += np.exp(  43.9762 + -46.88*tf.T913i + -0.536629*tf.T913
                  + 0.144715*tf.T9 + -0.197624*tf.T953 + -0.666667*tf.lnT9)
    # nacrr
    rate += np.exp(  -7.26831 + -13.2638*tf.T9i + 18.0748*tf.T913
                  + -0.981883*tf.T9 + -1.5*tf.lnT9)

    rate_eval.He4_Ne21__n_Mg24 = rate

@numba.njit()
def p_Na23__n_Mg23(rate_eval, tf):
    # Na23 + p --> n + Mg23
    rate = 0.0

    # nacr 
    rate += np.exp(  19.4638 + -56.1542*tf.T9i + 0.993488*tf.T913
                  + -0.257094*tf.T9 + 0.0284334*tf.T953)

    rate_eval.p_Na23__n_Mg23 = rate

@numba.njit()
def p_Na23__He4_Ne20(rate_eval, tf):
    # Na23 + p --> He4 + Ne20
    rate = 0.0

    # il10r
    rate += np.exp(  -6.58736 + -2.31577*tf.T9i + 19.7297*tf.T913
                  + -2.20987*tf.T9 + 0.153374*tf.T953 + -1.5*tf.lnT9)
    # il10r
    rate += np.exp(  0.0178295 + -1.86103*tf.T9i
                  + -1.5*tf.lnT9)
    # il10n
    rate += np.exp(  18.9756 + -20.0024*tf.T913i + 11.5988*tf.T913
                  + -1.37398*tf.T9 + -1.0*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_Na23__He4_Ne20 = rate

@numba.njit()
def p_Na23__C12_C12(rate_eval, tf):
    # Na23 + p --> C12 + C12
    rate = 0.0

    # cf88r
    rate += np.exp(  60.9438 + -26.0184*tf.T9i + -84.165*tf.T913i + -1.4191*tf.T913
                  + -0.114619*tf.T9 + -0.070307*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_Na23__C12_C12 = rate

@numba.njit()
def n_Mg23__p_Na23(rate_eval, tf):
    # Mg23 + n --> p + Na23
    rate = 0.0

    # nacr 
    rate += np.exp(  19.4638 + 0.993488*tf.T913
                  + -0.257094*tf.T9 + 0.0284334*tf.T953)

    rate_eval.n_Mg23__p_Na23 = rate

@numba.njit()
def n_Mg23__He4_Ne20(rate_eval, tf):
    # Mg23 + n --> He4 + Ne20
    rate = 0.0

    # ths8r
    rate += np.exp(  17.7448 + 1.83199*tf.T913
                  + -0.290485*tf.T9 + 0.0242929*tf.T953)

    rate_eval.n_Mg23__He4_Ne20 = rate

@numba.njit()
def n_Mg23__C12_C12(rate_eval, tf):
    # Mg23 + n --> C12 + C12
    rate = 0.0

    # cf88r
    rate += np.exp(  -12.8267 + 11.4826*tf.T913
                  + 1.82849*tf.T9 + -0.34844*tf.T953)

    rate_eval.n_Mg23__C12_C12 = rate

@numba.njit()
def n_Mg24__He4_Ne21(rate_eval, tf):
    # Mg24 + n --> He4 + Ne21
    rate = 0.0

    # nacrr
    rate += np.exp(  -4.69602 + -42.9133*tf.T9i + 18.0748*tf.T913
                  + -0.981883*tf.T9 + -1.5*tf.lnT9)
    # nacrn
    rate += np.exp(  46.5485 + -29.6495*tf.T9i + -46.88*tf.T913i + -0.536629*tf.T913
                  + 0.144715*tf.T9 + -0.197624*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.n_Mg24__He4_Ne21 = rate

@numba.njit()
def He4_Mg24__n_Si27(rate_eval, tf):
    # Mg24 + He4 --> n + Si27
    rate = 0.0

    # ths8r
    rate += np.exp(  18.6606 + -83.5022*tf.T9i + 0.942981*tf.T913
                  + -0.104624*tf.T9 + 0.00723421*tf.T953)

    rate_eval.He4_Mg24__n_Si27 = rate

@numba.njit()
def He4_Mg24__p_Al27(rate_eval, tf):
    # Mg24 + He4 --> p + Al27
    rate = 0.0

    # il10n
    rate += np.exp(  30.0397 + -18.5791*tf.T9i + -26.4162*tf.T913i
                  + -2.0*tf.T953 + -0.666667*tf.lnT9)
    # il10r
    rate += np.exp(  -26.2862 + -19.5422*tf.T9i + 5.18642*tf.T913i + -34.7936*tf.T913
                  + 168.225*tf.T9 + -115.825*tf.T953 + -1.5*tf.lnT9)
    # il10r
    rate += np.exp(  -6.44575 + -22.8216*tf.T9i + 18.0416*tf.T913
                  + -1.54137*tf.T9 + 0.0847506*tf.T953 + -1.5*tf.lnT9)

    rate_eval.He4_Mg24__p_Al27 = rate

@numba.njit()
def He4_Mg24__C12_O16(rate_eval, tf):
    # Mg24 + He4 --> C12 + O16
    rate = 0.0

    # cf88r
    rate += np.exp(  49.5738 + -78.202*tf.T9i + -133.413*tf.T913i + 50.1572*tf.T913
                  + -3.15987*tf.T9 + 0.0178251*tf.T953 + -23.7027*tf.lnT9)

    rate_eval.He4_Mg24__C12_O16 = rate

@numba.njit()
def p_Al27__n_Si27(rate_eval, tf):
    # Al27 + p --> n + Si27
    rate = 0.0

    # ths8r
    rate += np.exp(  17.84 + -64.9237*tf.T9i + 2.10552*tf.T913
                  + -0.318695*tf.T9 + 0.0257496*tf.T953)

    rate_eval.p_Al27__n_Si27 = rate

@numba.njit()
def p_Al27__He4_Mg24(rate_eval, tf):
    # Al27 + p --> He4 + Mg24
    rate = 0.0

    # il10r
    rate += np.exp(  -7.02789 + -4.2425*tf.T9i + 18.0416*tf.T913
                  + -1.54137*tf.T9 + 0.0847506*tf.T953 + -1.5*tf.lnT9)
    # il10r
    rate += np.exp(  -26.8683 + -0.963012*tf.T9i + 5.18642*tf.T913i + -34.7936*tf.T913
                  + 168.225*tf.T9 + -115.825*tf.T953 + -1.5*tf.lnT9)
    # il10n
    rate += np.exp(  29.4576 + -26.4162*tf.T913i
                  + -2.0*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_Al27__He4_Mg24 = rate

@numba.njit()
def p_Al27__C12_O16(rate_eval, tf):
    # Al27 + p --> C12 + O16
    rate = 0.0

    # cf88r
    rate += np.exp(  68.9829 + -59.8017*tf.T9i + -119.242*tf.T913i + 13.3667*tf.T913
                  + 0.295425*tf.T9 + -0.267288*tf.T953 + -9.91729*tf.lnT9)

    rate_eval.p_Al27__C12_O16 = rate

@numba.njit()
def n_Si27__p_Al27(rate_eval, tf):
    # Si27 + n --> p + Al27
    rate = 0.0

    # ths8r
    rate += np.exp(  17.84 + 2.10552*tf.T913
                  + -0.318695*tf.T9 + 0.0257496*tf.T953)

    rate_eval.n_Si27__p_Al27 = rate

@numba.njit()
def n_Si27__He4_Mg24(rate_eval, tf):
    # Si27 + n --> He4 + Mg24
    rate = 0.0

    # ths8r
    rate += np.exp(  18.0785 + 0.942981*tf.T913
                  + -0.104624*tf.T9 + 0.00723421*tf.T953)

    rate_eval.n_Si27__He4_Mg24 = rate

@numba.njit()
def n_Si27__C12_O16(rate_eval, tf):
    # Si27 + n --> C12 + O16
    rate = 0.0

    # cf88r
    rate += np.exp(  -131.755 + 3.44391*tf.T9i + -293.089*tf.T913i + 414.404*tf.T913
                  + -28.0562*tf.T9 + 1.61807*tf.T953 + -178.28*tf.lnT9)

    rate_eval.n_Si27__C12_O16 = rate

@numba.njit()
def He4_Si28__C12_Ne20(rate_eval, tf):
    # Si28 + He4 --> C12 + Ne20
    rate = 0.0

    # rolfr
    rate += np.exp(  -307.762 + -186.722*tf.T9i + 514.197*tf.T913i + -200.896*tf.T913
                  + -6.42713*tf.T9 + 0.758256*tf.T953 + 236.359*tf.lnT9)

    rate_eval.He4_Si28__C12_Ne20 = rate

@numba.njit()
def He4_Si28__O16_O16(rate_eval, tf):
    # Si28 + He4 --> O16 + O16
    rate = 0.0

    # cf88r
    rate += np.exp(  97.7904 + -111.595*tf.T9i + -119.324*tf.T913i + -32.2497*tf.T913
                  + 1.46214*tf.T9 + -0.200893*tf.T953 + 13.2148*tf.lnT9)

    rate_eval.He4_Si28__O16_O16 = rate

@numba.njit()
def He4_He4_He4__C12(rate_eval, tf):
    # He4 + He4 + He4 --> C12
    rate = 0.0

    # fy05r
    rate += np.exp(  -24.3505 + -4.12656*tf.T9i + -13.49*tf.T913i + 21.4259*tf.T913
                  + -1.34769*tf.T9 + 0.0879816*tf.T953 + -13.1653*tf.lnT9)
    # fy05r
    rate += np.exp(  -11.7884 + -1.02446*tf.T9i + -23.57*tf.T913i + 20.4886*tf.T913
                  + -12.9882*tf.T9 + -20.0*tf.T953 + -2.16667*tf.lnT9)
    # fy05n
    rate += np.exp(  -0.971052 + -37.06*tf.T913i + 29.3493*tf.T913
                  + -115.507*tf.T9 + -10.0*tf.T953 + -1.33333*tf.lnT9)

    rate_eval.He4_He4_He4__C12 = rate

def rhs(t, Y, rho, T, screen_func=None):
    return rhs_eq(t, Y, rho, T, screen_func)

@numba.njit()
def rhs_eq(t, Y, rho, T, screen_func):

    tf = Tfactors(T)
    rate_eval = RateEval()

    # reaclib rates
    n__p__weak__wc12(rate_eval, tf)
    Mg23__Na23__weak__wc12(rate_eval, tf)
    Si27__Al27__weak__wc12(rate_eval, tf)
    N13__p_C12(rate_eval, tf)
    N14__n_N13(rate_eval, tf)
    O16__He4_C12(rate_eval, tf)
    Ne20__He4_O16(rate_eval, tf)
    Ne21__n_Ne20(rate_eval, tf)
    Mg24__n_Mg23(rate_eval, tf)
    Mg24__p_Na23(rate_eval, tf)
    Mg24__He4_Ne20(rate_eval, tf)
    Al27__He4_Na23(rate_eval, tf)
    Si27__He4_Mg23(rate_eval, tf)
    Si28__n_Si27(rate_eval, tf)
    Si28__p_Al27(rate_eval, tf)
    Si28__He4_Mg24(rate_eval, tf)
    C12__He4_He4_He4(rate_eval, tf)
    p_C12__N13(rate_eval, tf)
    He4_C12__O16(rate_eval, tf)
    n_N13__N14(rate_eval, tf)
    He4_O16__Ne20(rate_eval, tf)
    n_Ne20__Ne21(rate_eval, tf)
    He4_Ne20__Mg24(rate_eval, tf)
    p_Na23__Mg24(rate_eval, tf)
    He4_Na23__Al27(rate_eval, tf)
    n_Mg23__Mg24(rate_eval, tf)
    He4_Mg23__Si27(rate_eval, tf)
    He4_Mg24__Si28(rate_eval, tf)
    p_Al27__Si28(rate_eval, tf)
    n_Si27__Si28(rate_eval, tf)
    C12_C12__n_Mg23(rate_eval, tf)
    C12_C12__p_Na23(rate_eval, tf)
    C12_C12__He4_Ne20(rate_eval, tf)
    He4_N13__p_O16(rate_eval, tf)
    p_O16__He4_N13(rate_eval, tf)
    C12_O16__n_Si27(rate_eval, tf)
    C12_O16__p_Al27(rate_eval, tf)
    C12_O16__He4_Mg24(rate_eval, tf)
    O16_O16__He4_Si28(rate_eval, tf)
    He4_Ne20__n_Mg23(rate_eval, tf)
    He4_Ne20__p_Na23(rate_eval, tf)
    He4_Ne20__C12_C12(rate_eval, tf)
    C12_Ne20__He4_Si28(rate_eval, tf)
    He4_Ne21__n_Mg24(rate_eval, tf)
    p_Na23__n_Mg23(rate_eval, tf)
    p_Na23__He4_Ne20(rate_eval, tf)
    p_Na23__C12_C12(rate_eval, tf)
    n_Mg23__p_Na23(rate_eval, tf)
    n_Mg23__He4_Ne20(rate_eval, tf)
    n_Mg23__C12_C12(rate_eval, tf)
    n_Mg24__He4_Ne21(rate_eval, tf)
    He4_Mg24__n_Si27(rate_eval, tf)
    He4_Mg24__p_Al27(rate_eval, tf)
    He4_Mg24__C12_O16(rate_eval, tf)
    p_Al27__n_Si27(rate_eval, tf)
    p_Al27__He4_Mg24(rate_eval, tf)
    p_Al27__C12_O16(rate_eval, tf)
    n_Si27__p_Al27(rate_eval, tf)
    n_Si27__He4_Mg24(rate_eval, tf)
    n_Si27__C12_O16(rate_eval, tf)
    He4_Si28__C12_Ne20(rate_eval, tf)
    He4_Si28__O16_O16(rate_eval, tf)
    He4_He4_He4__C12(rate_eval, tf)

    if screen_func is not None:
        plasma_state = PlasmaState(T, rho, Y, Z)

        scn_fac = ScreenFactors(1, 1, 6, 12)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_C12__N13 *= scor

        scn_fac = ScreenFactors(2, 4, 6, 12)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_C12__O16 *= scor

        scn_fac = ScreenFactors(2, 4, 8, 16)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_O16__Ne20 *= scor

        scn_fac = ScreenFactors(2, 4, 10, 20)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Ne20__Mg24 *= scor
        rate_eval.He4_Ne20__n_Mg23 *= scor
        rate_eval.He4_Ne20__p_Na23 *= scor
        rate_eval.He4_Ne20__C12_C12 *= scor

        scn_fac = ScreenFactors(1, 1, 11, 23)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_Na23__Mg24 *= scor
        rate_eval.p_Na23__n_Mg23 *= scor
        rate_eval.p_Na23__He4_Ne20 *= scor
        rate_eval.p_Na23__C12_C12 *= scor

        scn_fac = ScreenFactors(2, 4, 11, 23)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Na23__Al27 *= scor

        scn_fac = ScreenFactors(2, 4, 12, 23)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Mg23__Si27 *= scor

        scn_fac = ScreenFactors(2, 4, 12, 24)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Mg24__Si28 *= scor
        rate_eval.He4_Mg24__n_Si27 *= scor
        rate_eval.He4_Mg24__p_Al27 *= scor
        rate_eval.He4_Mg24__C12_O16 *= scor

        scn_fac = ScreenFactors(1, 1, 13, 27)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_Al27__Si28 *= scor
        rate_eval.p_Al27__n_Si27 *= scor
        rate_eval.p_Al27__He4_Mg24 *= scor
        rate_eval.p_Al27__C12_O16 *= scor

        scn_fac = ScreenFactors(6, 12, 6, 12)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.C12_C12__n_Mg23 *= scor
        rate_eval.C12_C12__p_Na23 *= scor
        rate_eval.C12_C12__He4_Ne20 *= scor

        scn_fac = ScreenFactors(2, 4, 7, 13)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_N13__p_O16 *= scor

        scn_fac = ScreenFactors(1, 1, 8, 16)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_O16__He4_N13 *= scor

        scn_fac = ScreenFactors(6, 12, 8, 16)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.C12_O16__n_Si27 *= scor
        rate_eval.C12_O16__p_Al27 *= scor
        rate_eval.C12_O16__He4_Mg24 *= scor

        scn_fac = ScreenFactors(8, 16, 8, 16)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.O16_O16__He4_Si28 *= scor

        scn_fac = ScreenFactors(6, 12, 10, 20)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.C12_Ne20__He4_Si28 *= scor

        scn_fac = ScreenFactors(2, 4, 10, 21)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Ne21__n_Mg24 *= scor

        scn_fac = ScreenFactors(2, 4, 14, 28)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Si28__C12_Ne20 *= scor
        rate_eval.He4_Si28__O16_O16 *= scor

        scn_fac = ScreenFactors(2, 4, 2, 4)
        scor = screen_func(plasma_state, scn_fac)
        scn_fac2 = ScreenFactors(2, 4, 4, 8)
        scor2 = screen_func(plasma_state, scn_fac2)
        rate_eval.He4_He4_He4__C12 *= scor * scor2

    dYdt = np.zeros((nnuc), dtype=np.float64)

    dYdt[jn] = (
       -Y[jn]*rate_eval.n__p__weak__wc12
       -rho*Y[jn]*Y[jn13]*rate_eval.n_N13__N14
       -rho*Y[jn]*Y[jne20]*rate_eval.n_Ne20__Ne21
       -rho*Y[jn]*Y[jmg23]*rate_eval.n_Mg23__Mg24
       -rho*Y[jn]*Y[jsi27]*rate_eval.n_Si27__Si28
       -rho*Y[jn]*Y[jmg23]*rate_eval.n_Mg23__p_Na23
       -rho*Y[jn]*Y[jmg23]*rate_eval.n_Mg23__He4_Ne20
       -rho*Y[jn]*Y[jmg23]*rate_eval.n_Mg23__C12_C12
       -rho*Y[jn]*Y[jmg24]*rate_eval.n_Mg24__He4_Ne21
       -rho*Y[jn]*Y[jsi27]*rate_eval.n_Si27__p_Al27
       -rho*Y[jn]*Y[jsi27]*rate_eval.n_Si27__He4_Mg24
       -rho*Y[jn]*Y[jsi27]*rate_eval.n_Si27__C12_O16
       +Y[jn14]*rate_eval.N14__n_N13
       +Y[jne21]*rate_eval.Ne21__n_Ne20
       +Y[jmg24]*rate_eval.Mg24__n_Mg23
       +Y[jsi28]*rate_eval.Si28__n_Si27
       +5.00000000000000e-01*rho*Y[jc12]**2*rate_eval.C12_C12__n_Mg23
       +rho*Y[jc12]*Y[jo16]*rate_eval.C12_O16__n_Si27
       +rho*Y[jhe4]*Y[jne20]*rate_eval.He4_Ne20__n_Mg23
       +rho*Y[jhe4]*Y[jne21]*rate_eval.He4_Ne21__n_Mg24
       +rho*Y[jp]*Y[jna23]*rate_eval.p_Na23__n_Mg23
       +rho*Y[jhe4]*Y[jmg24]*rate_eval.He4_Mg24__n_Si27
       +rho*Y[jp]*Y[jal27]*rate_eval.p_Al27__n_Si27
       )

    dYdt[jp] = (
       -rho*Y[jp]*Y[jc12]*rate_eval.p_C12__N13
       -rho*Y[jp]*Y[jna23]*rate_eval.p_Na23__Mg24
       -rho*Y[jp]*Y[jal27]*rate_eval.p_Al27__Si28
       -rho*Y[jp]*Y[jo16]*rate_eval.p_O16__He4_N13
       -rho*Y[jp]*Y[jna23]*rate_eval.p_Na23__n_Mg23
       -rho*Y[jp]*Y[jna23]*rate_eval.p_Na23__He4_Ne20
       -rho*Y[jp]*Y[jna23]*rate_eval.p_Na23__C12_C12
       -rho*Y[jp]*Y[jal27]*rate_eval.p_Al27__n_Si27
       -rho*Y[jp]*Y[jal27]*rate_eval.p_Al27__He4_Mg24
       -rho*Y[jp]*Y[jal27]*rate_eval.p_Al27__C12_O16
       +Y[jn]*rate_eval.n__p__weak__wc12
       +Y[jn13]*rate_eval.N13__p_C12
       +Y[jmg24]*rate_eval.Mg24__p_Na23
       +Y[jsi28]*rate_eval.Si28__p_Al27
       +5.00000000000000e-01*rho*Y[jc12]**2*rate_eval.C12_C12__p_Na23
       +rho*Y[jhe4]*Y[jn13]*rate_eval.He4_N13__p_O16
       +rho*Y[jc12]*Y[jo16]*rate_eval.C12_O16__p_Al27
       +rho*Y[jhe4]*Y[jne20]*rate_eval.He4_Ne20__p_Na23
       +rho*Y[jn]*Y[jmg23]*rate_eval.n_Mg23__p_Na23
       +rho*Y[jhe4]*Y[jmg24]*rate_eval.He4_Mg24__p_Al27
       +rho*Y[jn]*Y[jsi27]*rate_eval.n_Si27__p_Al27
       )

    dYdt[jhe4] = (
       -rho*Y[jhe4]*Y[jc12]*rate_eval.He4_C12__O16
       -rho*Y[jhe4]*Y[jo16]*rate_eval.He4_O16__Ne20
       -rho*Y[jhe4]*Y[jne20]*rate_eval.He4_Ne20__Mg24
       -rho*Y[jhe4]*Y[jna23]*rate_eval.He4_Na23__Al27
       -rho*Y[jhe4]*Y[jmg23]*rate_eval.He4_Mg23__Si27
       -rho*Y[jhe4]*Y[jmg24]*rate_eval.He4_Mg24__Si28
       -rho*Y[jhe4]*Y[jn13]*rate_eval.He4_N13__p_O16
       -rho*Y[jhe4]*Y[jne20]*rate_eval.He4_Ne20__n_Mg23
       -rho*Y[jhe4]*Y[jne20]*rate_eval.He4_Ne20__p_Na23
       -rho*Y[jhe4]*Y[jne20]*rate_eval.He4_Ne20__C12_C12
       -rho*Y[jhe4]*Y[jne21]*rate_eval.He4_Ne21__n_Mg24
       -rho*Y[jhe4]*Y[jmg24]*rate_eval.He4_Mg24__n_Si27
       -rho*Y[jhe4]*Y[jmg24]*rate_eval.He4_Mg24__p_Al27
       -rho*Y[jhe4]*Y[jmg24]*rate_eval.He4_Mg24__C12_O16
       -rho*Y[jhe4]*Y[jsi28]*rate_eval.He4_Si28__C12_Ne20
       -rho*Y[jhe4]*Y[jsi28]*rate_eval.He4_Si28__O16_O16
       -3*1.66666666666667e-01*rho**2*Y[jhe4]**3*rate_eval.He4_He4_He4__C12
       +Y[jo16]*rate_eval.O16__He4_C12
       +Y[jne20]*rate_eval.Ne20__He4_O16
       +Y[jmg24]*rate_eval.Mg24__He4_Ne20
       +Y[jal27]*rate_eval.Al27__He4_Na23
       +Y[jsi27]*rate_eval.Si27__He4_Mg23
       +Y[jsi28]*rate_eval.Si28__He4_Mg24
       +3*Y[jc12]*rate_eval.C12__He4_He4_He4
       +5.00000000000000e-01*rho*Y[jc12]**2*rate_eval.C12_C12__He4_Ne20
       +rho*Y[jp]*Y[jo16]*rate_eval.p_O16__He4_N13
       +rho*Y[jc12]*Y[jo16]*rate_eval.C12_O16__He4_Mg24
       +5.00000000000000e-01*rho*Y[jo16]**2*rate_eval.O16_O16__He4_Si28
       +rho*Y[jc12]*Y[jne20]*rate_eval.C12_Ne20__He4_Si28
       +rho*Y[jp]*Y[jna23]*rate_eval.p_Na23__He4_Ne20
       +rho*Y[jn]*Y[jmg23]*rate_eval.n_Mg23__He4_Ne20
       +rho*Y[jn]*Y[jmg24]*rate_eval.n_Mg24__He4_Ne21
       +rho*Y[jp]*Y[jal27]*rate_eval.p_Al27__He4_Mg24
       +rho*Y[jn]*Y[jsi27]*rate_eval.n_Si27__He4_Mg24
       )

    dYdt[jc12] = (
       -Y[jc12]*rate_eval.C12__He4_He4_He4
       -rho*Y[jp]*Y[jc12]*rate_eval.p_C12__N13
       -rho*Y[jhe4]*Y[jc12]*rate_eval.He4_C12__O16
       -2*5.00000000000000e-01*rho*Y[jc12]**2*rate_eval.C12_C12__n_Mg23
       -2*5.00000000000000e-01*rho*Y[jc12]**2*rate_eval.C12_C12__p_Na23
       -2*5.00000000000000e-01*rho*Y[jc12]**2*rate_eval.C12_C12__He4_Ne20
       -rho*Y[jc12]*Y[jo16]*rate_eval.C12_O16__n_Si27
       -rho*Y[jc12]*Y[jo16]*rate_eval.C12_O16__p_Al27
       -rho*Y[jc12]*Y[jo16]*rate_eval.C12_O16__He4_Mg24
       -rho*Y[jc12]*Y[jne20]*rate_eval.C12_Ne20__He4_Si28
       +Y[jn13]*rate_eval.N13__p_C12
       +Y[jo16]*rate_eval.O16__He4_C12
       +2*rho*Y[jhe4]*Y[jne20]*rate_eval.He4_Ne20__C12_C12
       +2*rho*Y[jp]*Y[jna23]*rate_eval.p_Na23__C12_C12
       +2*rho*Y[jn]*Y[jmg23]*rate_eval.n_Mg23__C12_C12
       +rho*Y[jhe4]*Y[jmg24]*rate_eval.He4_Mg24__C12_O16
       +rho*Y[jp]*Y[jal27]*rate_eval.p_Al27__C12_O16
       +rho*Y[jn]*Y[jsi27]*rate_eval.n_Si27__C12_O16
       +rho*Y[jhe4]*Y[jsi28]*rate_eval.He4_Si28__C12_Ne20
       +1.66666666666667e-01*rho**2*Y[jhe4]**3*rate_eval.He4_He4_He4__C12
       )

    dYdt[jn13] = (
       -Y[jn13]*rate_eval.N13__p_C12
       -rho*Y[jn]*Y[jn13]*rate_eval.n_N13__N14
       -rho*Y[jhe4]*Y[jn13]*rate_eval.He4_N13__p_O16
       +Y[jn14]*rate_eval.N14__n_N13
       +rho*Y[jp]*Y[jc12]*rate_eval.p_C12__N13
       +rho*Y[jp]*Y[jo16]*rate_eval.p_O16__He4_N13
       )

    dYdt[jn14] = (
       -Y[jn14]*rate_eval.N14__n_N13
       +rho*Y[jn]*Y[jn13]*rate_eval.n_N13__N14
       )

    dYdt[jo16] = (
       -Y[jo16]*rate_eval.O16__He4_C12
       -rho*Y[jhe4]*Y[jo16]*rate_eval.He4_O16__Ne20
       -rho*Y[jp]*Y[jo16]*rate_eval.p_O16__He4_N13
       -rho*Y[jc12]*Y[jo16]*rate_eval.C12_O16__n_Si27
       -rho*Y[jc12]*Y[jo16]*rate_eval.C12_O16__p_Al27
       -rho*Y[jc12]*Y[jo16]*rate_eval.C12_O16__He4_Mg24
       -2*5.00000000000000e-01*rho*Y[jo16]**2*rate_eval.O16_O16__He4_Si28
       +Y[jne20]*rate_eval.Ne20__He4_O16
       +rho*Y[jhe4]*Y[jc12]*rate_eval.He4_C12__O16
       +rho*Y[jhe4]*Y[jn13]*rate_eval.He4_N13__p_O16
       +rho*Y[jhe4]*Y[jmg24]*rate_eval.He4_Mg24__C12_O16
       +rho*Y[jp]*Y[jal27]*rate_eval.p_Al27__C12_O16
       +rho*Y[jn]*Y[jsi27]*rate_eval.n_Si27__C12_O16
       +2*rho*Y[jhe4]*Y[jsi28]*rate_eval.He4_Si28__O16_O16
       )

    dYdt[jne20] = (
       -Y[jne20]*rate_eval.Ne20__He4_O16
       -rho*Y[jn]*Y[jne20]*rate_eval.n_Ne20__Ne21
       -rho*Y[jhe4]*Y[jne20]*rate_eval.He4_Ne20__Mg24
       -rho*Y[jhe4]*Y[jne20]*rate_eval.He4_Ne20__n_Mg23
       -rho*Y[jhe4]*Y[jne20]*rate_eval.He4_Ne20__p_Na23
       -rho*Y[jhe4]*Y[jne20]*rate_eval.He4_Ne20__C12_C12
       -rho*Y[jc12]*Y[jne20]*rate_eval.C12_Ne20__He4_Si28
       +Y[jne21]*rate_eval.Ne21__n_Ne20
       +Y[jmg24]*rate_eval.Mg24__He4_Ne20
       +rho*Y[jhe4]*Y[jo16]*rate_eval.He4_O16__Ne20
       +5.00000000000000e-01*rho*Y[jc12]**2*rate_eval.C12_C12__He4_Ne20
       +rho*Y[jp]*Y[jna23]*rate_eval.p_Na23__He4_Ne20
       +rho*Y[jn]*Y[jmg23]*rate_eval.n_Mg23__He4_Ne20
       +rho*Y[jhe4]*Y[jsi28]*rate_eval.He4_Si28__C12_Ne20
       )

    dYdt[jne21] = (
       -Y[jne21]*rate_eval.Ne21__n_Ne20
       -rho*Y[jhe4]*Y[jne21]*rate_eval.He4_Ne21__n_Mg24
       +rho*Y[jn]*Y[jne20]*rate_eval.n_Ne20__Ne21
       +rho*Y[jn]*Y[jmg24]*rate_eval.n_Mg24__He4_Ne21
       )

    dYdt[jna23] = (
       -rho*Y[jp]*Y[jna23]*rate_eval.p_Na23__Mg24
       -rho*Y[jhe4]*Y[jna23]*rate_eval.He4_Na23__Al27
       -rho*Y[jp]*Y[jna23]*rate_eval.p_Na23__n_Mg23
       -rho*Y[jp]*Y[jna23]*rate_eval.p_Na23__He4_Ne20
       -rho*Y[jp]*Y[jna23]*rate_eval.p_Na23__C12_C12
       +Y[jmg23]*rate_eval.Mg23__Na23__weak__wc12
       +Y[jmg24]*rate_eval.Mg24__p_Na23
       +Y[jal27]*rate_eval.Al27__He4_Na23
       +5.00000000000000e-01*rho*Y[jc12]**2*rate_eval.C12_C12__p_Na23
       +rho*Y[jhe4]*Y[jne20]*rate_eval.He4_Ne20__p_Na23
       +rho*Y[jn]*Y[jmg23]*rate_eval.n_Mg23__p_Na23
       )

    dYdt[jmg23] = (
       -Y[jmg23]*rate_eval.Mg23__Na23__weak__wc12
       -rho*Y[jn]*Y[jmg23]*rate_eval.n_Mg23__Mg24
       -rho*Y[jhe4]*Y[jmg23]*rate_eval.He4_Mg23__Si27
       -rho*Y[jn]*Y[jmg23]*rate_eval.n_Mg23__p_Na23
       -rho*Y[jn]*Y[jmg23]*rate_eval.n_Mg23__He4_Ne20
       -rho*Y[jn]*Y[jmg23]*rate_eval.n_Mg23__C12_C12
       +Y[jmg24]*rate_eval.Mg24__n_Mg23
       +Y[jsi27]*rate_eval.Si27__He4_Mg23
       +5.00000000000000e-01*rho*Y[jc12]**2*rate_eval.C12_C12__n_Mg23
       +rho*Y[jhe4]*Y[jne20]*rate_eval.He4_Ne20__n_Mg23
       +rho*Y[jp]*Y[jna23]*rate_eval.p_Na23__n_Mg23
       )

    dYdt[jmg24] = (
       -Y[jmg24]*rate_eval.Mg24__n_Mg23
       -Y[jmg24]*rate_eval.Mg24__p_Na23
       -Y[jmg24]*rate_eval.Mg24__He4_Ne20
       -rho*Y[jhe4]*Y[jmg24]*rate_eval.He4_Mg24__Si28
       -rho*Y[jn]*Y[jmg24]*rate_eval.n_Mg24__He4_Ne21
       -rho*Y[jhe4]*Y[jmg24]*rate_eval.He4_Mg24__n_Si27
       -rho*Y[jhe4]*Y[jmg24]*rate_eval.He4_Mg24__p_Al27
       -rho*Y[jhe4]*Y[jmg24]*rate_eval.He4_Mg24__C12_O16
       +Y[jsi28]*rate_eval.Si28__He4_Mg24
       +rho*Y[jhe4]*Y[jne20]*rate_eval.He4_Ne20__Mg24
       +rho*Y[jp]*Y[jna23]*rate_eval.p_Na23__Mg24
       +rho*Y[jn]*Y[jmg23]*rate_eval.n_Mg23__Mg24
       +rho*Y[jc12]*Y[jo16]*rate_eval.C12_O16__He4_Mg24
       +rho*Y[jhe4]*Y[jne21]*rate_eval.He4_Ne21__n_Mg24
       +rho*Y[jp]*Y[jal27]*rate_eval.p_Al27__He4_Mg24
       +rho*Y[jn]*Y[jsi27]*rate_eval.n_Si27__He4_Mg24
       )

    dYdt[jal27] = (
       -Y[jal27]*rate_eval.Al27__He4_Na23
       -rho*Y[jp]*Y[jal27]*rate_eval.p_Al27__Si28
       -rho*Y[jp]*Y[jal27]*rate_eval.p_Al27__n_Si27
       -rho*Y[jp]*Y[jal27]*rate_eval.p_Al27__He4_Mg24
       -rho*Y[jp]*Y[jal27]*rate_eval.p_Al27__C12_O16
       +Y[jsi27]*rate_eval.Si27__Al27__weak__wc12
       +Y[jsi28]*rate_eval.Si28__p_Al27
       +rho*Y[jhe4]*Y[jna23]*rate_eval.He4_Na23__Al27
       +rho*Y[jc12]*Y[jo16]*rate_eval.C12_O16__p_Al27
       +rho*Y[jhe4]*Y[jmg24]*rate_eval.He4_Mg24__p_Al27
       +rho*Y[jn]*Y[jsi27]*rate_eval.n_Si27__p_Al27
       )

    dYdt[jsi27] = (
       -Y[jsi27]*rate_eval.Si27__Al27__weak__wc12
       -Y[jsi27]*rate_eval.Si27__He4_Mg23
       -rho*Y[jn]*Y[jsi27]*rate_eval.n_Si27__Si28
       -rho*Y[jn]*Y[jsi27]*rate_eval.n_Si27__p_Al27
       -rho*Y[jn]*Y[jsi27]*rate_eval.n_Si27__He4_Mg24
       -rho*Y[jn]*Y[jsi27]*rate_eval.n_Si27__C12_O16
       +Y[jsi28]*rate_eval.Si28__n_Si27
       +rho*Y[jhe4]*Y[jmg23]*rate_eval.He4_Mg23__Si27
       +rho*Y[jc12]*Y[jo16]*rate_eval.C12_O16__n_Si27
       +rho*Y[jhe4]*Y[jmg24]*rate_eval.He4_Mg24__n_Si27
       +rho*Y[jp]*Y[jal27]*rate_eval.p_Al27__n_Si27
       )

    dYdt[jsi28] = (
       -Y[jsi28]*rate_eval.Si28__n_Si27
       -Y[jsi28]*rate_eval.Si28__p_Al27
       -Y[jsi28]*rate_eval.Si28__He4_Mg24
       -rho*Y[jhe4]*Y[jsi28]*rate_eval.He4_Si28__C12_Ne20
       -rho*Y[jhe4]*Y[jsi28]*rate_eval.He4_Si28__O16_O16
       +rho*Y[jhe4]*Y[jmg24]*rate_eval.He4_Mg24__Si28
       +rho*Y[jp]*Y[jal27]*rate_eval.p_Al27__Si28
       +rho*Y[jn]*Y[jsi27]*rate_eval.n_Si27__Si28
       +5.00000000000000e-01*rho*Y[jo16]**2*rate_eval.O16_O16__He4_Si28
       +rho*Y[jc12]*Y[jne20]*rate_eval.C12_Ne20__He4_Si28
       )

    return dYdt

def jacobian(t, Y, rho, T, screen_func=None):
    return jacobian_eq(t, Y, rho, T, screen_func)

@numba.njit()
def jacobian_eq(t, Y, rho, T, screen_func):

    tf = Tfactors(T)
    rate_eval = RateEval()

    # reaclib rates
    n__p__weak__wc12(rate_eval, tf)
    Mg23__Na23__weak__wc12(rate_eval, tf)
    Si27__Al27__weak__wc12(rate_eval, tf)
    N13__p_C12(rate_eval, tf)
    N14__n_N13(rate_eval, tf)
    O16__He4_C12(rate_eval, tf)
    Ne20__He4_O16(rate_eval, tf)
    Ne21__n_Ne20(rate_eval, tf)
    Mg24__n_Mg23(rate_eval, tf)
    Mg24__p_Na23(rate_eval, tf)
    Mg24__He4_Ne20(rate_eval, tf)
    Al27__He4_Na23(rate_eval, tf)
    Si27__He4_Mg23(rate_eval, tf)
    Si28__n_Si27(rate_eval, tf)
    Si28__p_Al27(rate_eval, tf)
    Si28__He4_Mg24(rate_eval, tf)
    C12__He4_He4_He4(rate_eval, tf)
    p_C12__N13(rate_eval, tf)
    He4_C12__O16(rate_eval, tf)
    n_N13__N14(rate_eval, tf)
    He4_O16__Ne20(rate_eval, tf)
    n_Ne20__Ne21(rate_eval, tf)
    He4_Ne20__Mg24(rate_eval, tf)
    p_Na23__Mg24(rate_eval, tf)
    He4_Na23__Al27(rate_eval, tf)
    n_Mg23__Mg24(rate_eval, tf)
    He4_Mg23__Si27(rate_eval, tf)
    He4_Mg24__Si28(rate_eval, tf)
    p_Al27__Si28(rate_eval, tf)
    n_Si27__Si28(rate_eval, tf)
    C12_C12__n_Mg23(rate_eval, tf)
    C12_C12__p_Na23(rate_eval, tf)
    C12_C12__He4_Ne20(rate_eval, tf)
    He4_N13__p_O16(rate_eval, tf)
    p_O16__He4_N13(rate_eval, tf)
    C12_O16__n_Si27(rate_eval, tf)
    C12_O16__p_Al27(rate_eval, tf)
    C12_O16__He4_Mg24(rate_eval, tf)
    O16_O16__He4_Si28(rate_eval, tf)
    He4_Ne20__n_Mg23(rate_eval, tf)
    He4_Ne20__p_Na23(rate_eval, tf)
    He4_Ne20__C12_C12(rate_eval, tf)
    C12_Ne20__He4_Si28(rate_eval, tf)
    He4_Ne21__n_Mg24(rate_eval, tf)
    p_Na23__n_Mg23(rate_eval, tf)
    p_Na23__He4_Ne20(rate_eval, tf)
    p_Na23__C12_C12(rate_eval, tf)
    n_Mg23__p_Na23(rate_eval, tf)
    n_Mg23__He4_Ne20(rate_eval, tf)
    n_Mg23__C12_C12(rate_eval, tf)
    n_Mg24__He4_Ne21(rate_eval, tf)
    He4_Mg24__n_Si27(rate_eval, tf)
    He4_Mg24__p_Al27(rate_eval, tf)
    He4_Mg24__C12_O16(rate_eval, tf)
    p_Al27__n_Si27(rate_eval, tf)
    p_Al27__He4_Mg24(rate_eval, tf)
    p_Al27__C12_O16(rate_eval, tf)
    n_Si27__p_Al27(rate_eval, tf)
    n_Si27__He4_Mg24(rate_eval, tf)
    n_Si27__C12_O16(rate_eval, tf)
    He4_Si28__C12_Ne20(rate_eval, tf)
    He4_Si28__O16_O16(rate_eval, tf)
    He4_He4_He4__C12(rate_eval, tf)

    if screen_func is not None:
        plasma_state = PlasmaState(T, rho, Y, Z)

        scn_fac = ScreenFactors(1, 1, 6, 12)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_C12__N13 *= scor

        scn_fac = ScreenFactors(2, 4, 6, 12)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_C12__O16 *= scor

        scn_fac = ScreenFactors(2, 4, 8, 16)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_O16__Ne20 *= scor

        scn_fac = ScreenFactors(2, 4, 10, 20)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Ne20__Mg24 *= scor
        rate_eval.He4_Ne20__n_Mg23 *= scor
        rate_eval.He4_Ne20__p_Na23 *= scor
        rate_eval.He4_Ne20__C12_C12 *= scor

        scn_fac = ScreenFactors(1, 1, 11, 23)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_Na23__Mg24 *= scor
        rate_eval.p_Na23__n_Mg23 *= scor
        rate_eval.p_Na23__He4_Ne20 *= scor
        rate_eval.p_Na23__C12_C12 *= scor

        scn_fac = ScreenFactors(2, 4, 11, 23)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Na23__Al27 *= scor

        scn_fac = ScreenFactors(2, 4, 12, 23)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Mg23__Si27 *= scor

        scn_fac = ScreenFactors(2, 4, 12, 24)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Mg24__Si28 *= scor
        rate_eval.He4_Mg24__n_Si27 *= scor
        rate_eval.He4_Mg24__p_Al27 *= scor
        rate_eval.He4_Mg24__C12_O16 *= scor

        scn_fac = ScreenFactors(1, 1, 13, 27)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_Al27__Si28 *= scor
        rate_eval.p_Al27__n_Si27 *= scor
        rate_eval.p_Al27__He4_Mg24 *= scor
        rate_eval.p_Al27__C12_O16 *= scor

        scn_fac = ScreenFactors(6, 12, 6, 12)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.C12_C12__n_Mg23 *= scor
        rate_eval.C12_C12__p_Na23 *= scor
        rate_eval.C12_C12__He4_Ne20 *= scor

        scn_fac = ScreenFactors(2, 4, 7, 13)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_N13__p_O16 *= scor

        scn_fac = ScreenFactors(1, 1, 8, 16)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_O16__He4_N13 *= scor

        scn_fac = ScreenFactors(6, 12, 8, 16)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.C12_O16__n_Si27 *= scor
        rate_eval.C12_O16__p_Al27 *= scor
        rate_eval.C12_O16__He4_Mg24 *= scor

        scn_fac = ScreenFactors(8, 16, 8, 16)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.O16_O16__He4_Si28 *= scor

        scn_fac = ScreenFactors(6, 12, 10, 20)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.C12_Ne20__He4_Si28 *= scor

        scn_fac = ScreenFactors(2, 4, 10, 21)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Ne21__n_Mg24 *= scor

        scn_fac = ScreenFactors(2, 4, 14, 28)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Si28__C12_Ne20 *= scor
        rate_eval.He4_Si28__O16_O16 *= scor

        scn_fac = ScreenFactors(2, 4, 2, 4)
        scor = screen_func(plasma_state, scn_fac)
        scn_fac2 = ScreenFactors(2, 4, 4, 8)
        scor2 = screen_func(plasma_state, scn_fac2)
        rate_eval.He4_He4_He4__C12 *= scor * scor2

    jac = np.zeros((nnuc, nnuc), dtype=np.float64)

    jac[jn, jn] = (
       -rate_eval.n__p__weak__wc12
       -rho*Y[jn13]*rate_eval.n_N13__N14
       -rho*Y[jne20]*rate_eval.n_Ne20__Ne21
       -rho*Y[jmg23]*rate_eval.n_Mg23__Mg24
       -rho*Y[jsi27]*rate_eval.n_Si27__Si28
       -rho*Y[jmg23]*rate_eval.n_Mg23__p_Na23
       -rho*Y[jmg23]*rate_eval.n_Mg23__He4_Ne20
       -rho*Y[jmg23]*rate_eval.n_Mg23__C12_C12
       -rho*Y[jmg24]*rate_eval.n_Mg24__He4_Ne21
       -rho*Y[jsi27]*rate_eval.n_Si27__p_Al27
       -rho*Y[jsi27]*rate_eval.n_Si27__He4_Mg24
       -rho*Y[jsi27]*rate_eval.n_Si27__C12_O16
       )

    jac[jn, jp] = (
       +rho*Y[jna23]*rate_eval.p_Na23__n_Mg23
       +rho*Y[jal27]*rate_eval.p_Al27__n_Si27
       )

    jac[jn, jhe4] = (
       +rho*Y[jne20]*rate_eval.He4_Ne20__n_Mg23
       +rho*Y[jne21]*rate_eval.He4_Ne21__n_Mg24
       +rho*Y[jmg24]*rate_eval.He4_Mg24__n_Si27
       )

    jac[jn, jc12] = (
       +5.00000000000000e-01*rho*2*Y[jc12]*rate_eval.C12_C12__n_Mg23
       +rho*Y[jo16]*rate_eval.C12_O16__n_Si27
       )

    jac[jn, jn13] = (
       -rho*Y[jn]*rate_eval.n_N13__N14
       )

    jac[jn, jn14] = (
       +rate_eval.N14__n_N13
       )

    jac[jn, jo16] = (
       +rho*Y[jc12]*rate_eval.C12_O16__n_Si27
       )

    jac[jn, jne20] = (
       -rho*Y[jn]*rate_eval.n_Ne20__Ne21
       +rho*Y[jhe4]*rate_eval.He4_Ne20__n_Mg23
       )

    jac[jn, jne21] = (
       +rate_eval.Ne21__n_Ne20
       +rho*Y[jhe4]*rate_eval.He4_Ne21__n_Mg24
       )

    jac[jn, jna23] = (
       +rho*Y[jp]*rate_eval.p_Na23__n_Mg23
       )

    jac[jn, jmg23] = (
       -rho*Y[jn]*rate_eval.n_Mg23__Mg24
       -rho*Y[jn]*rate_eval.n_Mg23__p_Na23
       -rho*Y[jn]*rate_eval.n_Mg23__He4_Ne20
       -rho*Y[jn]*rate_eval.n_Mg23__C12_C12
       )

    jac[jn, jmg24] = (
       -rho*Y[jn]*rate_eval.n_Mg24__He4_Ne21
       +rate_eval.Mg24__n_Mg23
       +rho*Y[jhe4]*rate_eval.He4_Mg24__n_Si27
       )

    jac[jn, jal27] = (
       +rho*Y[jp]*rate_eval.p_Al27__n_Si27
       )

    jac[jn, jsi27] = (
       -rho*Y[jn]*rate_eval.n_Si27__Si28
       -rho*Y[jn]*rate_eval.n_Si27__p_Al27
       -rho*Y[jn]*rate_eval.n_Si27__He4_Mg24
       -rho*Y[jn]*rate_eval.n_Si27__C12_O16
       )

    jac[jn, jsi28] = (
       +rate_eval.Si28__n_Si27
       )

    jac[jp, jn] = (
       +rate_eval.n__p__weak__wc12
       +rho*Y[jmg23]*rate_eval.n_Mg23__p_Na23
       +rho*Y[jsi27]*rate_eval.n_Si27__p_Al27
       )

    jac[jp, jp] = (
       -rho*Y[jc12]*rate_eval.p_C12__N13
       -rho*Y[jna23]*rate_eval.p_Na23__Mg24
       -rho*Y[jal27]*rate_eval.p_Al27__Si28
       -rho*Y[jo16]*rate_eval.p_O16__He4_N13
       -rho*Y[jna23]*rate_eval.p_Na23__n_Mg23
       -rho*Y[jna23]*rate_eval.p_Na23__He4_Ne20
       -rho*Y[jna23]*rate_eval.p_Na23__C12_C12
       -rho*Y[jal27]*rate_eval.p_Al27__n_Si27
       -rho*Y[jal27]*rate_eval.p_Al27__He4_Mg24
       -rho*Y[jal27]*rate_eval.p_Al27__C12_O16
       )

    jac[jp, jhe4] = (
       +rho*Y[jn13]*rate_eval.He4_N13__p_O16
       +rho*Y[jne20]*rate_eval.He4_Ne20__p_Na23
       +rho*Y[jmg24]*rate_eval.He4_Mg24__p_Al27
       )

    jac[jp, jc12] = (
       -rho*Y[jp]*rate_eval.p_C12__N13
       +5.00000000000000e-01*rho*2*Y[jc12]*rate_eval.C12_C12__p_Na23
       +rho*Y[jo16]*rate_eval.C12_O16__p_Al27
       )

    jac[jp, jn13] = (
       +rate_eval.N13__p_C12
       +rho*Y[jhe4]*rate_eval.He4_N13__p_O16
       )

    jac[jp, jo16] = (
       -rho*Y[jp]*rate_eval.p_O16__He4_N13
       +rho*Y[jc12]*rate_eval.C12_O16__p_Al27
       )

    jac[jp, jne20] = (
       +rho*Y[jhe4]*rate_eval.He4_Ne20__p_Na23
       )

    jac[jp, jna23] = (
       -rho*Y[jp]*rate_eval.p_Na23__Mg24
       -rho*Y[jp]*rate_eval.p_Na23__n_Mg23
       -rho*Y[jp]*rate_eval.p_Na23__He4_Ne20
       -rho*Y[jp]*rate_eval.p_Na23__C12_C12
       )

    jac[jp, jmg23] = (
       +rho*Y[jn]*rate_eval.n_Mg23__p_Na23
       )

    jac[jp, jmg24] = (
       +rate_eval.Mg24__p_Na23
       +rho*Y[jhe4]*rate_eval.He4_Mg24__p_Al27
       )

    jac[jp, jal27] = (
       -rho*Y[jp]*rate_eval.p_Al27__Si28
       -rho*Y[jp]*rate_eval.p_Al27__n_Si27
       -rho*Y[jp]*rate_eval.p_Al27__He4_Mg24
       -rho*Y[jp]*rate_eval.p_Al27__C12_O16
       )

    jac[jp, jsi27] = (
       +rho*Y[jn]*rate_eval.n_Si27__p_Al27
       )

    jac[jp, jsi28] = (
       +rate_eval.Si28__p_Al27
       )

    jac[jhe4, jn] = (
       +rho*Y[jmg23]*rate_eval.n_Mg23__He4_Ne20
       +rho*Y[jmg24]*rate_eval.n_Mg24__He4_Ne21
       +rho*Y[jsi27]*rate_eval.n_Si27__He4_Mg24
       )

    jac[jhe4, jp] = (
       +rho*Y[jo16]*rate_eval.p_O16__He4_N13
       +rho*Y[jna23]*rate_eval.p_Na23__He4_Ne20
       +rho*Y[jal27]*rate_eval.p_Al27__He4_Mg24
       )

    jac[jhe4, jhe4] = (
       -rho*Y[jc12]*rate_eval.He4_C12__O16
       -rho*Y[jo16]*rate_eval.He4_O16__Ne20
       -rho*Y[jne20]*rate_eval.He4_Ne20__Mg24
       -rho*Y[jna23]*rate_eval.He4_Na23__Al27
       -rho*Y[jmg23]*rate_eval.He4_Mg23__Si27
       -rho*Y[jmg24]*rate_eval.He4_Mg24__Si28
       -rho*Y[jn13]*rate_eval.He4_N13__p_O16
       -rho*Y[jne20]*rate_eval.He4_Ne20__n_Mg23
       -rho*Y[jne20]*rate_eval.He4_Ne20__p_Na23
       -rho*Y[jne20]*rate_eval.He4_Ne20__C12_C12
       -rho*Y[jne21]*rate_eval.He4_Ne21__n_Mg24
       -rho*Y[jmg24]*rate_eval.He4_Mg24__n_Si27
       -rho*Y[jmg24]*rate_eval.He4_Mg24__p_Al27
       -rho*Y[jmg24]*rate_eval.He4_Mg24__C12_O16
       -rho*Y[jsi28]*rate_eval.He4_Si28__C12_Ne20
       -rho*Y[jsi28]*rate_eval.He4_Si28__O16_O16
       -3*1.66666666666667e-01*rho**2*3*Y[jhe4]**2*rate_eval.He4_He4_He4__C12
       )

    jac[jhe4, jc12] = (
       -rho*Y[jhe4]*rate_eval.He4_C12__O16
       +3*rate_eval.C12__He4_He4_He4
       +5.00000000000000e-01*rho*2*Y[jc12]*rate_eval.C12_C12__He4_Ne20
       +rho*Y[jo16]*rate_eval.C12_O16__He4_Mg24
       +rho*Y[jne20]*rate_eval.C12_Ne20__He4_Si28
       )

    jac[jhe4, jn13] = (
       -rho*Y[jhe4]*rate_eval.He4_N13__p_O16
       )

    jac[jhe4, jo16] = (
       -rho*Y[jhe4]*rate_eval.He4_O16__Ne20
       +rate_eval.O16__He4_C12
       +rho*Y[jp]*rate_eval.p_O16__He4_N13
       +rho*Y[jc12]*rate_eval.C12_O16__He4_Mg24
       +5.00000000000000e-01*rho*2*Y[jo16]*rate_eval.O16_O16__He4_Si28
       )

    jac[jhe4, jne20] = (
       -rho*Y[jhe4]*rate_eval.He4_Ne20__Mg24
       -rho*Y[jhe4]*rate_eval.He4_Ne20__n_Mg23
       -rho*Y[jhe4]*rate_eval.He4_Ne20__p_Na23
       -rho*Y[jhe4]*rate_eval.He4_Ne20__C12_C12
       +rate_eval.Ne20__He4_O16
       +rho*Y[jc12]*rate_eval.C12_Ne20__He4_Si28
       )

    jac[jhe4, jne21] = (
       -rho*Y[jhe4]*rate_eval.He4_Ne21__n_Mg24
       )

    jac[jhe4, jna23] = (
       -rho*Y[jhe4]*rate_eval.He4_Na23__Al27
       +rho*Y[jp]*rate_eval.p_Na23__He4_Ne20
       )

    jac[jhe4, jmg23] = (
       -rho*Y[jhe4]*rate_eval.He4_Mg23__Si27
       +rho*Y[jn]*rate_eval.n_Mg23__He4_Ne20
       )

    jac[jhe4, jmg24] = (
       -rho*Y[jhe4]*rate_eval.He4_Mg24__Si28
       -rho*Y[jhe4]*rate_eval.He4_Mg24__n_Si27
       -rho*Y[jhe4]*rate_eval.He4_Mg24__p_Al27
       -rho*Y[jhe4]*rate_eval.He4_Mg24__C12_O16
       +rate_eval.Mg24__He4_Ne20
       +rho*Y[jn]*rate_eval.n_Mg24__He4_Ne21
       )

    jac[jhe4, jal27] = (
       +rate_eval.Al27__He4_Na23
       +rho*Y[jp]*rate_eval.p_Al27__He4_Mg24
       )

    jac[jhe4, jsi27] = (
       +rate_eval.Si27__He4_Mg23
       +rho*Y[jn]*rate_eval.n_Si27__He4_Mg24
       )

    jac[jhe4, jsi28] = (
       -rho*Y[jhe4]*rate_eval.He4_Si28__C12_Ne20
       -rho*Y[jhe4]*rate_eval.He4_Si28__O16_O16
       +rate_eval.Si28__He4_Mg24
       )

    jac[jc12, jn] = (
       +2*rho*Y[jmg23]*rate_eval.n_Mg23__C12_C12
       +rho*Y[jsi27]*rate_eval.n_Si27__C12_O16
       )

    jac[jc12, jp] = (
       -rho*Y[jc12]*rate_eval.p_C12__N13
       +2*rho*Y[jna23]*rate_eval.p_Na23__C12_C12
       +rho*Y[jal27]*rate_eval.p_Al27__C12_O16
       )

    jac[jc12, jhe4] = (
       -rho*Y[jc12]*rate_eval.He4_C12__O16
       +2*rho*Y[jne20]*rate_eval.He4_Ne20__C12_C12
       +rho*Y[jmg24]*rate_eval.He4_Mg24__C12_O16
       +rho*Y[jsi28]*rate_eval.He4_Si28__C12_Ne20
       +1.66666666666667e-01*rho**2*3*Y[jhe4]**2*rate_eval.He4_He4_He4__C12
       )

    jac[jc12, jc12] = (
       -rate_eval.C12__He4_He4_He4
       -rho*Y[jp]*rate_eval.p_C12__N13
       -rho*Y[jhe4]*rate_eval.He4_C12__O16
       -2*5.00000000000000e-01*rho*2*Y[jc12]*rate_eval.C12_C12__n_Mg23
       -2*5.00000000000000e-01*rho*2*Y[jc12]*rate_eval.C12_C12__p_Na23
       -2*5.00000000000000e-01*rho*2*Y[jc12]*rate_eval.C12_C12__He4_Ne20
       -rho*Y[jo16]*rate_eval.C12_O16__n_Si27
       -rho*Y[jo16]*rate_eval.C12_O16__p_Al27
       -rho*Y[jo16]*rate_eval.C12_O16__He4_Mg24
       -rho*Y[jne20]*rate_eval.C12_Ne20__He4_Si28
       )

    jac[jc12, jn13] = (
       +rate_eval.N13__p_C12
       )

    jac[jc12, jo16] = (
       -rho*Y[jc12]*rate_eval.C12_O16__n_Si27
       -rho*Y[jc12]*rate_eval.C12_O16__p_Al27
       -rho*Y[jc12]*rate_eval.C12_O16__He4_Mg24
       +rate_eval.O16__He4_C12
       )

    jac[jc12, jne20] = (
       -rho*Y[jc12]*rate_eval.C12_Ne20__He4_Si28
       +2*rho*Y[jhe4]*rate_eval.He4_Ne20__C12_C12
       )

    jac[jc12, jna23] = (
       +2*rho*Y[jp]*rate_eval.p_Na23__C12_C12
       )

    jac[jc12, jmg23] = (
       +2*rho*Y[jn]*rate_eval.n_Mg23__C12_C12
       )

    jac[jc12, jmg24] = (
       +rho*Y[jhe4]*rate_eval.He4_Mg24__C12_O16
       )

    jac[jc12, jal27] = (
       +rho*Y[jp]*rate_eval.p_Al27__C12_O16
       )

    jac[jc12, jsi27] = (
       +rho*Y[jn]*rate_eval.n_Si27__C12_O16
       )

    jac[jc12, jsi28] = (
       +rho*Y[jhe4]*rate_eval.He4_Si28__C12_Ne20
       )

    jac[jn13, jn] = (
       -rho*Y[jn13]*rate_eval.n_N13__N14
       )

    jac[jn13, jp] = (
       +rho*Y[jc12]*rate_eval.p_C12__N13
       +rho*Y[jo16]*rate_eval.p_O16__He4_N13
       )

    jac[jn13, jhe4] = (
       -rho*Y[jn13]*rate_eval.He4_N13__p_O16
       )

    jac[jn13, jc12] = (
       +rho*Y[jp]*rate_eval.p_C12__N13
       )

    jac[jn13, jn13] = (
       -rate_eval.N13__p_C12
       -rho*Y[jn]*rate_eval.n_N13__N14
       -rho*Y[jhe4]*rate_eval.He4_N13__p_O16
       )

    jac[jn13, jn14] = (
       +rate_eval.N14__n_N13
       )

    jac[jn13, jo16] = (
       +rho*Y[jp]*rate_eval.p_O16__He4_N13
       )

    jac[jn14, jn] = (
       +rho*Y[jn13]*rate_eval.n_N13__N14
       )

    jac[jn14, jn13] = (
       +rho*Y[jn]*rate_eval.n_N13__N14
       )

    jac[jn14, jn14] = (
       -rate_eval.N14__n_N13
       )

    jac[jo16, jn] = (
       +rho*Y[jsi27]*rate_eval.n_Si27__C12_O16
       )

    jac[jo16, jp] = (
       -rho*Y[jo16]*rate_eval.p_O16__He4_N13
       +rho*Y[jal27]*rate_eval.p_Al27__C12_O16
       )

    jac[jo16, jhe4] = (
       -rho*Y[jo16]*rate_eval.He4_O16__Ne20
       +rho*Y[jc12]*rate_eval.He4_C12__O16
       +rho*Y[jn13]*rate_eval.He4_N13__p_O16
       +rho*Y[jmg24]*rate_eval.He4_Mg24__C12_O16
       +2*rho*Y[jsi28]*rate_eval.He4_Si28__O16_O16
       )

    jac[jo16, jc12] = (
       -rho*Y[jo16]*rate_eval.C12_O16__n_Si27
       -rho*Y[jo16]*rate_eval.C12_O16__p_Al27
       -rho*Y[jo16]*rate_eval.C12_O16__He4_Mg24
       +rho*Y[jhe4]*rate_eval.He4_C12__O16
       )

    jac[jo16, jn13] = (
       +rho*Y[jhe4]*rate_eval.He4_N13__p_O16
       )

    jac[jo16, jo16] = (
       -rate_eval.O16__He4_C12
       -rho*Y[jhe4]*rate_eval.He4_O16__Ne20
       -rho*Y[jp]*rate_eval.p_O16__He4_N13
       -rho*Y[jc12]*rate_eval.C12_O16__n_Si27
       -rho*Y[jc12]*rate_eval.C12_O16__p_Al27
       -rho*Y[jc12]*rate_eval.C12_O16__He4_Mg24
       -2*5.00000000000000e-01*rho*2*Y[jo16]*rate_eval.O16_O16__He4_Si28
       )

    jac[jo16, jne20] = (
       +rate_eval.Ne20__He4_O16
       )

    jac[jo16, jmg24] = (
       +rho*Y[jhe4]*rate_eval.He4_Mg24__C12_O16
       )

    jac[jo16, jal27] = (
       +rho*Y[jp]*rate_eval.p_Al27__C12_O16
       )

    jac[jo16, jsi27] = (
       +rho*Y[jn]*rate_eval.n_Si27__C12_O16
       )

    jac[jo16, jsi28] = (
       +2*rho*Y[jhe4]*rate_eval.He4_Si28__O16_O16
       )

    jac[jne20, jn] = (
       -rho*Y[jne20]*rate_eval.n_Ne20__Ne21
       +rho*Y[jmg23]*rate_eval.n_Mg23__He4_Ne20
       )

    jac[jne20, jp] = (
       +rho*Y[jna23]*rate_eval.p_Na23__He4_Ne20
       )

    jac[jne20, jhe4] = (
       -rho*Y[jne20]*rate_eval.He4_Ne20__Mg24
       -rho*Y[jne20]*rate_eval.He4_Ne20__n_Mg23
       -rho*Y[jne20]*rate_eval.He4_Ne20__p_Na23
       -rho*Y[jne20]*rate_eval.He4_Ne20__C12_C12
       +rho*Y[jo16]*rate_eval.He4_O16__Ne20
       +rho*Y[jsi28]*rate_eval.He4_Si28__C12_Ne20
       )

    jac[jne20, jc12] = (
       -rho*Y[jne20]*rate_eval.C12_Ne20__He4_Si28
       +5.00000000000000e-01*rho*2*Y[jc12]*rate_eval.C12_C12__He4_Ne20
       )

    jac[jne20, jo16] = (
       +rho*Y[jhe4]*rate_eval.He4_O16__Ne20
       )

    jac[jne20, jne20] = (
       -rate_eval.Ne20__He4_O16
       -rho*Y[jn]*rate_eval.n_Ne20__Ne21
       -rho*Y[jhe4]*rate_eval.He4_Ne20__Mg24
       -rho*Y[jhe4]*rate_eval.He4_Ne20__n_Mg23
       -rho*Y[jhe4]*rate_eval.He4_Ne20__p_Na23
       -rho*Y[jhe4]*rate_eval.He4_Ne20__C12_C12
       -rho*Y[jc12]*rate_eval.C12_Ne20__He4_Si28
       )

    jac[jne20, jne21] = (
       +rate_eval.Ne21__n_Ne20
       )

    jac[jne20, jna23] = (
       +rho*Y[jp]*rate_eval.p_Na23__He4_Ne20
       )

    jac[jne20, jmg23] = (
       +rho*Y[jn]*rate_eval.n_Mg23__He4_Ne20
       )

    jac[jne20, jmg24] = (
       +rate_eval.Mg24__He4_Ne20
       )

    jac[jne20, jsi28] = (
       +rho*Y[jhe4]*rate_eval.He4_Si28__C12_Ne20
       )

    jac[jne21, jn] = (
       +rho*Y[jne20]*rate_eval.n_Ne20__Ne21
       +rho*Y[jmg24]*rate_eval.n_Mg24__He4_Ne21
       )

    jac[jne21, jhe4] = (
       -rho*Y[jne21]*rate_eval.He4_Ne21__n_Mg24
       )

    jac[jne21, jne20] = (
       +rho*Y[jn]*rate_eval.n_Ne20__Ne21
       )

    jac[jne21, jne21] = (
       -rate_eval.Ne21__n_Ne20
       -rho*Y[jhe4]*rate_eval.He4_Ne21__n_Mg24
       )

    jac[jne21, jmg24] = (
       +rho*Y[jn]*rate_eval.n_Mg24__He4_Ne21
       )

    jac[jna23, jn] = (
       +rho*Y[jmg23]*rate_eval.n_Mg23__p_Na23
       )

    jac[jna23, jp] = (
       -rho*Y[jna23]*rate_eval.p_Na23__Mg24
       -rho*Y[jna23]*rate_eval.p_Na23__n_Mg23
       -rho*Y[jna23]*rate_eval.p_Na23__He4_Ne20
       -rho*Y[jna23]*rate_eval.p_Na23__C12_C12
       )

    jac[jna23, jhe4] = (
       -rho*Y[jna23]*rate_eval.He4_Na23__Al27
       +rho*Y[jne20]*rate_eval.He4_Ne20__p_Na23
       )

    jac[jna23, jc12] = (
       +5.00000000000000e-01*rho*2*Y[jc12]*rate_eval.C12_C12__p_Na23
       )

    jac[jna23, jne20] = (
       +rho*Y[jhe4]*rate_eval.He4_Ne20__p_Na23
       )

    jac[jna23, jna23] = (
       -rho*Y[jp]*rate_eval.p_Na23__Mg24
       -rho*Y[jhe4]*rate_eval.He4_Na23__Al27
       -rho*Y[jp]*rate_eval.p_Na23__n_Mg23
       -rho*Y[jp]*rate_eval.p_Na23__He4_Ne20
       -rho*Y[jp]*rate_eval.p_Na23__C12_C12
       )

    jac[jna23, jmg23] = (
       +rate_eval.Mg23__Na23__weak__wc12
       +rho*Y[jn]*rate_eval.n_Mg23__p_Na23
       )

    jac[jna23, jmg24] = (
       +rate_eval.Mg24__p_Na23
       )

    jac[jna23, jal27] = (
       +rate_eval.Al27__He4_Na23
       )

    jac[jmg23, jn] = (
       -rho*Y[jmg23]*rate_eval.n_Mg23__Mg24
       -rho*Y[jmg23]*rate_eval.n_Mg23__p_Na23
       -rho*Y[jmg23]*rate_eval.n_Mg23__He4_Ne20
       -rho*Y[jmg23]*rate_eval.n_Mg23__C12_C12
       )

    jac[jmg23, jp] = (
       +rho*Y[jna23]*rate_eval.p_Na23__n_Mg23
       )

    jac[jmg23, jhe4] = (
       -rho*Y[jmg23]*rate_eval.He4_Mg23__Si27
       +rho*Y[jne20]*rate_eval.He4_Ne20__n_Mg23
       )

    jac[jmg23, jc12] = (
       +5.00000000000000e-01*rho*2*Y[jc12]*rate_eval.C12_C12__n_Mg23
       )

    jac[jmg23, jne20] = (
       +rho*Y[jhe4]*rate_eval.He4_Ne20__n_Mg23
       )

    jac[jmg23, jna23] = (
       +rho*Y[jp]*rate_eval.p_Na23__n_Mg23
       )

    jac[jmg23, jmg23] = (
       -rate_eval.Mg23__Na23__weak__wc12
       -rho*Y[jn]*rate_eval.n_Mg23__Mg24
       -rho*Y[jhe4]*rate_eval.He4_Mg23__Si27
       -rho*Y[jn]*rate_eval.n_Mg23__p_Na23
       -rho*Y[jn]*rate_eval.n_Mg23__He4_Ne20
       -rho*Y[jn]*rate_eval.n_Mg23__C12_C12
       )

    jac[jmg23, jmg24] = (
       +rate_eval.Mg24__n_Mg23
       )

    jac[jmg23, jsi27] = (
       +rate_eval.Si27__He4_Mg23
       )

    jac[jmg24, jn] = (
       -rho*Y[jmg24]*rate_eval.n_Mg24__He4_Ne21
       +rho*Y[jmg23]*rate_eval.n_Mg23__Mg24
       +rho*Y[jsi27]*rate_eval.n_Si27__He4_Mg24
       )

    jac[jmg24, jp] = (
       +rho*Y[jna23]*rate_eval.p_Na23__Mg24
       +rho*Y[jal27]*rate_eval.p_Al27__He4_Mg24
       )

    jac[jmg24, jhe4] = (
       -rho*Y[jmg24]*rate_eval.He4_Mg24__Si28
       -rho*Y[jmg24]*rate_eval.He4_Mg24__n_Si27
       -rho*Y[jmg24]*rate_eval.He4_Mg24__p_Al27
       -rho*Y[jmg24]*rate_eval.He4_Mg24__C12_O16
       +rho*Y[jne20]*rate_eval.He4_Ne20__Mg24
       +rho*Y[jne21]*rate_eval.He4_Ne21__n_Mg24
       )

    jac[jmg24, jc12] = (
       +rho*Y[jo16]*rate_eval.C12_O16__He4_Mg24
       )

    jac[jmg24, jo16] = (
       +rho*Y[jc12]*rate_eval.C12_O16__He4_Mg24
       )

    jac[jmg24, jne20] = (
       +rho*Y[jhe4]*rate_eval.He4_Ne20__Mg24
       )

    jac[jmg24, jne21] = (
       +rho*Y[jhe4]*rate_eval.He4_Ne21__n_Mg24
       )

    jac[jmg24, jna23] = (
       +rho*Y[jp]*rate_eval.p_Na23__Mg24
       )

    jac[jmg24, jmg23] = (
       +rho*Y[jn]*rate_eval.n_Mg23__Mg24
       )

    jac[jmg24, jmg24] = (
       -rate_eval.Mg24__n_Mg23
       -rate_eval.Mg24__p_Na23
       -rate_eval.Mg24__He4_Ne20
       -rho*Y[jhe4]*rate_eval.He4_Mg24__Si28
       -rho*Y[jn]*rate_eval.n_Mg24__He4_Ne21
       -rho*Y[jhe4]*rate_eval.He4_Mg24__n_Si27
       -rho*Y[jhe4]*rate_eval.He4_Mg24__p_Al27
       -rho*Y[jhe4]*rate_eval.He4_Mg24__C12_O16
       )

    jac[jmg24, jal27] = (
       +rho*Y[jp]*rate_eval.p_Al27__He4_Mg24
       )

    jac[jmg24, jsi27] = (
       +rho*Y[jn]*rate_eval.n_Si27__He4_Mg24
       )

    jac[jmg24, jsi28] = (
       +rate_eval.Si28__He4_Mg24
       )

    jac[jal27, jn] = (
       +rho*Y[jsi27]*rate_eval.n_Si27__p_Al27
       )

    jac[jal27, jp] = (
       -rho*Y[jal27]*rate_eval.p_Al27__Si28
       -rho*Y[jal27]*rate_eval.p_Al27__n_Si27
       -rho*Y[jal27]*rate_eval.p_Al27__He4_Mg24
       -rho*Y[jal27]*rate_eval.p_Al27__C12_O16
       )

    jac[jal27, jhe4] = (
       +rho*Y[jna23]*rate_eval.He4_Na23__Al27
       +rho*Y[jmg24]*rate_eval.He4_Mg24__p_Al27
       )

    jac[jal27, jc12] = (
       +rho*Y[jo16]*rate_eval.C12_O16__p_Al27
       )

    jac[jal27, jo16] = (
       +rho*Y[jc12]*rate_eval.C12_O16__p_Al27
       )

    jac[jal27, jna23] = (
       +rho*Y[jhe4]*rate_eval.He4_Na23__Al27
       )

    jac[jal27, jmg24] = (
       +rho*Y[jhe4]*rate_eval.He4_Mg24__p_Al27
       )

    jac[jal27, jal27] = (
       -rate_eval.Al27__He4_Na23
       -rho*Y[jp]*rate_eval.p_Al27__Si28
       -rho*Y[jp]*rate_eval.p_Al27__n_Si27
       -rho*Y[jp]*rate_eval.p_Al27__He4_Mg24
       -rho*Y[jp]*rate_eval.p_Al27__C12_O16
       )

    jac[jal27, jsi27] = (
       +rate_eval.Si27__Al27__weak__wc12
       +rho*Y[jn]*rate_eval.n_Si27__p_Al27
       )

    jac[jal27, jsi28] = (
       +rate_eval.Si28__p_Al27
       )

    jac[jsi27, jn] = (
       -rho*Y[jsi27]*rate_eval.n_Si27__Si28
       -rho*Y[jsi27]*rate_eval.n_Si27__p_Al27
       -rho*Y[jsi27]*rate_eval.n_Si27__He4_Mg24
       -rho*Y[jsi27]*rate_eval.n_Si27__C12_O16
       )

    jac[jsi27, jp] = (
       +rho*Y[jal27]*rate_eval.p_Al27__n_Si27
       )

    jac[jsi27, jhe4] = (
       +rho*Y[jmg23]*rate_eval.He4_Mg23__Si27
       +rho*Y[jmg24]*rate_eval.He4_Mg24__n_Si27
       )

    jac[jsi27, jc12] = (
       +rho*Y[jo16]*rate_eval.C12_O16__n_Si27
       )

    jac[jsi27, jo16] = (
       +rho*Y[jc12]*rate_eval.C12_O16__n_Si27
       )

    jac[jsi27, jmg23] = (
       +rho*Y[jhe4]*rate_eval.He4_Mg23__Si27
       )

    jac[jsi27, jmg24] = (
       +rho*Y[jhe4]*rate_eval.He4_Mg24__n_Si27
       )

    jac[jsi27, jal27] = (
       +rho*Y[jp]*rate_eval.p_Al27__n_Si27
       )

    jac[jsi27, jsi27] = (
       -rate_eval.Si27__Al27__weak__wc12
       -rate_eval.Si27__He4_Mg23
       -rho*Y[jn]*rate_eval.n_Si27__Si28
       -rho*Y[jn]*rate_eval.n_Si27__p_Al27
       -rho*Y[jn]*rate_eval.n_Si27__He4_Mg24
       -rho*Y[jn]*rate_eval.n_Si27__C12_O16
       )

    jac[jsi27, jsi28] = (
       +rate_eval.Si28__n_Si27
       )

    jac[jsi28, jn] = (
       +rho*Y[jsi27]*rate_eval.n_Si27__Si28
       )

    jac[jsi28, jp] = (
       +rho*Y[jal27]*rate_eval.p_Al27__Si28
       )

    jac[jsi28, jhe4] = (
       -rho*Y[jsi28]*rate_eval.He4_Si28__C12_Ne20
       -rho*Y[jsi28]*rate_eval.He4_Si28__O16_O16
       +rho*Y[jmg24]*rate_eval.He4_Mg24__Si28
       )

    jac[jsi28, jc12] = (
       +rho*Y[jne20]*rate_eval.C12_Ne20__He4_Si28
       )

    jac[jsi28, jo16] = (
       +5.00000000000000e-01*rho*2*Y[jo16]*rate_eval.O16_O16__He4_Si28
       )

    jac[jsi28, jne20] = (
       +rho*Y[jc12]*rate_eval.C12_Ne20__He4_Si28
       )

    jac[jsi28, jmg24] = (
       +rho*Y[jhe4]*rate_eval.He4_Mg24__Si28
       )

    jac[jsi28, jal27] = (
       +rho*Y[jp]*rate_eval.p_Al27__Si28
       )

    jac[jsi28, jsi27] = (
       +rho*Y[jn]*rate_eval.n_Si27__Si28
       )

    jac[jsi28, jsi28] = (
       -rate_eval.Si28__n_Si27
       -rate_eval.Si28__p_Al27
       -rate_eval.Si28__He4_Mg24
       -rho*Y[jhe4]*rate_eval.He4_Si28__C12_Ne20
       -rho*Y[jhe4]*rate_eval.He4_Si28__O16_O16
       )

    return jac
