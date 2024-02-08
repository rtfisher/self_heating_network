import numba
import numpy as np
from scipy import constants
from numba.experimental import jitclass

from pynucastro.rates import TableIndex, TableInterpolator, TabularRate, Tfactors
from pynucastro.screening import PlasmaState, ScreenFactors

jn = 0
jp = 1
jhe4 = 2
jb11 = 3
jc12 = 4
jc13 = 5
jn13 = 6
jn14 = 7
jn15 = 8
jo15 = 9
jo16 = 10
jo17 = 11
jf18 = 12
jne19 = 13
jne20 = 14
jne21 = 15
jne22 = 16
jna22 = 17
jna23 = 18
jmg23 = 19
jmg24 = 20
jmg25 = 21
jmg26 = 22
jal25 = 23
jal26 = 24
jal27 = 25
jsi26 = 26
jsi28 = 27
jsi29 = 28
jsi30 = 29
jp29 = 30
jp30 = 31
jp31 = 32
js30 = 33
js31 = 34
js32 = 35
js33 = 36
jcl33 = 37
jcl34 = 38
jcl35 = 39
jar34 = 40
jar36 = 41
jar37 = 42
jar38 = 43
jar39 = 44
jk39 = 45
jca40 = 46
jsc43 = 47
jti44 = 48
jv47 = 49
jcr48 = 50
jmn51 = 51
jfe52 = 52
jfe55 = 53
jco55 = 54
jni56 = 55
jni58 = 56
jni59 = 57
nnuc = 58

A = np.zeros((nnuc), dtype=np.int32)

A[jn] = 1
A[jp] = 1
A[jhe4] = 4
A[jb11] = 11
A[jc12] = 12
A[jc13] = 13
A[jn13] = 13
A[jn14] = 14
A[jn15] = 15
A[jo15] = 15
A[jo16] = 16
A[jo17] = 17
A[jf18] = 18
A[jne19] = 19
A[jne20] = 20
A[jne21] = 21
A[jne22] = 22
A[jna22] = 22
A[jna23] = 23
A[jmg23] = 23
A[jmg24] = 24
A[jmg25] = 25
A[jmg26] = 26
A[jal25] = 25
A[jal26] = 26
A[jal27] = 27
A[jsi26] = 26
A[jsi28] = 28
A[jsi29] = 29
A[jsi30] = 30
A[jp29] = 29
A[jp30] = 30
A[jp31] = 31
A[js30] = 30
A[js31] = 31
A[js32] = 32
A[js33] = 33
A[jcl33] = 33
A[jcl34] = 34
A[jcl35] = 35
A[jar34] = 34
A[jar36] = 36
A[jar37] = 37
A[jar38] = 38
A[jar39] = 39
A[jk39] = 39
A[jca40] = 40
A[jsc43] = 43
A[jti44] = 44
A[jv47] = 47
A[jcr48] = 48
A[jmn51] = 51
A[jfe52] = 52
A[jfe55] = 55
A[jco55] = 55
A[jni56] = 56
A[jni58] = 58
A[jni59] = 59

Z = np.zeros((nnuc), dtype=np.int32)

Z[jn] = 0
Z[jp] = 1
Z[jhe4] = 2
Z[jb11] = 5
Z[jc12] = 6
Z[jc13] = 6
Z[jn13] = 7
Z[jn14] = 7
Z[jn15] = 7
Z[jo15] = 8
Z[jo16] = 8
Z[jo17] = 8
Z[jf18] = 9
Z[jne19] = 10
Z[jne20] = 10
Z[jne21] = 10
Z[jne22] = 10
Z[jna22] = 11
Z[jna23] = 11
Z[jmg23] = 12
Z[jmg24] = 12
Z[jmg25] = 12
Z[jmg26] = 12
Z[jal25] = 13
Z[jal26] = 13
Z[jal27] = 13
Z[jsi26] = 14
Z[jsi28] = 14
Z[jsi29] = 14
Z[jsi30] = 14
Z[jp29] = 15
Z[jp30] = 15
Z[jp31] = 15
Z[js30] = 16
Z[js31] = 16
Z[js32] = 16
Z[js33] = 16
Z[jcl33] = 17
Z[jcl34] = 17
Z[jcl35] = 17
Z[jar34] = 18
Z[jar36] = 18
Z[jar37] = 18
Z[jar38] = 18
Z[jar39] = 18
Z[jk39] = 19
Z[jca40] = 20
Z[jsc43] = 21
Z[jti44] = 22
Z[jv47] = 23
Z[jcr48] = 24
Z[jmn51] = 25
Z[jfe52] = 26
Z[jfe55] = 26
Z[jco55] = 27
Z[jni56] = 28
Z[jni58] = 28
Z[jni59] = 28

# masses in ergs
mass = np.zeros((nnuc), dtype=np.float64)

mass[jn] = 0.0015053497659156634
mass[jp] = 0.0015040963030260536
mass[jhe4] = 0.0059735574925878256
mass[jb11] = 0.01643048614409968
mass[jc12] = 0.017909017027273523
mass[jc13] = 0.019406441930882663
mass[jn13] = 0.01940999951603316
mass[jn14] = 0.020898440903103103
mass[jn15] = 0.0223864338056853
mass[jo15] = 0.02239084645968795
mass[jo16] = 0.023871099858982767
mass[jo17] = 0.02536981167252093
mass[jf18] = 0.026864924401329426
mass[jne19] = 0.02835875072008801
mass[jne20] = 0.02983707929641827
mass[jne21] = 0.03133159647374143
mass[jne22] = 0.0328203408644564
mass[jna22] = 0.03282489638134515
mass[jna23] = 0.034310347465945384
mass[jmg23] = 0.03431684618276469
mass[jmg24] = 0.03579570996619953
mass[jmg25] = 0.03728931494425613
mass[jmg26] = 0.038776891732727296
mass[jal25] = 0.03729616718134972
mass[jal26] = 0.038783307488840485
mass[jal27] = 0.04026773584000819
mass[jsi26] = 0.0387914290824159
mass[jsi28] = 0.041753271135012315
mass[jsi29] = 0.043245044664958585
mass[jsi30] = 0.04473339658648528
mass[jp29] = 0.0432529631025368
mass[jp30] = 0.04474017715821803
mass[jp31] = 0.046225802655766646
mass[js30] = 0.044750017086233405
mass[js31] = 0.04623445120523698
mass[js32] = 0.047715697313174224
mass[js33] = 0.049207201517228315
mass[jcl33] = 0.04921614582850528
mass[jcl34] = 0.05070305755345568
mass[jcl35] = 0.05218814824444387
mass[jar34] = 0.05071276962777567
mass[jar36] = 0.053678614878909785
mass[jar37] = 0.05516988548561064
mass[jar38] = 0.056656268011618834
mass[jar39] = 0.05815104578297151
mass[jk39] = 0.05815014023273798
mass[jca40] = 0.059640893336386044
mass[jsc43] = 0.06411599795281459
mass[jti44] = 0.06560623627711017
mass[jv47] = 0.07007634722940585
mass[jcr48] = 0.07156746130344957
mass[jmn51] = 0.07603602823812308
mass[jfe52] = 0.07752830389022643
mass[jfe55] = 0.08199089935236736
mass[jco55] = 0.08199642910480195
mass[jni56] = 0.08348904329682807
mass[jni58] = 0.08646375162888383
mass[jni59] = 0.08795468305479126

names = []
names.append("n")
names.append("H1")
names.append("He4")
names.append("B11")
names.append("C12")
names.append("C13")
names.append("N13")
names.append("N14")
names.append("N15")
names.append("O15")
names.append("O16")
names.append("O17")
names.append("F18")
names.append("Ne19")
names.append("Ne20")
names.append("Ne21")
names.append("Ne22")
names.append("Na22")
names.append("Na23")
names.append("Mg23")
names.append("Mg24")
names.append("Mg25")
names.append("Mg26")
names.append("Al25")
names.append("Al26")
names.append("Al27")
names.append("Si26")
names.append("Si28")
names.append("Si29")
names.append("Si30")
names.append("P29")
names.append("P30")
names.append("P31")
names.append("S30")
names.append("S31")
names.append("S32")
names.append("S33")
names.append("Cl33")
names.append("Cl34")
names.append("Cl35")
names.append("Ar34")
names.append("Ar36")
names.append("Ar37")
names.append("Ar38")
names.append("Ar39")
names.append("K39")
names.append("Ca40")
names.append("Sc43")
names.append("Ti44")
names.append("V47")
names.append("Cr48")
names.append("Mn51")
names.append("Fe52")
names.append("Fe55")
names.append("Co55")
names.append("Ni56")
names.append("Ni58")
names.append("Ni59")

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
    ("N13__C13__weak__wc12", numba.float64),
    ("O15__N15__weak__wc12", numba.float64),
    ("Na22__Ne22__weak__wc12", numba.float64),
    ("Mg23__Na23__weak__wc12", numba.float64),
    ("Al25__Mg25__weak__wc12", numba.float64),
    ("Al26__Mg26__weak__wc12", numba.float64),
    ("Si26__Al26__weak__wc12", numba.float64),
    ("P29__Si29__weak__wc12", numba.float64),
    ("P30__Si30__weak__wc12", numba.float64),
    ("S30__P30__weak__wc12", numba.float64),
    ("S31__P31__weak__wc12", numba.float64),
    ("Cl33__S33__weak__wc12", numba.float64),
    ("Ar34__Cl34__weak__wc12", numba.float64),
    ("Ar39__K39__weak__wc12", numba.float64),
    ("Co55__Fe55__weak__wc12", numba.float64),
    ("C12__p_B11", numba.float64),
    ("C13__n_C12", numba.float64),
    ("N13__p_C12", numba.float64),
    ("N14__n_N13", numba.float64),
    ("N14__p_C13", numba.float64),
    ("N15__n_N14", numba.float64),
    ("O15__p_N14", numba.float64),
    ("O16__n_O15", numba.float64),
    ("O16__p_N15", numba.float64),
    ("O16__He4_C12", numba.float64),
    ("O17__n_O16", numba.float64),
    ("F18__p_O17", numba.float64),
    ("F18__He4_N14", numba.float64),
    ("Ne19__p_F18", numba.float64),
    ("Ne19__He4_O15", numba.float64),
    ("Ne20__n_Ne19", numba.float64),
    ("Ne20__He4_O16", numba.float64),
    ("Ne21__n_Ne20", numba.float64),
    ("Ne21__He4_O17", numba.float64),
    ("Ne22__n_Ne21", numba.float64),
    ("Na22__p_Ne21", numba.float64),
    ("Na22__He4_F18", numba.float64),
    ("Na23__n_Na22", numba.float64),
    ("Na23__p_Ne22", numba.float64),
    ("Mg23__p_Na22", numba.float64),
    ("Mg23__He4_Ne19", numba.float64),
    ("Mg24__n_Mg23", numba.float64),
    ("Mg24__p_Na23", numba.float64),
    ("Mg24__He4_Ne20", numba.float64),
    ("Mg25__n_Mg24", numba.float64),
    ("Mg25__He4_Ne21", numba.float64),
    ("Mg26__n_Mg25", numba.float64),
    ("Mg26__He4_Ne22", numba.float64),
    ("Al25__p_Mg24", numba.float64),
    ("Al26__n_Al25", numba.float64),
    ("Al26__p_Mg25", numba.float64),
    ("Al26__He4_Na22", numba.float64),
    ("Al27__n_Al26", numba.float64),
    ("Al27__p_Mg26", numba.float64),
    ("Al27__He4_Na23", numba.float64),
    ("Si26__p_Al25", numba.float64),
    ("Si28__p_Al27", numba.float64),
    ("Si28__He4_Mg24", numba.float64),
    ("Si29__n_Si28", numba.float64),
    ("Si29__He4_Mg25", numba.float64),
    ("Si30__n_Si29", numba.float64),
    ("Si30__He4_Mg26", numba.float64),
    ("P29__p_Si28", numba.float64),
    ("P29__He4_Al25", numba.float64),
    ("P30__n_P29", numba.float64),
    ("P30__p_Si29", numba.float64),
    ("P30__He4_Al26", numba.float64),
    ("P31__n_P30", numba.float64),
    ("P31__p_Si30", numba.float64),
    ("P31__He4_Al27", numba.float64),
    ("S30__p_P29", numba.float64),
    ("S30__He4_Si26", numba.float64),
    ("S31__n_S30", numba.float64),
    ("S31__p_P30", numba.float64),
    ("S32__n_S31", numba.float64),
    ("S32__p_P31", numba.float64),
    ("S32__He4_Si28", numba.float64),
    ("S33__n_S32", numba.float64),
    ("S33__He4_Si29", numba.float64),
    ("Cl33__p_S32", numba.float64),
    ("Cl33__He4_P29", numba.float64),
    ("Cl34__n_Cl33", numba.float64),
    ("Cl34__p_S33", numba.float64),
    ("Cl34__He4_P30", numba.float64),
    ("Cl35__n_Cl34", numba.float64),
    ("Cl35__He4_P31", numba.float64),
    ("Ar34__p_Cl33", numba.float64),
    ("Ar34__He4_S30", numba.float64),
    ("Ar36__p_Cl35", numba.float64),
    ("Ar36__He4_S32", numba.float64),
    ("Ar37__n_Ar36", numba.float64),
    ("Ar37__He4_S33", numba.float64),
    ("Ar38__n_Ar37", numba.float64),
    ("Ar39__n_Ar38", numba.float64),
    ("K39__p_Ar38", numba.float64),
    ("K39__He4_Cl35", numba.float64),
    ("Ca40__p_K39", numba.float64),
    ("Ca40__He4_Ar36", numba.float64),
    ("Sc43__He4_K39", numba.float64),
    ("Ti44__p_Sc43", numba.float64),
    ("Ti44__He4_Ca40", numba.float64),
    ("V47__He4_Sc43", numba.float64),
    ("Cr48__p_V47", numba.float64),
    ("Cr48__He4_Ti44", numba.float64),
    ("Mn51__He4_V47", numba.float64),
    ("Fe52__p_Mn51", numba.float64),
    ("Fe52__He4_Cr48", numba.float64),
    ("Co55__He4_Mn51", numba.float64),
    ("Ni56__p_Co55", numba.float64),
    ("Ni56__He4_Fe52", numba.float64),
    ("Ni59__n_Ni58", numba.float64),
    ("Ni59__He4_Fe55", numba.float64),
    ("C12__He4_He4_He4", numba.float64),
    ("p_B11__C12", numba.float64),
    ("n_C12__C13", numba.float64),
    ("p_C12__N13", numba.float64),
    ("He4_C12__O16", numba.float64),
    ("p_C13__N14", numba.float64),
    ("n_N13__N14", numba.float64),
    ("n_N14__N15", numba.float64),
    ("p_N14__O15", numba.float64),
    ("He4_N14__F18", numba.float64),
    ("p_N15__O16", numba.float64),
    ("n_O15__O16", numba.float64),
    ("He4_O15__Ne19", numba.float64),
    ("n_O16__O17", numba.float64),
    ("He4_O16__Ne20", numba.float64),
    ("p_O17__F18", numba.float64),
    ("He4_O17__Ne21", numba.float64),
    ("p_F18__Ne19", numba.float64),
    ("He4_F18__Na22", numba.float64),
    ("n_Ne19__Ne20", numba.float64),
    ("He4_Ne19__Mg23", numba.float64),
    ("n_Ne20__Ne21", numba.float64),
    ("He4_Ne20__Mg24", numba.float64),
    ("n_Ne21__Ne22", numba.float64),
    ("p_Ne21__Na22", numba.float64),
    ("He4_Ne21__Mg25", numba.float64),
    ("p_Ne22__Na23", numba.float64),
    ("He4_Ne22__Mg26", numba.float64),
    ("n_Na22__Na23", numba.float64),
    ("p_Na22__Mg23", numba.float64),
    ("He4_Na22__Al26", numba.float64),
    ("p_Na23__Mg24", numba.float64),
    ("He4_Na23__Al27", numba.float64),
    ("n_Mg23__Mg24", numba.float64),
    ("n_Mg24__Mg25", numba.float64),
    ("p_Mg24__Al25", numba.float64),
    ("He4_Mg24__Si28", numba.float64),
    ("n_Mg25__Mg26", numba.float64),
    ("p_Mg25__Al26", numba.float64),
    ("He4_Mg25__Si29", numba.float64),
    ("p_Mg26__Al27", numba.float64),
    ("He4_Mg26__Si30", numba.float64),
    ("n_Al25__Al26", numba.float64),
    ("p_Al25__Si26", numba.float64),
    ("He4_Al25__P29", numba.float64),
    ("n_Al26__Al27", numba.float64),
    ("He4_Al26__P30", numba.float64),
    ("p_Al27__Si28", numba.float64),
    ("He4_Al27__P31", numba.float64),
    ("He4_Si26__S30", numba.float64),
    ("n_Si28__Si29", numba.float64),
    ("p_Si28__P29", numba.float64),
    ("He4_Si28__S32", numba.float64),
    ("n_Si29__Si30", numba.float64),
    ("p_Si29__P30", numba.float64),
    ("He4_Si29__S33", numba.float64),
    ("p_Si30__P31", numba.float64),
    ("n_P29__P30", numba.float64),
    ("p_P29__S30", numba.float64),
    ("He4_P29__Cl33", numba.float64),
    ("n_P30__P31", numba.float64),
    ("p_P30__S31", numba.float64),
    ("He4_P30__Cl34", numba.float64),
    ("p_P31__S32", numba.float64),
    ("He4_P31__Cl35", numba.float64),
    ("n_S30__S31", numba.float64),
    ("He4_S30__Ar34", numba.float64),
    ("n_S31__S32", numba.float64),
    ("n_S32__S33", numba.float64),
    ("p_S32__Cl33", numba.float64),
    ("He4_S32__Ar36", numba.float64),
    ("p_S33__Cl34", numba.float64),
    ("He4_S33__Ar37", numba.float64),
    ("n_Cl33__Cl34", numba.float64),
    ("p_Cl33__Ar34", numba.float64),
    ("n_Cl34__Cl35", numba.float64),
    ("p_Cl35__Ar36", numba.float64),
    ("He4_Cl35__K39", numba.float64),
    ("n_Ar36__Ar37", numba.float64),
    ("He4_Ar36__Ca40", numba.float64),
    ("n_Ar37__Ar38", numba.float64),
    ("n_Ar38__Ar39", numba.float64),
    ("p_Ar38__K39", numba.float64),
    ("p_K39__Ca40", numba.float64),
    ("He4_K39__Sc43", numba.float64),
    ("He4_Ca40__Ti44", numba.float64),
    ("p_Sc43__Ti44", numba.float64),
    ("He4_Sc43__V47", numba.float64),
    ("He4_Ti44__Cr48", numba.float64),
    ("p_V47__Cr48", numba.float64),
    ("He4_V47__Mn51", numba.float64),
    ("He4_Cr48__Fe52", numba.float64),
    ("p_Mn51__Fe52", numba.float64),
    ("He4_Mn51__Co55", numba.float64),
    ("He4_Fe52__Ni56", numba.float64),
    ("He4_Fe55__Ni59", numba.float64),
    ("p_Co55__Ni56", numba.float64),
    ("n_Ni58__Ni59", numba.float64),
    ("He4_B11__n_N14", numba.float64),
    ("He4_C12__n_O15", numba.float64),
    ("He4_C12__p_N15", numba.float64),
    ("C12_C12__n_Mg23", numba.float64),
    ("C12_C12__p_Na23", numba.float64),
    ("C12_C12__He4_Ne20", numba.float64),
    ("p_C13__n_N13", numba.float64),
    ("He4_C13__n_O16", numba.float64),
    ("n_N13__p_C13", numba.float64),
    ("He4_N13__p_O16", numba.float64),
    ("n_N14__He4_B11", numba.float64),
    ("He4_N14__p_O17", numba.float64),
    ("p_N15__n_O15", numba.float64),
    ("p_N15__He4_C12", numba.float64),
    ("He4_N15__n_F18", numba.float64),
    ("n_O15__p_N15", numba.float64),
    ("n_O15__He4_C12", numba.float64),
    ("He4_O15__p_F18", numba.float64),
    ("n_O16__He4_C13", numba.float64),
    ("p_O16__He4_N13", numba.float64),
    ("He4_O16__n_Ne19", numba.float64),
    ("C12_O16__p_Al27", numba.float64),
    ("C12_O16__He4_Mg24", numba.float64),
    ("O16_O16__n_S31", numba.float64),
    ("O16_O16__p_P31", numba.float64),
    ("O16_O16__He4_Si28", numba.float64),
    ("p_O17__He4_N14", numba.float64),
    ("He4_O17__n_Ne20", numba.float64),
    ("n_F18__He4_N15", numba.float64),
    ("p_F18__He4_O15", numba.float64),
    ("He4_F18__p_Ne21", numba.float64),
    ("n_Ne19__He4_O16", numba.float64),
    ("He4_Ne19__p_Na22", numba.float64),
    ("n_Ne20__He4_O17", numba.float64),
    ("He4_Ne20__n_Mg23", numba.float64),
    ("He4_Ne20__p_Na23", numba.float64),
    ("He4_Ne20__C12_C12", numba.float64),
    ("C12_Ne20__n_S31", numba.float64),
    ("C12_Ne20__p_P31", numba.float64),
    ("C12_Ne20__He4_Si28", numba.float64),
    ("p_Ne21__He4_F18", numba.float64),
    ("He4_Ne21__n_Mg24", numba.float64),
    ("p_Ne22__n_Na22", numba.float64),
    ("He4_Ne22__n_Mg25", numba.float64),
    ("n_Na22__p_Ne22", numba.float64),
    ("p_Na22__He4_Ne19", numba.float64),
    ("He4_Na22__n_Al25", numba.float64),
    ("He4_Na22__p_Mg25", numba.float64),
    ("p_Na23__n_Mg23", numba.float64),
    ("p_Na23__He4_Ne20", numba.float64),
    ("p_Na23__C12_C12", numba.float64),
    ("He4_Na23__n_Al26", numba.float64),
    ("He4_Na23__p_Mg26", numba.float64),
    ("n_Mg23__p_Na23", numba.float64),
    ("n_Mg23__He4_Ne20", numba.float64),
    ("n_Mg23__C12_C12", numba.float64),
    ("He4_Mg23__n_Si26", numba.float64),
    ("He4_Mg23__p_Al26", numba.float64),
    ("n_Mg24__He4_Ne21", numba.float64),
    ("He4_Mg24__p_Al27", numba.float64),
    ("He4_Mg24__C12_O16", numba.float64),
    ("n_Mg25__He4_Ne22", numba.float64),
    ("p_Mg25__n_Al25", numba.float64),
    ("p_Mg25__He4_Na22", numba.float64),
    ("He4_Mg25__n_Si28", numba.float64),
    ("p_Mg26__n_Al26", numba.float64),
    ("p_Mg26__He4_Na23", numba.float64),
    ("He4_Mg26__n_Si29", numba.float64),
    ("n_Al25__p_Mg25", numba.float64),
    ("n_Al25__He4_Na22", numba.float64),
    ("He4_Al25__p_Si28", numba.float64),
    ("n_Al26__p_Mg26", numba.float64),
    ("n_Al26__He4_Na23", numba.float64),
    ("p_Al26__n_Si26", numba.float64),
    ("p_Al26__He4_Mg23", numba.float64),
    ("He4_Al26__n_P29", numba.float64),
    ("He4_Al26__p_Si29", numba.float64),
    ("p_Al27__He4_Mg24", numba.float64),
    ("p_Al27__C12_O16", numba.float64),
    ("He4_Al27__n_P30", numba.float64),
    ("He4_Al27__p_Si30", numba.float64),
    ("n_Si26__p_Al26", numba.float64),
    ("n_Si26__He4_Mg23", numba.float64),
    ("He4_Si26__p_P29", numba.float64),
    ("n_Si28__He4_Mg25", numba.float64),
    ("p_Si28__He4_Al25", numba.float64),
    ("He4_Si28__n_S31", numba.float64),
    ("He4_Si28__p_P31", numba.float64),
    ("He4_Si28__C12_Ne20", numba.float64),
    ("He4_Si28__O16_O16", numba.float64),
    ("n_Si29__He4_Mg26", numba.float64),
    ("p_Si29__n_P29", numba.float64),
    ("p_Si29__He4_Al26", numba.float64),
    ("He4_Si29__n_S32", numba.float64),
    ("p_Si30__n_P30", numba.float64),
    ("p_Si30__He4_Al27", numba.float64),
    ("He4_Si30__n_S33", numba.float64),
    ("n_P29__p_Si29", numba.float64),
    ("n_P29__He4_Al26", numba.float64),
    ("p_P29__He4_Si26", numba.float64),
    ("He4_P29__p_S32", numba.float64),
    ("n_P30__p_Si30", numba.float64),
    ("n_P30__He4_Al27", numba.float64),
    ("p_P30__n_S30", numba.float64),
    ("He4_P30__n_Cl33", numba.float64),
    ("He4_P30__p_S33", numba.float64),
    ("p_P31__n_S31", numba.float64),
    ("p_P31__He4_Si28", numba.float64),
    ("p_P31__C12_Ne20", numba.float64),
    ("p_P31__O16_O16", numba.float64),
    ("He4_P31__n_Cl34", numba.float64),
    ("n_S30__p_P30", numba.float64),
    ("He4_S30__p_Cl33", numba.float64),
    ("n_S31__p_P31", numba.float64),
    ("n_S31__He4_Si28", numba.float64),
    ("n_S31__C12_Ne20", numba.float64),
    ("n_S31__O16_O16", numba.float64),
    ("He4_S31__n_Ar34", numba.float64),
    ("He4_S31__p_Cl34", numba.float64),
    ("n_S32__He4_Si29", numba.float64),
    ("p_S32__He4_P29", numba.float64),
    ("He4_S32__p_Cl35", numba.float64),
    ("n_S33__He4_Si30", numba.float64),
    ("p_S33__n_Cl33", numba.float64),
    ("p_S33__He4_P30", numba.float64),
    ("He4_S33__n_Ar36", numba.float64),
    ("n_Cl33__p_S33", numba.float64),
    ("n_Cl33__He4_P30", numba.float64),
    ("p_Cl33__He4_S30", numba.float64),
    ("He4_Cl33__p_Ar36", numba.float64),
    ("n_Cl34__He4_P31", numba.float64),
    ("p_Cl34__n_Ar34", numba.float64),
    ("p_Cl34__He4_S31", numba.float64),
    ("He4_Cl34__p_Ar37", numba.float64),
    ("p_Cl35__He4_S32", numba.float64),
    ("He4_Cl35__p_Ar38", numba.float64),
    ("n_Ar34__p_Cl34", numba.float64),
    ("n_Ar34__He4_S31", numba.float64),
    ("n_Ar36__He4_S33", numba.float64),
    ("p_Ar36__He4_Cl33", numba.float64),
    ("He4_Ar36__p_K39", numba.float64),
    ("p_Ar37__He4_Cl34", numba.float64),
    ("He4_Ar37__n_Ca40", numba.float64),
    ("p_Ar38__He4_Cl35", numba.float64),
    ("p_Ar39__n_K39", numba.float64),
    ("n_K39__p_Ar39", numba.float64),
    ("p_K39__He4_Ar36", numba.float64),
    ("n_Ca40__He4_Ar37", numba.float64),
    ("He4_Ca40__p_Sc43", numba.float64),
    ("p_Sc43__He4_Ca40", numba.float64),
    ("He4_Ti44__p_V47", numba.float64),
    ("p_V47__He4_Ti44", numba.float64),
    ("He4_Cr48__p_Mn51", numba.float64),
    ("p_Mn51__He4_Cr48", numba.float64),
    ("He4_Fe52__p_Co55", numba.float64),
    ("p_Fe55__n_Co55", numba.float64),
    ("He4_Fe55__n_Ni58", numba.float64),
    ("n_Co55__p_Fe55", numba.float64),
    ("p_Co55__He4_Fe52", numba.float64),
    ("He4_Co55__p_Ni58", numba.float64),
    ("n_Ni58__He4_Fe55", numba.float64),
    ("p_Ni58__He4_Co55", numba.float64),
    ("p_B11__He4_He4_He4", numba.float64),
    ("He4_He4_He4__C12", numba.float64),
    ("He4_He4_He4__p_B11", numba.float64),
])
class RateEval:
    def __init__(self):
        self.n__p__weak__wc12 = np.nan
        self.N13__C13__weak__wc12 = np.nan
        self.O15__N15__weak__wc12 = np.nan
        self.Na22__Ne22__weak__wc12 = np.nan
        self.Mg23__Na23__weak__wc12 = np.nan
        self.Al25__Mg25__weak__wc12 = np.nan
        self.Al26__Mg26__weak__wc12 = np.nan
        self.Si26__Al26__weak__wc12 = np.nan
        self.P29__Si29__weak__wc12 = np.nan
        self.P30__Si30__weak__wc12 = np.nan
        self.S30__P30__weak__wc12 = np.nan
        self.S31__P31__weak__wc12 = np.nan
        self.Cl33__S33__weak__wc12 = np.nan
        self.Ar34__Cl34__weak__wc12 = np.nan
        self.Ar39__K39__weak__wc12 = np.nan
        self.Co55__Fe55__weak__wc12 = np.nan
        self.C12__p_B11 = np.nan
        self.C13__n_C12 = np.nan
        self.N13__p_C12 = np.nan
        self.N14__n_N13 = np.nan
        self.N14__p_C13 = np.nan
        self.N15__n_N14 = np.nan
        self.O15__p_N14 = np.nan
        self.O16__n_O15 = np.nan
        self.O16__p_N15 = np.nan
        self.O16__He4_C12 = np.nan
        self.O17__n_O16 = np.nan
        self.F18__p_O17 = np.nan
        self.F18__He4_N14 = np.nan
        self.Ne19__p_F18 = np.nan
        self.Ne19__He4_O15 = np.nan
        self.Ne20__n_Ne19 = np.nan
        self.Ne20__He4_O16 = np.nan
        self.Ne21__n_Ne20 = np.nan
        self.Ne21__He4_O17 = np.nan
        self.Ne22__n_Ne21 = np.nan
        self.Na22__p_Ne21 = np.nan
        self.Na22__He4_F18 = np.nan
        self.Na23__n_Na22 = np.nan
        self.Na23__p_Ne22 = np.nan
        self.Mg23__p_Na22 = np.nan
        self.Mg23__He4_Ne19 = np.nan
        self.Mg24__n_Mg23 = np.nan
        self.Mg24__p_Na23 = np.nan
        self.Mg24__He4_Ne20 = np.nan
        self.Mg25__n_Mg24 = np.nan
        self.Mg25__He4_Ne21 = np.nan
        self.Mg26__n_Mg25 = np.nan
        self.Mg26__He4_Ne22 = np.nan
        self.Al25__p_Mg24 = np.nan
        self.Al26__n_Al25 = np.nan
        self.Al26__p_Mg25 = np.nan
        self.Al26__He4_Na22 = np.nan
        self.Al27__n_Al26 = np.nan
        self.Al27__p_Mg26 = np.nan
        self.Al27__He4_Na23 = np.nan
        self.Si26__p_Al25 = np.nan
        self.Si28__p_Al27 = np.nan
        self.Si28__He4_Mg24 = np.nan
        self.Si29__n_Si28 = np.nan
        self.Si29__He4_Mg25 = np.nan
        self.Si30__n_Si29 = np.nan
        self.Si30__He4_Mg26 = np.nan
        self.P29__p_Si28 = np.nan
        self.P29__He4_Al25 = np.nan
        self.P30__n_P29 = np.nan
        self.P30__p_Si29 = np.nan
        self.P30__He4_Al26 = np.nan
        self.P31__n_P30 = np.nan
        self.P31__p_Si30 = np.nan
        self.P31__He4_Al27 = np.nan
        self.S30__p_P29 = np.nan
        self.S30__He4_Si26 = np.nan
        self.S31__n_S30 = np.nan
        self.S31__p_P30 = np.nan
        self.S32__n_S31 = np.nan
        self.S32__p_P31 = np.nan
        self.S32__He4_Si28 = np.nan
        self.S33__n_S32 = np.nan
        self.S33__He4_Si29 = np.nan
        self.Cl33__p_S32 = np.nan
        self.Cl33__He4_P29 = np.nan
        self.Cl34__n_Cl33 = np.nan
        self.Cl34__p_S33 = np.nan
        self.Cl34__He4_P30 = np.nan
        self.Cl35__n_Cl34 = np.nan
        self.Cl35__He4_P31 = np.nan
        self.Ar34__p_Cl33 = np.nan
        self.Ar34__He4_S30 = np.nan
        self.Ar36__p_Cl35 = np.nan
        self.Ar36__He4_S32 = np.nan
        self.Ar37__n_Ar36 = np.nan
        self.Ar37__He4_S33 = np.nan
        self.Ar38__n_Ar37 = np.nan
        self.Ar39__n_Ar38 = np.nan
        self.K39__p_Ar38 = np.nan
        self.K39__He4_Cl35 = np.nan
        self.Ca40__p_K39 = np.nan
        self.Ca40__He4_Ar36 = np.nan
        self.Sc43__He4_K39 = np.nan
        self.Ti44__p_Sc43 = np.nan
        self.Ti44__He4_Ca40 = np.nan
        self.V47__He4_Sc43 = np.nan
        self.Cr48__p_V47 = np.nan
        self.Cr48__He4_Ti44 = np.nan
        self.Mn51__He4_V47 = np.nan
        self.Fe52__p_Mn51 = np.nan
        self.Fe52__He4_Cr48 = np.nan
        self.Co55__He4_Mn51 = np.nan
        self.Ni56__p_Co55 = np.nan
        self.Ni56__He4_Fe52 = np.nan
        self.Ni59__n_Ni58 = np.nan
        self.Ni59__He4_Fe55 = np.nan
        self.C12__He4_He4_He4 = np.nan
        self.p_B11__C12 = np.nan
        self.n_C12__C13 = np.nan
        self.p_C12__N13 = np.nan
        self.He4_C12__O16 = np.nan
        self.p_C13__N14 = np.nan
        self.n_N13__N14 = np.nan
        self.n_N14__N15 = np.nan
        self.p_N14__O15 = np.nan
        self.He4_N14__F18 = np.nan
        self.p_N15__O16 = np.nan
        self.n_O15__O16 = np.nan
        self.He4_O15__Ne19 = np.nan
        self.n_O16__O17 = np.nan
        self.He4_O16__Ne20 = np.nan
        self.p_O17__F18 = np.nan
        self.He4_O17__Ne21 = np.nan
        self.p_F18__Ne19 = np.nan
        self.He4_F18__Na22 = np.nan
        self.n_Ne19__Ne20 = np.nan
        self.He4_Ne19__Mg23 = np.nan
        self.n_Ne20__Ne21 = np.nan
        self.He4_Ne20__Mg24 = np.nan
        self.n_Ne21__Ne22 = np.nan
        self.p_Ne21__Na22 = np.nan
        self.He4_Ne21__Mg25 = np.nan
        self.p_Ne22__Na23 = np.nan
        self.He4_Ne22__Mg26 = np.nan
        self.n_Na22__Na23 = np.nan
        self.p_Na22__Mg23 = np.nan
        self.He4_Na22__Al26 = np.nan
        self.p_Na23__Mg24 = np.nan
        self.He4_Na23__Al27 = np.nan
        self.n_Mg23__Mg24 = np.nan
        self.n_Mg24__Mg25 = np.nan
        self.p_Mg24__Al25 = np.nan
        self.He4_Mg24__Si28 = np.nan
        self.n_Mg25__Mg26 = np.nan
        self.p_Mg25__Al26 = np.nan
        self.He4_Mg25__Si29 = np.nan
        self.p_Mg26__Al27 = np.nan
        self.He4_Mg26__Si30 = np.nan
        self.n_Al25__Al26 = np.nan
        self.p_Al25__Si26 = np.nan
        self.He4_Al25__P29 = np.nan
        self.n_Al26__Al27 = np.nan
        self.He4_Al26__P30 = np.nan
        self.p_Al27__Si28 = np.nan
        self.He4_Al27__P31 = np.nan
        self.He4_Si26__S30 = np.nan
        self.n_Si28__Si29 = np.nan
        self.p_Si28__P29 = np.nan
        self.He4_Si28__S32 = np.nan
        self.n_Si29__Si30 = np.nan
        self.p_Si29__P30 = np.nan
        self.He4_Si29__S33 = np.nan
        self.p_Si30__P31 = np.nan
        self.n_P29__P30 = np.nan
        self.p_P29__S30 = np.nan
        self.He4_P29__Cl33 = np.nan
        self.n_P30__P31 = np.nan
        self.p_P30__S31 = np.nan
        self.He4_P30__Cl34 = np.nan
        self.p_P31__S32 = np.nan
        self.He4_P31__Cl35 = np.nan
        self.n_S30__S31 = np.nan
        self.He4_S30__Ar34 = np.nan
        self.n_S31__S32 = np.nan
        self.n_S32__S33 = np.nan
        self.p_S32__Cl33 = np.nan
        self.He4_S32__Ar36 = np.nan
        self.p_S33__Cl34 = np.nan
        self.He4_S33__Ar37 = np.nan
        self.n_Cl33__Cl34 = np.nan
        self.p_Cl33__Ar34 = np.nan
        self.n_Cl34__Cl35 = np.nan
        self.p_Cl35__Ar36 = np.nan
        self.He4_Cl35__K39 = np.nan
        self.n_Ar36__Ar37 = np.nan
        self.He4_Ar36__Ca40 = np.nan
        self.n_Ar37__Ar38 = np.nan
        self.n_Ar38__Ar39 = np.nan
        self.p_Ar38__K39 = np.nan
        self.p_K39__Ca40 = np.nan
        self.He4_K39__Sc43 = np.nan
        self.He4_Ca40__Ti44 = np.nan
        self.p_Sc43__Ti44 = np.nan
        self.He4_Sc43__V47 = np.nan
        self.He4_Ti44__Cr48 = np.nan
        self.p_V47__Cr48 = np.nan
        self.He4_V47__Mn51 = np.nan
        self.He4_Cr48__Fe52 = np.nan
        self.p_Mn51__Fe52 = np.nan
        self.He4_Mn51__Co55 = np.nan
        self.He4_Fe52__Ni56 = np.nan
        self.He4_Fe55__Ni59 = np.nan
        self.p_Co55__Ni56 = np.nan
        self.n_Ni58__Ni59 = np.nan
        self.He4_B11__n_N14 = np.nan
        self.He4_C12__n_O15 = np.nan
        self.He4_C12__p_N15 = np.nan
        self.C12_C12__n_Mg23 = np.nan
        self.C12_C12__p_Na23 = np.nan
        self.C12_C12__He4_Ne20 = np.nan
        self.p_C13__n_N13 = np.nan
        self.He4_C13__n_O16 = np.nan
        self.n_N13__p_C13 = np.nan
        self.He4_N13__p_O16 = np.nan
        self.n_N14__He4_B11 = np.nan
        self.He4_N14__p_O17 = np.nan
        self.p_N15__n_O15 = np.nan
        self.p_N15__He4_C12 = np.nan
        self.He4_N15__n_F18 = np.nan
        self.n_O15__p_N15 = np.nan
        self.n_O15__He4_C12 = np.nan
        self.He4_O15__p_F18 = np.nan
        self.n_O16__He4_C13 = np.nan
        self.p_O16__He4_N13 = np.nan
        self.He4_O16__n_Ne19 = np.nan
        self.C12_O16__p_Al27 = np.nan
        self.C12_O16__He4_Mg24 = np.nan
        self.O16_O16__n_S31 = np.nan
        self.O16_O16__p_P31 = np.nan
        self.O16_O16__He4_Si28 = np.nan
        self.p_O17__He4_N14 = np.nan
        self.He4_O17__n_Ne20 = np.nan
        self.n_F18__He4_N15 = np.nan
        self.p_F18__He4_O15 = np.nan
        self.He4_F18__p_Ne21 = np.nan
        self.n_Ne19__He4_O16 = np.nan
        self.He4_Ne19__p_Na22 = np.nan
        self.n_Ne20__He4_O17 = np.nan
        self.He4_Ne20__n_Mg23 = np.nan
        self.He4_Ne20__p_Na23 = np.nan
        self.He4_Ne20__C12_C12 = np.nan
        self.C12_Ne20__n_S31 = np.nan
        self.C12_Ne20__p_P31 = np.nan
        self.C12_Ne20__He4_Si28 = np.nan
        self.p_Ne21__He4_F18 = np.nan
        self.He4_Ne21__n_Mg24 = np.nan
        self.p_Ne22__n_Na22 = np.nan
        self.He4_Ne22__n_Mg25 = np.nan
        self.n_Na22__p_Ne22 = np.nan
        self.p_Na22__He4_Ne19 = np.nan
        self.He4_Na22__n_Al25 = np.nan
        self.He4_Na22__p_Mg25 = np.nan
        self.p_Na23__n_Mg23 = np.nan
        self.p_Na23__He4_Ne20 = np.nan
        self.p_Na23__C12_C12 = np.nan
        self.He4_Na23__n_Al26 = np.nan
        self.He4_Na23__p_Mg26 = np.nan
        self.n_Mg23__p_Na23 = np.nan
        self.n_Mg23__He4_Ne20 = np.nan
        self.n_Mg23__C12_C12 = np.nan
        self.He4_Mg23__n_Si26 = np.nan
        self.He4_Mg23__p_Al26 = np.nan
        self.n_Mg24__He4_Ne21 = np.nan
        self.He4_Mg24__p_Al27 = np.nan
        self.He4_Mg24__C12_O16 = np.nan
        self.n_Mg25__He4_Ne22 = np.nan
        self.p_Mg25__n_Al25 = np.nan
        self.p_Mg25__He4_Na22 = np.nan
        self.He4_Mg25__n_Si28 = np.nan
        self.p_Mg26__n_Al26 = np.nan
        self.p_Mg26__He4_Na23 = np.nan
        self.He4_Mg26__n_Si29 = np.nan
        self.n_Al25__p_Mg25 = np.nan
        self.n_Al25__He4_Na22 = np.nan
        self.He4_Al25__p_Si28 = np.nan
        self.n_Al26__p_Mg26 = np.nan
        self.n_Al26__He4_Na23 = np.nan
        self.p_Al26__n_Si26 = np.nan
        self.p_Al26__He4_Mg23 = np.nan
        self.He4_Al26__n_P29 = np.nan
        self.He4_Al26__p_Si29 = np.nan
        self.p_Al27__He4_Mg24 = np.nan
        self.p_Al27__C12_O16 = np.nan
        self.He4_Al27__n_P30 = np.nan
        self.He4_Al27__p_Si30 = np.nan
        self.n_Si26__p_Al26 = np.nan
        self.n_Si26__He4_Mg23 = np.nan
        self.He4_Si26__p_P29 = np.nan
        self.n_Si28__He4_Mg25 = np.nan
        self.p_Si28__He4_Al25 = np.nan
        self.He4_Si28__n_S31 = np.nan
        self.He4_Si28__p_P31 = np.nan
        self.He4_Si28__C12_Ne20 = np.nan
        self.He4_Si28__O16_O16 = np.nan
        self.n_Si29__He4_Mg26 = np.nan
        self.p_Si29__n_P29 = np.nan
        self.p_Si29__He4_Al26 = np.nan
        self.He4_Si29__n_S32 = np.nan
        self.p_Si30__n_P30 = np.nan
        self.p_Si30__He4_Al27 = np.nan
        self.He4_Si30__n_S33 = np.nan
        self.n_P29__p_Si29 = np.nan
        self.n_P29__He4_Al26 = np.nan
        self.p_P29__He4_Si26 = np.nan
        self.He4_P29__p_S32 = np.nan
        self.n_P30__p_Si30 = np.nan
        self.n_P30__He4_Al27 = np.nan
        self.p_P30__n_S30 = np.nan
        self.He4_P30__n_Cl33 = np.nan
        self.He4_P30__p_S33 = np.nan
        self.p_P31__n_S31 = np.nan
        self.p_P31__He4_Si28 = np.nan
        self.p_P31__C12_Ne20 = np.nan
        self.p_P31__O16_O16 = np.nan
        self.He4_P31__n_Cl34 = np.nan
        self.n_S30__p_P30 = np.nan
        self.He4_S30__p_Cl33 = np.nan
        self.n_S31__p_P31 = np.nan
        self.n_S31__He4_Si28 = np.nan
        self.n_S31__C12_Ne20 = np.nan
        self.n_S31__O16_O16 = np.nan
        self.He4_S31__n_Ar34 = np.nan
        self.He4_S31__p_Cl34 = np.nan
        self.n_S32__He4_Si29 = np.nan
        self.p_S32__He4_P29 = np.nan
        self.He4_S32__p_Cl35 = np.nan
        self.n_S33__He4_Si30 = np.nan
        self.p_S33__n_Cl33 = np.nan
        self.p_S33__He4_P30 = np.nan
        self.He4_S33__n_Ar36 = np.nan
        self.n_Cl33__p_S33 = np.nan
        self.n_Cl33__He4_P30 = np.nan
        self.p_Cl33__He4_S30 = np.nan
        self.He4_Cl33__p_Ar36 = np.nan
        self.n_Cl34__He4_P31 = np.nan
        self.p_Cl34__n_Ar34 = np.nan
        self.p_Cl34__He4_S31 = np.nan
        self.He4_Cl34__p_Ar37 = np.nan
        self.p_Cl35__He4_S32 = np.nan
        self.He4_Cl35__p_Ar38 = np.nan
        self.n_Ar34__p_Cl34 = np.nan
        self.n_Ar34__He4_S31 = np.nan
        self.n_Ar36__He4_S33 = np.nan
        self.p_Ar36__He4_Cl33 = np.nan
        self.He4_Ar36__p_K39 = np.nan
        self.p_Ar37__He4_Cl34 = np.nan
        self.He4_Ar37__n_Ca40 = np.nan
        self.p_Ar38__He4_Cl35 = np.nan
        self.p_Ar39__n_K39 = np.nan
        self.n_K39__p_Ar39 = np.nan
        self.p_K39__He4_Ar36 = np.nan
        self.n_Ca40__He4_Ar37 = np.nan
        self.He4_Ca40__p_Sc43 = np.nan
        self.p_Sc43__He4_Ca40 = np.nan
        self.He4_Ti44__p_V47 = np.nan
        self.p_V47__He4_Ti44 = np.nan
        self.He4_Cr48__p_Mn51 = np.nan
        self.p_Mn51__He4_Cr48 = np.nan
        self.He4_Fe52__p_Co55 = np.nan
        self.p_Fe55__n_Co55 = np.nan
        self.He4_Fe55__n_Ni58 = np.nan
        self.n_Co55__p_Fe55 = np.nan
        self.p_Co55__He4_Fe52 = np.nan
        self.He4_Co55__p_Ni58 = np.nan
        self.n_Ni58__He4_Fe55 = np.nan
        self.p_Ni58__He4_Co55 = np.nan
        self.p_B11__He4_He4_He4 = np.nan
        self.He4_He4_He4__C12 = np.nan
        self.He4_He4_He4__p_B11 = np.nan

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
def N13__C13__weak__wc12(rate_eval, tf):
    # N13 --> C13
    rate = 0.0

    # wc12w
    rate += np.exp(  -6.7601)

    rate_eval.N13__C13__weak__wc12 = rate

@numba.njit()
def O15__N15__weak__wc12(rate_eval, tf):
    # O15 --> N15
    rate = 0.0

    # wc12w
    rate += np.exp(  -5.17053)

    rate_eval.O15__N15__weak__wc12 = rate

@numba.njit()
def Na22__Ne22__weak__wc12(rate_eval, tf):
    # Na22 --> Ne22
    rate = 0.0

    # wc12w
    rate += np.exp(  -18.59)

    rate_eval.Na22__Ne22__weak__wc12 = rate

@numba.njit()
def Mg23__Na23__weak__wc12(rate_eval, tf):
    # Mg23 --> Na23
    rate = 0.0

    # wc12w
    rate += np.exp(  -2.79132)

    rate_eval.Mg23__Na23__weak__wc12 = rate

@numba.njit()
def Al25__Mg25__weak__wc12(rate_eval, tf):
    # Al25 --> Mg25
    rate = 0.0

    # wc12w
    rate += np.exp(  -2.33781)

    rate_eval.Al25__Mg25__weak__wc12 = rate

@numba.njit()
def Al26__Mg26__weak__wc12(rate_eval, tf):
    # Al26 --> Mg26
    rate = 0.0

    # wc12w
    rate += np.exp(  -4.62175 + -2.64931*tf.T9i + -0.025978*tf.T913
                  + -0.0291284*tf.T9 + 0.00389774*tf.T953)

    rate_eval.Al26__Mg26__weak__wc12 = rate

@numba.njit()
def Si26__Al26__weak__wc12(rate_eval, tf):
    # Si26 --> Al26
    rate = 0.0

    # wc12w
    rate += np.exp(  -1.16851)

    rate_eval.Si26__Al26__weak__wc12 = rate

@numba.njit()
def P29__Si29__weak__wc12(rate_eval, tf):
    # P29 --> Si29
    rate = 0.0

    # wc12w
    rate += np.exp(  -1.78721)

    rate_eval.P29__Si29__weak__wc12 = rate

@numba.njit()
def P30__Si30__weak__wc12(rate_eval, tf):
    # P30 --> Si30
    rate = 0.0

    # wc12w
    rate += np.exp(  -5.37715)

    rate_eval.P30__Si30__weak__wc12 = rate

@numba.njit()
def S30__P30__weak__wc12(rate_eval, tf):
    # S30 --> P30
    rate = 0.0

    # wc12w
    rate += np.exp(  -0.532027)

    rate_eval.S30__P30__weak__wc12 = rate

@numba.njit()
def S31__P31__weak__wc12(rate_eval, tf):
    # S31 --> P31
    rate = 0.0

    # wc12w
    rate += np.exp(  -1.31042)

    rate_eval.S31__P31__weak__wc12 = rate

@numba.njit()
def Cl33__S33__weak__wc12(rate_eval, tf):
    # Cl33 --> S33
    rate = 0.0

    # wc12w
    rate += np.exp(  -1.2868)

    rate_eval.Cl33__S33__weak__wc12 = rate

@numba.njit()
def Ar34__Cl34__weak__wc12(rate_eval, tf):
    # Ar34 --> Cl34
    rate = 0.0

    # wc12w
    rate += np.exp(  -0.198094)

    rate_eval.Ar34__Cl34__weak__wc12 = rate

@numba.njit()
def Ar39__K39__weak__wc12(rate_eval, tf):
    # Ar39 --> K39
    rate = 0.0

    # wc12w
    rate += np.exp(  -23.2287)

    rate_eval.Ar39__K39__weak__wc12 = rate

@numba.njit()
def Co55__Fe55__weak__wc12(rate_eval, tf):
    # Co55 --> Fe55
    rate = 0.0

    # wc12w
    rate += np.exp(  -11.419)

    rate_eval.Co55__Fe55__weak__wc12 = rate

@numba.njit()
def C12__p_B11(rate_eval, tf):
    # C12 --> p + B11
    rate = 0.0

    # nw00r
    rate += np.exp(  33.6351 + -186.885*tf.T9i)
    # nw00n
    rate += np.exp(  50.5262 + -185.173*tf.T9i + -12.095*tf.T913i + -6.68421*tf.T913
                  + -0.0148736*tf.T9 + 0.0364288*tf.T953 + 2.83333*tf.lnT9)
    # nw00n
    rate += np.exp(  43.578 + -185.173*tf.T9i + -12.095*tf.T913i + -1.95046*tf.T913
                  + 9.56928*tf.T9 + -10.0637*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.C12__p_B11 = rate

@numba.njit()
def C13__n_C12(rate_eval, tf):
    # C13 --> n + C12
    rate = 0.0

    # ks03 
    rate += np.exp(  30.8808 + -57.4077*tf.T9i + 1.49573*tf.T913i + -0.841102*tf.T913
                  + 0.0340543*tf.T9 + -0.0026392*tf.T953 + 3.1662*tf.lnT9)

    rate_eval.C13__n_C12 = rate

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
def N14__p_C13(rate_eval, tf):
    # N14 --> p + C13
    rate = 0.0

    # nacrr
    rate += np.exp(  37.1528 + -93.4071*tf.T9i + -0.196703*tf.T913
                  + 0.142126*tf.T9 + -0.0238912*tf.T953)
    # nacrr
    rate += np.exp(  38.3716 + -101.18*tf.T9i)
    # nacrn
    rate += np.exp(  41.7046 + -87.6256*tf.T9i + -13.72*tf.T913i + -0.450018*tf.T913
                  + 3.70823*tf.T9 + -1.70545*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.N14__p_C13 = rate

@numba.njit()
def N15__n_N14(rate_eval, tf):
    # N15 --> n + N14
    rate = 0.0

    # ks03 
    rate += np.exp(  34.1728 + -125.726*tf.T9i + 1.396*tf.T913i + -3.47552*tf.T913
                  + 0.351773*tf.T9 + -0.0229344*tf.T953 + 2.52161*tf.lnT9)

    rate_eval.N15__n_N14 = rate

@numba.njit()
def O15__p_N14(rate_eval, tf):
    # O15 --> p + N14
    rate = 0.0

    # im05r
    rate += np.exp(  30.7435 + -89.5667*tf.T9i
                  + 1.5682*tf.lnT9)
    # im05r
    rate += np.exp(  31.6622 + -87.6737*tf.T9i)
    # im05n
    rate += np.exp(  44.1246 + -84.6757*tf.T9i + -15.193*tf.T913i + -4.63975*tf.T913
                  + 9.73458*tf.T9 + -9.55051*tf.T953 + 1.83333*tf.lnT9)
    # im05n
    rate += np.exp(  41.0177 + -84.6757*tf.T9i + -15.193*tf.T913i + -0.161954*tf.T913
                  + -7.52123*tf.T9 + -0.987565*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.O15__p_N14 = rate

@numba.njit()
def O16__n_O15(rate_eval, tf):
    # O16 --> n + O15
    rate = 0.0

    # rpsmr
    rate += np.exp(  32.3869 + -181.759*tf.T9i + -1.11761*tf.T913i + 1.0167*tf.T913
                  + 0.0449976*tf.T9 + -0.00204682*tf.T953 + 0.710783*tf.lnT9)

    rate_eval.O16__n_O15 = rate

@numba.njit()
def O16__p_N15(rate_eval, tf):
    # O16 --> p + N15
    rate = 0.0

    # li10r
    rate += np.exp(  38.8465 + -150.962*tf.T9i
                  + 0.0459037*tf.T9)
    # li10r
    rate += np.exp(  30.8927 + -143.656*tf.T9i)
    # li10n
    rate += np.exp(  44.3197 + -140.732*tf.T9i + -15.24*tf.T913i + 0.334926*tf.T913
                  + 4.59088*tf.T9 + -4.78468*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.O16__p_N15 = rate

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
def O17__n_O16(rate_eval, tf):
    # O17 --> n + O16
    rate = 0.0

    # ks03 
    rate += np.exp(  29.0385 + -48.0574*tf.T9i + -2.11246*tf.T913i + 4.87742*tf.T913
                  + -0.314426*tf.T9 + 0.0169515*tf.T953 + 0.515216*tf.lnT9)

    rate_eval.O17__n_O16 = rate

@numba.njit()
def F18__p_O17(rate_eval, tf):
    # F18 --> p + O17
    rate = 0.0

    # il10r
    rate += np.exp(  33.7037 + -71.2889*tf.T9i + 2.31435*tf.T913
                  + -0.302835*tf.T9 + 0.020133*tf.T953)
    # il10r
    rate += np.exp(  11.2362 + -65.8069*tf.T9i)
    # il10n
    rate += np.exp(  40.2061 + -65.0606*tf.T9i + -16.4035*tf.T913i + 4.31885*tf.T913
                  + -0.709921*tf.T9 + -2.0*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.F18__p_O17 = rate

@numba.njit()
def F18__He4_N14(rate_eval, tf):
    # F18 --> He4 + N14
    rate = 0.0

    # il10n
    rate += np.exp(  46.249 + -51.2292*tf.T9i + -36.2504*tf.T913i
                  + -5.0*tf.T953 + 0.833333*tf.lnT9)
    # il10r
    rate += np.exp(  38.6146 + -62.1948*tf.T9i + -5.6227*tf.T913i)
    # il10r
    rate += np.exp(  24.9119 + -56.3896*tf.T9i)

    rate_eval.F18__He4_N14 = rate

@numba.njit()
def Ne19__p_F18(rate_eval, tf):
    # Ne19 --> p + F18
    rate = 0.0

    # il10r
    rate += np.exp(  -5.41887 + -74.7977*tf.T9i + 22.4903*tf.T913
                  + 0.307872*tf.T9 + -0.296226*tf.T953)
    # il10n
    rate += np.exp(  81.4385 + -74.3988*tf.T9i + -21.4023*tf.T913i + -93.766*tf.T913
                  + 179.258*tf.T9 + -202.561*tf.T953 + 0.833333*tf.lnT9)
    # il10r
    rate += np.exp(  18.1729 + -77.2902*tf.T9i + 13.1683*tf.T913
                  + -1.92023*tf.T9 + 0.16901*tf.T953)

    rate_eval.Ne19__p_F18 = rate

@numba.njit()
def Ne19__He4_O15(rate_eval, tf):
    # Ne19 --> He4 + O15
    rate = 0.0

    # dc11n
    rate += np.exp(  51.0289 + -40.9534*tf.T9i + -39.578*tf.T913i
                  + -3.0*tf.T953 + 0.833333*tf.lnT9)
    # dc11r
    rate += np.exp(  -7.51212 + -45.1578*tf.T9i + -3.24609*tf.T913i + 44.4647*tf.T913
                  + -9.79962*tf.T9 + 0.841782*tf.T953)
    # dc11r
    rate += np.exp(  24.6922 + -46.8378*tf.T9i)

    rate_eval.Ne19__He4_O15 = rate

@numba.njit()
def Ne20__n_Ne19(rate_eval, tf):
    # Ne20 --> n + Ne19
    rate = 0.0

    # ths8r
    rate += np.exp(  30.7283 + -195.706*tf.T9i + 1.57592*tf.T913
                  + -0.11175*tf.T9 + 0.00226473*tf.T953 + 1.5*tf.lnT9)

    rate_eval.Ne20__n_Ne19 = rate

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
def Ne21__He4_O17(rate_eval, tf):
    # Ne21 --> He4 + O17
    rate = 0.0

    # be13r
    rate += np.exp(  27.3205 + -91.2722*tf.T9i + 2.87641*tf.T913i + -3.54489*tf.T913
                  + -2.11222e-08*tf.T9 + -3.90649e-09*tf.T953 + 6.25778*tf.lnT9)
    # be13r
    rate += np.exp(  0.0906657 + -90.782*tf.T9i + 123.363*tf.T913i + -87.4351*tf.T913
                  + -3.40974e-06*tf.T9 + -57.0469*tf.T953 + 83.7218*tf.lnT9)
    # be13r
    rate += np.exp(  -91.954 + -98.9487*tf.T9i + 3.31162e-08*tf.T913i + 130.258*tf.T913
                  + -7.92551e-05*tf.T9 + -4.13772*tf.T953 + -41.2753*tf.lnT9)

    rate_eval.Ne21__He4_O17 = rate

@numba.njit()
def Ne22__n_Ne21(rate_eval, tf):
    # Ne22 --> n + Ne21
    rate = 0.0

    # ks03 
    rate += np.exp(  48.5428 + -120.224*tf.T9i + -0.238173*tf.T913i + -12.2336*tf.T913
                  + 1.14968*tf.T9 + -0.0768882*tf.T953 + 4.13636*tf.lnT9)

    rate_eval.Ne22__n_Ne21 = rate

@numba.njit()
def Na22__p_Ne21(rate_eval, tf):
    # Na22 --> p + Ne21
    rate = 0.0

    # il10r
    rate += np.exp(  -16.4098 + -82.4235*tf.T9i + 21.1176*tf.T913i + 34.0411*tf.T913
                  + -4.45593*tf.T9 + 0.328613*tf.T953)
    # il10r
    rate += np.exp(  24.8334 + -79.6093*tf.T9i)
    # il10r
    rate += np.exp(  -24.579 + -78.4059*tf.T9i)
    # il10n
    rate += np.exp(  42.146 + -78.2097*tf.T9i + -19.2096*tf.T913i
                  + -1.0*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.Na22__p_Ne21 = rate

@numba.njit()
def Na22__He4_F18(rate_eval, tf):
    # Na22 --> He4 + F18
    rate = 0.0

    # rpsmr
    rate += np.exp(  59.3224 + -100.236*tf.T9i + 18.8956*tf.T913i + -65.6134*tf.T913
                  + 1.71114*tf.T9 + -0.0260999*tf.T953 + 39.3396*tf.lnT9)

    rate_eval.Na22__He4_F18 = rate

@numba.njit()
def Na23__n_Na22(rate_eval, tf):
    # Na23 --> n + Na22
    rate = 0.0

    # ths8r
    rate += np.exp(  37.0665 + -144.113*tf.T9i + 1.02148*tf.T913
                  + -0.334638*tf.T9 + 0.0258708*tf.T953 + 1.5*tf.lnT9)

    rate_eval.Na23__n_Na22 = rate

@numba.njit()
def Na23__p_Ne22(rate_eval, tf):
    # Na23 --> p + Ne22
    rate = 0.0

    # ke17r
    rate += np.exp(  18.2467 + -104.673*tf.T9i
                  + -2.79964*tf.lnT9)
    # ke17r
    rate += np.exp(  21.6534 + -103.776*tf.T9i
                  + 1.18923*tf.lnT9)
    # ke17r
    rate += np.exp(  0.818178 + -102.466*tf.T9i
                  + 0.009812*tf.lnT9)
    # ke17r
    rate += np.exp(  18.1624 + -102.855*tf.T9i
                  + 4.73558*tf.lnT9)
    # ke17r
    rate += np.exp(  36.29 + -110.779*tf.T9i
                  + 0.732533*tf.lnT9)
    # ke17r
    rate += np.exp(  33.8935 + -106.655*tf.T9i
                  + 1.65623*tf.lnT9)

    rate_eval.Na23__p_Ne22 = rate

@numba.njit()
def Mg23__p_Na22(rate_eval, tf):
    # Mg23 --> p + Na22
    rate = 0.0

    # il10r
    rate += np.exp(  12.9256 + -90.3923*tf.T9i + 4.86658*tf.T913i + 16.4592*tf.T913
                  + -1.95129*tf.T9 + 0.132972*tf.T953)
    # il10r
    rate += np.exp(  7.95641 + -88.7434*tf.T9i)
    # il10r
    rate += np.exp(  -1.07519 + -88.4655*tf.T9i)

    rate_eval.Mg23__p_Na22 = rate

@numba.njit()
def Mg23__He4_Ne19(rate_eval, tf):
    # Mg23 --> He4 + Ne19
    rate = 0.0

    # ths8r
    rate += np.exp(  61.3121 + -111.985*tf.T9i + -46.6346*tf.T913i + -1.1007*tf.T913
                  + -0.794097*tf.T9 + 0.0813036*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.Mg23__He4_Ne19 = rate

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
def Mg25__n_Mg24(rate_eval, tf):
    # Mg25 --> n + Mg24
    rate = 0.0

    # ks03 
    rate += np.exp(  86.4748 + -84.9032*tf.T9i + -0.142939*tf.T913i + -57.7499*tf.T913
                  + 7.01981*tf.T9 + -0.582057*tf.T953 + 14.3133*tf.lnT9)

    rate_eval.Mg25__n_Mg24 = rate

@numba.njit()
def Mg25__He4_Ne21(rate_eval, tf):
    # Mg25 --> He4 + Ne21
    rate = 0.0

    # cf88r
    rate += np.exp(  50.668 + -136.725*tf.T9i + -29.4583*tf.T913
                  + 14.6328*tf.T9 + -3.47392*tf.T953)
    # cf88n
    rate += np.exp(  61.1178 + -114.676*tf.T9i + -46.89*tf.T913i + -0.72642*tf.T913
                  + -0.76406*tf.T9 + 0.0797483*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.Mg25__He4_Ne21 = rate

@numba.njit()
def Mg26__n_Mg25(rate_eval, tf):
    # Mg26 --> n + Mg25
    rate = 0.0

    # ks03 
    rate += np.exp(  63.7787 + -128.778*tf.T9i + 9.392*tf.T913i + -36.6784*tf.T913
                  + 3.09567*tf.T9 + -0.223882*tf.T953 + 13.8852*tf.lnT9)

    rate_eval.Mg26__n_Mg25 = rate

@numba.njit()
def Mg26__He4_Ne22(rate_eval, tf):
    # Mg26 --> He4 + Ne22
    rate = 0.0

    # li12r
    rate += np.exp(  1.08878 + -127.062*tf.T9i)
    # li12r
    rate += np.exp(  -18.0225 + -125.401*tf.T9i)
    # li12r
    rate += np.exp(  -67.5662 + -124.09*tf.T9i)
    # li12r
    rate += np.exp(  -9.88392 + -129.544*tf.T9i + 35.9878*tf.T913
                  + -4.10684*tf.T9 + 0.259345*tf.T953)
    # li12r
    rate += np.exp(  -4.47312 + -129.627*tf.T9i + 43.2654*tf.T913
                  + -18.5982*tf.T9 + 2.80101*tf.T953)

    rate_eval.Mg26__He4_Ne22 = rate

@numba.njit()
def Al25__p_Mg24(rate_eval, tf):
    # Al25 --> p + Mg24
    rate = 0.0

    # il10n
    rate += np.exp(  41.7494 + -26.3608*tf.T9i + -22.0227*tf.T913i + 0.361297*tf.T913
                  + 2.61292*tf.T9 + -1.0*tf.T953 + 0.833333*tf.lnT9)
    # il10r
    rate += np.exp(  30.093 + -28.8453*tf.T9i + -1.57811*tf.T913
                  + 1.52232*tf.T9 + -0.183001*tf.T953)

    rate_eval.Al25__p_Mg24 = rate

@numba.njit()
def Al26__n_Al25(rate_eval, tf):
    # Al26 --> n + Al25
    rate = 0.0

    # ths8r
    rate += np.exp(  30.9667 + -131.891*tf.T9i + 1.17141*tf.T913
                  + -0.162515*tf.T9 + 0.0126275*tf.T953 + 1.5*tf.lnT9)

    rate_eval.Al26__n_Al25 = rate

@numba.njit()
def Al26__p_Mg25(rate_eval, tf):
    # Al26 --> p + Mg25
    rate = 0.0

    # il10r
    rate += np.exp(  25.2686 + -76.4067*tf.T9i + 8.46334*tf.T913
                  + -0.907024*tf.T9 + 0.0642981*tf.T953)
    # il10r
    rate += np.exp(  27.2591 + -73.903*tf.T9i + -88.9297*tf.T913
                  + 302.948*tf.T9 + -346.461*tf.T953)
    # il10r
    rate += np.exp(  -14.1555 + -73.6126*tf.T9i)

    rate_eval.Al26__p_Mg25 = rate

@numba.njit()
def Al26__He4_Na22(rate_eval, tf):
    # Al26 --> He4 + Na22
    rate = 0.0

    # ths8r
    rate += np.exp(  60.7692 + -109.695*tf.T9i + -50.0924*tf.T913i + -0.390826*tf.T913
                  + -0.99531*tf.T9 + 0.101354*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.Al26__He4_Na22 = rate

@numba.njit()
def Al27__n_Al26(rate_eval, tf):
    # Al27 --> n + Al26
    rate = 0.0

    # ks03 
    rate += np.exp(  39.0178 + -151.532*tf.T9i + -0.171158*tf.T913i + -1.77283*tf.T913
                  + 0.206192*tf.T9 + -0.0191705*tf.T953 + 1.63961*tf.lnT9)

    rate_eval.Al27__n_Al26 = rate

@numba.njit()
def Al27__p_Mg26(rate_eval, tf):
    # Al27 --> p + Mg26
    rate = 0.0

    # il10r
    rate += np.exp(  27.118 + -99.3406*tf.T9i + 6.78105*tf.T913
                  + -1.25771*tf.T9 + 0.140754*tf.T953)
    # il10r
    rate += np.exp(  -5.3594 + -96.8701*tf.T9i + 35.6312*tf.T913
                  + -5.27265*tf.T9 + 0.392932*tf.T953)
    # il10r
    rate += np.exp(  -62.6356 + -96.4509*tf.T9i + 251.281*tf.T913
                  + -730.009*tf.T9 + -224.016*tf.T953)

    rate_eval.Al27__p_Mg26 = rate

@numba.njit()
def Al27__He4_Na23(rate_eval, tf):
    # Al27 --> He4 + Na23
    rate = 0.0

    # ths8r
    rate += np.exp(  69.2185 + -117.109*tf.T9i + -50.2042*tf.T913i + -1.64239*tf.T913
                  + -1.59995*tf.T9 + 0.184933*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.Al27__He4_Na23 = rate

@numba.njit()
def Si26__p_Al25(rate_eval, tf):
    # Si26 --> p + Al25
    rate = 0.0

    # li20r
    rate += np.exp(  34.1845 + -68.7787*tf.T9i
                  + 0.01455*tf.lnT9)
    # li20r
    rate += np.exp(  30.8265 + -76.4488*tf.T9i
                  + 2.98721*tf.lnT9)
    # li20r
    rate += np.exp(  19.2308 + -65.7211*tf.T9i
                  + 1.25571*tf.lnT9)

    rate_eval.Si26__p_Al25 = rate

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
def Si29__n_Si28(rate_eval, tf):
    # Si29 --> n + Si28
    rate = 0.0

    # ka02r
    rate += np.exp(  29.8758 + -98.7165*tf.T9i + 7.68863*tf.T913
                  + -1.7991*tf.T9)
    # ka02 
    rate += np.exp(  31.7355 + -98.3365*tf.T9i
                  + 1.5*tf.lnT9)

    rate_eval.Si29__n_Si28 = rate

@numba.njit()
def Si29__He4_Mg25(rate_eval, tf):
    # Si29 --> He4 + Mg25
    rate = 0.0

    # cf88n
    rate += np.exp(  66.3395 + -129.123*tf.T9i + -53.41*tf.T913i + -1.83266*tf.T913
                  + -0.573073*tf.T9 + 0.0462678*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.Si29__He4_Mg25 = rate

@numba.njit()
def Si30__n_Si29(rate_eval, tf):
    # Si30 --> n + Si29
    rate = 0.0

    # ka02r
    rate += np.exp(  33.9492 + -123.292*tf.T9i + 5.50678*tf.T913
                  + -2.85656*tf.T9)
    # ka02n
    rate += np.exp(  36.1504 + -123.112*tf.T9i + 0.650904*tf.T913
                  + 1.5*tf.lnT9)

    rate_eval.Si30__n_Si29 = rate

@numba.njit()
def Si30__He4_Mg26(rate_eval, tf):
    # Si30 --> He4 + Mg26
    rate = 0.0

    # cf88r
    rate += np.exp(  26.2068 + -142.235*tf.T9i + -1.87411*tf.T913
                  + 3.41299*tf.T9 + -0.43226*tf.T953)
    # cf88n
    rate += np.exp(  70.7561 + -123.518*tf.T9i + -53.7518*tf.T913i + -4.8647*tf.T913
                  + -1.51467*tf.T9 + 0.833333*tf.lnT9)

    rate_eval.Si30__He4_Mg26 = rate

@numba.njit()
def P29__p_Si28(rate_eval, tf):
    # P29 --> p + Si28
    rate = 0.0

    # il10r
    rate += np.exp(  28.6997 + -36.0408*tf.T9i)
    # il10n
    rate += np.exp(  39.1379 + -31.8984*tf.T9i + -23.8173*tf.T913i + 7.08203*tf.T913
                  + -1.44753*tf.T9 + 0.0804296*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.P29__p_Si28 = rate

@numba.njit()
def P29__He4_Al25(rate_eval, tf):
    # P29 --> He4 + Al25
    rate = 0.0

    # ths8r
    rate += np.exp(  63.8779 + -121.399*tf.T9i + -56.3424*tf.T913i + 0.542998*tf.T913
                  + -0.721716*tf.T9 + 0.0469712*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.P29__He4_Al25 = rate

@numba.njit()
def P30__n_P29(rate_eval, tf):
    # P30 --> n + P29
    rate = 0.0

    # ths8r
    rate += np.exp(  32.0379 + -131.355*tf.T9i + 0.15555*tf.T913
                  + 0.155359*tf.T9 + -0.0208019*tf.T953 + 1.5*tf.lnT9)

    rate_eval.P30__n_P29 = rate

@numba.njit()
def P30__p_Si29(rate_eval, tf):
    # P30 --> p + Si29
    rate = 0.0

    # il10r
    rate += np.exp(  22.0015 + -68.2607*tf.T9i + 14.0921*tf.T913
                  + -3.92096*tf.T9 + 0.447706*tf.T953)
    # il10r
    rate += np.exp(  9.77935 + -66.1716*tf.T9i)
    # il10n
    rate += np.exp(  39.7677 + -64.9214*tf.T9i + -23.9101*tf.T913i + 10.7796*tf.T913
                  + -3.04181*tf.T9 + 0.274565*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.P30__p_Si29 = rate

@numba.njit()
def P30__He4_Al26(rate_eval, tf):
    # P30 --> He4 + Al26
    rate = 0.0

    # ths8r
    rate += np.exp(  69.1545 + -120.863*tf.T9i + -56.4422*tf.T913i + -2.44848*tf.T913
                  + -1.17578*tf.T9 + 0.150757*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.P30__He4_Al26 = rate

@numba.njit()
def P31__n_P30(rate_eval, tf):
    # P31 --> n + P30
    rate = 0.0

    # ths8r
    rate += np.exp(  36.8808 + -142.87*tf.T9i + 0.909911*tf.T913
                  + -0.162367*tf.T9 + 0.00668293*tf.T953 + 1.5*tf.lnT9)

    rate_eval.P31__n_P30 = rate

@numba.njit()
def P31__p_Si30(rate_eval, tf):
    # P31 --> p + Si30
    rate = 0.0

    # de20r
    rate += np.exp(  31.761 + -89.8588*tf.T9i
                  + 2.7883*tf.lnT9)
    # de20r
    rate += np.exp(  -311.303 + -85.8097*tf.T9i
                  + -77.047*tf.lnT9)
    # de20r
    rate += np.exp(  18.7213 + -85.9282*tf.T9i
                  + 6.49034*tf.lnT9)
    # de20r
    rate += np.exp(  4.04359 + -85.6217*tf.T9i
                  + 2.80331*tf.lnT9)
    # de20r
    rate += np.exp(  -1115.38 + -180.553*tf.T9i
                  + -895.258*tf.lnT9)
    # de20r
    rate += np.exp(  32.9288 + -90.2661*tf.T9i
                  + -0.070816*tf.lnT9)
    # de20r
    rate += np.exp(  -11.8961 + -85.2694*tf.T9i
                  + -0.128387*tf.lnT9)
    # de20r
    rate += np.exp(  35.3329 + -91.3175*tf.T9i
                  + 0.3809*tf.lnT9)

    rate_eval.P31__p_Si30 = rate

@numba.njit()
def P31__He4_Al27(rate_eval, tf):
    # P31 --> He4 + Al27
    rate = 0.0

    # ths8r
    rate += np.exp(  73.2168 + -112.206*tf.T9i + -56.5351*tf.T913i + -0.896208*tf.T913
                  + -1.72024*tf.T9 + 0.185409*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.P31__He4_Al27 = rate

@numba.njit()
def S30__p_P29(rate_eval, tf):
    # S30 --> p + P29
    rate = 0.0

    # il10n
    rate += np.exp(  45.3211 + -51.0504*tf.T9i + -25.6007*tf.T913i
                  + -2.0*tf.T953 + 0.833333*tf.lnT9)
    # il10r
    rate += np.exp(  -1.65202 + -54.7698*tf.T9i + 8.48834*tf.T913i + 25.65*tf.T913
                  + -3.79773*tf.T9 + 0.320391*tf.T953)

    rate_eval.S30__p_P29 = rate

@numba.njit()
def S30__He4_Si26(rate_eval, tf):
    # S30 --> He4 + Si26
    rate = 0.0

    # ths8r
    rate += np.exp(  63.7475 + -108.419*tf.T9i + -59.3013*tf.T913i + 0.642868*tf.T913
                  + -0.958008*tf.T9 + 0.0715476*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.S30__He4_Si26 = rate

@numba.njit()
def S31__n_S30(rate_eval, tf):
    # S31 --> n + S30
    rate = 0.0

    # ths8r
    rate += np.exp(  33.2919 + -151.478*tf.T9i + 1.62298*tf.T913
                  + -0.278802*tf.T9 + 0.0210647*tf.T953 + 1.5*tf.lnT9)

    rate_eval.S31__n_S30 = rate

@numba.njit()
def S31__p_P30(rate_eval, tf):
    # S31 --> p + P30
    rate = 0.0

    # mb07 
    rate += np.exp(  -7732.57 + -60.6616*tf.T9i + -1999.51*tf.T913i + 11886.5*tf.T913
                  + -2668.72*tf.T9 + 354.294*tf.T953 + -2898.95*tf.lnT9)
    # mb07 
    rate += np.exp(  48.4487 + -80.154*tf.T9i + 156.029*tf.T913i + -174.377*tf.T913
                  + 7.4644*tf.T9 + -0.342232*tf.T953 + 100.758*tf.lnT9)

    rate_eval.S31__p_P30 = rate

@numba.njit()
def S32__n_S31(rate_eval, tf):
    # S32 --> n + S31
    rate = 0.0

    # ths8r
    rate += np.exp(  31.9171 + -174.56*tf.T9i + 1.71463*tf.T913
                  + -0.221804*tf.T9 + 0.00880104*tf.T953 + 1.5*tf.lnT9)

    rate_eval.S32__n_S31 = rate

@numba.njit()
def S32__p_P31(rate_eval, tf):
    # S32 --> p + P31
    rate = 0.0

    # il10r
    rate += np.exp(  25.1729 + -106.637*tf.T9i + 8.09341*tf.T913
                  + -0.615971*tf.T9 + 0.031159*tf.T953)
    # il10r
    rate += np.exp(  21.6829 + -105.119*tf.T9i)
    # il10n
    rate += np.exp(  43.6109 + -102.86*tf.T9i + -25.3278*tf.T913i + 6.4931*tf.T913
                  + -9.27513*tf.T9 + -0.610439*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.S32__p_P31 = rate

@numba.njit()
def S32__He4_Si28(rate_eval, tf):
    # S32 --> He4 + Si28
    rate = 0.0

    # ths8r
    rate += np.exp(  72.813 + -80.626*tf.T9i + -59.4896*tf.T913i + 4.47205*tf.T913
                  + -4.78989*tf.T9 + 0.557201*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.S32__He4_Si28 = rate

@numba.njit()
def S33__n_S32(rate_eval, tf):
    # S33 --> n + S32
    rate = 0.0

    # ks03 
    rate += np.exp(  34.7199 + -100.079*tf.T9i + -15.0178*tf.T913i + 16.3567*tf.T913
                  + -0.436839*tf.T9 + -0.00574462*tf.T953 + -8.28034*tf.lnT9)

    rate_eval.S33__n_S32 = rate

@numba.njit()
def S33__He4_Si29(rate_eval, tf):
    # S33 --> He4 + Si29
    rate = 0.0

    # ths8r
    rate += np.exp(  73.7708 + -82.576*tf.T9i + -59.5755*tf.T913i + 1.06274*tf.T913
                  + -3.07529*tf.T9 + 0.372011*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.S33__He4_Si29 = rate

@numba.njit()
def Cl33__p_S32(rate_eval, tf):
    # Cl33 --> p + S32
    rate = 0.0

    # il10n
    rate += np.exp(  74.7432 + -26.4211*tf.T9i + -29.7741*tf.T913i + -87.4473*tf.T913
                  + 182.189*tf.T9 + -128.625*tf.T953 + 0.833333*tf.lnT9)
    # il10r
    rate += np.exp(  -4.96497 + -27.2952*tf.T9i)
    # il10r
    rate += np.exp(  91.6078 + -29.4248*tf.T9i + -33.7204*tf.T913i + -32.7355*tf.T913
                  + 3.92526*tf.T9 + -0.250479*tf.T953)

    rate_eval.Cl33__p_S32 = rate

@numba.njit()
def Cl33__He4_P29(rate_eval, tf):
    # Cl33 --> He4 + P29
    rate = 0.0

    # ths8r
    rate += np.exp(  66.203 + -75.1475*tf.T9i + -62.3802*tf.T913i + 0.593062*tf.T913
                  + -1.14226*tf.T9 + 0.0934776*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.Cl33__He4_P29 = rate

@numba.njit()
def Cl34__n_Cl33(rate_eval, tf):
    # Cl34 --> n + Cl33
    rate = 0.0

    # ths8r
    rate += np.exp(  33.1968 + -133.541*tf.T9i + 0.921411*tf.T913
                  + -0.0823764*tf.T9 + 0.000852746*tf.T953 + 1.5*tf.lnT9)

    rate_eval.Cl34__n_Cl33 = rate

@numba.njit()
def Cl34__p_S33(rate_eval, tf):
    # Cl34 --> p + S33
    rate = 0.0

    # ths8r
    rate += np.exp(  61.5381 + -59.679*tf.T9i + -26.777*tf.T913i + -5.96882*tf.T913
                  + -1.0706*tf.T9 + 0.19692*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.Cl34__p_S33 = rate

@numba.njit()
def Cl34__He4_P30(rate_eval, tf):
    # Cl34 --> He4 + P30
    rate = 0.0

    # ths8r
    rate += np.exp(  71.335 + -77.3338*tf.T9i + -62.4643*tf.T913i + -3.19028*tf.T913
                  + -0.832633*tf.T9 + 0.0987525*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.Cl34__He4_P30 = rate

@numba.njit()
def Cl35__n_Cl34(rate_eval, tf):
    # Cl35 --> n + Cl34
    rate = 0.0

    # ths8r
    rate += np.exp(  34.9299 + -146.74*tf.T9i + 0.990222*tf.T913
                  + -0.146686*tf.T9 + 0.00560251*tf.T953 + 1.5*tf.lnT9)

    rate_eval.Cl35__n_Cl34 = rate

@numba.njit()
def Cl35__He4_P31(rate_eval, tf):
    # Cl35 --> He4 + P31
    rate = 0.0

    # ths8r
    rate += np.exp(  74.6679 + -81.2033*tf.T9i + -62.5433*tf.T913i + -2.95026*tf.T913
                  + -0.89652*tf.T9 + 0.0774126*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.Cl35__He4_P31 = rate

@numba.njit()
def Ar34__p_Cl33(rate_eval, tf):
    # Ar34 --> p + Cl33
    rate = 0.0

    # ths8r
    rate += np.exp(  60.177 + -54.109*tf.T9i + -27.8815*tf.T913i + -3.18731*tf.T913
                  + -1.76254*tf.T9 + 0.264735*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.Ar34__p_Cl33 = rate

@numba.njit()
def Ar34__He4_S30(rate_eval, tf):
    # Ar34 --> He4 + S30
    rate = 0.0

    # ths8r
    rate += np.exp(  68.017 + -78.2097*tf.T9i + -65.211*tf.T913i + -1.41447*tf.T913
                  + -0.542976*tf.T9 + 0.0211165*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.Ar34__He4_S30 = rate

@numba.njit()
def Ar36__p_Cl35(rate_eval, tf):
    # Ar36 --> p + Cl35
    rate = 0.0

    # il10n
    rate += np.exp(  60.7366 + -98.7191*tf.T9i + -27.8971*tf.T913i + -16.2304*tf.T913
                  + 35.255*tf.T9 + -25.8411*tf.T953 + 0.833333*tf.lnT9)
    # il10r
    rate += np.exp(  17.2028 + -102.37*tf.T9i + 18.0179*tf.T913
                  + -2.86304*tf.T9 + 0.250854*tf.T953)
    # il10r
    rate += np.exp(  16.0169 + -100.729*tf.T9i)
    # il10r
    rate += np.exp(  -17.4751 + -99.2838*tf.T9i)

    rate_eval.Ar36__p_Cl35 = rate

@numba.njit()
def Ar36__He4_S32(rate_eval, tf):
    # Ar36 --> He4 + S32
    rate = 0.0

    # ths8r
    rate += np.exp(  73.8164 + -77.0627*tf.T9i + -65.3709*tf.T913i + 5.68294*tf.T913
                  + -5.00388*tf.T9 + 0.571407*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.Ar36__He4_S32 = rate

@numba.njit()
def Ar37__n_Ar36(rate_eval, tf):
    # Ar37 --> n + Ar36
    rate = 0.0

    # ks03 
    rate += np.exp(  34.2933 + -101.941*tf.T9i + -3.1764*tf.T913i + 5.13191*tf.T913
                  + -0.00639688*tf.T9 + -0.0292833*tf.T953 + -1.24683*tf.lnT9)

    rate_eval.Ar37__n_Ar36 = rate

@numba.njit()
def Ar37__He4_S33(rate_eval, tf):
    # Ar37 --> He4 + S33
    rate = 0.0

    # ths8r
    rate += np.exp(  74.852 + -78.7549*tf.T9i + -65.4446*tf.T913i + 3.59607*tf.T913
                  + -3.40501*tf.T9 + 0.363961*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.Ar37__He4_S33 = rate

@numba.njit()
def Ar38__n_Ar37(rate_eval, tf):
    # Ar38 --> n + Ar37
    rate = 0.0

    # ths8r
    rate += np.exp(  39.8454 + -137.376*tf.T9i + -0.825362*tf.T913
                  + 0.336634*tf.T9 + -0.0509617*tf.T953 + 1.5*tf.lnT9)

    rate_eval.Ar38__n_Ar37 = rate

@numba.njit()
def Ar39__n_Ar38(rate_eval, tf):
    # Ar39 --> n + Ar38
    rate = 0.0

    # ks03 
    rate += np.exp(  36.3134 + -76.6032*tf.T9i + 2.38837*tf.T913i + -4.76536*tf.T913
                  + 0.701311*tf.T9 + -0.0705226*tf.T953 + 3.30517*tf.lnT9)

    rate_eval.Ar39__n_Ar38 = rate

@numba.njit()
def K39__p_Ar38(rate_eval, tf):
    # K39 --> p + Ar38
    rate = 0.0

    # ths8r
    rate += np.exp(  57.5639 + -74.0533*tf.T9i + -29.0021*tf.T913i + -0.525968*tf.T913
                  + -1.94216*tf.T9 + 0.267346*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.K39__p_Ar38 = rate

@numba.njit()
def K39__He4_Cl35(rate_eval, tf):
    # K39 --> He4 + Cl35
    rate = 0.0

    # ths8r
    rate += np.exp(  77.6477 + -83.7658*tf.T9i + -68.2848*tf.T913i + 0.0178545*tf.T913
                  + -2.06783*tf.T9 + 0.199659*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.K39__He4_Cl35 = rate

@numba.njit()
def Ca40__p_K39(rate_eval, tf):
    # Ca40 --> p + K39
    rate = 0.0

    # lo18r
    rate += np.exp(  613.153 + -109.213*tf.T9i + 641.844*tf.T913i + -1248.49*tf.T913
                  + 566.426*tf.lnT9)
    # lo18r
    rate += np.exp(  127.306 + -98.3134*tf.T9i + 41.1723*tf.T913i + -149.299*tf.T913
                  + 10.5229*tf.T9 + -0.68208*tf.T953 + 60.7367*tf.lnT9)
    # lo18r
    rate += np.exp(  2786.44 + -101.871*tf.T9i + 802.18*tf.T913i + -4010.27*tf.T913
                  + 1137.69*tf.lnT9)

    rate_eval.Ca40__p_K39 = rate

@numba.njit()
def Ca40__He4_Ar36(rate_eval, tf):
    # Ca40 --> He4 + Ar36
    rate = 0.0

    # ths8r
    rate += np.exp(  77.2826 + -81.6916*tf.T9i + -71.0046*tf.T913i + 4.0656*tf.T913
                  + -5.26509*tf.T9 + 0.683546*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.Ca40__He4_Ar36 = rate

@numba.njit()
def Sc43__He4_K39(rate_eval, tf):
    # Sc43 --> He4 + K39
    rate = 0.0

    # ths8r
    rate += np.exp(  78.3727 + -55.7693*tf.T9i + -73.8006*tf.T913i + 1.87885*tf.T913
                  + -2.75862*tf.T9 + 0.279678*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.Sc43__He4_K39 = rate

@numba.njit()
def Ti44__p_Sc43(rate_eval, tf):
    # Ti44 --> p + Sc43
    rate = 0.0

    # ths8r
    rate += np.exp(  62.5939 + -100.373*tf.T9i + -32.1734*tf.T913i + -1.77078*tf.T913
                  + -2.21706*tf.T9 + 0.298499*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.Ti44__p_Sc43 = rate

@numba.njit()
def Ti44__He4_Ca40(rate_eval, tf):
    # Ti44 --> He4 + Ca40
    rate = 0.0

    # chw0 
    rate += np.exp(  78.6991 + -59.4974*tf.T9i + -76.4273*tf.T913i + 3.87451*tf.T913
                  + -3.61477*tf.T9 + 0.367451*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.Ti44__He4_Ca40 = rate

@numba.njit()
def V47__He4_Sc43(rate_eval, tf):
    # V47 --> He4 + Sc43
    rate = 0.0

    # ths8r
    rate += np.exp(  84.6713 + -95.6099*tf.T9i + -79.122*tf.T913i + -7.07006*tf.T913
                  + 0.424183*tf.T9 + -0.0665231*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.V47__He4_Sc43 = rate

@numba.njit()
def Cr48__p_V47(rate_eval, tf):
    # Cr48 --> p + V47
    rate = 0.0

    # nfisn
    rate += np.exp(  65.6231 + -94.5854*tf.T9i + -110.655*tf.T913i + 83.0232*tf.T913
                  + -19.7762*tf.T9 + 3.03961*tf.T953 + -47.9742*tf.lnT9)
    # nfisn
    rate += np.exp(  67.7403 + -100.13*tf.T9i + -34.0548*tf.T913i + -3.41973*tf.T913
                  + 1.16501*tf.T9 + -0.105543*tf.T953 + -6.20886*tf.lnT9)
    # nfisn
    rate += np.exp(  536.523 + -99.3659*tf.T9i + 317.171*tf.T913i + -911.679*tf.T913
                  + 94.4245*tf.T9 + -10.1973*tf.T953 + 332.227*tf.lnT9)
    # nfisn
    rate += np.exp(  48.892 + -93.8243*tf.T9i + -45.9868*tf.T913i + 13.6822*tf.T913
                  + -0.376902*tf.T9 + -0.0194875*tf.T953 + -6.92325*tf.lnT9)

    rate_eval.Cr48__p_V47 = rate

@numba.njit()
def Cr48__He4_Ti44(rate_eval, tf):
    # Cr48 --> He4 + Ti44
    rate = 0.0

    # ths8r
    rate += np.exp(  89.7573 + -89.3041*tf.T9i + -81.667*tf.T913i + -10.6333*tf.T913
                  + -0.672613*tf.T9 + 0.161209*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.Cr48__He4_Ti44 = rate

@numba.njit()
def Mn51__He4_V47(rate_eval, tf):
    # Mn51 --> He4 + V47
    rate = 0.0

    # ths8r
    rate += np.exp(  81.4259 + -100.544*tf.T9i + -84.2732*tf.T913i + -2.98061*tf.T913
                  + -0.531361*tf.T9 + 0.023612*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.Mn51__He4_V47 = rate

@numba.njit()
def Fe52__p_Mn51(rate_eval, tf):
    # Fe52 --> p + Mn51
    rate = 0.0

    # ths8r
    rate += np.exp(  61.728 + -85.6325*tf.T9i + -36.1825*tf.T913i + 0.873042*tf.T913
                  + -2.89731*tf.T9 + 0.364394*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.Fe52__p_Mn51 = rate

@numba.njit()
def Fe52__He4_Cr48(rate_eval, tf):
    # Fe52 --> He4 + Cr48
    rate = 0.0

    # ths8r
    rate += np.exp(  90.1474 + -92.109*tf.T9i + -86.7459*tf.T913i + -9.79373*tf.T913
                  + -0.772169*tf.T9 + 0.155883*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.Fe52__He4_Cr48 = rate

@numba.njit()
def Co55__He4_Mn51(rate_eval, tf):
    # Co55 --> He4 + Mn51
    rate = 0.0

    # ths8r
    rate += np.exp(  90.613 + -95.2861*tf.T9i + -89.274*tf.T913i + -10.4373*tf.T913
                  + 1.00492*tf.T9 + -0.125548*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.Co55__He4_Mn51 = rate

@numba.njit()
def Ni56__p_Co55(rate_eval, tf):
    # Ni56 --> p + Co55
    rate = 0.0

    # ths8r
    rate += np.exp(  63.1318 + -83.1473*tf.T9i + -38.1053*tf.T913i + -0.210947*tf.T913
                  + -2.68377*tf.T9 + 0.355814*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.Ni56__p_Co55 = rate

@numba.njit()
def Ni56__He4_Fe52(rate_eval, tf):
    # Ni56 --> He4 + Fe52
    rate = 0.0

    # ths8r
    rate += np.exp(  91.6226 + -92.801*tf.T9i + -91.6819*tf.T913i + -9.51885*tf.T913
                  + -0.533014*tf.T9 + 0.0892607*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.Ni56__He4_Fe52 = rate

@numba.njit()
def Ni59__n_Ni58(rate_eval, tf):
    # Ni59 --> n + Ni58
    rate = 0.0

    # ks03 
    rate += np.exp(  30.9258 + -104.3*tf.T9i + -11.785*tf.T913i + 19.5347*tf.T913
                  + -0.857179*tf.T9 + 0.00111653*tf.T953 + -7.85642*tf.lnT9)

    rate_eval.Ni59__n_Ni58 = rate

@numba.njit()
def Ni59__He4_Fe55(rate_eval, tf):
    # Ni59 --> He4 + Fe55
    rate = 0.0

    # ths8r
    rate += np.exp(  85.76 + -70.8014*tf.T9i + -91.8012*tf.T913i + 4.12067*tf.T913
                  + -4.13271*tf.T9 + 0.450006*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.Ni59__He4_Fe55 = rate

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
def p_B11__C12(rate_eval, tf):
    # B11 + p --> C12
    rate = 0.0

    # nw00r
    rate += np.exp(  8.67352 + -1.71197*tf.T9i
                  + -1.5*tf.lnT9)
    # nw00n
    rate += np.exp(  25.5647 + -12.095*tf.T913i + -6.68421*tf.T913
                  + -0.0148736*tf.T9 + 0.0364288*tf.T953 + 1.33333*tf.lnT9)
    # nw00n
    rate += np.exp(  18.6165 + -12.095*tf.T913i + -1.95046*tf.T913
                  + 9.56928*tf.T9 + -10.0637*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_B11__C12 = rate

@numba.njit()
def n_C12__C13(rate_eval, tf):
    # C12 + n --> C13
    rate = 0.0

    # ks03 
    rate += np.exp(  7.98821 + -0.00836732*tf.T9i + 1.49573*tf.T913i + -0.841102*tf.T913
                  + 0.0340543*tf.T9 + -0.0026392*tf.T953 + 1.6662*tf.lnT9)

    rate_eval.n_C12__C13 = rate

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
def p_C13__N14(rate_eval, tf):
    # C13 + p --> N14
    rate = 0.0

    # nacrr
    rate += np.exp(  15.1825 + -13.5543*tf.T9i
                  + -1.5*tf.lnT9)
    # nacrn
    rate += np.exp(  18.5155 + -13.72*tf.T913i + -0.450018*tf.T913
                  + 3.70823*tf.T9 + -1.70545*tf.T953 + -0.666667*tf.lnT9)
    # nacrr
    rate += np.exp(  13.9637 + -5.78147*tf.T9i + -0.196703*tf.T913
                  + 0.142126*tf.T9 + -0.0238912*tf.T953 + -1.5*tf.lnT9)

    rate_eval.p_C13__N14 = rate

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
def n_N14__N15(rate_eval, tf):
    # N14 + n --> N15
    rate = 0.0

    # ks03 
    rate += np.exp(  10.1651 + -0.0114078*tf.T9i + 1.396*tf.T913i + -3.47552*tf.T913
                  + 0.351773*tf.T9 + -0.0229344*tf.T953 + 1.02161*tf.lnT9)

    rate_eval.n_N14__N15 = rate

@numba.njit()
def p_N14__O15(rate_eval, tf):
    # N14 + p --> O15
    rate = 0.0

    # im05n
    rate += np.exp(  17.01 + -15.193*tf.T913i + -0.161954*tf.T913
                  + -7.52123*tf.T9 + -0.987565*tf.T953 + -0.666667*tf.lnT9)
    # im05r
    rate += np.exp(  6.73578 + -4.891*tf.T9i
                  + 0.0682*tf.lnT9)
    # im05r
    rate += np.exp(  7.65444 + -2.998*tf.T9i
                  + -1.5*tf.lnT9)
    # im05n
    rate += np.exp(  20.1169 + -15.193*tf.T913i + -4.63975*tf.T913
                  + 9.73458*tf.T9 + -9.55051*tf.T953 + 0.333333*tf.lnT9)

    rate_eval.p_N14__O15 = rate

@numba.njit()
def He4_N14__F18(rate_eval, tf):
    # N14 + He4 --> F18
    rate = 0.0

    # il10n
    rate += np.exp(  21.5339 + -36.2504*tf.T913i
                  + -5.0*tf.T953 + -0.666667*tf.lnT9)
    # il10r
    rate += np.exp(  13.8995 + -10.9656*tf.T9i + -5.6227*tf.T913i
                  + -1.5*tf.lnT9)
    # il10r
    rate += np.exp(  0.196838 + -5.16034*tf.T9i
                  + -1.5*tf.lnT9)

    rate_eval.He4_N14__F18 = rate

@numba.njit()
def p_N15__O16(rate_eval, tf):
    # N15 + p --> O16
    rate = 0.0

    # li10n
    rate += np.exp(  20.0176 + -15.24*tf.T913i + 0.334926*tf.T913
                  + 4.59088*tf.T9 + -4.78468*tf.T953 + -0.666667*tf.lnT9)
    # li10r
    rate += np.exp(  14.5444 + -10.2295*tf.T9i
                  + 0.0459037*tf.T9 + -1.5*tf.lnT9)
    # li10r
    rate += np.exp(  6.59056 + -2.92315*tf.T9i
                  + -1.5*tf.lnT9)

    rate_eval.p_N15__O16 = rate

@numba.njit()
def n_O15__O16(rate_eval, tf):
    # O15 + n --> O16
    rate = 0.0

    # rpsmr
    rate += np.exp(  8.08476 + 0.0135346*tf.T9i + -1.11761*tf.T913i + 1.0167*tf.T913
                  + 0.0449976*tf.T9 + -0.00204682*tf.T953 + -0.789217*tf.lnT9)

    rate_eval.n_O15__O16 = rate

@numba.njit()
def He4_O15__Ne19(rate_eval, tf):
    # O15 + He4 --> Ne19
    rate = 0.0

    # dc11r
    rate += np.exp(  -32.2496 + -4.20439*tf.T9i + -3.24609*tf.T913i + 44.4647*tf.T913
                  + -9.79962*tf.T9 + 0.841782*tf.T953 + -1.5*tf.lnT9)
    # dc11r
    rate += np.exp(  -0.0452465 + -5.88439*tf.T9i
                  + -1.5*tf.lnT9)
    # dc11n
    rate += np.exp(  26.2914 + -39.578*tf.T913i
                  + -3.0*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He4_O15__Ne19 = rate

@numba.njit()
def n_O16__O17(rate_eval, tf):
    # O16 + n --> O17
    rate = 0.0

    # ks03 
    rate += np.exp(  7.21546 + 0.0235015*tf.T9i + -2.11246*tf.T913i + 4.87742*tf.T913
                  + -0.314426*tf.T9 + 0.0169515*tf.T953 + -0.984784*tf.lnT9)

    rate_eval.n_O16__O17 = rate

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
def p_O17__F18(rate_eval, tf):
    # O17 + p --> F18
    rate = 0.0

    # il10n
    rate += np.exp(  15.8929 + -16.4035*tf.T913i + 4.31885*tf.T913
                  + -0.709921*tf.T9 + -2.0*tf.T953 + -0.666667*tf.lnT9)
    # il10r
    rate += np.exp(  9.39048 + -6.22828*tf.T9i + 2.31435*tf.T913
                  + -0.302835*tf.T9 + 0.020133*tf.T953 + -1.5*tf.lnT9)
    # il10r
    rate += np.exp(  -13.077 + -0.746296*tf.T9i
                  + -1.5*tf.lnT9)

    rate_eval.p_O17__F18 = rate

@numba.njit()
def He4_O17__Ne21(rate_eval, tf):
    # O17 + He4 --> Ne21
    rate = 0.0

    # be13r
    rate += np.exp(  -25.0898 + -5.50926*tf.T9i + 123.363*tf.T913i + -87.4351*tf.T913
                  + -3.40974e-06*tf.T9 + -57.0469*tf.T953 + 82.2218*tf.lnT9)
    # be13r
    rate += np.exp(  -117.134 + -13.6759*tf.T9i + 3.31162e-08*tf.T913i + 130.258*tf.T913
                  + -7.92551e-05*tf.T9 + -4.13772*tf.T953 + -42.7753*tf.lnT9)
    # be13r
    rate += np.exp(  2.14 + -5.99952*tf.T9i + 2.87641*tf.T913i + -3.54489*tf.T913
                  + -2.11222e-08*tf.T9 + -3.90649e-09*tf.T953 + 4.75778*tf.lnT9)

    rate_eval.He4_O17__Ne21 = rate

@numba.njit()
def p_F18__Ne19(rate_eval, tf):
    # F18 + p --> Ne19
    rate = 0.0

    # il10n
    rate += np.exp(  57.4084 + -21.4023*tf.T913i + -93.766*tf.T913
                  + 179.258*tf.T9 + -202.561*tf.T953 + -0.666667*tf.lnT9)
    # il10r
    rate += np.exp(  -5.85727 + -2.89147*tf.T9i + 13.1683*tf.T913
                  + -1.92023*tf.T9 + 0.16901*tf.T953 + -1.5*tf.lnT9)
    # il10r
    rate += np.exp(  -29.449 + -0.39895*tf.T9i + 22.4903*tf.T913
                  + 0.307872*tf.T9 + -0.296226*tf.T953 + -1.5*tf.lnT9)

    rate_eval.p_F18__Ne19 = rate

@numba.njit()
def He4_F18__Na22(rate_eval, tf):
    # F18 + He4 --> Na22
    rate = 0.0

    # rpsmr
    rate += np.exp(  35.3786 + -1.82957*tf.T9i + 18.8956*tf.T913i + -65.6134*tf.T913
                  + 1.71114*tf.T9 + -0.0260999*tf.T953 + 37.8396*tf.lnT9)

    rate_eval.He4_F18__Na22 = rate

@numba.njit()
def n_Ne19__Ne20(rate_eval, tf):
    # Ne19 + n --> Ne20
    rate = 0.0

    # ths8r
    rate += np.exp(  6.40633 + 1.57592*tf.T913
                  + -0.11175*tf.T9 + 0.00226473*tf.T953)

    rate_eval.n_Ne19__Ne20 = rate

@numba.njit()
def He4_Ne19__Mg23(rate_eval, tf):
    # Ne19 + He4 --> Mg23
    rate = 0.0

    # ths8r
    rate += np.exp(  37.1998 + -46.6346*tf.T913i + -1.1007*tf.T913
                  + -0.794097*tf.T9 + 0.0813036*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He4_Ne19__Mg23 = rate

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
def n_Ne21__Ne22(rate_eval, tf):
    # Ne21 + n --> Ne22
    rate = 0.0

    # ks03 
    rate += np.exp(  23.5205 + 0.0482005*tf.T9i + -0.238173*tf.T913i + -12.2336*tf.T913
                  + 1.14968*tf.T9 + -0.0768882*tf.T953 + 2.63636*tf.lnT9)

    rate_eval.n_Ne21__Ne22 = rate

@numba.njit()
def p_Ne21__Na22(rate_eval, tf):
    # Ne21 + p --> Na22
    rate = 0.0

    # il10r
    rate += np.exp(  -47.6554 + -0.19618*tf.T9i
                  + -1.5*tf.lnT9)
    # il10n
    rate += np.exp(  19.0696 + -19.2096*tf.T913i
                  + -1.0*tf.T953 + -0.666667*tf.lnT9)
    # il10r
    rate += np.exp(  -39.4862 + -4.21385*tf.T9i + 21.1176*tf.T913i + 34.0411*tf.T913
                  + -4.45593*tf.T9 + 0.328613*tf.T953 + -1.5*tf.lnT9)
    # il10r
    rate += np.exp(  1.75704 + -1.39957*tf.T9i
                  + -1.5*tf.lnT9)

    rate_eval.p_Ne21__Na22 = rate

@numba.njit()
def He4_Ne21__Mg25(rate_eval, tf):
    # Ne21 + He4 --> Mg25
    rate = 0.0

    # cf88r
    rate += np.exp(  26.2429 + -22.049*tf.T9i + -29.4583*tf.T913
                  + 14.6328*tf.T9 + -3.47392*tf.T953 + -1.5*tf.lnT9)
    # cf88n
    rate += np.exp(  36.6927 + -46.89*tf.T913i + -0.72642*tf.T913
                  + -0.76406*tf.T9 + 0.0797483*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He4_Ne21__Mg25 = rate

@numba.njit()
def p_Ne22__Na23(rate_eval, tf):
    # Ne22 + p --> Na23
    rate = 0.0

    # ke17r
    rate += np.exp(  -4.00597 + -2.6179*tf.T9i
                  + -4.29964*tf.lnT9)
    # ke17r
    rate += np.exp(  -0.599331 + -1.72007*tf.T9i
                  + -0.310765*tf.lnT9)
    # ke17r
    rate += np.exp(  -21.4345 + -0.410962*tf.T9i
                  + -1.49019*tf.lnT9)
    # ke17r
    rate += np.exp(  -4.09035 + -0.799756*tf.T9i
                  + 3.23558*tf.lnT9)
    # ke17r
    rate += np.exp(  14.0373 + -8.72377*tf.T9i
                  + -0.767467*tf.lnT9)
    # ke17r
    rate += np.exp(  11.6408 + -4.59936*tf.T9i
                  + 0.156226*tf.lnT9)

    rate_eval.p_Ne22__Na23 = rate

@numba.njit()
def He4_Ne22__Mg26(rate_eval, tf):
    # Ne22 + He4 --> Mg26
    rate = 0.0

    # li12r
    rate += np.exp(  -23.7527 + -3.88217*tf.T9i
                  + -1.5*tf.lnT9)
    # li12r
    rate += np.exp(  -42.864 + -2.22115*tf.T9i
                  + -1.5*tf.lnT9)
    # li12r
    rate += np.exp(  -92.4077 + -0.910477*tf.T9i
                  + -1.5*tf.lnT9)
    # li12r
    rate += np.exp(  -34.7254 + -6.36421*tf.T9i + 35.9878*tf.T913
                  + -4.10684*tf.T9 + 0.259345*tf.T953 + -1.5*tf.lnT9)
    # li12r
    rate += np.exp(  -29.3146 + -6.44772*tf.T9i + 43.2654*tf.T913
                  + -18.5982*tf.T9 + 2.80101*tf.T953 + -1.5*tf.lnT9)

    rate_eval.He4_Ne22__Mg26 = rate

@numba.njit()
def n_Na22__Na23(rate_eval, tf):
    # Na22 + n --> Na23
    rate = 0.0

    # ths8r
    rate += np.exp(  12.8678 + 1.02148*tf.T913
                  + -0.334638*tf.T9 + 0.0258708*tf.T953)

    rate_eval.n_Na22__Na23 = rate

@numba.njit()
def p_Na22__Mg23(rate_eval, tf):
    # Na22 + p --> Mg23
    rate = 0.0

    # il10r
    rate += np.exp(  -11.2731 + -2.42669*tf.T9i + 4.86658*tf.T913i + 16.4592*tf.T913
                  + -1.95129*tf.T9 + 0.132972*tf.T953 + -1.5*tf.lnT9)
    # il10r
    rate += np.exp(  -16.2423 + -0.777841*tf.T9i
                  + -1.5*tf.lnT9)
    # il10r
    rate += np.exp(  -25.2739 + -0.499888*tf.T9i
                  + -1.5*tf.lnT9)

    rate_eval.p_Na22__Mg23 = rate

@numba.njit()
def He4_Na22__Al26(rate_eval, tf):
    # Na22 + He4 --> Al26
    rate = 0.0

    # ths8r
    rate += np.exp(  36.3797 + -50.0924*tf.T913i + -0.390826*tf.T913
                  + -0.99531*tf.T9 + 0.101354*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He4_Na22__Al26 = rate

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
def n_Mg24__Mg25(rate_eval, tf):
    # Mg24 + n --> Mg25
    rate = 0.0

    # ks03 
    rate += np.exp(  64.622 + 0.161296*tf.T9i + -0.142939*tf.T913i + -57.7499*tf.T913
                  + 7.01981*tf.T9 + -0.582057*tf.T953 + 12.8133*tf.lnT9)

    rate_eval.n_Mg24__Mg25 = rate

@numba.njit()
def p_Mg24__Al25(rate_eval, tf):
    # Mg24 + p --> Al25
    rate = 0.0

    # il10r
    rate += np.exp(  8.24021 + -2.48451*tf.T9i + -1.57811*tf.T913
                  + 1.52232*tf.T9 + -0.183001*tf.T953 + -1.5*tf.lnT9)
    # il10n
    rate += np.exp(  19.8966 + -22.0227*tf.T913i + 0.361297*tf.T913
                  + 2.61292*tf.T9 + -1.0*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_Mg24__Al25 = rate

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
def n_Mg25__Mg26(rate_eval, tf):
    # Mg25 + n --> Mg26
    rate = 0.0

    # ks03 
    rate += np.exp(  38.34 + -0.0457591*tf.T9i + 9.392*tf.T913i + -36.6784*tf.T913
                  + 3.09567*tf.T9 + -0.223882*tf.T953 + 12.3852*tf.lnT9)

    rate_eval.n_Mg25__Mg26 = rate

@numba.njit()
def p_Mg25__Al26(rate_eval, tf):
    # Mg25 + p --> Al26
    rate = 0.0

    # il10r
    rate += np.exp(  2.22778 + -3.22353*tf.T9i + 8.46334*tf.T913
                  + -0.907024*tf.T9 + 0.0642981*tf.T953 + -1.5*tf.lnT9)
    # il10r
    rate += np.exp(  4.21826 + -0.71983*tf.T9i + -88.9297*tf.T913
                  + 302.948*tf.T9 + -346.461*tf.T953 + -1.5*tf.lnT9)
    # il10r
    rate += np.exp(  -37.1963 + -0.429366*tf.T9i
                  + -1.5*tf.lnT9)

    rate_eval.p_Mg25__Al26 = rate

@numba.njit()
def He4_Mg25__Si29(rate_eval, tf):
    # Mg25 + He4 --> Si29
    rate = 0.0

    # cf88n
    rate += np.exp(  40.3715 + -53.41*tf.T913i + -1.83266*tf.T913
                  + -0.573073*tf.T9 + 0.0462678*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He4_Mg25__Si29 = rate

@numba.njit()
def p_Mg26__Al27(rate_eval, tf):
    # Mg26 + p --> Al27
    rate = 0.0

    # il10r
    rate += np.exp(  5.26056 + -3.35921*tf.T9i + 6.78105*tf.T913
                  + -1.25771*tf.T9 + 0.140754*tf.T953 + -1.5*tf.lnT9)
    # il10r
    rate += np.exp(  -27.2168 + -0.888689*tf.T9i + 35.6312*tf.T913
                  + -5.27265*tf.T9 + 0.392932*tf.T953 + -1.5*tf.lnT9)
    # il10r
    rate += np.exp(  -84.493 + -0.469464*tf.T9i + 251.281*tf.T913
                  + -730.009*tf.T9 + -224.016*tf.T953 + -1.5*tf.lnT9)

    rate_eval.p_Mg26__Al27 = rate

@numba.njit()
def He4_Mg26__Si30(rate_eval, tf):
    # Mg26 + He4 --> Si30
    rate = 0.0

    # cf88r
    rate += np.exp(  1.32941 + -18.7164*tf.T9i + -1.87411*tf.T913
                  + 3.41299*tf.T9 + -0.43226*tf.T953 + -1.5*tf.lnT9)
    # cf88n
    rate += np.exp(  45.8787 + -53.7518*tf.T913i + -4.8647*tf.T913
                  + -1.51467*tf.T9 + -0.666667*tf.lnT9)

    rate_eval.He4_Mg26__Si30 = rate

@numba.njit()
def n_Al25__Al26(rate_eval, tf):
    # Al25 + n --> Al26
    rate = 0.0

    # ths8r
    rate += np.exp(  7.92587 + 1.17141*tf.T913
                  + -0.162515*tf.T9 + 0.0126275*tf.T953)

    rate_eval.n_Al25__Al26 = rate

@numba.njit()
def p_Al25__Si26(rate_eval, tf):
    # Al25 + p --> Si26
    rate = 0.0

    # li20r
    rate += np.exp(  8.74592 + -4.78862*tf.T9i
                  + -1.48545*tf.lnT9)
    # li20r
    rate += np.exp(  5.38793 + -12.4587*tf.T9i
                  + 1.48721*tf.lnT9)
    # li20r
    rate += np.exp(  -6.20781 + -1.73102*tf.T9i
                  + -0.244294*tf.lnT9)

    rate_eval.p_Al25__Si26 = rate

@numba.njit()
def He4_Al25__P29(rate_eval, tf):
    # Al25 + He4 --> P29
    rate = 0.0

    # ths8r
    rate += np.exp(  37.9099 + -56.3424*tf.T913i + 0.542998*tf.T913
                  + -0.721716*tf.T9 + 0.0469712*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He4_Al25__P29 = rate

@numba.njit()
def n_Al26__Al27(rate_eval, tf):
    # Al26 + n --> Al27
    rate = 0.0

    # ks03 
    rate += np.exp(  14.7625 + 0.00350938*tf.T9i + -0.171158*tf.T913i + -1.77283*tf.T913
                  + 0.206192*tf.T9 + -0.0191705*tf.T953 + 0.139609*tf.lnT9)

    rate_eval.n_Al26__Al27 = rate

@numba.njit()
def He4_Al26__P30(rate_eval, tf):
    # Al26 + He4 --> P30
    rate = 0.0

    # ths8r
    rate += np.exp(  42.9778 + -56.4422*tf.T913i + -2.44848*tf.T913
                  + -1.17578*tf.T9 + 0.150757*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He4_Al26__P30 = rate

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
def He4_Al27__P31(rate_eval, tf):
    # Al27 + He4 --> P31
    rate = 0.0

    # ths8r
    rate += np.exp(  47.2333 + -56.5351*tf.T913i + -0.896208*tf.T913
                  + -1.72024*tf.T9 + 0.185409*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He4_Al27__P31 = rate

@numba.njit()
def He4_Si26__S30(rate_eval, tf):
    # Si26 + He4 --> S30
    rate = 0.0

    # ths8r
    rate += np.exp(  38.8701 + -59.3013*tf.T913i + 0.642868*tf.T913
                  + -0.958008*tf.T9 + 0.0715476*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He4_Si26__S30 = rate

@numba.njit()
def n_Si28__Si29(rate_eval, tf):
    # Si28 + n --> Si29
    rate = 0.0

    # ka02r
    rate += np.exp(  6.9158 + -0.38*tf.T9i + 7.68863*tf.T913
                  + -1.7991*tf.T9 + -1.5*tf.lnT9)
    # ka02 
    rate += np.exp(  8.77553)

    rate_eval.n_Si28__Si29 = rate

@numba.njit()
def p_Si28__P29(rate_eval, tf):
    # Si28 + p --> P29
    rate = 0.0

    # il10n
    rate += np.exp(  16.1779 + -23.8173*tf.T913i + 7.08203*tf.T913
                  + -1.44753*tf.T9 + 0.0804296*tf.T953 + -0.666667*tf.lnT9)
    # il10r
    rate += np.exp(  5.73975 + -4.14232*tf.T9i
                  + -1.5*tf.lnT9)

    rate_eval.p_Si28__P29 = rate

@numba.njit()
def He4_Si28__S32(rate_eval, tf):
    # Si28 + He4 --> S32
    rate = 0.0

    # ths8r
    rate += np.exp(  47.9212 + -59.4896*tf.T913i + 4.47205*tf.T913
                  + -4.78989*tf.T9 + 0.557201*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He4_Si28__S32 = rate

@numba.njit()
def n_Si29__Si30(rate_eval, tf):
    # Si29 + n --> Si30
    rate = 0.0

    # ka02r
    rate += np.exp(  9.60115 + -0.179366*tf.T9i + 5.50678*tf.T913
                  + -2.85656*tf.T9 + -1.5*tf.lnT9)
    # ka02n
    rate += np.exp(  11.8023 + 0.650904*tf.T913)

    rate_eval.n_Si29__Si30 = rate

@numba.njit()
def p_Si29__P30(rate_eval, tf):
    # Si29 + p --> P30
    rate = 0.0

    # il10n
    rate += np.exp(  16.5182 + -23.9101*tf.T913i + 10.7796*tf.T913
                  + -3.04181*tf.T9 + 0.274565*tf.T953 + -0.666667*tf.lnT9)
    # il10r
    rate += np.exp(  -1.24791 + -3.33929*tf.T9i + 14.0921*tf.T913
                  + -3.92096*tf.T9 + 0.447706*tf.T953 + -1.5*tf.lnT9)
    # il10r
    rate += np.exp(  -13.4701 + -1.25026*tf.T9i
                  + -1.5*tf.lnT9)

    rate_eval.p_Si29__P30 = rate

@numba.njit()
def He4_Si29__S33(rate_eval, tf):
    # Si29 + He4 --> S33
    rate = 0.0

    # ths8r
    rate += np.exp(  49.5657 + -59.5755*tf.T913i + 1.06274*tf.T913
                  + -3.07529*tf.T9 + 0.372011*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He4_Si29__S33 = rate

@numba.njit()
def p_Si30__P31(rate_eval, tf):
    # Si30 + p --> P31
    rate = 0.0

    # de20r
    rate += np.exp(  9.96544 + -5.58963*tf.T9i
                  + -1.57082*tf.lnT9)
    # de20r
    rate += np.exp(  -34.8594 + -0.592934*tf.T9i
                  + -1.62839*tf.lnT9)
    # de20r
    rate += np.exp(  12.3695 + -6.64105*tf.T9i
                  + -1.1191*tf.lnT9)
    # de20r
    rate += np.exp(  8.79766 + -5.18231*tf.T9i
                  + 1.2883*tf.lnT9)
    # de20r
    rate += np.exp(  -334.266 + -1.13327*tf.T9i
                  + -78.547*tf.lnT9)
    # de20r
    rate += np.exp(  -4.24208 + -1.25174*tf.T9i
                  + 4.99034*tf.lnT9)
    # de20r
    rate += np.exp(  -18.9198 + -0.945261*tf.T9i
                  + 1.30331*tf.lnT9)
    # de20r
    rate += np.exp(  -1138.34 + -95.8769*tf.T9i
                  + -896.758*tf.lnT9)

    rate_eval.p_Si30__P31 = rate

@numba.njit()
def n_P29__P30(rate_eval, tf):
    # P29 + n --> P30
    rate = 0.0

    # ths8r
    rate += np.exp(  8.78841 + 0.15555*tf.T913
                  + 0.155359*tf.T9 + -0.0208019*tf.T953)

    rate_eval.n_P29__P30 = rate

@numba.njit()
def p_P29__S30(rate_eval, tf):
    # P29 + p --> S30
    rate = 0.0

    # il10r
    rate += np.exp(  -26.0 + -3.71938*tf.T9i + 8.48834*tf.T913i + 25.65*tf.T913
                  + -3.79773*tf.T9 + 0.320391*tf.T953 + -1.5*tf.lnT9)
    # il10n
    rate += np.exp(  20.9731 + -25.6007*tf.T913i
                  + -2.0*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_P29__S30 = rate

@numba.njit()
def He4_P29__Cl33(rate_eval, tf):
    # P29 + He4 --> Cl33
    rate = 0.0

    # ths8r
    rate += np.exp(  41.9979 + -62.3802*tf.T913i + 0.593062*tf.T913
                  + -1.14226*tf.T9 + 0.0934776*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He4_P29__Cl33 = rate

@numba.njit()
def n_P30__P31(rate_eval, tf):
    # P30 + n --> P31
    rate = 0.0

    # ths8r
    rate += np.exp(  12.8187 + 0.909911*tf.T913
                  + -0.162367*tf.T9 + 0.00668293*tf.T953)

    rate_eval.n_P30__P31 = rate

@numba.njit()
def p_P30__S31(rate_eval, tf):
    # P30 + p --> S31
    rate = 0.0

    # mb07 
    rate += np.exp(  -7756.63 + 10.5093*tf.T9i + -1999.51*tf.T913i + 11886.5*tf.T913
                  + -2668.72*tf.T9 + 354.294*tf.T953 + -2900.45*tf.lnT9)
    # mb07 
    rate += np.exp(  24.3866 + -8.98316*tf.T9i + 156.029*tf.T913i + -174.377*tf.T913
                  + 7.4644*tf.T9 + -0.342232*tf.T953 + 99.2579*tf.lnT9)

    rate_eval.p_P30__S31 = rate

@numba.njit()
def He4_P30__Cl34(rate_eval, tf):
    # P30 + He4 --> Cl34
    rate = 0.0

    # ths8r
    rate += np.exp(  45.3321 + -62.4643*tf.T913i + -3.19028*tf.T913
                  + -0.832633*tf.T9 + 0.0987525*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He4_P30__Cl34 = rate

@numba.njit()
def p_P31__S32(rate_eval, tf):
    # P31 + p --> S32
    rate = 0.0

    # il10r
    rate += np.exp(  0.821556 + -3.77704*tf.T9i + 8.09341*tf.T913
                  + -0.615971*tf.T9 + 0.031159*tf.T953 + -1.5*tf.lnT9)
    # il10r
    rate += np.exp(  -2.66839 + -2.25958*tf.T9i
                  + -1.5*tf.lnT9)
    # il10n
    rate += np.exp(  19.2596 + -25.3278*tf.T913i + 6.4931*tf.T913
                  + -9.27513*tf.T9 + -0.610439*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_P31__S32 = rate

@numba.njit()
def He4_P31__Cl35(rate_eval, tf):
    # P31 + He4 --> Cl35
    rate = 0.0

    # ths8r
    rate += np.exp(  50.451 + -62.5433*tf.T913i + -2.95026*tf.T913
                  + -0.89652*tf.T9 + 0.0774126*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He4_P31__Cl35 = rate

@numba.njit()
def n_S30__S31(rate_eval, tf):
    # S30 + n --> S31
    rate = 0.0

    # ths8r
    rate += np.exp(  10.3285 + 1.62298*tf.T913
                  + -0.278802*tf.T9 + 0.0210647*tf.T953)

    rate_eval.n_S30__S31 = rate

@numba.njit()
def He4_S30__Ar34(rate_eval, tf):
    # S30 + He4 --> Ar34
    rate = 0.0

    # ths8r
    rate += np.exp(  43.1127 + -65.211*tf.T913i + -1.41447*tf.T913
                  + -0.542976*tf.T9 + 0.0211165*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He4_S30__Ar34 = rate

@numba.njit()
def n_S31__S32(rate_eval, tf):
    # S31 + n --> S32
    rate = 0.0

    # ths8r
    rate += np.exp(  7.56582 + 1.71463*tf.T913
                  + -0.221804*tf.T9 + 0.00880104*tf.T953)

    rate_eval.n_S31__S32 = rate

@numba.njit()
def n_S32__S33(rate_eval, tf):
    # S32 + n --> S33
    rate = 0.0

    # ks03 
    rate += np.exp(  12.4466 + 0.198828*tf.T9i + -15.0178*tf.T913i + 16.3567*tf.T913
                  + -0.436839*tf.T9 + -0.00574462*tf.T953 + -9.78034*tf.lnT9)

    rate_eval.n_S32__S33 = rate

@numba.njit()
def p_S32__Cl33(rate_eval, tf):
    # S32 + p --> Cl33
    rate = 0.0

    # il10r
    rate += np.exp(  -27.2382 + -0.874107*tf.T9i
                  + -1.5*tf.lnT9)
    # il10n
    rate += np.exp(  52.47 + -29.7741*tf.T913i + -87.4473*tf.T913
                  + 182.189*tf.T9 + -128.625*tf.T953 + -0.666667*tf.lnT9)
    # il10r
    rate += np.exp(  69.3346 + -3.00371*tf.T9i + -33.7204*tf.T913i + -32.7355*tf.T913
                  + 3.92526*tf.T9 + -0.250479*tf.T953 + -1.5*tf.lnT9)

    rate_eval.p_S32__Cl33 = rate

@numba.njit()
def He4_S32__Ar36(rate_eval, tf):
    # S32 + He4 --> Ar36
    rate = 0.0

    # ths8r
    rate += np.exp(  48.901 + -65.3709*tf.T913i + 5.68294*tf.T913
                  + -5.00388*tf.T9 + 0.571407*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He4_S32__Ar36 = rate

@numba.njit()
def p_S33__Cl34(rate_eval, tf):
    # S33 + p --> Cl34
    rate = 0.0

    # ths8r
    rate += np.exp(  36.4908 + -26.777*tf.T913i + -5.96882*tf.T913
                  + -1.0706*tf.T9 + 0.19692*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_S33__Cl34 = rate

@numba.njit()
def He4_S33__Ar37(rate_eval, tf):
    # S33 + He4 --> Ar37
    rate = 0.0

    # ths8r
    rate += np.exp(  49.9315 + -65.4446*tf.T913i + 3.59607*tf.T913
                  + -3.40501*tf.T9 + 0.363961*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He4_S33__Ar37 = rate

@numba.njit()
def n_Cl33__Cl34(rate_eval, tf):
    # Cl33 + n --> Cl34
    rate = 0.0

    # ths8r
    rate += np.exp(  8.14947 + 0.921411*tf.T913
                  + -0.0823764*tf.T9 + 0.000852746*tf.T953)

    rate_eval.n_Cl33__Cl34 = rate

@numba.njit()
def p_Cl33__Ar34(rate_eval, tf):
    # Cl33 + p --> Ar34
    rate = 0.0

    # ths8r
    rate += np.exp(  35.1297 + -27.8815*tf.T913i + -3.18731*tf.T913
                  + -1.76254*tf.T9 + 0.264735*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_Cl33__Ar34 = rate

@numba.njit()
def n_Cl34__Cl35(rate_eval, tf):
    # Cl34 + n --> Cl35
    rate = 0.0

    # ths8r
    rate += np.exp(  12.6539 + 0.990222*tf.T913
                  + -0.146686*tf.T9 + 0.00560251*tf.T953)

    rate_eval.n_Cl34__Cl35 = rate

@numba.njit()
def p_Cl35__Ar36(rate_eval, tf):
    # Cl35 + p --> Ar36
    rate = 0.0

    # il10r
    rate += np.exp(  -9.03294 + -2.00996*tf.T9i
                  + -1.5*tf.lnT9)
    # il10r
    rate += np.exp(  -42.5249 + -0.564651*tf.T9i
                  + -1.5*tf.lnT9)
    # il10n
    rate += np.exp(  35.6868 + -27.8971*tf.T913i + -16.2304*tf.T913
                  + 35.255*tf.T9 + -25.8411*tf.T953 + -0.666667*tf.lnT9)
    # il10r
    rate += np.exp(  -7.84699 + -3.65092*tf.T9i + 18.0179*tf.T913
                  + -2.86304*tf.T9 + 0.250854*tf.T953 + -1.5*tf.lnT9)

    rate_eval.p_Cl35__Ar36 = rate

@numba.njit()
def He4_Cl35__K39(rate_eval, tf):
    # Cl35 + He4 --> K39
    rate = 0.0

    # ths8r
    rate += np.exp(  52.718 + -68.2848*tf.T913i + 0.0178545*tf.T913
                  + -2.06783*tf.T9 + 0.199659*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He4_Cl35__K39 = rate

@numba.njit()
def n_Ar36__Ar37(rate_eval, tf):
    # Ar36 + n --> Ar37
    rate = 0.0

    # ks03 
    rate += np.exp(  12.0149 + 0.0317044*tf.T9i + -3.1764*tf.T913i + 5.13191*tf.T913
                  + -0.00639688*tf.T9 + -0.0292833*tf.T953 + -2.74683*tf.lnT9)

    rate_eval.n_Ar36__Ar37 = rate

@numba.njit()
def He4_Ar36__Ca40(rate_eval, tf):
    # Ar36 + He4 --> Ca40
    rate = 0.0

    # ths8r
    rate += np.exp(  52.3486 + -71.0046*tf.T913i + 4.0656*tf.T913
                  + -5.26509*tf.T9 + 0.683546*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He4_Ar36__Ca40 = rate

@numba.njit()
def n_Ar37__Ar38(rate_eval, tf):
    # Ar37 + n --> Ar38
    rate = 0.0

    # ths8r
    rate += np.exp(  14.7933 + -0.825362*tf.T913
                  + 0.336634*tf.T9 + -0.0509617*tf.T953)

    rate_eval.n_Ar37__Ar38 = rate

@numba.njit()
def n_Ar38__Ar39(rate_eval, tf):
    # Ar38 + n --> Ar39
    rate = 0.0

    # ks03 
    rate += np.exp(  14.726 + -0.0331959*tf.T9i + 2.38837*tf.T913i + -4.76536*tf.T913
                  + 0.701311*tf.T9 + -0.0705226*tf.T953 + 1.80517*tf.lnT9)

    rate_eval.n_Ar38__Ar39 = rate

@numba.njit()
def p_Ar38__K39(rate_eval, tf):
    # Ar38 + p --> K39
    rate = 0.0

    # ths8r
    rate += np.exp(  35.2834 + -29.0021*tf.T913i + -0.525968*tf.T913
                  + -1.94216*tf.T9 + 0.267346*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_Ar38__K39 = rate

@numba.njit()
def p_K39__Ca40(rate_eval, tf):
    # K39 + p --> Ca40
    rate = 0.0

    # lo18r
    rate += np.exp(  2761.38 + -5.22234*tf.T9i + 802.18*tf.T913i + -4010.27*tf.T913
                  + 1136.19*tf.lnT9)
    # lo18r
    rate += np.exp(  588.099 + -12.5647*tf.T9i + 641.844*tf.T913i + -1248.49*tf.T913
                  + 564.926*tf.lnT9)
    # lo18r
    rate += np.exp(  102.252 + -1.66508*tf.T9i + 41.1723*tf.T913i + -149.299*tf.T913
                  + 10.5229*tf.T9 + -0.68208*tf.T953 + 59.2367*tf.lnT9)

    rate_eval.p_K39__Ca40 = rate

@numba.njit()
def He4_K39__Sc43(rate_eval, tf):
    # K39 + He4 --> Sc43
    rate = 0.0

    # ths8r
    rate += np.exp(  54.1202 + -73.8006*tf.T913i + 1.87885*tf.T913
                  + -2.75862*tf.T9 + 0.279678*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He4_K39__Sc43 = rate

@numba.njit()
def He4_Ca40__Ti44(rate_eval, tf):
    # Ca40 + He4 --> Ti44
    rate = 0.0

    # chw0 
    rate += np.exp(  53.75 + -76.4273*tf.T913i + 3.87451*tf.T913
                  + -3.61477*tf.T9 + 0.367451*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He4_Ca40__Ti44 = rate

@numba.njit()
def p_Sc43__Ti44(rate_eval, tf):
    # Sc43 + p --> Ti44
    rate = 0.0

    # ths8r
    rate += np.exp(  36.8432 + -32.1734*tf.T913i + -1.77078*tf.T913
                  + -2.21706*tf.T9 + 0.298499*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_Sc43__Ti44 = rate

@numba.njit()
def He4_Sc43__V47(rate_eval, tf):
    # Sc43 + He4 --> V47
    rate = 0.0

    # ths8r
    rate += np.exp(  59.0195 + -79.122*tf.T913i + -7.07006*tf.T913
                  + 0.424183*tf.T9 + -0.0665231*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He4_Sc43__V47 = rate

@numba.njit()
def He4_Ti44__Cr48(rate_eval, tf):
    # Ti44 + He4 --> Cr48
    rate = 0.0

    # ths8r
    rate += np.exp(  64.7958 + -81.667*tf.T913i + -10.6333*tf.T913
                  + -0.672613*tf.T9 + 0.161209*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He4_Ti44__Cr48 = rate

@numba.njit()
def p_V47__Cr48(rate_eval, tf):
    # V47 + p --> Cr48
    rate = 0.0

    # nfisn
    rate += np.exp(  42.6798 + -6.0593*tf.T9i + -34.0548*tf.T913i + -3.41973*tf.T913
                  + 1.16501*tf.T9 + -0.105543*tf.T953 + -7.70886*tf.lnT9)
    # nfisn
    rate += np.exp(  511.463 + -5.29491*tf.T9i + 317.171*tf.T913i + -911.679*tf.T913
                  + 94.4245*tf.T9 + -10.1973*tf.T953 + 330.727*tf.lnT9)
    # nfisn
    rate += np.exp(  23.8315 + 0.246665*tf.T9i + -45.9868*tf.T913i + 13.6822*tf.T913
                  + -0.376902*tf.T9 + -0.0194875*tf.T953 + -8.42325*tf.lnT9)
    # nfisn
    rate += np.exp(  40.5626 + -0.514414*tf.T9i + -110.655*tf.T913i + 83.0232*tf.T913
                  + -19.7762*tf.T9 + 3.03961*tf.T953 + -49.4742*tf.lnT9)

    rate_eval.p_V47__Cr48 = rate

@numba.njit()
def He4_V47__Mn51(rate_eval, tf):
    # V47 + He4 --> Mn51
    rate = 0.0

    # ths8r
    rate += np.exp(  56.8618 + -84.2732*tf.T913i + -2.98061*tf.T913
                  + -0.531361*tf.T9 + 0.023612*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He4_V47__Mn51 = rate

@numba.njit()
def He4_Cr48__Fe52(rate_eval, tf):
    # Cr48 + He4 --> Fe52
    rate = 0.0

    # ths8r
    rate += np.exp(  65.1754 + -86.7459*tf.T913i + -9.79373*tf.T913
                  + -0.772169*tf.T9 + 0.155883*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He4_Cr48__Fe52 = rate

@numba.njit()
def p_Mn51__Fe52(rate_eval, tf):
    # Mn51 + p --> Fe52
    rate = 0.0

    # ths8r
    rate += np.exp(  36.2596 + -36.1825*tf.T913i + 0.873042*tf.T913
                  + -2.89731*tf.T9 + 0.364394*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_Mn51__Fe52 = rate

@numba.njit()
def He4_Mn51__Co55(rate_eval, tf):
    # Mn51 + He4 --> Co55
    rate = 0.0

    # ths8r
    rate += np.exp(  65.9219 + -89.274*tf.T913i + -10.4373*tf.T913
                  + 1.00492*tf.T9 + -0.125548*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He4_Mn51__Co55 = rate

@numba.njit()
def He4_Fe52__Ni56(rate_eval, tf):
    # Fe52 + He4 --> Ni56
    rate = 0.0

    # ths8r
    rate += np.exp(  66.6417 + -91.6819*tf.T913i + -9.51885*tf.T913
                  + -0.533014*tf.T9 + 0.0892607*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He4_Fe52__Ni56 = rate

@numba.njit()
def He4_Fe55__Ni59(rate_eval, tf):
    # Fe55 + He4 --> Ni59
    rate = 0.0

    # ths8r
    rate += np.exp(  60.7732 + -91.8012*tf.T913i + 4.12067*tf.T913
                  + -4.13271*tf.T9 + 0.450006*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He4_Fe55__Ni59 = rate

@numba.njit()
def p_Co55__Ni56(rate_eval, tf):
    # Co55 + p --> Ni56
    rate = 0.0

    # ths8r
    rate += np.exp(  37.3736 + -38.1053*tf.T913i + -0.210947*tf.T913
                  + -2.68377*tf.T9 + 0.355814*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_Co55__Ni56 = rate

@numba.njit()
def n_Ni58__Ni59(rate_eval, tf):
    # Ni58 + n --> Ni59
    rate = 0.0

    # ks03 
    rate += np.exp(  8.63197 + 0.13279*tf.T9i + -11.785*tf.T913i + 19.5347*tf.T913
                  + -0.857179*tf.T9 + 0.00111653*tf.T953 + -9.35642*tf.lnT9)

    rate_eval.n_Ni58__Ni59 = rate

@numba.njit()
def He4_B11__n_N14(rate_eval, tf):
    # B11 + He4 --> n + N14
    rate = 0.0

    # cf88r
    rate += np.exp(  0.582216 + -2.827*tf.T9i
                  + -1.5*tf.lnT9)
    # cf88n
    rate += np.exp(  29.5726 + -28.234*tf.T913i + -0.325987*tf.T913
                  + 30.135*tf.T9 + -78.4165*tf.T953 + -0.666667*tf.lnT9)
    # cf88r
    rate += np.exp(  15.3084 + -8.596*tf.T9i
                  + 0.6*tf.lnT9)
    # cf88r
    rate += np.exp(  7.44425 + -5.178*tf.T9i
                  + -1.5*tf.lnT9)

    rate_eval.He4_B11__n_N14 = rate

@numba.njit()
def He4_C12__n_O15(rate_eval, tf):
    # C12 + He4 --> n + O15
    rate = 0.0

    # cf88n
    rate += np.exp(  17.0115 + -98.6615*tf.T9i + 0.124787*tf.T913
                  + 0.0588937*tf.T9 + -0.00679206*tf.T953)

    rate_eval.He4_C12__n_O15 = rate

@numba.njit()
def He4_C12__p_N15(rate_eval, tf):
    # C12 + He4 --> p + N15
    rate = 0.0

    # nacrn
    rate += np.exp(  27.118 + -57.6279*tf.T9i + -15.253*tf.T913i + 1.59318*tf.T913
                  + 2.4479*tf.T9 + -2.19708*tf.T953 + -0.666667*tf.lnT9)
    # nacrr
    rate += np.exp(  -6.93365 + -58.7917*tf.T9i + 22.7105*tf.T913
                  + -2.90707*tf.T9 + 0.205754*tf.T953 + -1.5*tf.lnT9)
    # nacrr
    rate += np.exp(  20.5388 + -65.034*tf.T9i
                  + -1.5*tf.lnT9)
    # nacrr
    rate += np.exp(  -5.2319 + -59.6491*tf.T9i + 30.8497*tf.T913
                  + -8.50433*tf.T9 + -1.54426*tf.T953 + -1.5*tf.lnT9)

    rate_eval.He4_C12__p_N15 = rate

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
def p_C13__n_N13(rate_eval, tf):
    # C13 + p --> n + N13
    rate = 0.0

    # nacrn
    rate += np.exp(  17.7625 + -34.8483*tf.T9i + 1.26126*tf.T913
                  + -0.204952*tf.T9 + 0.0310523*tf.T953)

    rate_eval.p_C13__n_N13 = rate

@numba.njit()
def He4_C13__n_O16(rate_eval, tf):
    # C13 + He4 --> n + O16
    rate = 0.0

    # gl12 
    rate += np.exp(  79.3008 + -0.30489*tf.T9i + 7.43132*tf.T913i + -84.8689*tf.T913
                  + 3.65083*tf.T9 + -0.148015*tf.T953 + 37.6008*tf.lnT9)
    # gl12 
    rate += np.exp(  62.5775 + -0.0277331*tf.T9i + -32.3917*tf.T913i + -48.934*tf.T913
                  + 44.1843*tf.T9 + -20.8743*tf.T953 + 2.02494*tf.lnT9)

    rate_eval.He4_C13__n_O16 = rate

@numba.njit()
def n_N13__p_C13(rate_eval, tf):
    # N13 + n --> p + C13
    rate = 0.0

    # nacrn
    rate += np.exp(  17.7625 + 1.26126*tf.T913
                  + -0.204952*tf.T9 + 0.0310523*tf.T953)

    rate_eval.n_N13__p_C13 = rate

@numba.njit()
def He4_N13__p_O16(rate_eval, tf):
    # N13 + He4 --> p + O16
    rate = 0.0

    # cf88n
    rate += np.exp(  40.4644 + -35.829*tf.T913i + -0.530275*tf.T913
                  + -0.982462*tf.T9 + 0.0808059*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He4_N13__p_O16 = rate

@numba.njit()
def n_N14__He4_B11(rate_eval, tf):
    # N14 + n --> He4 + B11
    rate = 0.0

    # cf88r
    rate += np.exp(  1.89445 + -4.66051*tf.T9i
                  + -1.5*tf.lnT9)
    # cf88n
    rate += np.exp(  30.8848 + -1.83351*tf.T9i + -28.234*tf.T913i + -0.325987*tf.T913
                  + 30.135*tf.T9 + -78.4165*tf.T953 + -0.666667*tf.lnT9)
    # cf88r
    rate += np.exp(  16.6206 + -10.4295*tf.T9i
                  + 0.6*tf.lnT9)
    # cf88r
    rate += np.exp(  8.75648 + -7.01151*tf.T9i
                  + -1.5*tf.lnT9)

    rate_eval.n_N14__He4_B11 = rate

@numba.njit()
def He4_N14__p_O17(rate_eval, tf):
    # N14 + He4 --> p + O17
    rate = 0.0

    # il10r
    rate += np.exp(  -7.60954 + -14.5839*tf.T9i
                  + -1.5*tf.lnT9)
    # il10n
    rate += np.exp(  19.1771 + -13.8305*tf.T9i + -16.9078*tf.T913i
                  + -2.0*tf.T953 + -0.666667*tf.lnT9)
    # il10r
    rate += np.exp(  9.77209 + -18.7891*tf.T9i + 5.10182*tf.T913
                  + 0.379373*tf.T9 + -0.0672515*tf.T953 + -1.5*tf.lnT9)
    # il10r
    rate += np.exp(  5.13169 + -15.9452*tf.T9i
                  + -1.5*tf.lnT9)

    rate_eval.He4_N14__p_O17 = rate

@numba.njit()
def p_N15__n_O15(rate_eval, tf):
    # N15 + p --> n + O15
    rate = 0.0

    # nacrn
    rate += np.exp(  18.3942 + -41.0335*tf.T9i + 0.331392*tf.T913
                  + 0.0171473*tf.T9)

    rate_eval.p_N15__n_O15 = rate

@numba.njit()
def p_N15__He4_C12(rate_eval, tf):
    # N15 + p --> He4 + C12
    rate = 0.0

    # nacrn
    rate += np.exp(  27.4764 + -15.253*tf.T913i + 1.59318*tf.T913
                  + 2.4479*tf.T9 + -2.19708*tf.T953 + -0.666667*tf.lnT9)
    # nacrr
    rate += np.exp(  -6.57522 + -1.1638*tf.T9i + 22.7105*tf.T913
                  + -2.90707*tf.T9 + 0.205754*tf.T953 + -1.5*tf.lnT9)
    # nacrr
    rate += np.exp(  20.8972 + -7.406*tf.T9i
                  + -1.5*tf.lnT9)
    # nacrr
    rate += np.exp(  -4.87347 + -2.02117*tf.T9i + 30.8497*tf.T913
                  + -8.50433*tf.T9 + -1.54426*tf.T953 + -1.5*tf.lnT9)

    rate_eval.p_N15__He4_C12 = rate

@numba.njit()
def He4_N15__n_F18(rate_eval, tf):
    # N15 + He4 --> n + F18
    rate = 0.0

    # cf88n
    rate += np.exp(  18.0938 + -74.4713*tf.T9i + 1.74308*tf.T913
                  + -1.15123*tf.T9 + 0.135196*tf.T953)

    rate_eval.He4_N15__n_F18 = rate

@numba.njit()
def n_O15__p_N15(rate_eval, tf):
    # O15 + n --> p + N15
    rate = 0.0

    # nacrn
    rate += np.exp(  18.3942 + 0.331392*tf.T913
                  + 0.0171473*tf.T9)

    rate_eval.n_O15__p_N15 = rate

@numba.njit()
def n_O15__He4_C12(rate_eval, tf):
    # O15 + n --> He4 + C12
    rate = 0.0

    # cf88n
    rate += np.exp(  17.3699 + 0.124787*tf.T913
                  + 0.0588937*tf.T9 + -0.00679206*tf.T953)

    rate_eval.n_O15__He4_C12 = rate

@numba.njit()
def He4_O15__p_F18(rate_eval, tf):
    # O15 + He4 --> p + F18
    rate = 0.0

    # il10r
    rate += np.exp(  1.04969 + -36.4627*tf.T9i + 13.3223*tf.T913
                  + -1.36696*tf.T9 + 0.0757363*tf.T953 + -1.5*tf.lnT9)
    # il10r
    rate += np.exp(  -32.4461 + -33.8223*tf.T9i + 61.738*tf.T913
                  + -108.29*tf.T9 + -34.2365*tf.T953 + -1.5*tf.lnT9)
    # il10n
    rate += np.exp(  61.2985 + -33.4459*tf.T9i + -21.4023*tf.T913i + -80.8891*tf.T913
                  + 134.6*tf.T9 + -126.504*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He4_O15__p_F18 = rate

@numba.njit()
def n_O16__He4_C13(rate_eval, tf):
    # O16 + n --> He4 + C13
    rate = 0.0

    # gl12 
    rate += np.exp(  81.0688 + -26.0159*tf.T9i + 7.43132*tf.T913i + -84.8689*tf.T913
                  + 3.65083*tf.T9 + -0.148015*tf.T953 + 37.6008*tf.lnT9)
    # gl12 
    rate += np.exp(  64.3455 + -25.7388*tf.T9i + -32.3917*tf.T913i + -48.934*tf.T913
                  + 44.1843*tf.T9 + -20.8743*tf.T953 + 2.02494*tf.lnT9)

    rate_eval.n_O16__He4_C13 = rate

@numba.njit()
def p_O16__He4_N13(rate_eval, tf):
    # O16 + p --> He4 + N13
    rate = 0.0

    # cf88n
    rate += np.exp(  42.2324 + -60.5523*tf.T9i + -35.829*tf.T913i + -0.530275*tf.T913
                  + -0.982462*tf.T9 + 0.0808059*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_O16__He4_N13 = rate

@numba.njit()
def He4_O16__n_Ne19(rate_eval, tf):
    # O16 + He4 --> n + Ne19
    rate = 0.0

    # ths8r
    rate += np.exp(  17.2055 + -140.818*tf.T9i + 1.70736*tf.T913
                  + -0.132579*tf.T9 + 0.00454218*tf.T953)

    rate_eval.He4_O16__n_Ne19 = rate

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
def O16_O16__n_S31(rate_eval, tf):
    # O16 + O16 --> n + S31
    rate = 0.0

    # cf88r
    rate += np.exp(  77.5491 + -0.373641*tf.T9i + -120.83*tf.T913i + -7.72334*tf.T913
                  + -2.27939*tf.T9 + 0.167655*tf.T953 + 7.62001*tf.lnT9)

    rate_eval.O16_O16__n_S31 = rate

@numba.njit()
def O16_O16__p_P31(rate_eval, tf):
    # O16 + O16 --> p + P31
    rate = 0.0

    # cf88r
    rate += np.exp(  85.2628 + 0.223453*tf.T9i + -145.844*tf.T913i + 8.72612*tf.T913
                  + -0.554035*tf.T9 + -0.137562*tf.T953 + -6.88807*tf.lnT9)

    rate_eval.O16_O16__p_P31 = rate

@numba.njit()
def O16_O16__He4_Si28(rate_eval, tf):
    # O16 + O16 --> He4 + Si28
    rate = 0.0

    # cf88r
    rate += np.exp(  97.2435 + -0.268514*tf.T9i + -119.324*tf.T913i + -32.2497*tf.T913
                  + 1.46214*tf.T9 + -0.200893*tf.T953 + 13.2148*tf.lnT9)

    rate_eval.O16_O16__He4_Si28 = rate

@numba.njit()
def p_O17__He4_N14(rate_eval, tf):
    # O17 + p --> He4 + N14
    rate = 0.0

    # il10r
    rate += np.exp(  5.5336 + -2.11477*tf.T9i
                  + -1.5*tf.lnT9)
    # il10r
    rate += np.exp(  -7.20763 + -0.753395*tf.T9i
                  + -1.5*tf.lnT9)
    # il10n
    rate += np.exp(  19.579 + -16.9078*tf.T913i
                  + -2.0*tf.T953 + -0.666667*tf.lnT9)
    # il10r
    rate += np.exp(  10.174 + -4.95865*tf.T9i + 5.10182*tf.T913
                  + 0.379373*tf.T9 + -0.0672515*tf.T953 + -1.5*tf.lnT9)

    rate_eval.p_O17__He4_N14 = rate

@numba.njit()
def He4_O17__n_Ne20(rate_eval, tf):
    # O17 + He4 --> n + Ne20
    rate = 0.0

    # nacrn
    rate += np.exp(  40.621 + -39.918*tf.T913i
                  + 0.227017*tf.T9 + -0.900234*tf.T953 + -0.666667*tf.lnT9)
    # nacrr
    rate += np.exp(  1.80342 + -13.8*tf.T9i + 12.6501*tf.T913
                  + -1.10938*tf.T9 + 0.0696232*tf.T953 + -1.5*tf.lnT9)
    # nacrr
    rate += np.exp(  7.45588 + -8.55*tf.T9i
                  + -1.5*tf.lnT9)

    rate_eval.He4_O17__n_Ne20 = rate

@numba.njit()
def n_F18__He4_N15(rate_eval, tf):
    # F18 + n --> He4 + N15
    rate = 0.0

    # cf88n
    rate += np.exp(  18.8011 + 1.74308*tf.T913
                  + -1.15123*tf.T9 + 0.135196*tf.T953)

    rate_eval.n_F18__He4_N15 = rate

@numba.njit()
def p_F18__He4_O15(rate_eval, tf):
    # F18 + p --> He4 + O15
    rate = 0.0

    # il10n
    rate += np.exp(  62.0058 + -21.4023*tf.T913i + -80.8891*tf.T913
                  + 134.6*tf.T9 + -126.504*tf.T953 + -0.666667*tf.lnT9)
    # il10r
    rate += np.exp(  1.75704 + -3.01675*tf.T9i + 13.3223*tf.T913
                  + -1.36696*tf.T9 + 0.0757363*tf.T953 + -1.5*tf.lnT9)
    # il10r
    rate += np.exp(  -31.7388 + -0.376432*tf.T9i + 61.738*tf.T913
                  + -108.29*tf.T9 + -34.2365*tf.T953 + -1.5*tf.lnT9)

    rate_eval.p_F18__He4_O15 = rate

@numba.njit()
def He4_F18__p_Ne21(rate_eval, tf):
    # F18 + He4 --> p + Ne21
    rate = 0.0

    # rpsmr
    rate += np.exp(  49.7863 + -1.84559*tf.T9i + 21.4461*tf.T913i + -73.252*tf.T913
                  + 2.42329*tf.T9 + -0.077278*tf.T953 + 40.7604*tf.lnT9)

    rate_eval.He4_F18__p_Ne21 = rate

@numba.njit()
def n_Ne19__He4_O16(rate_eval, tf):
    # Ne19 + n --> He4 + O16
    rate = 0.0

    # ths8r
    rate += np.exp(  17.6409 + 1.70736*tf.T913
                  + -0.132579*tf.T9 + 0.00454218*tf.T953)

    rate_eval.n_Ne19__He4_O16 = rate

@numba.njit()
def He4_Ne19__p_Na22(rate_eval, tf):
    # Ne19 + He4 --> p + Na22
    rate = 0.0

    # ths8r
    rate += np.exp(  43.1874 + -46.6346*tf.T913i + 0.866532*tf.T913
                  + -0.893541*tf.T9 + 0.0747971*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He4_Ne19__p_Na22 = rate

@numba.njit()
def n_Ne20__He4_O17(rate_eval, tf):
    # Ne20 + n --> He4 + O17
    rate = 0.0

    # nacrr
    rate += np.exp(  4.7377 + -20.6002*tf.T9i + 12.6501*tf.T913
                  + -1.10938*tf.T9 + 0.0696232*tf.T953 + -1.5*tf.lnT9)
    # nacrr
    rate += np.exp(  10.3902 + -15.3502*tf.T9i
                  + -1.5*tf.lnT9)
    # nacrn
    rate += np.exp(  43.5553 + -6.80024*tf.T9i + -39.918*tf.T913i
                  + 0.227017*tf.T9 + -0.900234*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.n_Ne20__He4_O17 = rate

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
def C12_Ne20__n_S31(rate_eval, tf):
    # Ne20 + C12 --> n + S31
    rate = 0.0

    # rolfr
    rate += np.exp(  -342.129 + -54.118*tf.T9i + 638.957*tf.T913i + -289.04*tf.T913
                  + -3.53144*tf.T9 + 0.648991*tf.T953 + 297.721*tf.lnT9)

    rate_eval.C12_Ne20__n_S31 = rate

@numba.njit()
def C12_Ne20__p_P31(rate_eval, tf):
    # Ne20 + C12 --> p + P31
    rate = 0.0

    # rolfr
    rate += np.exp(  -268.136 + -38.7624*tf.T9i + 361.154*tf.T913i + -92.643*tf.T913
                  + -9.98738*tf.T9 + 0.892737*tf.T953 + 161.042*tf.lnT9)

    rate_eval.C12_Ne20__p_P31 = rate

@numba.njit()
def C12_Ne20__He4_Si28(rate_eval, tf):
    # Ne20 + C12 --> He4 + Si28
    rate = 0.0

    # rolfr
    rate += np.exp(  -308.905 + -47.2175*tf.T9i + 514.197*tf.T913i + -200.896*tf.T913
                  + -6.42713*tf.T9 + 0.758256*tf.T953 + 236.359*tf.lnT9)

    rate_eval.C12_Ne20__He4_Si28 = rate

@numba.njit()
def p_Ne21__He4_F18(rate_eval, tf):
    # Ne21 + p --> He4 + F18
    rate = 0.0

    # rpsmr
    rate += np.exp(  50.6536 + -22.049*tf.T9i + 21.4461*tf.T913i + -73.252*tf.T913
                  + 2.42329*tf.T9 + -0.077278*tf.T953 + 40.7604*tf.lnT9)

    rate_eval.p_Ne21__He4_F18 = rate

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
def p_Ne22__n_Na22(rate_eval, tf):
    # Ne22 + p --> n + Na22
    rate = 0.0

    # cf88n
    rate += np.exp(  21.5948 + -42.0547*tf.T9i + -0.0514777*tf.T913
                  + 0.0274055*tf.T9 + -0.00690277*tf.T953)

    rate_eval.p_Ne22__n_Na22 = rate

@numba.njit()
def He4_Ne22__n_Mg25(rate_eval, tf):
    # Ne22 + He4 --> n + Mg25
    rate = 0.0

    # li12r
    rate += np.exp(  -27.5027 + -7.38607*tf.T9i + 35.987*tf.T913
                  + -4.12183*tf.T9 + 0.263326*tf.T953 + -1.5*tf.lnT9)
    # li12n
    rate += np.exp(  -10.4729 + -5.55032*tf.T9i + 15.4898*tf.T913
                  + -30.8154*tf.T9 + -3.0*tf.T953 + 2.0*tf.lnT9)
    # li12n
    rate += np.exp(  -52.326 + -5.55032*tf.T9i + 88.2725*tf.T913
                  + -40.1578*tf.T9 + -3.0*tf.T953)

    rate_eval.He4_Ne22__n_Mg25 = rate

@numba.njit()
def n_Na22__p_Ne22(rate_eval, tf):
    # Na22 + n --> p + Ne22
    rate = 0.0

    # cf88n
    rate += np.exp(  19.6489 + -0.0514777*tf.T913
                  + 0.0274055*tf.T9 + -0.00690277*tf.T953)

    rate_eval.n_Na22__p_Ne22 = rate

@numba.njit()
def p_Na22__He4_Ne19(rate_eval, tf):
    # Na22 + p --> He4 + Ne19
    rate = 0.0

    # ths8r
    rate += np.exp(  43.101 + -24.0192*tf.T9i + -46.6346*tf.T913i + 0.866532*tf.T913
                  + -0.893541*tf.T9 + 0.0747971*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_Na22__He4_Ne19 = rate

@numba.njit()
def He4_Na22__n_Al25(rate_eval, tf):
    # Na22 + He4 --> n + Al25
    rate = 0.0

    # ths8r
    rate += np.exp(  7.59058 + -22.1956*tf.T9i + 2.92382*tf.T913
                  + 0.706669*tf.T9 + -0.0950292*tf.T953)

    rate_eval.He4_Na22__n_Al25 = rate

@numba.njit()
def He4_Na22__p_Mg25(rate_eval, tf):
    # Na22 + He4 --> p + Mg25
    rate = 0.0

    # ths8r
    rate += np.exp(  44.973 + -50.0924*tf.T913i + 0.807739*tf.T913
                  + -0.956029*tf.T9 + 0.0793321*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He4_Na22__p_Mg25 = rate

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
def He4_Na23__n_Al26(rate_eval, tf):
    # Na23 + He4 --> n + Al26
    rate = 0.0

    # ol11r
    rate += np.exp(  14.5219 + -34.8285*tf.T9i
                  + -1.5*tf.lnT9)
    # ol11r
    rate += np.exp(  13.4292 + -34.4845*tf.T9i
                  + -1.5*tf.lnT9)
    # ol11n
    rate += np.exp(  11.4506 + -34.4184*tf.T9i + 5.07134*tf.T913
                  + -0.557537*tf.T9 + 0.0451737*tf.T953)

    rate_eval.He4_Na23__n_Al26 = rate

@numba.njit()
def He4_Na23__p_Mg26(rate_eval, tf):
    # Na23 + He4 --> p + Mg26
    rate = 0.0

    # ths8r
    rate += np.exp(  44.527 + -50.2042*tf.T913i + 1.76141*tf.T913
                  + -1.36813*tf.T9 + 0.123087*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He4_Na23__p_Mg26 = rate

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
def He4_Mg23__n_Si26(rate_eval, tf):
    # Mg23 + He4 --> n + Si26
    rate = 0.0

    # ths8r
    rate += np.exp(  13.5183 + -46.1342*tf.T9i + 1.08923*tf.T913
                  + -0.0248723*tf.T9 + 0.00450822*tf.T953)

    rate_eval.He4_Mg23__n_Si26 = rate

@numba.njit()
def He4_Mg23__p_Al26(rate_eval, tf):
    # Mg23 + He4 --> p + Al26
    rate = 0.0

    # ths8r
    rate += np.exp(  46.215 + -53.203*tf.T913i + 0.71292*tf.T913
                  + -0.892548*tf.T9 + 0.0709813*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He4_Mg23__p_Al26 = rate

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
def n_Mg25__He4_Ne22(rate_eval, tf):
    # Mg25 + n --> He4 + Ne22
    rate = 0.0

    # li12n
    rate += np.exp(  -52.9232 + 88.2725*tf.T913
                  + -40.1578*tf.T9 + -3.0*tf.T953)
    # li12r
    rate += np.exp(  -28.0999 + -1.83575*tf.T9i + 35.987*tf.T913
                  + -4.12183*tf.T9 + 0.263326*tf.T953 + -1.5*tf.lnT9)
    # li12n
    rate += np.exp(  -11.0701 + 15.4898*tf.T913
                  + -30.8154*tf.T9 + -3.0*tf.T953 + 2.0*tf.lnT9)

    rate_eval.n_Mg25__He4_Ne22 = rate

@numba.njit()
def p_Mg25__n_Al25(rate_eval, tf):
    # Mg25 + p --> n + Al25
    rate = 0.0

    # ths8r
    rate += np.exp(  18.4104 + -58.7072*tf.T9i + 2.28536*tf.T913
                  + -0.38512*tf.T9 + 0.0288056*tf.T953)

    rate_eval.p_Mg25__n_Al25 = rate

@numba.njit()
def p_Mg25__He4_Na22(rate_eval, tf):
    # Mg25 + p --> He4 + Na22
    rate = 0.0

    # ths8r
    rate += np.exp(  46.3217 + -36.5117*tf.T9i + -50.0924*tf.T913i + 0.807739*tf.T913
                  + -0.956029*tf.T9 + 0.0793321*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_Mg25__He4_Na22 = rate

@numba.njit()
def He4_Mg25__n_Si28(rate_eval, tf):
    # Mg25 + He4 --> n + Si28
    rate = 0.0

    # nacr 
    rate += np.exp(  38.337 + -53.416*tf.T913i + 8.01209*tf.T913
                  + -2.64791*tf.T9 + 0.218637*tf.T953 + 0.333333*tf.lnT9)
    # nacrn
    rate += np.exp(  45.7613 + -53.415*tf.T913i + -1.46489*tf.T913
                  + 1.7777*tf.T9 + -0.903499*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He4_Mg25__n_Si28 = rate

@numba.njit()
def p_Mg26__n_Al26(rate_eval, tf):
    # Mg26 + p --> n + Al26
    rate = 0.0

    # ol11r
    rate += np.exp(  14.8989 + -55.6072*tf.T9i
                  + -1.5*tf.lnT9)
    # ol11n
    rate += np.exp(  13.5366 + -55.5463*tf.T9i + 7.36773*tf.T913
                  + -2.42424*tf.T9 + 0.313743*tf.T953)
    # ol11r
    rate += np.exp(  15.9629 + -55.7499*tf.T9i + 3.87515*tf.T913
                  + 0.228327*tf.T9 + -0.045872*tf.T953 + -1.5*tf.lnT9)

    rate_eval.p_Mg26__n_Al26 = rate

@numba.njit()
def p_Mg26__He4_Na23(rate_eval, tf):
    # Mg26 + p --> He4 + Na23
    rate = 0.0

    # ths8r
    rate += np.exp(  47.1157 + -21.128*tf.T9i + -50.2042*tf.T913i + 1.76141*tf.T913
                  + -1.36813*tf.T9 + 0.123087*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_Mg26__He4_Na23 = rate

@numba.njit()
def He4_Mg26__n_Si29(rate_eval, tf):
    # Mg26 + He4 --> n + Si29
    rate = 0.0

    # nacrr
    rate += np.exp(  15.7057 + -18.73*tf.T9i + -9.54056*tf.T913
                  + 4.71712*tf.T9 + -0.461053*tf.T953 + -1.5*tf.lnT9)
    # nacrn
    rate += np.exp(  45.8397 + -53.505*tf.T913i + 0.045598*tf.T913
                  + -0.194481*tf.T9 + -0.0748153*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He4_Mg26__n_Si29 = rate

@numba.njit()
def n_Al25__p_Mg25(rate_eval, tf):
    # Al25 + n --> p + Mg25
    rate = 0.0

    # ths8r
    rate += np.exp(  18.4104 + 2.28536*tf.T913
                  + -0.38512*tf.T9 + 0.0288056*tf.T953)

    rate_eval.n_Al25__p_Mg25 = rate

@numba.njit()
def n_Al25__He4_Na22(rate_eval, tf):
    # Al25 + n --> He4 + Na22
    rate = 0.0

    # ths8r
    rate += np.exp(  8.93927 + 2.92382*tf.T913
                  + 0.706669*tf.T9 + -0.0950292*tf.T953)

    rate_eval.n_Al25__He4_Na22 = rate

@numba.njit()
def He4_Al25__p_Si28(rate_eval, tf):
    # Al25 + He4 --> p + Si28
    rate = 0.0

    # ths8r
    rate += np.exp(  47.6167 + -56.3424*tf.T913i + 0.553763*tf.T913
                  + -0.84072*tf.T9 + 0.0634219*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He4_Al25__p_Si28 = rate

@numba.njit()
def n_Al26__p_Mg26(rate_eval, tf):
    # Al26 + n --> p + Mg26
    rate = 0.0

    # ol11r
    rate += np.exp(  13.565 + -0.203581*tf.T9i + 3.87515*tf.T913
                  + 0.228327*tf.T9 + -0.045872*tf.T953 + -1.5*tf.lnT9)
    # ol11r
    rate += np.exp(  12.501 + -0.0608268*tf.T9i
                  + -1.5*tf.lnT9)
    # ol11n
    rate += np.exp(  11.1387 + 7.36773*tf.T913
                  + -2.42424*tf.T9 + 0.313743*tf.T953)

    rate_eval.n_Al26__p_Mg26 = rate

@numba.njit()
def n_Al26__He4_Na23(rate_eval, tf):
    # Al26 + n --> He4 + Na23
    rate = 0.0

    # ol11r
    rate += np.exp(  13.62 + -0.0661138*tf.T9i
                  + -1.5*tf.lnT9)
    # ol11n
    rate += np.exp(  11.6414 + 5.07134*tf.T913
                  + -0.557537*tf.T9 + 0.0451737*tf.T953)
    # ol11r
    rate += np.exp(  14.7127 + -0.41015*tf.T9i
                  + -1.5*tf.lnT9)

    rate_eval.n_Al26__He4_Na23 = rate

@numba.njit()
def p_Al26__n_Si26(rate_eval, tf):
    # Al26 + p --> n + Si26
    rate = 0.0

    # ths8r
    rate += np.exp(  15.9928 + -67.8633*tf.T9i + 2.01618*tf.T913
                  + -0.300371*tf.T9 + 0.0228631*tf.T953)

    rate_eval.p_Al26__n_Si26 = rate

@numba.njit()
def p_Al26__He4_Mg23(rate_eval, tf):
    # Al26 + p --> He4 + Mg23
    rate = 0.0

    # ths8r
    rate += np.exp(  46.4058 + -21.7293*tf.T9i + -53.203*tf.T913i + 0.71292*tf.T913
                  + -0.892548*tf.T9 + 0.0709813*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_Al26__He4_Mg23 = rate

@numba.njit()
def He4_Al26__n_P29(rate_eval, tf):
    # Al26 + He4 --> n + P29
    rate = 0.0

    # ths8r
    rate += np.exp(  -42.1718 + -10.4914*tf.T9i + 41.9938*tf.T913
                  + -4.54859*tf.T9 + 0.243841*tf.T953)

    rate_eval.He4_Al26__n_P29 = rate

@numba.njit()
def He4_Al26__p_Si29(rate_eval, tf):
    # Al26 + He4 --> p + Si29
    rate = 0.0

    # ths8r
    rate += np.exp(  47.7092 + -56.4422*tf.T913i + 0.705353*tf.T913
                  + -0.957427*tf.T9 + 0.0756045*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He4_Al26__p_Si29 = rate

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
def He4_Al27__n_P30(rate_eval, tf):
    # Al27 + He4 --> n + P30
    rate = 0.0

    # nacr 
    rate += np.exp(  11.3074 + -30.667*tf.T9i + 0.311974*tf.T913
                  + -2.02044*tf.T9)
    # nacr 
    rate += np.exp(  6.09163 + -30.667*tf.T9i + 5.40982*tf.T913
                  + -0.265676*tf.T9 + 1.0*tf.lnT9)

    rate_eval.He4_Al27__n_P30 = rate

@numba.njit()
def He4_Al27__p_Si30(rate_eval, tf):
    # Al27 + He4 --> p + Si30
    rate = 0.0

    # ths8r
    rate += np.exp(  47.4856 + -56.5351*tf.T913i + 1.60477*tf.T913
                  + -1.40594*tf.T9 + 0.127353*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He4_Al27__p_Si30 = rate

@numba.njit()
def n_Si26__p_Al26(rate_eval, tf):
    # Si26 + n --> p + Al26
    rate = 0.0

    # ths8r
    rate += np.exp(  18.3907 + 2.01618*tf.T913
                  + -0.300371*tf.T9 + 0.0228631*tf.T953)

    rate_eval.n_Si26__p_Al26 = rate

@numba.njit()
def n_Si26__He4_Mg23(rate_eval, tf):
    # Si26 + n --> He4 + Mg23
    rate = 0.0

    # ths8r
    rate += np.exp(  16.107 + 1.08923*tf.T913
                  + -0.0248723*tf.T9 + 0.00450822*tf.T953)

    rate_eval.n_Si26__He4_Mg23 = rate

@numba.njit()
def He4_Si26__p_P29(rate_eval, tf):
    # Si26 + He4 --> p + P29
    rate = 0.0

    # ths8r
    rate += np.exp(  48.8732 + -59.3013*tf.T913i + 0.480742*tf.T913
                  + -0.834505*tf.T9 + 0.0621841*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He4_Si26__p_P29 = rate

@numba.njit()
def n_Si28__He4_Mg25(rate_eval, tf):
    # Si28 + n --> He4 + Mg25
    rate = 0.0

    # nacrn
    rate += np.exp(  48.7694 + -30.7983*tf.T9i + -53.415*tf.T913i + -1.46489*tf.T913
                  + 1.7777*tf.T9 + -0.903499*tf.T953 + -0.666667*tf.lnT9)
    # nacr 
    rate += np.exp(  41.3451 + -30.7983*tf.T9i + -53.416*tf.T913i + 8.01209*tf.T913
                  + -2.64791*tf.T9 + 0.218637*tf.T953 + 0.333333*tf.lnT9)

    rate_eval.n_Si28__He4_Mg25 = rate

@numba.njit()
def p_Si28__He4_Al25(rate_eval, tf):
    # Si28 + p --> He4 + Al25
    rate = 0.0

    # ths8r
    rate += np.exp(  50.6248 + -89.5005*tf.T9i + -56.3424*tf.T913i + 0.553763*tf.T913
                  + -0.84072*tf.T9 + 0.0634219*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_Si28__He4_Al25 = rate

@numba.njit()
def He4_Si28__n_S31(rate_eval, tf):
    # Si28 + He4 --> n + S31
    rate = 0.0

    # ths8r
    rate += np.exp(  17.5194 + -93.9332*tf.T9i + 1.12938*tf.T913
                  + -0.122539*tf.T9 + 0.00424404*tf.T953)

    rate_eval.He4_Si28__n_S31 = rate

@numba.njit()
def He4_Si28__p_P31(rate_eval, tf):
    # Si28 + He4 --> p + P31
    rate = 0.0

    # il10r
    rate += np.exp(  -11.4335 + -25.6606*tf.T9i + 21.521*tf.T913
                  + -1.90355*tf.T9 + 0.092724*tf.T953 + -1.5*tf.lnT9)
    # il10n
    rate += np.exp(  60.3424 + -22.2348*tf.T9i + -31.932*tf.T913i + -77.0334*tf.T913
                  + -43.6847*tf.T9 + -4.28955*tf.T953 + -0.666667*tf.lnT9)
    # il10r
    rate += np.exp(  -13.4595 + -24.112*tf.T9i
                  + -1.5*tf.lnT9)

    rate_eval.He4_Si28__p_P31 = rate

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
def n_Si29__He4_Mg26(rate_eval, tf):
    # Si29 + n --> He4 + Mg26
    rate = 0.0

    # nacrr
    rate += np.exp(  16.235 + -19.1246*tf.T9i + -9.54056*tf.T913
                  + 4.71712*tf.T9 + -0.461053*tf.T953 + -1.5*tf.lnT9)
    # nacrn
    rate += np.exp(  46.369 + -0.394553*tf.T9i + -53.505*tf.T913i + 0.045598*tf.T913
                  + -0.194481*tf.T9 + -0.0748153*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.n_Si29__He4_Mg26 = rate

@numba.njit()
def p_Si29__n_P29(rate_eval, tf):
    # Si29 + p --> n + P29
    rate = 0.0

    # ths8r
    rate += np.exp(  18.5351 + -66.4331*tf.T9i + 1.48018*tf.T913
                  + -0.177129*tf.T9 + 0.0127163*tf.T953)

    rate_eval.p_Si29__n_P29 = rate

@numba.njit()
def p_Si29__He4_Al26(rate_eval, tf):
    # Si29 + p --> He4 + Al26
    rate = 0.0

    # ths8r
    rate += np.exp(  50.6364 + -55.9416*tf.T9i + -56.4422*tf.T913i + 0.705353*tf.T913
                  + -0.957427*tf.T9 + 0.0756045*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_Si29__He4_Al26 = rate

@numba.njit()
def He4_Si29__n_S32(rate_eval, tf):
    # Si29 + He4 --> n + S32
    rate = 0.0

    # ths8r
    rate += np.exp(  -2.87932 + -17.7056*tf.T9i + 9.48125*tf.T913
                  + 0.4472*tf.T9 + -0.119237*tf.T953)

    rate_eval.He4_Si29__n_S32 = rate

@numba.njit()
def p_Si30__n_P30(rate_eval, tf):
    # Si30 + p --> n + P30
    rate = 0.0

    # ths8r
    rate += np.exp(  19.4345 + -58.1931*tf.T9i + 1.88379*tf.T913
                  + -0.330243*tf.T9 + 0.0239836*tf.T953)

    rate_eval.p_Si30__n_P30 = rate

@numba.njit()
def p_Si30__He4_Al27(rate_eval, tf):
    # Si30 + p --> He4 + Al27
    rate = 0.0

    # ths8r
    rate += np.exp(  50.5056 + -27.5284*tf.T9i + -56.5351*tf.T913i + 1.60477*tf.T913
                  + -1.40594*tf.T9 + 0.127353*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_Si30__He4_Al27 = rate

@numba.njit()
def He4_Si30__n_S33(rate_eval, tf):
    # Si30 + He4 --> n + S33
    rate = 0.0

    # ths8r
    rate += np.exp(  17.3228 + -40.5385*tf.T9i + -0.725571*tf.T913
                  + 0.430731*tf.T9 + -0.0323663*tf.T953)

    rate_eval.He4_Si30__n_S33 = rate

@numba.njit()
def n_P29__p_Si29(rate_eval, tf):
    # P29 + n --> p + Si29
    rate = 0.0

    # ths8r
    rate += np.exp(  18.5351 + 1.48018*tf.T913
                  + -0.177129*tf.T9 + 0.0127163*tf.T953)

    rate_eval.n_P29__p_Si29 = rate

@numba.njit()
def n_P29__He4_Al26(rate_eval, tf):
    # P29 + n --> He4 + Al26
    rate = 0.0

    # ths8r
    rate += np.exp(  -39.2446 + 41.9938*tf.T913
                  + -4.54859*tf.T9 + 0.243841*tf.T953)

    rate_eval.n_P29__He4_Al26 = rate

@numba.njit()
def p_P29__He4_Si26(rate_eval, tf):
    # P29 + p --> He4 + Si26
    rate = 0.0

    # ths8r
    rate += np.exp(  49.4025 + -57.372*tf.T9i + -59.3013*tf.T913i + 0.480742*tf.T913
                  + -0.834505*tf.T9 + 0.0621841*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_P29__He4_Si26 = rate

@numba.njit()
def He4_P29__p_S32(rate_eval, tf):
    # P29 + He4 --> p + S32
    rate = 0.0

    # ths8r
    rate += np.exp(  50.2824 + -62.3802*tf.T913i + 0.459085*tf.T913
                  + -0.870169*tf.T9 + 0.0631143*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He4_P29__p_S32 = rate

@numba.njit()
def n_P30__p_Si30(rate_eval, tf):
    # P30 + n --> p + Si30
    rate = 0.0

    # ths8r
    rate += np.exp(  18.3359 + 1.88379*tf.T913
                  + -0.330243*tf.T9 + 0.0239836*tf.T953)

    rate_eval.n_P30__p_Si30 = rate

@numba.njit()
def n_P30__He4_Al27(rate_eval, tf):
    # P30 + n --> He4 + Al27
    rate = 0.0

    # nacr 
    rate += np.exp(  8.01303 + 0.0036935*tf.T9i + 5.40982*tf.T913
                  + -0.265676*tf.T9 + 1.0*tf.lnT9)
    # nacr 
    rate += np.exp(  13.2288 + 0.0036935*tf.T9i + 0.311974*tf.T913
                  + -2.02044*tf.T9)

    rate_eval.n_P30__He4_Al27 = rate

@numba.njit()
def p_P30__n_S30(rate_eval, tf):
    # P30 + p --> n + S30
    rate = 0.0

    # ths8r
    rate += np.exp(  17.3211 + -80.3077*tf.T9i + 1.95764*tf.T913
                  + -0.276549*tf.T9 + 0.0196548*tf.T953)

    rate_eval.p_P30__n_S30 = rate

@numba.njit()
def He4_P30__n_Cl33(rate_eval, tf):
    # P30 + He4 --> n + Cl33
    rate = 0.0

    # ths8r
    rate += np.exp(  14.9462 + -56.2069*tf.T9i + 1.33535*tf.T913
                  + -0.149988*tf.T9 + 0.0279218*tf.T953)

    rate_eval.He4_P30__n_Cl33 = rate

@numba.njit()
def He4_P30__p_S33(rate_eval, tf):
    # P30 + He4 --> p + S33
    rate = 0.0

    # ths8r
    rate += np.exp(  49.813 + -62.4643*tf.T913i + 0.492934*tf.T913
                  + -0.841855*tf.T9 + 0.059263*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He4_P30__p_S33 = rate

@numba.njit()
def p_P31__n_S31(rate_eval, tf):
    # P31 + p --> n + S31
    rate = 0.0

    # ths8r
    rate += np.exp(  17.7037 + -71.6993*tf.T9i + 2.15006*tf.T913
                  + -0.300369*tf.T9 + 0.0229589*tf.T953)

    rate_eval.p_P31__n_S31 = rate

@numba.njit()
def p_P31__He4_Si28(rate_eval, tf):
    # P31 + p --> He4 + Si28
    rate = 0.0

    # il10r
    rate += np.exp(  -10.893 + -3.42575*tf.T9i + 21.521*tf.T913
                  + -1.90355*tf.T9 + 0.092724*tf.T953 + -1.5*tf.lnT9)
    # il10r
    rate += np.exp(  -12.919 + -1.87716*tf.T9i
                  + -1.5*tf.lnT9)
    # il10n
    rate += np.exp(  60.8829 + -31.932*tf.T913i + -77.0334*tf.T913
                  + -43.6847*tf.T9 + -4.28955*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_P31__He4_Si28 = rate

@numba.njit()
def p_P31__C12_Ne20(rate_eval, tf):
    # P31 + p --> C12 + Ne20
    rate = 0.0

    # rolfr
    rate += np.exp(  -266.452 + -156.019*tf.T9i + 361.154*tf.T913i + -92.643*tf.T913
                  + -9.98738*tf.T9 + 0.892737*tf.T953 + 161.042*tf.lnT9)

    rate_eval.p_P31__C12_Ne20 = rate

@numba.njit()
def p_P31__O16_O16(rate_eval, tf):
    # P31 + p --> O16 + O16
    rate = 0.0

    # cf88r
    rate += np.exp(  86.3501 + -88.8797*tf.T9i + -145.844*tf.T913i + 8.72612*tf.T913
                  + -0.554035*tf.T9 + -0.137562*tf.T953 + -6.88807*tf.lnT9)

    rate_eval.p_P31__O16_O16 = rate

@numba.njit()
def He4_P31__n_Cl34(rate_eval, tf):
    # P31 + He4 --> n + Cl34
    rate = 0.0

    # ths8r
    rate += np.exp(  16.9244 + -65.5365*tf.T9i + -0.466069*tf.T913
                  + 0.167169*tf.T9 + -0.00537463*tf.T953)

    rate_eval.He4_P31__n_Cl34 = rate

@numba.njit()
def n_S30__p_P30(rate_eval, tf):
    # S30 + n --> p + P30
    rate = 0.0

    # ths8r
    rate += np.exp(  18.4197 + 1.95764*tf.T913
                  + -0.276549*tf.T9 + 0.0196548*tf.T953)

    rate_eval.n_S30__p_P30 = rate

@numba.njit()
def He4_S30__p_Cl33(rate_eval, tf):
    # S30 + He4 --> p + Cl33
    rate = 0.0

    # ths8r
    rate += np.exp(  50.7513 + -65.211*tf.T913i + 0.403887*tf.T913
                  + -0.808239*tf.T9 + 0.0581461*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He4_S30__p_Cl33 = rate

@numba.njit()
def n_S31__p_P31(rate_eval, tf):
    # S31 + n --> p + P31
    rate = 0.0

    # ths8r
    rate += np.exp(  17.7037 + 2.15006*tf.T913
                  + -0.300369*tf.T9 + 0.0229589*tf.T953)

    rate_eval.n_S31__p_P31 = rate

@numba.njit()
def n_S31__He4_Si28(rate_eval, tf):
    # S31 + n --> He4 + Si28
    rate = 0.0

    # ths8r
    rate += np.exp(  18.0599 + 1.12938*tf.T913
                  + -0.122539*tf.T9 + 0.00424404*tf.T953)

    rate_eval.n_S31__He4_Si28 = rate

@numba.njit()
def n_S31__C12_Ne20(rate_eval, tf):
    # S31 + n --> C12 + Ne20
    rate = 0.0

    # rolfr
    rate += np.exp(  -340.445 + -99.6981*tf.T9i + 638.957*tf.T913i + -289.04*tf.T913
                  + -3.53144*tf.T9 + 0.648991*tf.T953 + 297.721*tf.lnT9)

    rate_eval.n_S31__C12_Ne20 = rate

@numba.njit()
def n_S31__O16_O16(rate_eval, tf):
    # S31 + n --> O16 + O16
    rate = 0.0

    # cf88r
    rate += np.exp(  78.6364 + -17.7811*tf.T9i + -120.83*tf.T913i + -7.72334*tf.T913
                  + -2.27939*tf.T9 + 0.167655*tf.T953 + 7.62001*tf.lnT9)

    rate_eval.n_S31__O16_O16 = rate

@numba.njit()
def He4_S31__n_Ar34(rate_eval, tf):
    # S31 + He4 --> n + Ar34
    rate = 0.0

    # ths8r
    rate += np.exp(  14.2205 + -73.2688*tf.T9i + 1.91616*tf.T913
                  + -0.544035*tf.T9 + 0.0771515*tf.T953)

    rate_eval.He4_S31__n_Ar34 = rate

@numba.njit()
def He4_S31__p_Cl34(rate_eval, tf):
    # S31 + He4 --> p + Cl34
    rate = 0.0

    # ths8r
    rate += np.exp(  50.2997 + -65.2934*tf.T913i + 1.2687*tf.T913
                  + -1.08925*tf.T9 + 0.0898609*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He4_S31__p_Cl34 = rate

@numba.njit()
def n_S32__He4_Si29(rate_eval, tf):
    # S32 + n --> He4 + Si29
    rate = 0.0

    # ths8r
    rate += np.exp(  -0.947538 + 9.48125*tf.T913
                  + 0.4472*tf.T9 + -0.119237*tf.T953)

    rate_eval.n_S32__He4_Si29 = rate

@numba.njit()
def p_S32__He4_P29(rate_eval, tf):
    # S32 + p --> He4 + P29
    rate = 0.0

    # ths8r
    rate += np.exp(  52.2142 + -48.7275*tf.T9i + -62.3802*tf.T913i + 0.459085*tf.T913
                  + -0.870169*tf.T9 + 0.0631143*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_S32__He4_P29 = rate

@numba.njit()
def He4_S32__p_Cl35(rate_eval, tf):
    # S32 + He4 --> p + Cl35
    rate = 0.0

    # il10r
    rate += np.exp(  2.42563 + -27.6662*tf.T9i + 5.33756*tf.T913
                  + 1.64418*tf.T9 + -0.246167*tf.T953 + -1.5*tf.lnT9)
    # il10r
    rate += np.exp(  -0.877602 + -25.5914*tf.T9i
                  + -1.5*tf.lnT9)
    # il10r
    rate += np.exp(  -57.395 + -22.1894*tf.T9i + 25.5338*tf.T913
                  + 6.45824*tf.T9 + -0.950294*tf.T953 + -1.5*tf.lnT9)
    # il10n
    rate += np.exp(  32.2544 + -21.6564*tf.T9i + -30.9147*tf.T913i + -1.2345*tf.T913
                  + 22.5118*tf.T9 + -33.0589*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He4_S32__p_Cl35 = rate

@numba.njit()
def n_S33__He4_Si30(rate_eval, tf):
    # S33 + n --> He4 + Si30
    rate = 0.0

    # ths8r
    rate += np.exp(  17.1798 + -0.725571*tf.T913
                  + 0.430731*tf.T9 + -0.0323663*tf.T953)

    rate_eval.n_S33__He4_Si30 = rate

@numba.njit()
def p_S33__n_Cl33(rate_eval, tf):
    # S33 + p --> n + Cl33
    rate = 0.0

    # ths8r
    rate += np.exp(  18.5772 + -73.8616*tf.T9i + 1.39633*tf.T913
                  + -0.136457*tf.T9 + 0.00585594*tf.T953)

    rate_eval.p_S33__n_Cl33 = rate

@numba.njit()
def p_S33__He4_P30(rate_eval, tf):
    # S33 + p --> He4 + P30
    rate = 0.0

    # ths8r
    rate += np.exp(  50.7686 + -17.6546*tf.T9i + -62.4643*tf.T913i + 0.492934*tf.T913
                  + -0.841855*tf.T9 + 0.059263*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_S33__He4_P30 = rate

@numba.njit()
def He4_S33__n_Ar36(rate_eval, tf):
    # S33 + He4 --> n + Ar36
    rate = 0.0

    # ths8r
    rate += np.exp(  0.398258 + -23.219*tf.T9i + 5.44923*tf.T913
                  + 1.17359*tf.T9 + -0.175811*tf.T953)

    rate_eval.He4_S33__n_Ar36 = rate

@numba.njit()
def n_Cl33__p_S33(rate_eval, tf):
    # Cl33 + n --> p + S33
    rate = 0.0

    # ths8r
    rate += np.exp(  18.5772 + 1.39633*tf.T913
                  + -0.136457*tf.T9 + 0.00585594*tf.T953)

    rate_eval.n_Cl33__p_S33 = rate

@numba.njit()
def n_Cl33__He4_P30(rate_eval, tf):
    # Cl33 + n --> He4 + P30
    rate = 0.0

    # ths8r
    rate += np.exp(  15.9018 + 1.33535*tf.T913
                  + -0.149988*tf.T9 + 0.0279218*tf.T953)

    rate_eval.n_Cl33__He4_P30 = rate

@numba.njit()
def p_Cl33__He4_S30(rate_eval, tf):
    # Cl33 + p --> He4 + S30
    rate = 0.0

    # ths8r
    rate += np.exp(  50.6083 + -24.1008*tf.T9i + -65.211*tf.T913i + 0.403887*tf.T913
                  + -0.808239*tf.T9 + 0.0581461*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_Cl33__He4_S30 = rate

@numba.njit()
def He4_Cl33__p_Ar36(rate_eval, tf):
    # Cl33 + He4 --> p + Ar36
    rate = 0.0

    # ths8r
    rate += np.exp(  52.1588 + -68.1442*tf.T913i + 0.291238*tf.T913
                  + -0.791384*tf.T9 + 0.0524823*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He4_Cl33__p_Ar36 = rate

@numba.njit()
def n_Cl34__He4_P31(rate_eval, tf):
    # Cl34 + n --> He4 + P31
    rate = 0.0

    # ths8r
    rate += np.exp(  18.8653 + -0.466069*tf.T913
                  + 0.167169*tf.T9 + -0.00537463*tf.T953)

    rate_eval.n_Cl34__He4_P31 = rate

@numba.njit()
def p_Cl34__n_Ar34(rate_eval, tf):
    # Cl34 + p --> n + Ar34
    rate = 0.0

    # ths8r
    rate += np.exp(  18.8161 + -79.4316*tf.T9i + 1.04995*tf.T913
                  + -0.0436573*tf.T9 + -0.00168124*tf.T953)

    rate_eval.p_Cl34__n_Ar34 = rate

@numba.njit()
def p_Cl34__He4_S31(rate_eval, tf):
    # Cl34 + p --> He4 + S31
    rate = 0.0

    # ths8r
    rate += np.exp(  52.2406 + -6.16284*tf.T9i + -65.2934*tf.T913i + 1.2687*tf.T913
                  + -1.08925*tf.T9 + 0.0898609*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_Cl34__He4_S31 = rate

@numba.njit()
def He4_Cl34__p_Ar37(rate_eval, tf):
    # Cl34 + He4 --> p + Ar37
    rate = 0.0

    # ths8r
    rate += np.exp(  52.795 + -68.2165*tf.T913i + 0.330057*tf.T913
                  + -0.873334*tf.T9 + 0.0592127*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He4_Cl34__p_Ar37 = rate

@numba.njit()
def p_Cl35__He4_S32(rate_eval, tf):
    # Cl35 + p --> He4 + S32
    rate = 0.0

    # il10r
    rate += np.exp(  2.29121 + -6.00976*tf.T9i + 5.33756*tf.T913
                  + 1.64418*tf.T9 + -0.246167*tf.T953 + -1.5*tf.lnT9)
    # il10r
    rate += np.exp(  -1.01202 + -3.93495*tf.T9i
                  + -1.5*tf.lnT9)
    # il10r
    rate += np.exp(  -57.5294 + -0.532931*tf.T9i + 25.5338*tf.T913
                  + 6.45824*tf.T9 + -0.950294*tf.T953 + -1.5*tf.lnT9)
    # il10n
    rate += np.exp(  32.12 + -30.9147*tf.T913i + -1.2345*tf.T913
                  + 22.5118*tf.T9 + -33.0589*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_Cl35__He4_S32 = rate

@numba.njit()
def He4_Cl35__p_Ar38(rate_eval, tf):
    # Cl35 + He4 --> p + Ar38
    rate = 0.0

    # ths8r
    rate += np.exp(  51.1272 + -68.2848*tf.T913i + 2.5993*tf.T913
                  + -1.59144*tf.T9 + 0.137745*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He4_Cl35__p_Ar38 = rate

@numba.njit()
def n_Ar34__p_Cl34(rate_eval, tf):
    # Ar34 + n --> p + Cl34
    rate = 0.0

    # ths8r
    rate += np.exp(  18.8161 + 1.04995*tf.T913
                  + -0.0436573*tf.T9 + -0.00168124*tf.T953)

    rate_eval.n_Ar34__p_Cl34 = rate

@numba.njit()
def n_Ar34__He4_S31(rate_eval, tf):
    # Ar34 + n --> He4 + S31
    rate = 0.0

    # ths8r
    rate += np.exp(  16.1614 + 1.91616*tf.T913
                  + -0.544035*tf.T9 + 0.0771515*tf.T953)

    rate_eval.n_Ar34__He4_S31 = rate

@numba.njit()
def n_Ar36__He4_S33(rate_eval, tf):
    # Ar36 + n --> He4 + S33
    rate = 0.0

    # ths8r
    rate += np.exp(  3.04033 + 5.44923*tf.T913
                  + 1.17359*tf.T9 + -0.175811*tf.T953)

    rate_eval.n_Ar36__He4_S33 = rate

@numba.njit()
def p_Ar36__He4_Cl33(rate_eval, tf):
    # Ar36 + p --> He4 + Cl33
    rate = 0.0

    # ths8r
    rate += np.exp(  54.8009 + -50.6426*tf.T9i + -68.1442*tf.T913i + 0.291238*tf.T913
                  + -0.791384*tf.T9 + 0.0524823*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_Ar36__He4_Cl33 = rate

@numba.njit()
def He4_Ar36__p_K39(rate_eval, tf):
    # Ar36 + He4 --> p + K39
    rate = 0.0

    # ths8r
    rate += np.exp(  20.6367 + -14.9533*tf.T9i + -30.0732*tf.T913i + 7.03263*tf.T913
                  + -1.10085*tf.T9 + 0.133768*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He4_Ar36__p_K39 = rate

@numba.njit()
def p_Ar37__He4_Cl34(rate_eval, tf):
    # Ar37 + p --> He4 + Cl34
    rate = 0.0

    # ths8r
    rate += np.exp(  52.6682 + -19.0758*tf.T9i + -68.2165*tf.T913i + 0.330057*tf.T913
                  + -0.873334*tf.T9 + 0.0592127*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_Ar37__He4_Cl34 = rate

@numba.njit()
def He4_Ar37__n_Ca40(rate_eval, tf):
    # Ar37 + He4 --> n + Ca40
    rate = 0.0

    # ths8r
    rate += np.exp(  -12.0914 + -20.2822*tf.T9i + 13.8882*tf.T913
                  + 0.260223*tf.T9 + -0.108063*tf.T953)

    rate_eval.He4_Ar37__n_Ca40 = rate

@numba.njit()
def p_Ar38__He4_Cl35(rate_eval, tf):
    # Ar38 + p --> He4 + Cl35
    rate = 0.0

    # ths8r
    rate += np.exp(  53.7764 + -9.71246*tf.T9i + -68.2848*tf.T913i + 2.5993*tf.T913
                  + -1.59144*tf.T9 + 0.137745*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_Ar38__He4_Cl35 = rate

@numba.njit()
def p_Ar39__n_K39(rate_eval, tf):
    # Ar39 + p --> n + K39
    rate = 0.0

    # ths8r
    rate += np.exp(  -15.9025 + -2.52219*tf.T9i + 26.5209*tf.T913
                  + -3.63684*tf.T9 + 0.276163*tf.T953)

    rate_eval.p_Ar39__n_K39 = rate

@numba.njit()
def n_K39__p_Ar39(rate_eval, tf):
    # K39 + n --> p + Ar39
    rate = 0.0

    # ths8r
    rate += np.exp(  -15.2094 + 26.5209*tf.T913
                  + -3.63684*tf.T9 + 0.276163*tf.T953)

    rate_eval.n_K39__p_Ar39 = rate

@numba.njit()
def p_K39__He4_Ar36(rate_eval, tf):
    # K39 + p --> He4 + Ar36
    rate = 0.0

    # ths8r
    rate += np.exp(  20.5166 + -30.0732*tf.T913i + 7.03263*tf.T913
                  + -1.10085*tf.T9 + 0.133768*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_K39__He4_Ar36 = rate

@numba.njit()
def n_Ca40__He4_Ar37(rate_eval, tf):
    # Ca40 + n --> He4 + Ar37
    rate = 0.0

    # ths8r
    rate += np.exp(  -9.43576 + 13.8882*tf.T913
                  + 0.260223*tf.T9 + -0.108063*tf.T953)

    rate_eval.n_Ca40__He4_Ar37 = rate

@numba.njit()
def He4_Ca40__p_Sc43(rate_eval, tf):
    # Ca40 + He4 --> p + Sc43
    rate = 0.0

    # ths8r
    rate += np.exp(  35.6575 + -40.8757*tf.T9i + -32.1734*tf.T913i + 0.0296879*tf.T913
                  + -0.95232*tf.T9 + 0.129022*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He4_Ca40__p_Sc43 = rate

@numba.njit()
def p_Sc43__He4_Ca40(rate_eval, tf):
    # Sc43 + p --> He4 + Ca40
    rate = 0.0

    # ths8r
    rate += np.exp(  34.8559 + -32.1734*tf.T913i + 0.0296879*tf.T913
                  + -0.95232*tf.T9 + 0.129022*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_Sc43__He4_Ca40 = rate

@numba.njit()
def He4_Ti44__p_V47(rate_eval, tf):
    # Ti44 + He4 --> p + V47
    rate = 0.0

    # chw0r
    rate += np.exp(  -76.5154 + -10.7931*tf.T9i + 70.2835*tf.T913
                  + -7.99061*tf.T9 + 0.486213*tf.T953 + -1.5*tf.lnT9)

    rate_eval.He4_Ti44__p_V47 = rate

@numba.njit()
def p_V47__He4_Ti44(rate_eval, tf):
    # V47 + p --> He4 + Ti44
    rate = 0.0

    # chw0r
    rate += np.exp(  -76.6143 + -6.02945*tf.T9i + 70.2835*tf.T913
                  + -7.99061*tf.T9 + 0.486213*tf.T953 + -1.5*tf.lnT9)

    rate_eval.p_V47__He4_Ti44 = rate

@numba.njit()
def He4_Cr48__p_Mn51(rate_eval, tf):
    # Cr48 + He4 --> p + Mn51
    rate = 0.0

    # ths8r
    rate += np.exp(  59.2276 + -86.7459*tf.T913i + 1.05653*tf.T913
                  + -1.15757*tf.T9 + 0.0877546*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He4_Cr48__p_Mn51 = rate

@numba.njit()
def p_Mn51__He4_Cr48(rate_eval, tf):
    # Mn51 + p --> He4 + Cr48
    rate = 0.0

    # ths8r
    rate += np.exp(  58.7312 + -6.47654*tf.T9i + -86.7459*tf.T913i + 1.05653*tf.T913
                  + -1.15757*tf.T9 + 0.0877546*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_Mn51__He4_Cr48 = rate

@numba.njit()
def He4_Fe52__p_Co55(rate_eval, tf):
    # Fe52 + He4 --> p + Co55
    rate = 0.0

    # ths8r
    rate += np.exp(  62.2207 + -91.6819*tf.T913i + -0.329235*tf.T913
                  + -0.780924*tf.T9 + 0.0425179*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He4_Fe52__p_Co55 = rate

@numba.njit()
def p_Fe55__n_Co55(rate_eval, tf):
    # Fe55 + p --> n + Co55
    rate = 0.0

    # ths8r
    rate += np.exp(  21.4329 + -49.1353*tf.T9i + -1.62382*tf.T913
                  + 0.58115*tf.T9 + -0.0537057*tf.T953)

    rate_eval.p_Fe55__n_Co55 = rate

@numba.njit()
def He4_Fe55__n_Ni58(rate_eval, tf):
    # Fe55 + He4 --> n + Ni58
    rate = 0.0

    # ths8r
    rate += np.exp(  -4.73193 + -33.6308*tf.T9i + 3.44996*tf.T913
                  + 2.98226*tf.T9 + -0.387699*tf.T953)

    rate_eval.He4_Fe55__n_Ni58 = rate

@numba.njit()
def n_Co55__p_Fe55(rate_eval, tf):
    # Co55 + n --> p + Fe55
    rate = 0.0

    # ths8r
    rate += np.exp(  20.7398 + -1.62382*tf.T913
                  + 0.58115*tf.T9 + -0.0537057*tf.T953)

    rate_eval.n_Co55__p_Fe55 = rate

@numba.njit()
def p_Co55__He4_Fe52(rate_eval, tf):
    # Co55 + p --> He4 + Fe52
    rate = 0.0

    # ths8r
    rate += np.exp(  61.4434 + -9.65363*tf.T9i + -91.6819*tf.T913i + -0.329235*tf.T913
                  + -0.780924*tf.T9 + 0.0425179*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_Co55__He4_Fe52 = rate

@numba.njit()
def He4_Co55__p_Ni58(rate_eval, tf):
    # Co55 + He4 --> p + Ni58
    rate = 0.0

    # ths8r
    rate += np.exp(  60.2281 + -94.1404*tf.T913i + 3.39179*tf.T913
                  + -1.71062*tf.T9 + 0.133003*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.He4_Co55__p_Ni58 = rate

@numba.njit()
def n_Ni58__He4_Fe55(rate_eval, tf):
    # Ni58 + n --> He4 + Fe55
    rate = 0.0

    # ths8r
    rate += np.exp(  -2.03901 + 3.44996*tf.T913
                  + 2.98226*tf.T9 + -0.387699*tf.T953)

    rate_eval.n_Ni58__He4_Fe55 = rate

@numba.njit()
def p_Ni58__He4_Co55(rate_eval, tf):
    # Ni58 + p --> He4 + Co55
    rate = 0.0

    # ths8r
    rate += np.exp(  63.6142 + -15.5045*tf.T9i + -94.1404*tf.T913i + 3.39179*tf.T913
                  + -1.71062*tf.T9 + 0.133003*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_Ni58__He4_Co55 = rate

@numba.njit()
def p_B11__He4_He4_He4(rate_eval, tf):
    # B11 + p --> He4 + He4 + He4
    rate = 0.0

    # nacrr
    rate += np.exp(  -14.9395 + -1.724*tf.T9i + 8.49175*tf.T913i + 27.3254*tf.T913
                  + -3.72071*tf.T9 + 0.275516*tf.T953 + -1.5*tf.lnT9)
    # nacrn
    rate += np.exp(  28.6442 + -12.097*tf.T913i + -0.0496312*tf.T913
                  + 0.687736*tf.T9 + -0.564229*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_B11__He4_He4_He4 = rate

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

@numba.njit()
def He4_He4_He4__p_B11(rate_eval, tf):
    # He4 + He4 + He4 --> p + B11
    rate = 0.0

    # nacrr
    rate += np.exp(  -36.7224 + -102.474*tf.T9i + 8.49175*tf.T913i + 27.3254*tf.T913
                  + -3.72071*tf.T9 + 0.275516*tf.T953 + -3.0*tf.lnT9)
    # nacrn
    rate += np.exp(  6.8613 + -100.75*tf.T9i + -12.097*tf.T913i + -0.0496312*tf.T913
                  + 0.687736*tf.T9 + -0.564229*tf.T953 + -2.16667*tf.lnT9)

    rate_eval.He4_He4_He4__p_B11 = rate

def rhs(t, Y, rho, T, screen_func=None):
    return rhs_eq(t, Y, rho, T, screen_func)

@numba.njit()
def rhs_eq(t, Y, rho, T, screen_func):

    tf = Tfactors(T)
    rate_eval = RateEval()

    # reaclib rates
    n__p__weak__wc12(rate_eval, tf)
    N13__C13__weak__wc12(rate_eval, tf)
    O15__N15__weak__wc12(rate_eval, tf)
    Na22__Ne22__weak__wc12(rate_eval, tf)
    Mg23__Na23__weak__wc12(rate_eval, tf)
    Al25__Mg25__weak__wc12(rate_eval, tf)
    Al26__Mg26__weak__wc12(rate_eval, tf)
    Si26__Al26__weak__wc12(rate_eval, tf)
    P29__Si29__weak__wc12(rate_eval, tf)
    P30__Si30__weak__wc12(rate_eval, tf)
    S30__P30__weak__wc12(rate_eval, tf)
    S31__P31__weak__wc12(rate_eval, tf)
    Cl33__S33__weak__wc12(rate_eval, tf)
    Ar34__Cl34__weak__wc12(rate_eval, tf)
    Ar39__K39__weak__wc12(rate_eval, tf)
    Co55__Fe55__weak__wc12(rate_eval, tf)
    C12__p_B11(rate_eval, tf)
    C13__n_C12(rate_eval, tf)
    N13__p_C12(rate_eval, tf)
    N14__n_N13(rate_eval, tf)
    N14__p_C13(rate_eval, tf)
    N15__n_N14(rate_eval, tf)
    O15__p_N14(rate_eval, tf)
    O16__n_O15(rate_eval, tf)
    O16__p_N15(rate_eval, tf)
    O16__He4_C12(rate_eval, tf)
    O17__n_O16(rate_eval, tf)
    F18__p_O17(rate_eval, tf)
    F18__He4_N14(rate_eval, tf)
    Ne19__p_F18(rate_eval, tf)
    Ne19__He4_O15(rate_eval, tf)
    Ne20__n_Ne19(rate_eval, tf)
    Ne20__He4_O16(rate_eval, tf)
    Ne21__n_Ne20(rate_eval, tf)
    Ne21__He4_O17(rate_eval, tf)
    Ne22__n_Ne21(rate_eval, tf)
    Na22__p_Ne21(rate_eval, tf)
    Na22__He4_F18(rate_eval, tf)
    Na23__n_Na22(rate_eval, tf)
    Na23__p_Ne22(rate_eval, tf)
    Mg23__p_Na22(rate_eval, tf)
    Mg23__He4_Ne19(rate_eval, tf)
    Mg24__n_Mg23(rate_eval, tf)
    Mg24__p_Na23(rate_eval, tf)
    Mg24__He4_Ne20(rate_eval, tf)
    Mg25__n_Mg24(rate_eval, tf)
    Mg25__He4_Ne21(rate_eval, tf)
    Mg26__n_Mg25(rate_eval, tf)
    Mg26__He4_Ne22(rate_eval, tf)
    Al25__p_Mg24(rate_eval, tf)
    Al26__n_Al25(rate_eval, tf)
    Al26__p_Mg25(rate_eval, tf)
    Al26__He4_Na22(rate_eval, tf)
    Al27__n_Al26(rate_eval, tf)
    Al27__p_Mg26(rate_eval, tf)
    Al27__He4_Na23(rate_eval, tf)
    Si26__p_Al25(rate_eval, tf)
    Si28__p_Al27(rate_eval, tf)
    Si28__He4_Mg24(rate_eval, tf)
    Si29__n_Si28(rate_eval, tf)
    Si29__He4_Mg25(rate_eval, tf)
    Si30__n_Si29(rate_eval, tf)
    Si30__He4_Mg26(rate_eval, tf)
    P29__p_Si28(rate_eval, tf)
    P29__He4_Al25(rate_eval, tf)
    P30__n_P29(rate_eval, tf)
    P30__p_Si29(rate_eval, tf)
    P30__He4_Al26(rate_eval, tf)
    P31__n_P30(rate_eval, tf)
    P31__p_Si30(rate_eval, tf)
    P31__He4_Al27(rate_eval, tf)
    S30__p_P29(rate_eval, tf)
    S30__He4_Si26(rate_eval, tf)
    S31__n_S30(rate_eval, tf)
    S31__p_P30(rate_eval, tf)
    S32__n_S31(rate_eval, tf)
    S32__p_P31(rate_eval, tf)
    S32__He4_Si28(rate_eval, tf)
    S33__n_S32(rate_eval, tf)
    S33__He4_Si29(rate_eval, tf)
    Cl33__p_S32(rate_eval, tf)
    Cl33__He4_P29(rate_eval, tf)
    Cl34__n_Cl33(rate_eval, tf)
    Cl34__p_S33(rate_eval, tf)
    Cl34__He4_P30(rate_eval, tf)
    Cl35__n_Cl34(rate_eval, tf)
    Cl35__He4_P31(rate_eval, tf)
    Ar34__p_Cl33(rate_eval, tf)
    Ar34__He4_S30(rate_eval, tf)
    Ar36__p_Cl35(rate_eval, tf)
    Ar36__He4_S32(rate_eval, tf)
    Ar37__n_Ar36(rate_eval, tf)
    Ar37__He4_S33(rate_eval, tf)
    Ar38__n_Ar37(rate_eval, tf)
    Ar39__n_Ar38(rate_eval, tf)
    K39__p_Ar38(rate_eval, tf)
    K39__He4_Cl35(rate_eval, tf)
    Ca40__p_K39(rate_eval, tf)
    Ca40__He4_Ar36(rate_eval, tf)
    Sc43__He4_K39(rate_eval, tf)
    Ti44__p_Sc43(rate_eval, tf)
    Ti44__He4_Ca40(rate_eval, tf)
    V47__He4_Sc43(rate_eval, tf)
    Cr48__p_V47(rate_eval, tf)
    Cr48__He4_Ti44(rate_eval, tf)
    Mn51__He4_V47(rate_eval, tf)
    Fe52__p_Mn51(rate_eval, tf)
    Fe52__He4_Cr48(rate_eval, tf)
    Co55__He4_Mn51(rate_eval, tf)
    Ni56__p_Co55(rate_eval, tf)
    Ni56__He4_Fe52(rate_eval, tf)
    Ni59__n_Ni58(rate_eval, tf)
    Ni59__He4_Fe55(rate_eval, tf)
    C12__He4_He4_He4(rate_eval, tf)
    p_B11__C12(rate_eval, tf)
    n_C12__C13(rate_eval, tf)
    p_C12__N13(rate_eval, tf)
    He4_C12__O16(rate_eval, tf)
    p_C13__N14(rate_eval, tf)
    n_N13__N14(rate_eval, tf)
    n_N14__N15(rate_eval, tf)
    p_N14__O15(rate_eval, tf)
    He4_N14__F18(rate_eval, tf)
    p_N15__O16(rate_eval, tf)
    n_O15__O16(rate_eval, tf)
    He4_O15__Ne19(rate_eval, tf)
    n_O16__O17(rate_eval, tf)
    He4_O16__Ne20(rate_eval, tf)
    p_O17__F18(rate_eval, tf)
    He4_O17__Ne21(rate_eval, tf)
    p_F18__Ne19(rate_eval, tf)
    He4_F18__Na22(rate_eval, tf)
    n_Ne19__Ne20(rate_eval, tf)
    He4_Ne19__Mg23(rate_eval, tf)
    n_Ne20__Ne21(rate_eval, tf)
    He4_Ne20__Mg24(rate_eval, tf)
    n_Ne21__Ne22(rate_eval, tf)
    p_Ne21__Na22(rate_eval, tf)
    He4_Ne21__Mg25(rate_eval, tf)
    p_Ne22__Na23(rate_eval, tf)
    He4_Ne22__Mg26(rate_eval, tf)
    n_Na22__Na23(rate_eval, tf)
    p_Na22__Mg23(rate_eval, tf)
    He4_Na22__Al26(rate_eval, tf)
    p_Na23__Mg24(rate_eval, tf)
    He4_Na23__Al27(rate_eval, tf)
    n_Mg23__Mg24(rate_eval, tf)
    n_Mg24__Mg25(rate_eval, tf)
    p_Mg24__Al25(rate_eval, tf)
    He4_Mg24__Si28(rate_eval, tf)
    n_Mg25__Mg26(rate_eval, tf)
    p_Mg25__Al26(rate_eval, tf)
    He4_Mg25__Si29(rate_eval, tf)
    p_Mg26__Al27(rate_eval, tf)
    He4_Mg26__Si30(rate_eval, tf)
    n_Al25__Al26(rate_eval, tf)
    p_Al25__Si26(rate_eval, tf)
    He4_Al25__P29(rate_eval, tf)
    n_Al26__Al27(rate_eval, tf)
    He4_Al26__P30(rate_eval, tf)
    p_Al27__Si28(rate_eval, tf)
    He4_Al27__P31(rate_eval, tf)
    He4_Si26__S30(rate_eval, tf)
    n_Si28__Si29(rate_eval, tf)
    p_Si28__P29(rate_eval, tf)
    He4_Si28__S32(rate_eval, tf)
    n_Si29__Si30(rate_eval, tf)
    p_Si29__P30(rate_eval, tf)
    He4_Si29__S33(rate_eval, tf)
    p_Si30__P31(rate_eval, tf)
    n_P29__P30(rate_eval, tf)
    p_P29__S30(rate_eval, tf)
    He4_P29__Cl33(rate_eval, tf)
    n_P30__P31(rate_eval, tf)
    p_P30__S31(rate_eval, tf)
    He4_P30__Cl34(rate_eval, tf)
    p_P31__S32(rate_eval, tf)
    He4_P31__Cl35(rate_eval, tf)
    n_S30__S31(rate_eval, tf)
    He4_S30__Ar34(rate_eval, tf)
    n_S31__S32(rate_eval, tf)
    n_S32__S33(rate_eval, tf)
    p_S32__Cl33(rate_eval, tf)
    He4_S32__Ar36(rate_eval, tf)
    p_S33__Cl34(rate_eval, tf)
    He4_S33__Ar37(rate_eval, tf)
    n_Cl33__Cl34(rate_eval, tf)
    p_Cl33__Ar34(rate_eval, tf)
    n_Cl34__Cl35(rate_eval, tf)
    p_Cl35__Ar36(rate_eval, tf)
    He4_Cl35__K39(rate_eval, tf)
    n_Ar36__Ar37(rate_eval, tf)
    He4_Ar36__Ca40(rate_eval, tf)
    n_Ar37__Ar38(rate_eval, tf)
    n_Ar38__Ar39(rate_eval, tf)
    p_Ar38__K39(rate_eval, tf)
    p_K39__Ca40(rate_eval, tf)
    He4_K39__Sc43(rate_eval, tf)
    He4_Ca40__Ti44(rate_eval, tf)
    p_Sc43__Ti44(rate_eval, tf)
    He4_Sc43__V47(rate_eval, tf)
    He4_Ti44__Cr48(rate_eval, tf)
    p_V47__Cr48(rate_eval, tf)
    He4_V47__Mn51(rate_eval, tf)
    He4_Cr48__Fe52(rate_eval, tf)
    p_Mn51__Fe52(rate_eval, tf)
    He4_Mn51__Co55(rate_eval, tf)
    He4_Fe52__Ni56(rate_eval, tf)
    He4_Fe55__Ni59(rate_eval, tf)
    p_Co55__Ni56(rate_eval, tf)
    n_Ni58__Ni59(rate_eval, tf)
    He4_B11__n_N14(rate_eval, tf)
    He4_C12__n_O15(rate_eval, tf)
    He4_C12__p_N15(rate_eval, tf)
    C12_C12__n_Mg23(rate_eval, tf)
    C12_C12__p_Na23(rate_eval, tf)
    C12_C12__He4_Ne20(rate_eval, tf)
    p_C13__n_N13(rate_eval, tf)
    He4_C13__n_O16(rate_eval, tf)
    n_N13__p_C13(rate_eval, tf)
    He4_N13__p_O16(rate_eval, tf)
    n_N14__He4_B11(rate_eval, tf)
    He4_N14__p_O17(rate_eval, tf)
    p_N15__n_O15(rate_eval, tf)
    p_N15__He4_C12(rate_eval, tf)
    He4_N15__n_F18(rate_eval, tf)
    n_O15__p_N15(rate_eval, tf)
    n_O15__He4_C12(rate_eval, tf)
    He4_O15__p_F18(rate_eval, tf)
    n_O16__He4_C13(rate_eval, tf)
    p_O16__He4_N13(rate_eval, tf)
    He4_O16__n_Ne19(rate_eval, tf)
    C12_O16__p_Al27(rate_eval, tf)
    C12_O16__He4_Mg24(rate_eval, tf)
    O16_O16__n_S31(rate_eval, tf)
    O16_O16__p_P31(rate_eval, tf)
    O16_O16__He4_Si28(rate_eval, tf)
    p_O17__He4_N14(rate_eval, tf)
    He4_O17__n_Ne20(rate_eval, tf)
    n_F18__He4_N15(rate_eval, tf)
    p_F18__He4_O15(rate_eval, tf)
    He4_F18__p_Ne21(rate_eval, tf)
    n_Ne19__He4_O16(rate_eval, tf)
    He4_Ne19__p_Na22(rate_eval, tf)
    n_Ne20__He4_O17(rate_eval, tf)
    He4_Ne20__n_Mg23(rate_eval, tf)
    He4_Ne20__p_Na23(rate_eval, tf)
    He4_Ne20__C12_C12(rate_eval, tf)
    C12_Ne20__n_S31(rate_eval, tf)
    C12_Ne20__p_P31(rate_eval, tf)
    C12_Ne20__He4_Si28(rate_eval, tf)
    p_Ne21__He4_F18(rate_eval, tf)
    He4_Ne21__n_Mg24(rate_eval, tf)
    p_Ne22__n_Na22(rate_eval, tf)
    He4_Ne22__n_Mg25(rate_eval, tf)
    n_Na22__p_Ne22(rate_eval, tf)
    p_Na22__He4_Ne19(rate_eval, tf)
    He4_Na22__n_Al25(rate_eval, tf)
    He4_Na22__p_Mg25(rate_eval, tf)
    p_Na23__n_Mg23(rate_eval, tf)
    p_Na23__He4_Ne20(rate_eval, tf)
    p_Na23__C12_C12(rate_eval, tf)
    He4_Na23__n_Al26(rate_eval, tf)
    He4_Na23__p_Mg26(rate_eval, tf)
    n_Mg23__p_Na23(rate_eval, tf)
    n_Mg23__He4_Ne20(rate_eval, tf)
    n_Mg23__C12_C12(rate_eval, tf)
    He4_Mg23__n_Si26(rate_eval, tf)
    He4_Mg23__p_Al26(rate_eval, tf)
    n_Mg24__He4_Ne21(rate_eval, tf)
    He4_Mg24__p_Al27(rate_eval, tf)
    He4_Mg24__C12_O16(rate_eval, tf)
    n_Mg25__He4_Ne22(rate_eval, tf)
    p_Mg25__n_Al25(rate_eval, tf)
    p_Mg25__He4_Na22(rate_eval, tf)
    He4_Mg25__n_Si28(rate_eval, tf)
    p_Mg26__n_Al26(rate_eval, tf)
    p_Mg26__He4_Na23(rate_eval, tf)
    He4_Mg26__n_Si29(rate_eval, tf)
    n_Al25__p_Mg25(rate_eval, tf)
    n_Al25__He4_Na22(rate_eval, tf)
    He4_Al25__p_Si28(rate_eval, tf)
    n_Al26__p_Mg26(rate_eval, tf)
    n_Al26__He4_Na23(rate_eval, tf)
    p_Al26__n_Si26(rate_eval, tf)
    p_Al26__He4_Mg23(rate_eval, tf)
    He4_Al26__n_P29(rate_eval, tf)
    He4_Al26__p_Si29(rate_eval, tf)
    p_Al27__He4_Mg24(rate_eval, tf)
    p_Al27__C12_O16(rate_eval, tf)
    He4_Al27__n_P30(rate_eval, tf)
    He4_Al27__p_Si30(rate_eval, tf)
    n_Si26__p_Al26(rate_eval, tf)
    n_Si26__He4_Mg23(rate_eval, tf)
    He4_Si26__p_P29(rate_eval, tf)
    n_Si28__He4_Mg25(rate_eval, tf)
    p_Si28__He4_Al25(rate_eval, tf)
    He4_Si28__n_S31(rate_eval, tf)
    He4_Si28__p_P31(rate_eval, tf)
    He4_Si28__C12_Ne20(rate_eval, tf)
    He4_Si28__O16_O16(rate_eval, tf)
    n_Si29__He4_Mg26(rate_eval, tf)
    p_Si29__n_P29(rate_eval, tf)
    p_Si29__He4_Al26(rate_eval, tf)
    He4_Si29__n_S32(rate_eval, tf)
    p_Si30__n_P30(rate_eval, tf)
    p_Si30__He4_Al27(rate_eval, tf)
    He4_Si30__n_S33(rate_eval, tf)
    n_P29__p_Si29(rate_eval, tf)
    n_P29__He4_Al26(rate_eval, tf)
    p_P29__He4_Si26(rate_eval, tf)
    He4_P29__p_S32(rate_eval, tf)
    n_P30__p_Si30(rate_eval, tf)
    n_P30__He4_Al27(rate_eval, tf)
    p_P30__n_S30(rate_eval, tf)
    He4_P30__n_Cl33(rate_eval, tf)
    He4_P30__p_S33(rate_eval, tf)
    p_P31__n_S31(rate_eval, tf)
    p_P31__He4_Si28(rate_eval, tf)
    p_P31__C12_Ne20(rate_eval, tf)
    p_P31__O16_O16(rate_eval, tf)
    He4_P31__n_Cl34(rate_eval, tf)
    n_S30__p_P30(rate_eval, tf)
    He4_S30__p_Cl33(rate_eval, tf)
    n_S31__p_P31(rate_eval, tf)
    n_S31__He4_Si28(rate_eval, tf)
    n_S31__C12_Ne20(rate_eval, tf)
    n_S31__O16_O16(rate_eval, tf)
    He4_S31__n_Ar34(rate_eval, tf)
    He4_S31__p_Cl34(rate_eval, tf)
    n_S32__He4_Si29(rate_eval, tf)
    p_S32__He4_P29(rate_eval, tf)
    He4_S32__p_Cl35(rate_eval, tf)
    n_S33__He4_Si30(rate_eval, tf)
    p_S33__n_Cl33(rate_eval, tf)
    p_S33__He4_P30(rate_eval, tf)
    He4_S33__n_Ar36(rate_eval, tf)
    n_Cl33__p_S33(rate_eval, tf)
    n_Cl33__He4_P30(rate_eval, tf)
    p_Cl33__He4_S30(rate_eval, tf)
    He4_Cl33__p_Ar36(rate_eval, tf)
    n_Cl34__He4_P31(rate_eval, tf)
    p_Cl34__n_Ar34(rate_eval, tf)
    p_Cl34__He4_S31(rate_eval, tf)
    He4_Cl34__p_Ar37(rate_eval, tf)
    p_Cl35__He4_S32(rate_eval, tf)
    He4_Cl35__p_Ar38(rate_eval, tf)
    n_Ar34__p_Cl34(rate_eval, tf)
    n_Ar34__He4_S31(rate_eval, tf)
    n_Ar36__He4_S33(rate_eval, tf)
    p_Ar36__He4_Cl33(rate_eval, tf)
    He4_Ar36__p_K39(rate_eval, tf)
    p_Ar37__He4_Cl34(rate_eval, tf)
    He4_Ar37__n_Ca40(rate_eval, tf)
    p_Ar38__He4_Cl35(rate_eval, tf)
    p_Ar39__n_K39(rate_eval, tf)
    n_K39__p_Ar39(rate_eval, tf)
    p_K39__He4_Ar36(rate_eval, tf)
    n_Ca40__He4_Ar37(rate_eval, tf)
    He4_Ca40__p_Sc43(rate_eval, tf)
    p_Sc43__He4_Ca40(rate_eval, tf)
    He4_Ti44__p_V47(rate_eval, tf)
    p_V47__He4_Ti44(rate_eval, tf)
    He4_Cr48__p_Mn51(rate_eval, tf)
    p_Mn51__He4_Cr48(rate_eval, tf)
    He4_Fe52__p_Co55(rate_eval, tf)
    p_Fe55__n_Co55(rate_eval, tf)
    He4_Fe55__n_Ni58(rate_eval, tf)
    n_Co55__p_Fe55(rate_eval, tf)
    p_Co55__He4_Fe52(rate_eval, tf)
    He4_Co55__p_Ni58(rate_eval, tf)
    n_Ni58__He4_Fe55(rate_eval, tf)
    p_Ni58__He4_Co55(rate_eval, tf)
    p_B11__He4_He4_He4(rate_eval, tf)
    He4_He4_He4__C12(rate_eval, tf)
    He4_He4_He4__p_B11(rate_eval, tf)

    if screen_func is not None:
        plasma_state = PlasmaState(T, rho, Y, Z)

        scn_fac = ScreenFactors(1, 1, 5, 11)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_B11__C12 *= scor
        rate_eval.p_B11__He4_He4_He4 *= scor

        scn_fac = ScreenFactors(1, 1, 6, 12)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_C12__N13 *= scor

        scn_fac = ScreenFactors(2, 4, 6, 12)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_C12__O16 *= scor
        rate_eval.He4_C12__n_O15 *= scor
        rate_eval.He4_C12__p_N15 *= scor

        scn_fac = ScreenFactors(1, 1, 6, 13)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_C13__N14 *= scor
        rate_eval.p_C13__n_N13 *= scor

        scn_fac = ScreenFactors(1, 1, 7, 14)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_N14__O15 *= scor

        scn_fac = ScreenFactors(2, 4, 7, 14)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_N14__F18 *= scor
        rate_eval.He4_N14__p_O17 *= scor

        scn_fac = ScreenFactors(1, 1, 7, 15)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_N15__O16 *= scor
        rate_eval.p_N15__n_O15 *= scor
        rate_eval.p_N15__He4_C12 *= scor

        scn_fac = ScreenFactors(2, 4, 8, 15)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_O15__Ne19 *= scor
        rate_eval.He4_O15__p_F18 *= scor

        scn_fac = ScreenFactors(2, 4, 8, 16)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_O16__Ne20 *= scor
        rate_eval.He4_O16__n_Ne19 *= scor

        scn_fac = ScreenFactors(1, 1, 8, 17)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_O17__F18 *= scor
        rate_eval.p_O17__He4_N14 *= scor

        scn_fac = ScreenFactors(2, 4, 8, 17)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_O17__Ne21 *= scor
        rate_eval.He4_O17__n_Ne20 *= scor

        scn_fac = ScreenFactors(1, 1, 9, 18)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_F18__Ne19 *= scor
        rate_eval.p_F18__He4_O15 *= scor

        scn_fac = ScreenFactors(2, 4, 9, 18)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_F18__Na22 *= scor
        rate_eval.He4_F18__p_Ne21 *= scor

        scn_fac = ScreenFactors(2, 4, 10, 19)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Ne19__Mg23 *= scor
        rate_eval.He4_Ne19__p_Na22 *= scor

        scn_fac = ScreenFactors(2, 4, 10, 20)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Ne20__Mg24 *= scor
        rate_eval.He4_Ne20__n_Mg23 *= scor
        rate_eval.He4_Ne20__p_Na23 *= scor
        rate_eval.He4_Ne20__C12_C12 *= scor

        scn_fac = ScreenFactors(1, 1, 10, 21)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_Ne21__Na22 *= scor
        rate_eval.p_Ne21__He4_F18 *= scor

        scn_fac = ScreenFactors(2, 4, 10, 21)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Ne21__Mg25 *= scor
        rate_eval.He4_Ne21__n_Mg24 *= scor

        scn_fac = ScreenFactors(1, 1, 10, 22)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_Ne22__Na23 *= scor
        rate_eval.p_Ne22__n_Na22 *= scor

        scn_fac = ScreenFactors(2, 4, 10, 22)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Ne22__Mg26 *= scor
        rate_eval.He4_Ne22__n_Mg25 *= scor

        scn_fac = ScreenFactors(1, 1, 11, 22)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_Na22__Mg23 *= scor
        rate_eval.p_Na22__He4_Ne19 *= scor

        scn_fac = ScreenFactors(2, 4, 11, 22)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Na22__Al26 *= scor
        rate_eval.He4_Na22__n_Al25 *= scor
        rate_eval.He4_Na22__p_Mg25 *= scor

        scn_fac = ScreenFactors(1, 1, 11, 23)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_Na23__Mg24 *= scor
        rate_eval.p_Na23__n_Mg23 *= scor
        rate_eval.p_Na23__He4_Ne20 *= scor
        rate_eval.p_Na23__C12_C12 *= scor

        scn_fac = ScreenFactors(2, 4, 11, 23)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Na23__Al27 *= scor
        rate_eval.He4_Na23__n_Al26 *= scor
        rate_eval.He4_Na23__p_Mg26 *= scor

        scn_fac = ScreenFactors(1, 1, 12, 24)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_Mg24__Al25 *= scor

        scn_fac = ScreenFactors(2, 4, 12, 24)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Mg24__Si28 *= scor
        rate_eval.He4_Mg24__p_Al27 *= scor
        rate_eval.He4_Mg24__C12_O16 *= scor

        scn_fac = ScreenFactors(1, 1, 12, 25)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_Mg25__Al26 *= scor
        rate_eval.p_Mg25__n_Al25 *= scor
        rate_eval.p_Mg25__He4_Na22 *= scor

        scn_fac = ScreenFactors(2, 4, 12, 25)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Mg25__Si29 *= scor
        rate_eval.He4_Mg25__n_Si28 *= scor

        scn_fac = ScreenFactors(1, 1, 12, 26)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_Mg26__Al27 *= scor
        rate_eval.p_Mg26__n_Al26 *= scor
        rate_eval.p_Mg26__He4_Na23 *= scor

        scn_fac = ScreenFactors(2, 4, 12, 26)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Mg26__Si30 *= scor
        rate_eval.He4_Mg26__n_Si29 *= scor

        scn_fac = ScreenFactors(1, 1, 13, 25)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_Al25__Si26 *= scor

        scn_fac = ScreenFactors(2, 4, 13, 25)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Al25__P29 *= scor
        rate_eval.He4_Al25__p_Si28 *= scor

        scn_fac = ScreenFactors(2, 4, 13, 26)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Al26__P30 *= scor
        rate_eval.He4_Al26__n_P29 *= scor
        rate_eval.He4_Al26__p_Si29 *= scor

        scn_fac = ScreenFactors(1, 1, 13, 27)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_Al27__Si28 *= scor
        rate_eval.p_Al27__He4_Mg24 *= scor
        rate_eval.p_Al27__C12_O16 *= scor

        scn_fac = ScreenFactors(2, 4, 13, 27)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Al27__P31 *= scor
        rate_eval.He4_Al27__n_P30 *= scor
        rate_eval.He4_Al27__p_Si30 *= scor

        scn_fac = ScreenFactors(2, 4, 14, 26)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Si26__S30 *= scor
        rate_eval.He4_Si26__p_P29 *= scor

        scn_fac = ScreenFactors(1, 1, 14, 28)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_Si28__P29 *= scor
        rate_eval.p_Si28__He4_Al25 *= scor

        scn_fac = ScreenFactors(2, 4, 14, 28)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Si28__S32 *= scor
        rate_eval.He4_Si28__n_S31 *= scor
        rate_eval.He4_Si28__p_P31 *= scor
        rate_eval.He4_Si28__C12_Ne20 *= scor
        rate_eval.He4_Si28__O16_O16 *= scor

        scn_fac = ScreenFactors(1, 1, 14, 29)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_Si29__P30 *= scor
        rate_eval.p_Si29__n_P29 *= scor
        rate_eval.p_Si29__He4_Al26 *= scor

        scn_fac = ScreenFactors(2, 4, 14, 29)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Si29__S33 *= scor
        rate_eval.He4_Si29__n_S32 *= scor

        scn_fac = ScreenFactors(1, 1, 14, 30)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_Si30__P31 *= scor
        rate_eval.p_Si30__n_P30 *= scor
        rate_eval.p_Si30__He4_Al27 *= scor

        scn_fac = ScreenFactors(1, 1, 15, 29)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_P29__S30 *= scor
        rate_eval.p_P29__He4_Si26 *= scor

        scn_fac = ScreenFactors(2, 4, 15, 29)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_P29__Cl33 *= scor
        rate_eval.He4_P29__p_S32 *= scor

        scn_fac = ScreenFactors(1, 1, 15, 30)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_P30__S31 *= scor
        rate_eval.p_P30__n_S30 *= scor

        scn_fac = ScreenFactors(2, 4, 15, 30)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_P30__Cl34 *= scor
        rate_eval.He4_P30__n_Cl33 *= scor
        rate_eval.He4_P30__p_S33 *= scor

        scn_fac = ScreenFactors(1, 1, 15, 31)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_P31__S32 *= scor
        rate_eval.p_P31__n_S31 *= scor
        rate_eval.p_P31__He4_Si28 *= scor
        rate_eval.p_P31__C12_Ne20 *= scor
        rate_eval.p_P31__O16_O16 *= scor

        scn_fac = ScreenFactors(2, 4, 15, 31)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_P31__Cl35 *= scor
        rate_eval.He4_P31__n_Cl34 *= scor

        scn_fac = ScreenFactors(2, 4, 16, 30)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_S30__Ar34 *= scor
        rate_eval.He4_S30__p_Cl33 *= scor

        scn_fac = ScreenFactors(1, 1, 16, 32)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_S32__Cl33 *= scor
        rate_eval.p_S32__He4_P29 *= scor

        scn_fac = ScreenFactors(2, 4, 16, 32)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_S32__Ar36 *= scor
        rate_eval.He4_S32__p_Cl35 *= scor

        scn_fac = ScreenFactors(1, 1, 16, 33)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_S33__Cl34 *= scor
        rate_eval.p_S33__n_Cl33 *= scor
        rate_eval.p_S33__He4_P30 *= scor

        scn_fac = ScreenFactors(2, 4, 16, 33)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_S33__Ar37 *= scor
        rate_eval.He4_S33__n_Ar36 *= scor

        scn_fac = ScreenFactors(1, 1, 17, 33)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_Cl33__Ar34 *= scor
        rate_eval.p_Cl33__He4_S30 *= scor

        scn_fac = ScreenFactors(1, 1, 17, 35)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_Cl35__Ar36 *= scor
        rate_eval.p_Cl35__He4_S32 *= scor

        scn_fac = ScreenFactors(2, 4, 17, 35)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Cl35__K39 *= scor
        rate_eval.He4_Cl35__p_Ar38 *= scor

        scn_fac = ScreenFactors(2, 4, 18, 36)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Ar36__Ca40 *= scor
        rate_eval.He4_Ar36__p_K39 *= scor

        scn_fac = ScreenFactors(1, 1, 18, 38)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_Ar38__K39 *= scor
        rate_eval.p_Ar38__He4_Cl35 *= scor

        scn_fac = ScreenFactors(1, 1, 19, 39)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_K39__Ca40 *= scor
        rate_eval.p_K39__He4_Ar36 *= scor

        scn_fac = ScreenFactors(2, 4, 19, 39)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_K39__Sc43 *= scor

        scn_fac = ScreenFactors(2, 4, 20, 40)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Ca40__Ti44 *= scor
        rate_eval.He4_Ca40__p_Sc43 *= scor

        scn_fac = ScreenFactors(1, 1, 21, 43)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_Sc43__Ti44 *= scor
        rate_eval.p_Sc43__He4_Ca40 *= scor

        scn_fac = ScreenFactors(2, 4, 21, 43)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Sc43__V47 *= scor

        scn_fac = ScreenFactors(2, 4, 22, 44)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Ti44__Cr48 *= scor
        rate_eval.He4_Ti44__p_V47 *= scor

        scn_fac = ScreenFactors(1, 1, 23, 47)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_V47__Cr48 *= scor
        rate_eval.p_V47__He4_Ti44 *= scor

        scn_fac = ScreenFactors(2, 4, 23, 47)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_V47__Mn51 *= scor

        scn_fac = ScreenFactors(2, 4, 24, 48)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Cr48__Fe52 *= scor
        rate_eval.He4_Cr48__p_Mn51 *= scor

        scn_fac = ScreenFactors(1, 1, 25, 51)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_Mn51__Fe52 *= scor
        rate_eval.p_Mn51__He4_Cr48 *= scor

        scn_fac = ScreenFactors(2, 4, 25, 51)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Mn51__Co55 *= scor

        scn_fac = ScreenFactors(2, 4, 26, 52)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Fe52__Ni56 *= scor
        rate_eval.He4_Fe52__p_Co55 *= scor

        scn_fac = ScreenFactors(2, 4, 26, 55)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Fe55__Ni59 *= scor
        rate_eval.He4_Fe55__n_Ni58 *= scor

        scn_fac = ScreenFactors(1, 1, 27, 55)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_Co55__Ni56 *= scor
        rate_eval.p_Co55__He4_Fe52 *= scor

        scn_fac = ScreenFactors(2, 4, 5, 11)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_B11__n_N14 *= scor

        scn_fac = ScreenFactors(6, 12, 6, 12)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.C12_C12__n_Mg23 *= scor
        rate_eval.C12_C12__p_Na23 *= scor
        rate_eval.C12_C12__He4_Ne20 *= scor

        scn_fac = ScreenFactors(2, 4, 6, 13)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_C13__n_O16 *= scor

        scn_fac = ScreenFactors(2, 4, 7, 13)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_N13__p_O16 *= scor

        scn_fac = ScreenFactors(2, 4, 7, 15)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_N15__n_F18 *= scor

        scn_fac = ScreenFactors(1, 1, 8, 16)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_O16__He4_N13 *= scor

        scn_fac = ScreenFactors(6, 12, 8, 16)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.C12_O16__p_Al27 *= scor
        rate_eval.C12_O16__He4_Mg24 *= scor

        scn_fac = ScreenFactors(8, 16, 8, 16)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.O16_O16__n_S31 *= scor
        rate_eval.O16_O16__p_P31 *= scor
        rate_eval.O16_O16__He4_Si28 *= scor

        scn_fac = ScreenFactors(6, 12, 10, 20)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.C12_Ne20__n_S31 *= scor
        rate_eval.C12_Ne20__p_P31 *= scor
        rate_eval.C12_Ne20__He4_Si28 *= scor

        scn_fac = ScreenFactors(2, 4, 12, 23)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Mg23__n_Si26 *= scor
        rate_eval.He4_Mg23__p_Al26 *= scor

        scn_fac = ScreenFactors(1, 1, 13, 26)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_Al26__n_Si26 *= scor
        rate_eval.p_Al26__He4_Mg23 *= scor

        scn_fac = ScreenFactors(2, 4, 14, 30)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Si30__n_S33 *= scor

        scn_fac = ScreenFactors(2, 4, 16, 31)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_S31__n_Ar34 *= scor
        rate_eval.He4_S31__p_Cl34 *= scor

        scn_fac = ScreenFactors(2, 4, 17, 33)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Cl33__p_Ar36 *= scor

        scn_fac = ScreenFactors(1, 1, 17, 34)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_Cl34__n_Ar34 *= scor
        rate_eval.p_Cl34__He4_S31 *= scor

        scn_fac = ScreenFactors(2, 4, 17, 34)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Cl34__p_Ar37 *= scor

        scn_fac = ScreenFactors(1, 1, 18, 36)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_Ar36__He4_Cl33 *= scor

        scn_fac = ScreenFactors(1, 1, 18, 37)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_Ar37__He4_Cl34 *= scor

        scn_fac = ScreenFactors(2, 4, 18, 37)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Ar37__n_Ca40 *= scor

        scn_fac = ScreenFactors(1, 1, 18, 39)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_Ar39__n_K39 *= scor

        scn_fac = ScreenFactors(1, 1, 26, 55)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_Fe55__n_Co55 *= scor

        scn_fac = ScreenFactors(2, 4, 27, 55)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Co55__p_Ni58 *= scor

        scn_fac = ScreenFactors(1, 1, 28, 58)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_Ni58__He4_Co55 *= scor

        scn_fac = ScreenFactors(2, 4, 2, 4)
        scor = screen_func(plasma_state, scn_fac)
        scn_fac2 = ScreenFactors(2, 4, 4, 8)
        scor2 = screen_func(plasma_state, scn_fac2)
        rate_eval.He4_He4_He4__C12 *= scor * scor2
        rate_eval.He4_He4_He4__p_B11 *= scor * scor2

    dYdt = np.zeros((nnuc), dtype=np.float64)

    dYdt[jn] = (
       -Y[jn]*rate_eval.n__p__weak__wc12
       -rho*Y[jn]*Y[jc12]*rate_eval.n_C12__C13
       -rho*Y[jn]*Y[jn13]*rate_eval.n_N13__N14
       -rho*Y[jn]*Y[jn14]*rate_eval.n_N14__N15
       -rho*Y[jn]*Y[jo15]*rate_eval.n_O15__O16
       -rho*Y[jn]*Y[jo16]*rate_eval.n_O16__O17
       -rho*Y[jn]*Y[jne19]*rate_eval.n_Ne19__Ne20
       -rho*Y[jn]*Y[jne20]*rate_eval.n_Ne20__Ne21
       -rho*Y[jn]*Y[jne21]*rate_eval.n_Ne21__Ne22
       -rho*Y[jn]*Y[jna22]*rate_eval.n_Na22__Na23
       -rho*Y[jn]*Y[jmg23]*rate_eval.n_Mg23__Mg24
       -rho*Y[jn]*Y[jmg24]*rate_eval.n_Mg24__Mg25
       -rho*Y[jn]*Y[jmg25]*rate_eval.n_Mg25__Mg26
       -rho*Y[jn]*Y[jal25]*rate_eval.n_Al25__Al26
       -rho*Y[jn]*Y[jal26]*rate_eval.n_Al26__Al27
       -rho*Y[jn]*Y[jsi28]*rate_eval.n_Si28__Si29
       -rho*Y[jn]*Y[jsi29]*rate_eval.n_Si29__Si30
       -rho*Y[jn]*Y[jp29]*rate_eval.n_P29__P30
       -rho*Y[jn]*Y[jp30]*rate_eval.n_P30__P31
       -rho*Y[jn]*Y[js30]*rate_eval.n_S30__S31
       -rho*Y[jn]*Y[js31]*rate_eval.n_S31__S32
       -rho*Y[jn]*Y[js32]*rate_eval.n_S32__S33
       -rho*Y[jn]*Y[jcl33]*rate_eval.n_Cl33__Cl34
       -rho*Y[jn]*Y[jcl34]*rate_eval.n_Cl34__Cl35
       -rho*Y[jn]*Y[jar36]*rate_eval.n_Ar36__Ar37
       -rho*Y[jn]*Y[jar37]*rate_eval.n_Ar37__Ar38
       -rho*Y[jn]*Y[jar38]*rate_eval.n_Ar38__Ar39
       -rho*Y[jn]*Y[jni58]*rate_eval.n_Ni58__Ni59
       -rho*Y[jn]*Y[jn13]*rate_eval.n_N13__p_C13
       -rho*Y[jn]*Y[jn14]*rate_eval.n_N14__He4_B11
       -rho*Y[jn]*Y[jo15]*rate_eval.n_O15__p_N15
       -rho*Y[jn]*Y[jo15]*rate_eval.n_O15__He4_C12
       -rho*Y[jn]*Y[jo16]*rate_eval.n_O16__He4_C13
       -rho*Y[jn]*Y[jf18]*rate_eval.n_F18__He4_N15
       -rho*Y[jn]*Y[jne19]*rate_eval.n_Ne19__He4_O16
       -rho*Y[jn]*Y[jne20]*rate_eval.n_Ne20__He4_O17
       -rho*Y[jn]*Y[jna22]*rate_eval.n_Na22__p_Ne22
       -rho*Y[jn]*Y[jmg23]*rate_eval.n_Mg23__p_Na23
       -rho*Y[jn]*Y[jmg23]*rate_eval.n_Mg23__He4_Ne20
       -rho*Y[jn]*Y[jmg23]*rate_eval.n_Mg23__C12_C12
       -rho*Y[jn]*Y[jmg24]*rate_eval.n_Mg24__He4_Ne21
       -rho*Y[jn]*Y[jmg25]*rate_eval.n_Mg25__He4_Ne22
       -rho*Y[jn]*Y[jal25]*rate_eval.n_Al25__p_Mg25
       -rho*Y[jn]*Y[jal25]*rate_eval.n_Al25__He4_Na22
       -rho*Y[jn]*Y[jal26]*rate_eval.n_Al26__p_Mg26
       -rho*Y[jn]*Y[jal26]*rate_eval.n_Al26__He4_Na23
       -rho*Y[jn]*Y[jsi26]*rate_eval.n_Si26__p_Al26
       -rho*Y[jn]*Y[jsi26]*rate_eval.n_Si26__He4_Mg23
       -rho*Y[jn]*Y[jsi28]*rate_eval.n_Si28__He4_Mg25
       -rho*Y[jn]*Y[jsi29]*rate_eval.n_Si29__He4_Mg26
       -rho*Y[jn]*Y[jp29]*rate_eval.n_P29__p_Si29
       -rho*Y[jn]*Y[jp29]*rate_eval.n_P29__He4_Al26
       -rho*Y[jn]*Y[jp30]*rate_eval.n_P30__p_Si30
       -rho*Y[jn]*Y[jp30]*rate_eval.n_P30__He4_Al27
       -rho*Y[jn]*Y[js30]*rate_eval.n_S30__p_P30
       -rho*Y[jn]*Y[js31]*rate_eval.n_S31__p_P31
       -rho*Y[jn]*Y[js31]*rate_eval.n_S31__He4_Si28
       -rho*Y[jn]*Y[js31]*rate_eval.n_S31__C12_Ne20
       -rho*Y[jn]*Y[js31]*rate_eval.n_S31__O16_O16
       -rho*Y[jn]*Y[js32]*rate_eval.n_S32__He4_Si29
       -rho*Y[jn]*Y[js33]*rate_eval.n_S33__He4_Si30
       -rho*Y[jn]*Y[jcl33]*rate_eval.n_Cl33__p_S33
       -rho*Y[jn]*Y[jcl33]*rate_eval.n_Cl33__He4_P30
       -rho*Y[jn]*Y[jcl34]*rate_eval.n_Cl34__He4_P31
       -rho*Y[jn]*Y[jar34]*rate_eval.n_Ar34__p_Cl34
       -rho*Y[jn]*Y[jar34]*rate_eval.n_Ar34__He4_S31
       -rho*Y[jn]*Y[jar36]*rate_eval.n_Ar36__He4_S33
       -rho*Y[jn]*Y[jk39]*rate_eval.n_K39__p_Ar39
       -rho*Y[jn]*Y[jca40]*rate_eval.n_Ca40__He4_Ar37
       -rho*Y[jn]*Y[jco55]*rate_eval.n_Co55__p_Fe55
       -rho*Y[jn]*Y[jni58]*rate_eval.n_Ni58__He4_Fe55
       +Y[jc13]*rate_eval.C13__n_C12
       +Y[jn14]*rate_eval.N14__n_N13
       +Y[jn15]*rate_eval.N15__n_N14
       +Y[jo16]*rate_eval.O16__n_O15
       +Y[jo17]*rate_eval.O17__n_O16
       +Y[jne20]*rate_eval.Ne20__n_Ne19
       +Y[jne21]*rate_eval.Ne21__n_Ne20
       +Y[jne22]*rate_eval.Ne22__n_Ne21
       +Y[jna23]*rate_eval.Na23__n_Na22
       +Y[jmg24]*rate_eval.Mg24__n_Mg23
       +Y[jmg25]*rate_eval.Mg25__n_Mg24
       +Y[jmg26]*rate_eval.Mg26__n_Mg25
       +Y[jal26]*rate_eval.Al26__n_Al25
       +Y[jal27]*rate_eval.Al27__n_Al26
       +Y[jsi29]*rate_eval.Si29__n_Si28
       +Y[jsi30]*rate_eval.Si30__n_Si29
       +Y[jp30]*rate_eval.P30__n_P29
       +Y[jp31]*rate_eval.P31__n_P30
       +Y[js31]*rate_eval.S31__n_S30
       +Y[js32]*rate_eval.S32__n_S31
       +Y[js33]*rate_eval.S33__n_S32
       +Y[jcl34]*rate_eval.Cl34__n_Cl33
       +Y[jcl35]*rate_eval.Cl35__n_Cl34
       +Y[jar37]*rate_eval.Ar37__n_Ar36
       +Y[jar38]*rate_eval.Ar38__n_Ar37
       +Y[jar39]*rate_eval.Ar39__n_Ar38
       +Y[jni59]*rate_eval.Ni59__n_Ni58
       +rho*Y[jhe4]*Y[jb11]*rate_eval.He4_B11__n_N14
       +rho*Y[jhe4]*Y[jc12]*rate_eval.He4_C12__n_O15
       +5.00000000000000e-01*rho*Y[jc12]**2*rate_eval.C12_C12__n_Mg23
       +rho*Y[jp]*Y[jc13]*rate_eval.p_C13__n_N13
       +rho*Y[jhe4]*Y[jc13]*rate_eval.He4_C13__n_O16
       +rho*Y[jp]*Y[jn15]*rate_eval.p_N15__n_O15
       +rho*Y[jhe4]*Y[jn15]*rate_eval.He4_N15__n_F18
       +rho*Y[jhe4]*Y[jo16]*rate_eval.He4_O16__n_Ne19
       +5.00000000000000e-01*rho*Y[jo16]**2*rate_eval.O16_O16__n_S31
       +rho*Y[jhe4]*Y[jo17]*rate_eval.He4_O17__n_Ne20
       +rho*Y[jhe4]*Y[jne20]*rate_eval.He4_Ne20__n_Mg23
       +rho*Y[jc12]*Y[jne20]*rate_eval.C12_Ne20__n_S31
       +rho*Y[jhe4]*Y[jne21]*rate_eval.He4_Ne21__n_Mg24
       +rho*Y[jp]*Y[jne22]*rate_eval.p_Ne22__n_Na22
       +rho*Y[jhe4]*Y[jne22]*rate_eval.He4_Ne22__n_Mg25
       +rho*Y[jhe4]*Y[jna22]*rate_eval.He4_Na22__n_Al25
       +rho*Y[jp]*Y[jna23]*rate_eval.p_Na23__n_Mg23
       +rho*Y[jhe4]*Y[jna23]*rate_eval.He4_Na23__n_Al26
       +rho*Y[jhe4]*Y[jmg23]*rate_eval.He4_Mg23__n_Si26
       +rho*Y[jp]*Y[jmg25]*rate_eval.p_Mg25__n_Al25
       +rho*Y[jhe4]*Y[jmg25]*rate_eval.He4_Mg25__n_Si28
       +rho*Y[jp]*Y[jmg26]*rate_eval.p_Mg26__n_Al26
       +rho*Y[jhe4]*Y[jmg26]*rate_eval.He4_Mg26__n_Si29
       +rho*Y[jp]*Y[jal26]*rate_eval.p_Al26__n_Si26
       +rho*Y[jhe4]*Y[jal26]*rate_eval.He4_Al26__n_P29
       +rho*Y[jhe4]*Y[jal27]*rate_eval.He4_Al27__n_P30
       +rho*Y[jhe4]*Y[jsi28]*rate_eval.He4_Si28__n_S31
       +rho*Y[jp]*Y[jsi29]*rate_eval.p_Si29__n_P29
       +rho*Y[jhe4]*Y[jsi29]*rate_eval.He4_Si29__n_S32
       +rho*Y[jp]*Y[jsi30]*rate_eval.p_Si30__n_P30
       +rho*Y[jhe4]*Y[jsi30]*rate_eval.He4_Si30__n_S33
       +rho*Y[jp]*Y[jp30]*rate_eval.p_P30__n_S30
       +rho*Y[jhe4]*Y[jp30]*rate_eval.He4_P30__n_Cl33
       +rho*Y[jp]*Y[jp31]*rate_eval.p_P31__n_S31
       +rho*Y[jhe4]*Y[jp31]*rate_eval.He4_P31__n_Cl34
       +rho*Y[jhe4]*Y[js31]*rate_eval.He4_S31__n_Ar34
       +rho*Y[jp]*Y[js33]*rate_eval.p_S33__n_Cl33
       +rho*Y[jhe4]*Y[js33]*rate_eval.He4_S33__n_Ar36
       +rho*Y[jp]*Y[jcl34]*rate_eval.p_Cl34__n_Ar34
       +rho*Y[jhe4]*Y[jar37]*rate_eval.He4_Ar37__n_Ca40
       +rho*Y[jp]*Y[jar39]*rate_eval.p_Ar39__n_K39
       +rho*Y[jp]*Y[jfe55]*rate_eval.p_Fe55__n_Co55
       +rho*Y[jhe4]*Y[jfe55]*rate_eval.He4_Fe55__n_Ni58
       )

    dYdt[jp] = (
       -rho*Y[jp]*Y[jb11]*rate_eval.p_B11__C12
       -rho*Y[jp]*Y[jc12]*rate_eval.p_C12__N13
       -rho*Y[jp]*Y[jc13]*rate_eval.p_C13__N14
       -rho*Y[jp]*Y[jn14]*rate_eval.p_N14__O15
       -rho*Y[jp]*Y[jn15]*rate_eval.p_N15__O16
       -rho*Y[jp]*Y[jo17]*rate_eval.p_O17__F18
       -rho*Y[jp]*Y[jf18]*rate_eval.p_F18__Ne19
       -rho*Y[jp]*Y[jne21]*rate_eval.p_Ne21__Na22
       -rho*Y[jp]*Y[jne22]*rate_eval.p_Ne22__Na23
       -rho*Y[jp]*Y[jna22]*rate_eval.p_Na22__Mg23
       -rho*Y[jp]*Y[jna23]*rate_eval.p_Na23__Mg24
       -rho*Y[jp]*Y[jmg24]*rate_eval.p_Mg24__Al25
       -rho*Y[jp]*Y[jmg25]*rate_eval.p_Mg25__Al26
       -rho*Y[jp]*Y[jmg26]*rate_eval.p_Mg26__Al27
       -rho*Y[jp]*Y[jal25]*rate_eval.p_Al25__Si26
       -rho*Y[jp]*Y[jal27]*rate_eval.p_Al27__Si28
       -rho*Y[jp]*Y[jsi28]*rate_eval.p_Si28__P29
       -rho*Y[jp]*Y[jsi29]*rate_eval.p_Si29__P30
       -rho*Y[jp]*Y[jsi30]*rate_eval.p_Si30__P31
       -rho*Y[jp]*Y[jp29]*rate_eval.p_P29__S30
       -rho*Y[jp]*Y[jp30]*rate_eval.p_P30__S31
       -rho*Y[jp]*Y[jp31]*rate_eval.p_P31__S32
       -rho*Y[jp]*Y[js32]*rate_eval.p_S32__Cl33
       -rho*Y[jp]*Y[js33]*rate_eval.p_S33__Cl34
       -rho*Y[jp]*Y[jcl33]*rate_eval.p_Cl33__Ar34
       -rho*Y[jp]*Y[jcl35]*rate_eval.p_Cl35__Ar36
       -rho*Y[jp]*Y[jar38]*rate_eval.p_Ar38__K39
       -rho*Y[jp]*Y[jk39]*rate_eval.p_K39__Ca40
       -rho*Y[jp]*Y[jsc43]*rate_eval.p_Sc43__Ti44
       -rho*Y[jp]*Y[jv47]*rate_eval.p_V47__Cr48
       -rho*Y[jp]*Y[jmn51]*rate_eval.p_Mn51__Fe52
       -rho*Y[jp]*Y[jco55]*rate_eval.p_Co55__Ni56
       -rho*Y[jp]*Y[jc13]*rate_eval.p_C13__n_N13
       -rho*Y[jp]*Y[jn15]*rate_eval.p_N15__n_O15
       -rho*Y[jp]*Y[jn15]*rate_eval.p_N15__He4_C12
       -rho*Y[jp]*Y[jo16]*rate_eval.p_O16__He4_N13
       -rho*Y[jp]*Y[jo17]*rate_eval.p_O17__He4_N14
       -rho*Y[jp]*Y[jf18]*rate_eval.p_F18__He4_O15
       -rho*Y[jp]*Y[jne21]*rate_eval.p_Ne21__He4_F18
       -rho*Y[jp]*Y[jne22]*rate_eval.p_Ne22__n_Na22
       -rho*Y[jp]*Y[jna22]*rate_eval.p_Na22__He4_Ne19
       -rho*Y[jp]*Y[jna23]*rate_eval.p_Na23__n_Mg23
       -rho*Y[jp]*Y[jna23]*rate_eval.p_Na23__He4_Ne20
       -rho*Y[jp]*Y[jna23]*rate_eval.p_Na23__C12_C12
       -rho*Y[jp]*Y[jmg25]*rate_eval.p_Mg25__n_Al25
       -rho*Y[jp]*Y[jmg25]*rate_eval.p_Mg25__He4_Na22
       -rho*Y[jp]*Y[jmg26]*rate_eval.p_Mg26__n_Al26
       -rho*Y[jp]*Y[jmg26]*rate_eval.p_Mg26__He4_Na23
       -rho*Y[jp]*Y[jal26]*rate_eval.p_Al26__n_Si26
       -rho*Y[jp]*Y[jal26]*rate_eval.p_Al26__He4_Mg23
       -rho*Y[jp]*Y[jal27]*rate_eval.p_Al27__He4_Mg24
       -rho*Y[jp]*Y[jal27]*rate_eval.p_Al27__C12_O16
       -rho*Y[jp]*Y[jsi28]*rate_eval.p_Si28__He4_Al25
       -rho*Y[jp]*Y[jsi29]*rate_eval.p_Si29__n_P29
       -rho*Y[jp]*Y[jsi29]*rate_eval.p_Si29__He4_Al26
       -rho*Y[jp]*Y[jsi30]*rate_eval.p_Si30__n_P30
       -rho*Y[jp]*Y[jsi30]*rate_eval.p_Si30__He4_Al27
       -rho*Y[jp]*Y[jp29]*rate_eval.p_P29__He4_Si26
       -rho*Y[jp]*Y[jp30]*rate_eval.p_P30__n_S30
       -rho*Y[jp]*Y[jp31]*rate_eval.p_P31__n_S31
       -rho*Y[jp]*Y[jp31]*rate_eval.p_P31__He4_Si28
       -rho*Y[jp]*Y[jp31]*rate_eval.p_P31__C12_Ne20
       -rho*Y[jp]*Y[jp31]*rate_eval.p_P31__O16_O16
       -rho*Y[jp]*Y[js32]*rate_eval.p_S32__He4_P29
       -rho*Y[jp]*Y[js33]*rate_eval.p_S33__n_Cl33
       -rho*Y[jp]*Y[js33]*rate_eval.p_S33__He4_P30
       -rho*Y[jp]*Y[jcl33]*rate_eval.p_Cl33__He4_S30
       -rho*Y[jp]*Y[jcl34]*rate_eval.p_Cl34__n_Ar34
       -rho*Y[jp]*Y[jcl34]*rate_eval.p_Cl34__He4_S31
       -rho*Y[jp]*Y[jcl35]*rate_eval.p_Cl35__He4_S32
       -rho*Y[jp]*Y[jar36]*rate_eval.p_Ar36__He4_Cl33
       -rho*Y[jp]*Y[jar37]*rate_eval.p_Ar37__He4_Cl34
       -rho*Y[jp]*Y[jar38]*rate_eval.p_Ar38__He4_Cl35
       -rho*Y[jp]*Y[jar39]*rate_eval.p_Ar39__n_K39
       -rho*Y[jp]*Y[jk39]*rate_eval.p_K39__He4_Ar36
       -rho*Y[jp]*Y[jsc43]*rate_eval.p_Sc43__He4_Ca40
       -rho*Y[jp]*Y[jv47]*rate_eval.p_V47__He4_Ti44
       -rho*Y[jp]*Y[jmn51]*rate_eval.p_Mn51__He4_Cr48
       -rho*Y[jp]*Y[jfe55]*rate_eval.p_Fe55__n_Co55
       -rho*Y[jp]*Y[jco55]*rate_eval.p_Co55__He4_Fe52
       -rho*Y[jp]*Y[jni58]*rate_eval.p_Ni58__He4_Co55
       -rho*Y[jp]*Y[jb11]*rate_eval.p_B11__He4_He4_He4
       +Y[jn]*rate_eval.n__p__weak__wc12
       +Y[jc12]*rate_eval.C12__p_B11
       +Y[jn13]*rate_eval.N13__p_C12
       +Y[jn14]*rate_eval.N14__p_C13
       +Y[jo15]*rate_eval.O15__p_N14
       +Y[jo16]*rate_eval.O16__p_N15
       +Y[jf18]*rate_eval.F18__p_O17
       +Y[jne19]*rate_eval.Ne19__p_F18
       +Y[jna22]*rate_eval.Na22__p_Ne21
       +Y[jna23]*rate_eval.Na23__p_Ne22
       +Y[jmg23]*rate_eval.Mg23__p_Na22
       +Y[jmg24]*rate_eval.Mg24__p_Na23
       +Y[jal25]*rate_eval.Al25__p_Mg24
       +Y[jal26]*rate_eval.Al26__p_Mg25
       +Y[jal27]*rate_eval.Al27__p_Mg26
       +Y[jsi26]*rate_eval.Si26__p_Al25
       +Y[jsi28]*rate_eval.Si28__p_Al27
       +Y[jp29]*rate_eval.P29__p_Si28
       +Y[jp30]*rate_eval.P30__p_Si29
       +Y[jp31]*rate_eval.P31__p_Si30
       +Y[js30]*rate_eval.S30__p_P29
       +Y[js31]*rate_eval.S31__p_P30
       +Y[js32]*rate_eval.S32__p_P31
       +Y[jcl33]*rate_eval.Cl33__p_S32
       +Y[jcl34]*rate_eval.Cl34__p_S33
       +Y[jar34]*rate_eval.Ar34__p_Cl33
       +Y[jar36]*rate_eval.Ar36__p_Cl35
       +Y[jk39]*rate_eval.K39__p_Ar38
       +Y[jca40]*rate_eval.Ca40__p_K39
       +Y[jti44]*rate_eval.Ti44__p_Sc43
       +Y[jcr48]*rate_eval.Cr48__p_V47
       +Y[jfe52]*rate_eval.Fe52__p_Mn51
       +Y[jni56]*rate_eval.Ni56__p_Co55
       +rho*Y[jhe4]*Y[jc12]*rate_eval.He4_C12__p_N15
       +5.00000000000000e-01*rho*Y[jc12]**2*rate_eval.C12_C12__p_Na23
       +rho*Y[jn]*Y[jn13]*rate_eval.n_N13__p_C13
       +rho*Y[jhe4]*Y[jn13]*rate_eval.He4_N13__p_O16
       +rho*Y[jhe4]*Y[jn14]*rate_eval.He4_N14__p_O17
       +rho*Y[jn]*Y[jo15]*rate_eval.n_O15__p_N15
       +rho*Y[jhe4]*Y[jo15]*rate_eval.He4_O15__p_F18
       +rho*Y[jc12]*Y[jo16]*rate_eval.C12_O16__p_Al27
       +5.00000000000000e-01*rho*Y[jo16]**2*rate_eval.O16_O16__p_P31
       +rho*Y[jhe4]*Y[jf18]*rate_eval.He4_F18__p_Ne21
       +rho*Y[jhe4]*Y[jne19]*rate_eval.He4_Ne19__p_Na22
       +rho*Y[jhe4]*Y[jne20]*rate_eval.He4_Ne20__p_Na23
       +rho*Y[jc12]*Y[jne20]*rate_eval.C12_Ne20__p_P31
       +rho*Y[jn]*Y[jna22]*rate_eval.n_Na22__p_Ne22
       +rho*Y[jhe4]*Y[jna22]*rate_eval.He4_Na22__p_Mg25
       +rho*Y[jhe4]*Y[jna23]*rate_eval.He4_Na23__p_Mg26
       +rho*Y[jn]*Y[jmg23]*rate_eval.n_Mg23__p_Na23
       +rho*Y[jhe4]*Y[jmg23]*rate_eval.He4_Mg23__p_Al26
       +rho*Y[jhe4]*Y[jmg24]*rate_eval.He4_Mg24__p_Al27
       +rho*Y[jn]*Y[jal25]*rate_eval.n_Al25__p_Mg25
       +rho*Y[jhe4]*Y[jal25]*rate_eval.He4_Al25__p_Si28
       +rho*Y[jn]*Y[jal26]*rate_eval.n_Al26__p_Mg26
       +rho*Y[jhe4]*Y[jal26]*rate_eval.He4_Al26__p_Si29
       +rho*Y[jhe4]*Y[jal27]*rate_eval.He4_Al27__p_Si30
       +rho*Y[jn]*Y[jsi26]*rate_eval.n_Si26__p_Al26
       +rho*Y[jhe4]*Y[jsi26]*rate_eval.He4_Si26__p_P29
       +rho*Y[jhe4]*Y[jsi28]*rate_eval.He4_Si28__p_P31
       +rho*Y[jn]*Y[jp29]*rate_eval.n_P29__p_Si29
       +rho*Y[jhe4]*Y[jp29]*rate_eval.He4_P29__p_S32
       +rho*Y[jn]*Y[jp30]*rate_eval.n_P30__p_Si30
       +rho*Y[jhe4]*Y[jp30]*rate_eval.He4_P30__p_S33
       +rho*Y[jn]*Y[js30]*rate_eval.n_S30__p_P30
       +rho*Y[jhe4]*Y[js30]*rate_eval.He4_S30__p_Cl33
       +rho*Y[jn]*Y[js31]*rate_eval.n_S31__p_P31
       +rho*Y[jhe4]*Y[js31]*rate_eval.He4_S31__p_Cl34
       +rho*Y[jhe4]*Y[js32]*rate_eval.He4_S32__p_Cl35
       +rho*Y[jn]*Y[jcl33]*rate_eval.n_Cl33__p_S33
       +rho*Y[jhe4]*Y[jcl33]*rate_eval.He4_Cl33__p_Ar36
       +rho*Y[jhe4]*Y[jcl34]*rate_eval.He4_Cl34__p_Ar37
       +rho*Y[jhe4]*Y[jcl35]*rate_eval.He4_Cl35__p_Ar38
       +rho*Y[jn]*Y[jar34]*rate_eval.n_Ar34__p_Cl34
       +rho*Y[jhe4]*Y[jar36]*rate_eval.He4_Ar36__p_K39
       +rho*Y[jn]*Y[jk39]*rate_eval.n_K39__p_Ar39
       +rho*Y[jhe4]*Y[jca40]*rate_eval.He4_Ca40__p_Sc43
       +rho*Y[jhe4]*Y[jti44]*rate_eval.He4_Ti44__p_V47
       +rho*Y[jhe4]*Y[jcr48]*rate_eval.He4_Cr48__p_Mn51
       +rho*Y[jhe4]*Y[jfe52]*rate_eval.He4_Fe52__p_Co55
       +rho*Y[jn]*Y[jco55]*rate_eval.n_Co55__p_Fe55
       +rho*Y[jhe4]*Y[jco55]*rate_eval.He4_Co55__p_Ni58
       +1.66666666666667e-01*rho**2*Y[jhe4]**3*rate_eval.He4_He4_He4__p_B11
       )

    dYdt[jhe4] = (
       -rho*Y[jhe4]*Y[jc12]*rate_eval.He4_C12__O16
       -rho*Y[jhe4]*Y[jn14]*rate_eval.He4_N14__F18
       -rho*Y[jhe4]*Y[jo15]*rate_eval.He4_O15__Ne19
       -rho*Y[jhe4]*Y[jo16]*rate_eval.He4_O16__Ne20
       -rho*Y[jhe4]*Y[jo17]*rate_eval.He4_O17__Ne21
       -rho*Y[jhe4]*Y[jf18]*rate_eval.He4_F18__Na22
       -rho*Y[jhe4]*Y[jne19]*rate_eval.He4_Ne19__Mg23
       -rho*Y[jhe4]*Y[jne20]*rate_eval.He4_Ne20__Mg24
       -rho*Y[jhe4]*Y[jne21]*rate_eval.He4_Ne21__Mg25
       -rho*Y[jhe4]*Y[jne22]*rate_eval.He4_Ne22__Mg26
       -rho*Y[jhe4]*Y[jna22]*rate_eval.He4_Na22__Al26
       -rho*Y[jhe4]*Y[jna23]*rate_eval.He4_Na23__Al27
       -rho*Y[jhe4]*Y[jmg24]*rate_eval.He4_Mg24__Si28
       -rho*Y[jhe4]*Y[jmg25]*rate_eval.He4_Mg25__Si29
       -rho*Y[jhe4]*Y[jmg26]*rate_eval.He4_Mg26__Si30
       -rho*Y[jhe4]*Y[jal25]*rate_eval.He4_Al25__P29
       -rho*Y[jhe4]*Y[jal26]*rate_eval.He4_Al26__P30
       -rho*Y[jhe4]*Y[jal27]*rate_eval.He4_Al27__P31
       -rho*Y[jhe4]*Y[jsi26]*rate_eval.He4_Si26__S30
       -rho*Y[jhe4]*Y[jsi28]*rate_eval.He4_Si28__S32
       -rho*Y[jhe4]*Y[jsi29]*rate_eval.He4_Si29__S33
       -rho*Y[jhe4]*Y[jp29]*rate_eval.He4_P29__Cl33
       -rho*Y[jhe4]*Y[jp30]*rate_eval.He4_P30__Cl34
       -rho*Y[jhe4]*Y[jp31]*rate_eval.He4_P31__Cl35
       -rho*Y[jhe4]*Y[js30]*rate_eval.He4_S30__Ar34
       -rho*Y[jhe4]*Y[js32]*rate_eval.He4_S32__Ar36
       -rho*Y[jhe4]*Y[js33]*rate_eval.He4_S33__Ar37
       -rho*Y[jhe4]*Y[jcl35]*rate_eval.He4_Cl35__K39
       -rho*Y[jhe4]*Y[jar36]*rate_eval.He4_Ar36__Ca40
       -rho*Y[jhe4]*Y[jk39]*rate_eval.He4_K39__Sc43
       -rho*Y[jhe4]*Y[jca40]*rate_eval.He4_Ca40__Ti44
       -rho*Y[jhe4]*Y[jsc43]*rate_eval.He4_Sc43__V47
       -rho*Y[jhe4]*Y[jti44]*rate_eval.He4_Ti44__Cr48
       -rho*Y[jhe4]*Y[jv47]*rate_eval.He4_V47__Mn51
       -rho*Y[jhe4]*Y[jcr48]*rate_eval.He4_Cr48__Fe52
       -rho*Y[jhe4]*Y[jmn51]*rate_eval.He4_Mn51__Co55
       -rho*Y[jhe4]*Y[jfe52]*rate_eval.He4_Fe52__Ni56
       -rho*Y[jhe4]*Y[jfe55]*rate_eval.He4_Fe55__Ni59
       -rho*Y[jhe4]*Y[jb11]*rate_eval.He4_B11__n_N14
       -rho*Y[jhe4]*Y[jc12]*rate_eval.He4_C12__n_O15
       -rho*Y[jhe4]*Y[jc12]*rate_eval.He4_C12__p_N15
       -rho*Y[jhe4]*Y[jc13]*rate_eval.He4_C13__n_O16
       -rho*Y[jhe4]*Y[jn13]*rate_eval.He4_N13__p_O16
       -rho*Y[jhe4]*Y[jn14]*rate_eval.He4_N14__p_O17
       -rho*Y[jhe4]*Y[jn15]*rate_eval.He4_N15__n_F18
       -rho*Y[jhe4]*Y[jo15]*rate_eval.He4_O15__p_F18
       -rho*Y[jhe4]*Y[jo16]*rate_eval.He4_O16__n_Ne19
       -rho*Y[jhe4]*Y[jo17]*rate_eval.He4_O17__n_Ne20
       -rho*Y[jhe4]*Y[jf18]*rate_eval.He4_F18__p_Ne21
       -rho*Y[jhe4]*Y[jne19]*rate_eval.He4_Ne19__p_Na22
       -rho*Y[jhe4]*Y[jne20]*rate_eval.He4_Ne20__n_Mg23
       -rho*Y[jhe4]*Y[jne20]*rate_eval.He4_Ne20__p_Na23
       -rho*Y[jhe4]*Y[jne20]*rate_eval.He4_Ne20__C12_C12
       -rho*Y[jhe4]*Y[jne21]*rate_eval.He4_Ne21__n_Mg24
       -rho*Y[jhe4]*Y[jne22]*rate_eval.He4_Ne22__n_Mg25
       -rho*Y[jhe4]*Y[jna22]*rate_eval.He4_Na22__n_Al25
       -rho*Y[jhe4]*Y[jna22]*rate_eval.He4_Na22__p_Mg25
       -rho*Y[jhe4]*Y[jna23]*rate_eval.He4_Na23__n_Al26
       -rho*Y[jhe4]*Y[jna23]*rate_eval.He4_Na23__p_Mg26
       -rho*Y[jhe4]*Y[jmg23]*rate_eval.He4_Mg23__n_Si26
       -rho*Y[jhe4]*Y[jmg23]*rate_eval.He4_Mg23__p_Al26
       -rho*Y[jhe4]*Y[jmg24]*rate_eval.He4_Mg24__p_Al27
       -rho*Y[jhe4]*Y[jmg24]*rate_eval.He4_Mg24__C12_O16
       -rho*Y[jhe4]*Y[jmg25]*rate_eval.He4_Mg25__n_Si28
       -rho*Y[jhe4]*Y[jmg26]*rate_eval.He4_Mg26__n_Si29
       -rho*Y[jhe4]*Y[jal25]*rate_eval.He4_Al25__p_Si28
       -rho*Y[jhe4]*Y[jal26]*rate_eval.He4_Al26__n_P29
       -rho*Y[jhe4]*Y[jal26]*rate_eval.He4_Al26__p_Si29
       -rho*Y[jhe4]*Y[jal27]*rate_eval.He4_Al27__n_P30
       -rho*Y[jhe4]*Y[jal27]*rate_eval.He4_Al27__p_Si30
       -rho*Y[jhe4]*Y[jsi26]*rate_eval.He4_Si26__p_P29
       -rho*Y[jhe4]*Y[jsi28]*rate_eval.He4_Si28__n_S31
       -rho*Y[jhe4]*Y[jsi28]*rate_eval.He4_Si28__p_P31
       -rho*Y[jhe4]*Y[jsi28]*rate_eval.He4_Si28__C12_Ne20
       -rho*Y[jhe4]*Y[jsi28]*rate_eval.He4_Si28__O16_O16
       -rho*Y[jhe4]*Y[jsi29]*rate_eval.He4_Si29__n_S32
       -rho*Y[jhe4]*Y[jsi30]*rate_eval.He4_Si30__n_S33
       -rho*Y[jhe4]*Y[jp29]*rate_eval.He4_P29__p_S32
       -rho*Y[jhe4]*Y[jp30]*rate_eval.He4_P30__n_Cl33
       -rho*Y[jhe4]*Y[jp30]*rate_eval.He4_P30__p_S33
       -rho*Y[jhe4]*Y[jp31]*rate_eval.He4_P31__n_Cl34
       -rho*Y[jhe4]*Y[js30]*rate_eval.He4_S30__p_Cl33
       -rho*Y[jhe4]*Y[js31]*rate_eval.He4_S31__n_Ar34
       -rho*Y[jhe4]*Y[js31]*rate_eval.He4_S31__p_Cl34
       -rho*Y[jhe4]*Y[js32]*rate_eval.He4_S32__p_Cl35
       -rho*Y[jhe4]*Y[js33]*rate_eval.He4_S33__n_Ar36
       -rho*Y[jhe4]*Y[jcl33]*rate_eval.He4_Cl33__p_Ar36
       -rho*Y[jhe4]*Y[jcl34]*rate_eval.He4_Cl34__p_Ar37
       -rho*Y[jhe4]*Y[jcl35]*rate_eval.He4_Cl35__p_Ar38
       -rho*Y[jhe4]*Y[jar36]*rate_eval.He4_Ar36__p_K39
       -rho*Y[jhe4]*Y[jar37]*rate_eval.He4_Ar37__n_Ca40
       -rho*Y[jhe4]*Y[jca40]*rate_eval.He4_Ca40__p_Sc43
       -rho*Y[jhe4]*Y[jti44]*rate_eval.He4_Ti44__p_V47
       -rho*Y[jhe4]*Y[jcr48]*rate_eval.He4_Cr48__p_Mn51
       -rho*Y[jhe4]*Y[jfe52]*rate_eval.He4_Fe52__p_Co55
       -rho*Y[jhe4]*Y[jfe55]*rate_eval.He4_Fe55__n_Ni58
       -rho*Y[jhe4]*Y[jco55]*rate_eval.He4_Co55__p_Ni58
       -3*1.66666666666667e-01*rho**2*Y[jhe4]**3*rate_eval.He4_He4_He4__C12
       -3*1.66666666666667e-01*rho**2*Y[jhe4]**3*rate_eval.He4_He4_He4__p_B11
       +Y[jo16]*rate_eval.O16__He4_C12
       +Y[jf18]*rate_eval.F18__He4_N14
       +Y[jne19]*rate_eval.Ne19__He4_O15
       +Y[jne20]*rate_eval.Ne20__He4_O16
       +Y[jne21]*rate_eval.Ne21__He4_O17
       +Y[jna22]*rate_eval.Na22__He4_F18
       +Y[jmg23]*rate_eval.Mg23__He4_Ne19
       +Y[jmg24]*rate_eval.Mg24__He4_Ne20
       +Y[jmg25]*rate_eval.Mg25__He4_Ne21
       +Y[jmg26]*rate_eval.Mg26__He4_Ne22
       +Y[jal26]*rate_eval.Al26__He4_Na22
       +Y[jal27]*rate_eval.Al27__He4_Na23
       +Y[jsi28]*rate_eval.Si28__He4_Mg24
       +Y[jsi29]*rate_eval.Si29__He4_Mg25
       +Y[jsi30]*rate_eval.Si30__He4_Mg26
       +Y[jp29]*rate_eval.P29__He4_Al25
       +Y[jp30]*rate_eval.P30__He4_Al26
       +Y[jp31]*rate_eval.P31__He4_Al27
       +Y[js30]*rate_eval.S30__He4_Si26
       +Y[js32]*rate_eval.S32__He4_Si28
       +Y[js33]*rate_eval.S33__He4_Si29
       +Y[jcl33]*rate_eval.Cl33__He4_P29
       +Y[jcl34]*rate_eval.Cl34__He4_P30
       +Y[jcl35]*rate_eval.Cl35__He4_P31
       +Y[jar34]*rate_eval.Ar34__He4_S30
       +Y[jar36]*rate_eval.Ar36__He4_S32
       +Y[jar37]*rate_eval.Ar37__He4_S33
       +Y[jk39]*rate_eval.K39__He4_Cl35
       +Y[jca40]*rate_eval.Ca40__He4_Ar36
       +Y[jsc43]*rate_eval.Sc43__He4_K39
       +Y[jti44]*rate_eval.Ti44__He4_Ca40
       +Y[jv47]*rate_eval.V47__He4_Sc43
       +Y[jcr48]*rate_eval.Cr48__He4_Ti44
       +Y[jmn51]*rate_eval.Mn51__He4_V47
       +Y[jfe52]*rate_eval.Fe52__He4_Cr48
       +Y[jco55]*rate_eval.Co55__He4_Mn51
       +Y[jni56]*rate_eval.Ni56__He4_Fe52
       +Y[jni59]*rate_eval.Ni59__He4_Fe55
       +3*Y[jc12]*rate_eval.C12__He4_He4_He4
       +5.00000000000000e-01*rho*Y[jc12]**2*rate_eval.C12_C12__He4_Ne20
       +rho*Y[jn]*Y[jn14]*rate_eval.n_N14__He4_B11
       +rho*Y[jp]*Y[jn15]*rate_eval.p_N15__He4_C12
       +rho*Y[jn]*Y[jo15]*rate_eval.n_O15__He4_C12
       +rho*Y[jn]*Y[jo16]*rate_eval.n_O16__He4_C13
       +rho*Y[jp]*Y[jo16]*rate_eval.p_O16__He4_N13
       +rho*Y[jc12]*Y[jo16]*rate_eval.C12_O16__He4_Mg24
       +5.00000000000000e-01*rho*Y[jo16]**2*rate_eval.O16_O16__He4_Si28
       +rho*Y[jp]*Y[jo17]*rate_eval.p_O17__He4_N14
       +rho*Y[jn]*Y[jf18]*rate_eval.n_F18__He4_N15
       +rho*Y[jp]*Y[jf18]*rate_eval.p_F18__He4_O15
       +rho*Y[jn]*Y[jne19]*rate_eval.n_Ne19__He4_O16
       +rho*Y[jn]*Y[jne20]*rate_eval.n_Ne20__He4_O17
       +rho*Y[jc12]*Y[jne20]*rate_eval.C12_Ne20__He4_Si28
       +rho*Y[jp]*Y[jne21]*rate_eval.p_Ne21__He4_F18
       +rho*Y[jp]*Y[jna22]*rate_eval.p_Na22__He4_Ne19
       +rho*Y[jp]*Y[jna23]*rate_eval.p_Na23__He4_Ne20
       +rho*Y[jn]*Y[jmg23]*rate_eval.n_Mg23__He4_Ne20
       +rho*Y[jn]*Y[jmg24]*rate_eval.n_Mg24__He4_Ne21
       +rho*Y[jn]*Y[jmg25]*rate_eval.n_Mg25__He4_Ne22
       +rho*Y[jp]*Y[jmg25]*rate_eval.p_Mg25__He4_Na22
       +rho*Y[jp]*Y[jmg26]*rate_eval.p_Mg26__He4_Na23
       +rho*Y[jn]*Y[jal25]*rate_eval.n_Al25__He4_Na22
       +rho*Y[jn]*Y[jal26]*rate_eval.n_Al26__He4_Na23
       +rho*Y[jp]*Y[jal26]*rate_eval.p_Al26__He4_Mg23
       +rho*Y[jp]*Y[jal27]*rate_eval.p_Al27__He4_Mg24
       +rho*Y[jn]*Y[jsi26]*rate_eval.n_Si26__He4_Mg23
       +rho*Y[jn]*Y[jsi28]*rate_eval.n_Si28__He4_Mg25
       +rho*Y[jp]*Y[jsi28]*rate_eval.p_Si28__He4_Al25
       +rho*Y[jn]*Y[jsi29]*rate_eval.n_Si29__He4_Mg26
       +rho*Y[jp]*Y[jsi29]*rate_eval.p_Si29__He4_Al26
       +rho*Y[jp]*Y[jsi30]*rate_eval.p_Si30__He4_Al27
       +rho*Y[jn]*Y[jp29]*rate_eval.n_P29__He4_Al26
       +rho*Y[jp]*Y[jp29]*rate_eval.p_P29__He4_Si26
       +rho*Y[jn]*Y[jp30]*rate_eval.n_P30__He4_Al27
       +rho*Y[jp]*Y[jp31]*rate_eval.p_P31__He4_Si28
       +rho*Y[jn]*Y[js31]*rate_eval.n_S31__He4_Si28
       +rho*Y[jn]*Y[js32]*rate_eval.n_S32__He4_Si29
       +rho*Y[jp]*Y[js32]*rate_eval.p_S32__He4_P29
       +rho*Y[jn]*Y[js33]*rate_eval.n_S33__He4_Si30
       +rho*Y[jp]*Y[js33]*rate_eval.p_S33__He4_P30
       +rho*Y[jn]*Y[jcl33]*rate_eval.n_Cl33__He4_P30
       +rho*Y[jp]*Y[jcl33]*rate_eval.p_Cl33__He4_S30
       +rho*Y[jn]*Y[jcl34]*rate_eval.n_Cl34__He4_P31
       +rho*Y[jp]*Y[jcl34]*rate_eval.p_Cl34__He4_S31
       +rho*Y[jp]*Y[jcl35]*rate_eval.p_Cl35__He4_S32
       +rho*Y[jn]*Y[jar34]*rate_eval.n_Ar34__He4_S31
       +rho*Y[jn]*Y[jar36]*rate_eval.n_Ar36__He4_S33
       +rho*Y[jp]*Y[jar36]*rate_eval.p_Ar36__He4_Cl33
       +rho*Y[jp]*Y[jar37]*rate_eval.p_Ar37__He4_Cl34
       +rho*Y[jp]*Y[jar38]*rate_eval.p_Ar38__He4_Cl35
       +rho*Y[jp]*Y[jk39]*rate_eval.p_K39__He4_Ar36
       +rho*Y[jn]*Y[jca40]*rate_eval.n_Ca40__He4_Ar37
       +rho*Y[jp]*Y[jsc43]*rate_eval.p_Sc43__He4_Ca40
       +rho*Y[jp]*Y[jv47]*rate_eval.p_V47__He4_Ti44
       +rho*Y[jp]*Y[jmn51]*rate_eval.p_Mn51__He4_Cr48
       +rho*Y[jp]*Y[jco55]*rate_eval.p_Co55__He4_Fe52
       +rho*Y[jn]*Y[jni58]*rate_eval.n_Ni58__He4_Fe55
       +rho*Y[jp]*Y[jni58]*rate_eval.p_Ni58__He4_Co55
       +3*rho*Y[jp]*Y[jb11]*rate_eval.p_B11__He4_He4_He4
       )

    dYdt[jb11] = (
       -rho*Y[jp]*Y[jb11]*rate_eval.p_B11__C12
       -rho*Y[jhe4]*Y[jb11]*rate_eval.He4_B11__n_N14
       -rho*Y[jp]*Y[jb11]*rate_eval.p_B11__He4_He4_He4
       +Y[jc12]*rate_eval.C12__p_B11
       +rho*Y[jn]*Y[jn14]*rate_eval.n_N14__He4_B11
       +1.66666666666667e-01*rho**2*Y[jhe4]**3*rate_eval.He4_He4_He4__p_B11
       )

    dYdt[jc12] = (
       -Y[jc12]*rate_eval.C12__p_B11
       -Y[jc12]*rate_eval.C12__He4_He4_He4
       -rho*Y[jn]*Y[jc12]*rate_eval.n_C12__C13
       -rho*Y[jp]*Y[jc12]*rate_eval.p_C12__N13
       -rho*Y[jhe4]*Y[jc12]*rate_eval.He4_C12__O16
       -rho*Y[jhe4]*Y[jc12]*rate_eval.He4_C12__n_O15
       -rho*Y[jhe4]*Y[jc12]*rate_eval.He4_C12__p_N15
       -2*5.00000000000000e-01*rho*Y[jc12]**2*rate_eval.C12_C12__n_Mg23
       -2*5.00000000000000e-01*rho*Y[jc12]**2*rate_eval.C12_C12__p_Na23
       -2*5.00000000000000e-01*rho*Y[jc12]**2*rate_eval.C12_C12__He4_Ne20
       -rho*Y[jc12]*Y[jo16]*rate_eval.C12_O16__p_Al27
       -rho*Y[jc12]*Y[jo16]*rate_eval.C12_O16__He4_Mg24
       -rho*Y[jc12]*Y[jne20]*rate_eval.C12_Ne20__n_S31
       -rho*Y[jc12]*Y[jne20]*rate_eval.C12_Ne20__p_P31
       -rho*Y[jc12]*Y[jne20]*rate_eval.C12_Ne20__He4_Si28
       +Y[jc13]*rate_eval.C13__n_C12
       +Y[jn13]*rate_eval.N13__p_C12
       +Y[jo16]*rate_eval.O16__He4_C12
       +rho*Y[jp]*Y[jb11]*rate_eval.p_B11__C12
       +rho*Y[jp]*Y[jn15]*rate_eval.p_N15__He4_C12
       +rho*Y[jn]*Y[jo15]*rate_eval.n_O15__He4_C12
       +2*rho*Y[jhe4]*Y[jne20]*rate_eval.He4_Ne20__C12_C12
       +2*rho*Y[jp]*Y[jna23]*rate_eval.p_Na23__C12_C12
       +2*rho*Y[jn]*Y[jmg23]*rate_eval.n_Mg23__C12_C12
       +rho*Y[jhe4]*Y[jmg24]*rate_eval.He4_Mg24__C12_O16
       +rho*Y[jp]*Y[jal27]*rate_eval.p_Al27__C12_O16
       +rho*Y[jhe4]*Y[jsi28]*rate_eval.He4_Si28__C12_Ne20
       +rho*Y[jp]*Y[jp31]*rate_eval.p_P31__C12_Ne20
       +rho*Y[jn]*Y[js31]*rate_eval.n_S31__C12_Ne20
       +1.66666666666667e-01*rho**2*Y[jhe4]**3*rate_eval.He4_He4_He4__C12
       )

    dYdt[jc13] = (
       -Y[jc13]*rate_eval.C13__n_C12
       -rho*Y[jp]*Y[jc13]*rate_eval.p_C13__N14
       -rho*Y[jp]*Y[jc13]*rate_eval.p_C13__n_N13
       -rho*Y[jhe4]*Y[jc13]*rate_eval.He4_C13__n_O16
       +Y[jn13]*rate_eval.N13__C13__weak__wc12
       +Y[jn14]*rate_eval.N14__p_C13
       +rho*Y[jn]*Y[jc12]*rate_eval.n_C12__C13
       +rho*Y[jn]*Y[jn13]*rate_eval.n_N13__p_C13
       +rho*Y[jn]*Y[jo16]*rate_eval.n_O16__He4_C13
       )

    dYdt[jn13] = (
       -Y[jn13]*rate_eval.N13__C13__weak__wc12
       -Y[jn13]*rate_eval.N13__p_C12
       -rho*Y[jn]*Y[jn13]*rate_eval.n_N13__N14
       -rho*Y[jn]*Y[jn13]*rate_eval.n_N13__p_C13
       -rho*Y[jhe4]*Y[jn13]*rate_eval.He4_N13__p_O16
       +Y[jn14]*rate_eval.N14__n_N13
       +rho*Y[jp]*Y[jc12]*rate_eval.p_C12__N13
       +rho*Y[jp]*Y[jc13]*rate_eval.p_C13__n_N13
       +rho*Y[jp]*Y[jo16]*rate_eval.p_O16__He4_N13
       )

    dYdt[jn14] = (
       -Y[jn14]*rate_eval.N14__n_N13
       -Y[jn14]*rate_eval.N14__p_C13
       -rho*Y[jn]*Y[jn14]*rate_eval.n_N14__N15
       -rho*Y[jp]*Y[jn14]*rate_eval.p_N14__O15
       -rho*Y[jhe4]*Y[jn14]*rate_eval.He4_N14__F18
       -rho*Y[jn]*Y[jn14]*rate_eval.n_N14__He4_B11
       -rho*Y[jhe4]*Y[jn14]*rate_eval.He4_N14__p_O17
       +Y[jn15]*rate_eval.N15__n_N14
       +Y[jo15]*rate_eval.O15__p_N14
       +Y[jf18]*rate_eval.F18__He4_N14
       +rho*Y[jp]*Y[jc13]*rate_eval.p_C13__N14
       +rho*Y[jn]*Y[jn13]*rate_eval.n_N13__N14
       +rho*Y[jhe4]*Y[jb11]*rate_eval.He4_B11__n_N14
       +rho*Y[jp]*Y[jo17]*rate_eval.p_O17__He4_N14
       )

    dYdt[jn15] = (
       -Y[jn15]*rate_eval.N15__n_N14
       -rho*Y[jp]*Y[jn15]*rate_eval.p_N15__O16
       -rho*Y[jp]*Y[jn15]*rate_eval.p_N15__n_O15
       -rho*Y[jp]*Y[jn15]*rate_eval.p_N15__He4_C12
       -rho*Y[jhe4]*Y[jn15]*rate_eval.He4_N15__n_F18
       +Y[jo15]*rate_eval.O15__N15__weak__wc12
       +Y[jo16]*rate_eval.O16__p_N15
       +rho*Y[jn]*Y[jn14]*rate_eval.n_N14__N15
       +rho*Y[jhe4]*Y[jc12]*rate_eval.He4_C12__p_N15
       +rho*Y[jn]*Y[jo15]*rate_eval.n_O15__p_N15
       +rho*Y[jn]*Y[jf18]*rate_eval.n_F18__He4_N15
       )

    dYdt[jo15] = (
       -Y[jo15]*rate_eval.O15__N15__weak__wc12
       -Y[jo15]*rate_eval.O15__p_N14
       -rho*Y[jn]*Y[jo15]*rate_eval.n_O15__O16
       -rho*Y[jhe4]*Y[jo15]*rate_eval.He4_O15__Ne19
       -rho*Y[jn]*Y[jo15]*rate_eval.n_O15__p_N15
       -rho*Y[jn]*Y[jo15]*rate_eval.n_O15__He4_C12
       -rho*Y[jhe4]*Y[jo15]*rate_eval.He4_O15__p_F18
       +Y[jo16]*rate_eval.O16__n_O15
       +Y[jne19]*rate_eval.Ne19__He4_O15
       +rho*Y[jp]*Y[jn14]*rate_eval.p_N14__O15
       +rho*Y[jhe4]*Y[jc12]*rate_eval.He4_C12__n_O15
       +rho*Y[jp]*Y[jn15]*rate_eval.p_N15__n_O15
       +rho*Y[jp]*Y[jf18]*rate_eval.p_F18__He4_O15
       )

    dYdt[jo16] = (
       -Y[jo16]*rate_eval.O16__n_O15
       -Y[jo16]*rate_eval.O16__p_N15
       -Y[jo16]*rate_eval.O16__He4_C12
       -rho*Y[jn]*Y[jo16]*rate_eval.n_O16__O17
       -rho*Y[jhe4]*Y[jo16]*rate_eval.He4_O16__Ne20
       -rho*Y[jn]*Y[jo16]*rate_eval.n_O16__He4_C13
       -rho*Y[jp]*Y[jo16]*rate_eval.p_O16__He4_N13
       -rho*Y[jhe4]*Y[jo16]*rate_eval.He4_O16__n_Ne19
       -rho*Y[jc12]*Y[jo16]*rate_eval.C12_O16__p_Al27
       -rho*Y[jc12]*Y[jo16]*rate_eval.C12_O16__He4_Mg24
       -2*5.00000000000000e-01*rho*Y[jo16]**2*rate_eval.O16_O16__n_S31
       -2*5.00000000000000e-01*rho*Y[jo16]**2*rate_eval.O16_O16__p_P31
       -2*5.00000000000000e-01*rho*Y[jo16]**2*rate_eval.O16_O16__He4_Si28
       +Y[jo17]*rate_eval.O17__n_O16
       +Y[jne20]*rate_eval.Ne20__He4_O16
       +rho*Y[jhe4]*Y[jc12]*rate_eval.He4_C12__O16
       +rho*Y[jp]*Y[jn15]*rate_eval.p_N15__O16
       +rho*Y[jn]*Y[jo15]*rate_eval.n_O15__O16
       +rho*Y[jhe4]*Y[jc13]*rate_eval.He4_C13__n_O16
       +rho*Y[jhe4]*Y[jn13]*rate_eval.He4_N13__p_O16
       +rho*Y[jn]*Y[jne19]*rate_eval.n_Ne19__He4_O16
       +rho*Y[jhe4]*Y[jmg24]*rate_eval.He4_Mg24__C12_O16
       +rho*Y[jp]*Y[jal27]*rate_eval.p_Al27__C12_O16
       +2*rho*Y[jhe4]*Y[jsi28]*rate_eval.He4_Si28__O16_O16
       +2*rho*Y[jp]*Y[jp31]*rate_eval.p_P31__O16_O16
       +2*rho*Y[jn]*Y[js31]*rate_eval.n_S31__O16_O16
       )

    dYdt[jo17] = (
       -Y[jo17]*rate_eval.O17__n_O16
       -rho*Y[jp]*Y[jo17]*rate_eval.p_O17__F18
       -rho*Y[jhe4]*Y[jo17]*rate_eval.He4_O17__Ne21
       -rho*Y[jp]*Y[jo17]*rate_eval.p_O17__He4_N14
       -rho*Y[jhe4]*Y[jo17]*rate_eval.He4_O17__n_Ne20
       +Y[jf18]*rate_eval.F18__p_O17
       +Y[jne21]*rate_eval.Ne21__He4_O17
       +rho*Y[jn]*Y[jo16]*rate_eval.n_O16__O17
       +rho*Y[jhe4]*Y[jn14]*rate_eval.He4_N14__p_O17
       +rho*Y[jn]*Y[jne20]*rate_eval.n_Ne20__He4_O17
       )

    dYdt[jf18] = (
       -Y[jf18]*rate_eval.F18__p_O17
       -Y[jf18]*rate_eval.F18__He4_N14
       -rho*Y[jp]*Y[jf18]*rate_eval.p_F18__Ne19
       -rho*Y[jhe4]*Y[jf18]*rate_eval.He4_F18__Na22
       -rho*Y[jn]*Y[jf18]*rate_eval.n_F18__He4_N15
       -rho*Y[jp]*Y[jf18]*rate_eval.p_F18__He4_O15
       -rho*Y[jhe4]*Y[jf18]*rate_eval.He4_F18__p_Ne21
       +Y[jne19]*rate_eval.Ne19__p_F18
       +Y[jna22]*rate_eval.Na22__He4_F18
       +rho*Y[jhe4]*Y[jn14]*rate_eval.He4_N14__F18
       +rho*Y[jp]*Y[jo17]*rate_eval.p_O17__F18
       +rho*Y[jhe4]*Y[jn15]*rate_eval.He4_N15__n_F18
       +rho*Y[jhe4]*Y[jo15]*rate_eval.He4_O15__p_F18
       +rho*Y[jp]*Y[jne21]*rate_eval.p_Ne21__He4_F18
       )

    dYdt[jne19] = (
       -Y[jne19]*rate_eval.Ne19__p_F18
       -Y[jne19]*rate_eval.Ne19__He4_O15
       -rho*Y[jn]*Y[jne19]*rate_eval.n_Ne19__Ne20
       -rho*Y[jhe4]*Y[jne19]*rate_eval.He4_Ne19__Mg23
       -rho*Y[jn]*Y[jne19]*rate_eval.n_Ne19__He4_O16
       -rho*Y[jhe4]*Y[jne19]*rate_eval.He4_Ne19__p_Na22
       +Y[jne20]*rate_eval.Ne20__n_Ne19
       +Y[jmg23]*rate_eval.Mg23__He4_Ne19
       +rho*Y[jhe4]*Y[jo15]*rate_eval.He4_O15__Ne19
       +rho*Y[jp]*Y[jf18]*rate_eval.p_F18__Ne19
       +rho*Y[jhe4]*Y[jo16]*rate_eval.He4_O16__n_Ne19
       +rho*Y[jp]*Y[jna22]*rate_eval.p_Na22__He4_Ne19
       )

    dYdt[jne20] = (
       -Y[jne20]*rate_eval.Ne20__n_Ne19
       -Y[jne20]*rate_eval.Ne20__He4_O16
       -rho*Y[jn]*Y[jne20]*rate_eval.n_Ne20__Ne21
       -rho*Y[jhe4]*Y[jne20]*rate_eval.He4_Ne20__Mg24
       -rho*Y[jn]*Y[jne20]*rate_eval.n_Ne20__He4_O17
       -rho*Y[jhe4]*Y[jne20]*rate_eval.He4_Ne20__n_Mg23
       -rho*Y[jhe4]*Y[jne20]*rate_eval.He4_Ne20__p_Na23
       -rho*Y[jhe4]*Y[jne20]*rate_eval.He4_Ne20__C12_C12
       -rho*Y[jc12]*Y[jne20]*rate_eval.C12_Ne20__n_S31
       -rho*Y[jc12]*Y[jne20]*rate_eval.C12_Ne20__p_P31
       -rho*Y[jc12]*Y[jne20]*rate_eval.C12_Ne20__He4_Si28
       +Y[jne21]*rate_eval.Ne21__n_Ne20
       +Y[jmg24]*rate_eval.Mg24__He4_Ne20
       +rho*Y[jhe4]*Y[jo16]*rate_eval.He4_O16__Ne20
       +rho*Y[jn]*Y[jne19]*rate_eval.n_Ne19__Ne20
       +5.00000000000000e-01*rho*Y[jc12]**2*rate_eval.C12_C12__He4_Ne20
       +rho*Y[jhe4]*Y[jo17]*rate_eval.He4_O17__n_Ne20
       +rho*Y[jp]*Y[jna23]*rate_eval.p_Na23__He4_Ne20
       +rho*Y[jn]*Y[jmg23]*rate_eval.n_Mg23__He4_Ne20
       +rho*Y[jhe4]*Y[jsi28]*rate_eval.He4_Si28__C12_Ne20
       +rho*Y[jp]*Y[jp31]*rate_eval.p_P31__C12_Ne20
       +rho*Y[jn]*Y[js31]*rate_eval.n_S31__C12_Ne20
       )

    dYdt[jne21] = (
       -Y[jne21]*rate_eval.Ne21__n_Ne20
       -Y[jne21]*rate_eval.Ne21__He4_O17
       -rho*Y[jn]*Y[jne21]*rate_eval.n_Ne21__Ne22
       -rho*Y[jp]*Y[jne21]*rate_eval.p_Ne21__Na22
       -rho*Y[jhe4]*Y[jne21]*rate_eval.He4_Ne21__Mg25
       -rho*Y[jp]*Y[jne21]*rate_eval.p_Ne21__He4_F18
       -rho*Y[jhe4]*Y[jne21]*rate_eval.He4_Ne21__n_Mg24
       +Y[jne22]*rate_eval.Ne22__n_Ne21
       +Y[jna22]*rate_eval.Na22__p_Ne21
       +Y[jmg25]*rate_eval.Mg25__He4_Ne21
       +rho*Y[jhe4]*Y[jo17]*rate_eval.He4_O17__Ne21
       +rho*Y[jn]*Y[jne20]*rate_eval.n_Ne20__Ne21
       +rho*Y[jhe4]*Y[jf18]*rate_eval.He4_F18__p_Ne21
       +rho*Y[jn]*Y[jmg24]*rate_eval.n_Mg24__He4_Ne21
       )

    dYdt[jne22] = (
       -Y[jne22]*rate_eval.Ne22__n_Ne21
       -rho*Y[jp]*Y[jne22]*rate_eval.p_Ne22__Na23
       -rho*Y[jhe4]*Y[jne22]*rate_eval.He4_Ne22__Mg26
       -rho*Y[jp]*Y[jne22]*rate_eval.p_Ne22__n_Na22
       -rho*Y[jhe4]*Y[jne22]*rate_eval.He4_Ne22__n_Mg25
       +Y[jna22]*rate_eval.Na22__Ne22__weak__wc12
       +Y[jna23]*rate_eval.Na23__p_Ne22
       +Y[jmg26]*rate_eval.Mg26__He4_Ne22
       +rho*Y[jn]*Y[jne21]*rate_eval.n_Ne21__Ne22
       +rho*Y[jn]*Y[jna22]*rate_eval.n_Na22__p_Ne22
       +rho*Y[jn]*Y[jmg25]*rate_eval.n_Mg25__He4_Ne22
       )

    dYdt[jna22] = (
       -Y[jna22]*rate_eval.Na22__Ne22__weak__wc12
       -Y[jna22]*rate_eval.Na22__p_Ne21
       -Y[jna22]*rate_eval.Na22__He4_F18
       -rho*Y[jn]*Y[jna22]*rate_eval.n_Na22__Na23
       -rho*Y[jp]*Y[jna22]*rate_eval.p_Na22__Mg23
       -rho*Y[jhe4]*Y[jna22]*rate_eval.He4_Na22__Al26
       -rho*Y[jn]*Y[jna22]*rate_eval.n_Na22__p_Ne22
       -rho*Y[jp]*Y[jna22]*rate_eval.p_Na22__He4_Ne19
       -rho*Y[jhe4]*Y[jna22]*rate_eval.He4_Na22__n_Al25
       -rho*Y[jhe4]*Y[jna22]*rate_eval.He4_Na22__p_Mg25
       +Y[jna23]*rate_eval.Na23__n_Na22
       +Y[jmg23]*rate_eval.Mg23__p_Na22
       +Y[jal26]*rate_eval.Al26__He4_Na22
       +rho*Y[jhe4]*Y[jf18]*rate_eval.He4_F18__Na22
       +rho*Y[jp]*Y[jne21]*rate_eval.p_Ne21__Na22
       +rho*Y[jhe4]*Y[jne19]*rate_eval.He4_Ne19__p_Na22
       +rho*Y[jp]*Y[jne22]*rate_eval.p_Ne22__n_Na22
       +rho*Y[jp]*Y[jmg25]*rate_eval.p_Mg25__He4_Na22
       +rho*Y[jn]*Y[jal25]*rate_eval.n_Al25__He4_Na22
       )

    dYdt[jna23] = (
       -Y[jna23]*rate_eval.Na23__n_Na22
       -Y[jna23]*rate_eval.Na23__p_Ne22
       -rho*Y[jp]*Y[jna23]*rate_eval.p_Na23__Mg24
       -rho*Y[jhe4]*Y[jna23]*rate_eval.He4_Na23__Al27
       -rho*Y[jp]*Y[jna23]*rate_eval.p_Na23__n_Mg23
       -rho*Y[jp]*Y[jna23]*rate_eval.p_Na23__He4_Ne20
       -rho*Y[jp]*Y[jna23]*rate_eval.p_Na23__C12_C12
       -rho*Y[jhe4]*Y[jna23]*rate_eval.He4_Na23__n_Al26
       -rho*Y[jhe4]*Y[jna23]*rate_eval.He4_Na23__p_Mg26
       +Y[jmg23]*rate_eval.Mg23__Na23__weak__wc12
       +Y[jmg24]*rate_eval.Mg24__p_Na23
       +Y[jal27]*rate_eval.Al27__He4_Na23
       +rho*Y[jp]*Y[jne22]*rate_eval.p_Ne22__Na23
       +rho*Y[jn]*Y[jna22]*rate_eval.n_Na22__Na23
       +5.00000000000000e-01*rho*Y[jc12]**2*rate_eval.C12_C12__p_Na23
       +rho*Y[jhe4]*Y[jne20]*rate_eval.He4_Ne20__p_Na23
       +rho*Y[jn]*Y[jmg23]*rate_eval.n_Mg23__p_Na23
       +rho*Y[jp]*Y[jmg26]*rate_eval.p_Mg26__He4_Na23
       +rho*Y[jn]*Y[jal26]*rate_eval.n_Al26__He4_Na23
       )

    dYdt[jmg23] = (
       -Y[jmg23]*rate_eval.Mg23__Na23__weak__wc12
       -Y[jmg23]*rate_eval.Mg23__p_Na22
       -Y[jmg23]*rate_eval.Mg23__He4_Ne19
       -rho*Y[jn]*Y[jmg23]*rate_eval.n_Mg23__Mg24
       -rho*Y[jn]*Y[jmg23]*rate_eval.n_Mg23__p_Na23
       -rho*Y[jn]*Y[jmg23]*rate_eval.n_Mg23__He4_Ne20
       -rho*Y[jn]*Y[jmg23]*rate_eval.n_Mg23__C12_C12
       -rho*Y[jhe4]*Y[jmg23]*rate_eval.He4_Mg23__n_Si26
       -rho*Y[jhe4]*Y[jmg23]*rate_eval.He4_Mg23__p_Al26
       +Y[jmg24]*rate_eval.Mg24__n_Mg23
       +rho*Y[jhe4]*Y[jne19]*rate_eval.He4_Ne19__Mg23
       +rho*Y[jp]*Y[jna22]*rate_eval.p_Na22__Mg23
       +5.00000000000000e-01*rho*Y[jc12]**2*rate_eval.C12_C12__n_Mg23
       +rho*Y[jhe4]*Y[jne20]*rate_eval.He4_Ne20__n_Mg23
       +rho*Y[jp]*Y[jna23]*rate_eval.p_Na23__n_Mg23
       +rho*Y[jp]*Y[jal26]*rate_eval.p_Al26__He4_Mg23
       +rho*Y[jn]*Y[jsi26]*rate_eval.n_Si26__He4_Mg23
       )

    dYdt[jmg24] = (
       -Y[jmg24]*rate_eval.Mg24__n_Mg23
       -Y[jmg24]*rate_eval.Mg24__p_Na23
       -Y[jmg24]*rate_eval.Mg24__He4_Ne20
       -rho*Y[jn]*Y[jmg24]*rate_eval.n_Mg24__Mg25
       -rho*Y[jp]*Y[jmg24]*rate_eval.p_Mg24__Al25
       -rho*Y[jhe4]*Y[jmg24]*rate_eval.He4_Mg24__Si28
       -rho*Y[jn]*Y[jmg24]*rate_eval.n_Mg24__He4_Ne21
       -rho*Y[jhe4]*Y[jmg24]*rate_eval.He4_Mg24__p_Al27
       -rho*Y[jhe4]*Y[jmg24]*rate_eval.He4_Mg24__C12_O16
       +Y[jmg25]*rate_eval.Mg25__n_Mg24
       +Y[jal25]*rate_eval.Al25__p_Mg24
       +Y[jsi28]*rate_eval.Si28__He4_Mg24
       +rho*Y[jhe4]*Y[jne20]*rate_eval.He4_Ne20__Mg24
       +rho*Y[jp]*Y[jna23]*rate_eval.p_Na23__Mg24
       +rho*Y[jn]*Y[jmg23]*rate_eval.n_Mg23__Mg24
       +rho*Y[jc12]*Y[jo16]*rate_eval.C12_O16__He4_Mg24
       +rho*Y[jhe4]*Y[jne21]*rate_eval.He4_Ne21__n_Mg24
       +rho*Y[jp]*Y[jal27]*rate_eval.p_Al27__He4_Mg24
       )

    dYdt[jmg25] = (
       -Y[jmg25]*rate_eval.Mg25__n_Mg24
       -Y[jmg25]*rate_eval.Mg25__He4_Ne21
       -rho*Y[jn]*Y[jmg25]*rate_eval.n_Mg25__Mg26
       -rho*Y[jp]*Y[jmg25]*rate_eval.p_Mg25__Al26
       -rho*Y[jhe4]*Y[jmg25]*rate_eval.He4_Mg25__Si29
       -rho*Y[jn]*Y[jmg25]*rate_eval.n_Mg25__He4_Ne22
       -rho*Y[jp]*Y[jmg25]*rate_eval.p_Mg25__n_Al25
       -rho*Y[jp]*Y[jmg25]*rate_eval.p_Mg25__He4_Na22
       -rho*Y[jhe4]*Y[jmg25]*rate_eval.He4_Mg25__n_Si28
       +Y[jal25]*rate_eval.Al25__Mg25__weak__wc12
       +Y[jmg26]*rate_eval.Mg26__n_Mg25
       +Y[jal26]*rate_eval.Al26__p_Mg25
       +Y[jsi29]*rate_eval.Si29__He4_Mg25
       +rho*Y[jhe4]*Y[jne21]*rate_eval.He4_Ne21__Mg25
       +rho*Y[jn]*Y[jmg24]*rate_eval.n_Mg24__Mg25
       +rho*Y[jhe4]*Y[jne22]*rate_eval.He4_Ne22__n_Mg25
       +rho*Y[jhe4]*Y[jna22]*rate_eval.He4_Na22__p_Mg25
       +rho*Y[jn]*Y[jal25]*rate_eval.n_Al25__p_Mg25
       +rho*Y[jn]*Y[jsi28]*rate_eval.n_Si28__He4_Mg25
       )

    dYdt[jmg26] = (
       -Y[jmg26]*rate_eval.Mg26__n_Mg25
       -Y[jmg26]*rate_eval.Mg26__He4_Ne22
       -rho*Y[jp]*Y[jmg26]*rate_eval.p_Mg26__Al27
       -rho*Y[jhe4]*Y[jmg26]*rate_eval.He4_Mg26__Si30
       -rho*Y[jp]*Y[jmg26]*rate_eval.p_Mg26__n_Al26
       -rho*Y[jp]*Y[jmg26]*rate_eval.p_Mg26__He4_Na23
       -rho*Y[jhe4]*Y[jmg26]*rate_eval.He4_Mg26__n_Si29
       +Y[jal26]*rate_eval.Al26__Mg26__weak__wc12
       +Y[jal27]*rate_eval.Al27__p_Mg26
       +Y[jsi30]*rate_eval.Si30__He4_Mg26
       +rho*Y[jhe4]*Y[jne22]*rate_eval.He4_Ne22__Mg26
       +rho*Y[jn]*Y[jmg25]*rate_eval.n_Mg25__Mg26
       +rho*Y[jhe4]*Y[jna23]*rate_eval.He4_Na23__p_Mg26
       +rho*Y[jn]*Y[jal26]*rate_eval.n_Al26__p_Mg26
       +rho*Y[jn]*Y[jsi29]*rate_eval.n_Si29__He4_Mg26
       )

    dYdt[jal25] = (
       -Y[jal25]*rate_eval.Al25__Mg25__weak__wc12
       -Y[jal25]*rate_eval.Al25__p_Mg24
       -rho*Y[jn]*Y[jal25]*rate_eval.n_Al25__Al26
       -rho*Y[jp]*Y[jal25]*rate_eval.p_Al25__Si26
       -rho*Y[jhe4]*Y[jal25]*rate_eval.He4_Al25__P29
       -rho*Y[jn]*Y[jal25]*rate_eval.n_Al25__p_Mg25
       -rho*Y[jn]*Y[jal25]*rate_eval.n_Al25__He4_Na22
       -rho*Y[jhe4]*Y[jal25]*rate_eval.He4_Al25__p_Si28
       +Y[jal26]*rate_eval.Al26__n_Al25
       +Y[jsi26]*rate_eval.Si26__p_Al25
       +Y[jp29]*rate_eval.P29__He4_Al25
       +rho*Y[jp]*Y[jmg24]*rate_eval.p_Mg24__Al25
       +rho*Y[jhe4]*Y[jna22]*rate_eval.He4_Na22__n_Al25
       +rho*Y[jp]*Y[jmg25]*rate_eval.p_Mg25__n_Al25
       +rho*Y[jp]*Y[jsi28]*rate_eval.p_Si28__He4_Al25
       )

    dYdt[jal26] = (
       -Y[jal26]*rate_eval.Al26__Mg26__weak__wc12
       -Y[jal26]*rate_eval.Al26__n_Al25
       -Y[jal26]*rate_eval.Al26__p_Mg25
       -Y[jal26]*rate_eval.Al26__He4_Na22
       -rho*Y[jn]*Y[jal26]*rate_eval.n_Al26__Al27
       -rho*Y[jhe4]*Y[jal26]*rate_eval.He4_Al26__P30
       -rho*Y[jn]*Y[jal26]*rate_eval.n_Al26__p_Mg26
       -rho*Y[jn]*Y[jal26]*rate_eval.n_Al26__He4_Na23
       -rho*Y[jp]*Y[jal26]*rate_eval.p_Al26__n_Si26
       -rho*Y[jp]*Y[jal26]*rate_eval.p_Al26__He4_Mg23
       -rho*Y[jhe4]*Y[jal26]*rate_eval.He4_Al26__n_P29
       -rho*Y[jhe4]*Y[jal26]*rate_eval.He4_Al26__p_Si29
       +Y[jsi26]*rate_eval.Si26__Al26__weak__wc12
       +Y[jal27]*rate_eval.Al27__n_Al26
       +Y[jp30]*rate_eval.P30__He4_Al26
       +rho*Y[jhe4]*Y[jna22]*rate_eval.He4_Na22__Al26
       +rho*Y[jp]*Y[jmg25]*rate_eval.p_Mg25__Al26
       +rho*Y[jn]*Y[jal25]*rate_eval.n_Al25__Al26
       +rho*Y[jhe4]*Y[jna23]*rate_eval.He4_Na23__n_Al26
       +rho*Y[jhe4]*Y[jmg23]*rate_eval.He4_Mg23__p_Al26
       +rho*Y[jp]*Y[jmg26]*rate_eval.p_Mg26__n_Al26
       +rho*Y[jn]*Y[jsi26]*rate_eval.n_Si26__p_Al26
       +rho*Y[jp]*Y[jsi29]*rate_eval.p_Si29__He4_Al26
       +rho*Y[jn]*Y[jp29]*rate_eval.n_P29__He4_Al26
       )

    dYdt[jal27] = (
       -Y[jal27]*rate_eval.Al27__n_Al26
       -Y[jal27]*rate_eval.Al27__p_Mg26
       -Y[jal27]*rate_eval.Al27__He4_Na23
       -rho*Y[jp]*Y[jal27]*rate_eval.p_Al27__Si28
       -rho*Y[jhe4]*Y[jal27]*rate_eval.He4_Al27__P31
       -rho*Y[jp]*Y[jal27]*rate_eval.p_Al27__He4_Mg24
       -rho*Y[jp]*Y[jal27]*rate_eval.p_Al27__C12_O16
       -rho*Y[jhe4]*Y[jal27]*rate_eval.He4_Al27__n_P30
       -rho*Y[jhe4]*Y[jal27]*rate_eval.He4_Al27__p_Si30
       +Y[jsi28]*rate_eval.Si28__p_Al27
       +Y[jp31]*rate_eval.P31__He4_Al27
       +rho*Y[jhe4]*Y[jna23]*rate_eval.He4_Na23__Al27
       +rho*Y[jp]*Y[jmg26]*rate_eval.p_Mg26__Al27
       +rho*Y[jn]*Y[jal26]*rate_eval.n_Al26__Al27
       +rho*Y[jc12]*Y[jo16]*rate_eval.C12_O16__p_Al27
       +rho*Y[jhe4]*Y[jmg24]*rate_eval.He4_Mg24__p_Al27
       +rho*Y[jp]*Y[jsi30]*rate_eval.p_Si30__He4_Al27
       +rho*Y[jn]*Y[jp30]*rate_eval.n_P30__He4_Al27
       )

    dYdt[jsi26] = (
       -Y[jsi26]*rate_eval.Si26__Al26__weak__wc12
       -Y[jsi26]*rate_eval.Si26__p_Al25
       -rho*Y[jhe4]*Y[jsi26]*rate_eval.He4_Si26__S30
       -rho*Y[jn]*Y[jsi26]*rate_eval.n_Si26__p_Al26
       -rho*Y[jn]*Y[jsi26]*rate_eval.n_Si26__He4_Mg23
       -rho*Y[jhe4]*Y[jsi26]*rate_eval.He4_Si26__p_P29
       +Y[js30]*rate_eval.S30__He4_Si26
       +rho*Y[jp]*Y[jal25]*rate_eval.p_Al25__Si26
       +rho*Y[jhe4]*Y[jmg23]*rate_eval.He4_Mg23__n_Si26
       +rho*Y[jp]*Y[jal26]*rate_eval.p_Al26__n_Si26
       +rho*Y[jp]*Y[jp29]*rate_eval.p_P29__He4_Si26
       )

    dYdt[jsi28] = (
       -Y[jsi28]*rate_eval.Si28__p_Al27
       -Y[jsi28]*rate_eval.Si28__He4_Mg24
       -rho*Y[jn]*Y[jsi28]*rate_eval.n_Si28__Si29
       -rho*Y[jp]*Y[jsi28]*rate_eval.p_Si28__P29
       -rho*Y[jhe4]*Y[jsi28]*rate_eval.He4_Si28__S32
       -rho*Y[jn]*Y[jsi28]*rate_eval.n_Si28__He4_Mg25
       -rho*Y[jp]*Y[jsi28]*rate_eval.p_Si28__He4_Al25
       -rho*Y[jhe4]*Y[jsi28]*rate_eval.He4_Si28__n_S31
       -rho*Y[jhe4]*Y[jsi28]*rate_eval.He4_Si28__p_P31
       -rho*Y[jhe4]*Y[jsi28]*rate_eval.He4_Si28__C12_Ne20
       -rho*Y[jhe4]*Y[jsi28]*rate_eval.He4_Si28__O16_O16
       +Y[jsi29]*rate_eval.Si29__n_Si28
       +Y[jp29]*rate_eval.P29__p_Si28
       +Y[js32]*rate_eval.S32__He4_Si28
       +rho*Y[jhe4]*Y[jmg24]*rate_eval.He4_Mg24__Si28
       +rho*Y[jp]*Y[jal27]*rate_eval.p_Al27__Si28
       +5.00000000000000e-01*rho*Y[jo16]**2*rate_eval.O16_O16__He4_Si28
       +rho*Y[jc12]*Y[jne20]*rate_eval.C12_Ne20__He4_Si28
       +rho*Y[jhe4]*Y[jmg25]*rate_eval.He4_Mg25__n_Si28
       +rho*Y[jhe4]*Y[jal25]*rate_eval.He4_Al25__p_Si28
       +rho*Y[jp]*Y[jp31]*rate_eval.p_P31__He4_Si28
       +rho*Y[jn]*Y[js31]*rate_eval.n_S31__He4_Si28
       )

    dYdt[jsi29] = (
       -Y[jsi29]*rate_eval.Si29__n_Si28
       -Y[jsi29]*rate_eval.Si29__He4_Mg25
       -rho*Y[jn]*Y[jsi29]*rate_eval.n_Si29__Si30
       -rho*Y[jp]*Y[jsi29]*rate_eval.p_Si29__P30
       -rho*Y[jhe4]*Y[jsi29]*rate_eval.He4_Si29__S33
       -rho*Y[jn]*Y[jsi29]*rate_eval.n_Si29__He4_Mg26
       -rho*Y[jp]*Y[jsi29]*rate_eval.p_Si29__n_P29
       -rho*Y[jp]*Y[jsi29]*rate_eval.p_Si29__He4_Al26
       -rho*Y[jhe4]*Y[jsi29]*rate_eval.He4_Si29__n_S32
       +Y[jp29]*rate_eval.P29__Si29__weak__wc12
       +Y[jsi30]*rate_eval.Si30__n_Si29
       +Y[jp30]*rate_eval.P30__p_Si29
       +Y[js33]*rate_eval.S33__He4_Si29
       +rho*Y[jhe4]*Y[jmg25]*rate_eval.He4_Mg25__Si29
       +rho*Y[jn]*Y[jsi28]*rate_eval.n_Si28__Si29
       +rho*Y[jhe4]*Y[jmg26]*rate_eval.He4_Mg26__n_Si29
       +rho*Y[jhe4]*Y[jal26]*rate_eval.He4_Al26__p_Si29
       +rho*Y[jn]*Y[jp29]*rate_eval.n_P29__p_Si29
       +rho*Y[jn]*Y[js32]*rate_eval.n_S32__He4_Si29
       )

    dYdt[jsi30] = (
       -Y[jsi30]*rate_eval.Si30__n_Si29
       -Y[jsi30]*rate_eval.Si30__He4_Mg26
       -rho*Y[jp]*Y[jsi30]*rate_eval.p_Si30__P31
       -rho*Y[jp]*Y[jsi30]*rate_eval.p_Si30__n_P30
       -rho*Y[jp]*Y[jsi30]*rate_eval.p_Si30__He4_Al27
       -rho*Y[jhe4]*Y[jsi30]*rate_eval.He4_Si30__n_S33
       +Y[jp30]*rate_eval.P30__Si30__weak__wc12
       +Y[jp31]*rate_eval.P31__p_Si30
       +rho*Y[jhe4]*Y[jmg26]*rate_eval.He4_Mg26__Si30
       +rho*Y[jn]*Y[jsi29]*rate_eval.n_Si29__Si30
       +rho*Y[jhe4]*Y[jal27]*rate_eval.He4_Al27__p_Si30
       +rho*Y[jn]*Y[jp30]*rate_eval.n_P30__p_Si30
       +rho*Y[jn]*Y[js33]*rate_eval.n_S33__He4_Si30
       )

    dYdt[jp29] = (
       -Y[jp29]*rate_eval.P29__Si29__weak__wc12
       -Y[jp29]*rate_eval.P29__p_Si28
       -Y[jp29]*rate_eval.P29__He4_Al25
       -rho*Y[jn]*Y[jp29]*rate_eval.n_P29__P30
       -rho*Y[jp]*Y[jp29]*rate_eval.p_P29__S30
       -rho*Y[jhe4]*Y[jp29]*rate_eval.He4_P29__Cl33
       -rho*Y[jn]*Y[jp29]*rate_eval.n_P29__p_Si29
       -rho*Y[jn]*Y[jp29]*rate_eval.n_P29__He4_Al26
       -rho*Y[jp]*Y[jp29]*rate_eval.p_P29__He4_Si26
       -rho*Y[jhe4]*Y[jp29]*rate_eval.He4_P29__p_S32
       +Y[jp30]*rate_eval.P30__n_P29
       +Y[js30]*rate_eval.S30__p_P29
       +Y[jcl33]*rate_eval.Cl33__He4_P29
       +rho*Y[jhe4]*Y[jal25]*rate_eval.He4_Al25__P29
       +rho*Y[jp]*Y[jsi28]*rate_eval.p_Si28__P29
       +rho*Y[jhe4]*Y[jal26]*rate_eval.He4_Al26__n_P29
       +rho*Y[jhe4]*Y[jsi26]*rate_eval.He4_Si26__p_P29
       +rho*Y[jp]*Y[jsi29]*rate_eval.p_Si29__n_P29
       +rho*Y[jp]*Y[js32]*rate_eval.p_S32__He4_P29
       )

    dYdt[jp30] = (
       -Y[jp30]*rate_eval.P30__Si30__weak__wc12
       -Y[jp30]*rate_eval.P30__n_P29
       -Y[jp30]*rate_eval.P30__p_Si29
       -Y[jp30]*rate_eval.P30__He4_Al26
       -rho*Y[jn]*Y[jp30]*rate_eval.n_P30__P31
       -rho*Y[jp]*Y[jp30]*rate_eval.p_P30__S31
       -rho*Y[jhe4]*Y[jp30]*rate_eval.He4_P30__Cl34
       -rho*Y[jn]*Y[jp30]*rate_eval.n_P30__p_Si30
       -rho*Y[jn]*Y[jp30]*rate_eval.n_P30__He4_Al27
       -rho*Y[jp]*Y[jp30]*rate_eval.p_P30__n_S30
       -rho*Y[jhe4]*Y[jp30]*rate_eval.He4_P30__n_Cl33
       -rho*Y[jhe4]*Y[jp30]*rate_eval.He4_P30__p_S33
       +Y[js30]*rate_eval.S30__P30__weak__wc12
       +Y[jp31]*rate_eval.P31__n_P30
       +Y[js31]*rate_eval.S31__p_P30
       +Y[jcl34]*rate_eval.Cl34__He4_P30
       +rho*Y[jhe4]*Y[jal26]*rate_eval.He4_Al26__P30
       +rho*Y[jp]*Y[jsi29]*rate_eval.p_Si29__P30
       +rho*Y[jn]*Y[jp29]*rate_eval.n_P29__P30
       +rho*Y[jhe4]*Y[jal27]*rate_eval.He4_Al27__n_P30
       +rho*Y[jp]*Y[jsi30]*rate_eval.p_Si30__n_P30
       +rho*Y[jn]*Y[js30]*rate_eval.n_S30__p_P30
       +rho*Y[jp]*Y[js33]*rate_eval.p_S33__He4_P30
       +rho*Y[jn]*Y[jcl33]*rate_eval.n_Cl33__He4_P30
       )

    dYdt[jp31] = (
       -Y[jp31]*rate_eval.P31__n_P30
       -Y[jp31]*rate_eval.P31__p_Si30
       -Y[jp31]*rate_eval.P31__He4_Al27
       -rho*Y[jp]*Y[jp31]*rate_eval.p_P31__S32
       -rho*Y[jhe4]*Y[jp31]*rate_eval.He4_P31__Cl35
       -rho*Y[jp]*Y[jp31]*rate_eval.p_P31__n_S31
       -rho*Y[jp]*Y[jp31]*rate_eval.p_P31__He4_Si28
       -rho*Y[jp]*Y[jp31]*rate_eval.p_P31__C12_Ne20
       -rho*Y[jp]*Y[jp31]*rate_eval.p_P31__O16_O16
       -rho*Y[jhe4]*Y[jp31]*rate_eval.He4_P31__n_Cl34
       +Y[js31]*rate_eval.S31__P31__weak__wc12
       +Y[js32]*rate_eval.S32__p_P31
       +Y[jcl35]*rate_eval.Cl35__He4_P31
       +rho*Y[jhe4]*Y[jal27]*rate_eval.He4_Al27__P31
       +rho*Y[jp]*Y[jsi30]*rate_eval.p_Si30__P31
       +rho*Y[jn]*Y[jp30]*rate_eval.n_P30__P31
       +5.00000000000000e-01*rho*Y[jo16]**2*rate_eval.O16_O16__p_P31
       +rho*Y[jc12]*Y[jne20]*rate_eval.C12_Ne20__p_P31
       +rho*Y[jhe4]*Y[jsi28]*rate_eval.He4_Si28__p_P31
       +rho*Y[jn]*Y[js31]*rate_eval.n_S31__p_P31
       +rho*Y[jn]*Y[jcl34]*rate_eval.n_Cl34__He4_P31
       )

    dYdt[js30] = (
       -Y[js30]*rate_eval.S30__P30__weak__wc12
       -Y[js30]*rate_eval.S30__p_P29
       -Y[js30]*rate_eval.S30__He4_Si26
       -rho*Y[jn]*Y[js30]*rate_eval.n_S30__S31
       -rho*Y[jhe4]*Y[js30]*rate_eval.He4_S30__Ar34
       -rho*Y[jn]*Y[js30]*rate_eval.n_S30__p_P30
       -rho*Y[jhe4]*Y[js30]*rate_eval.He4_S30__p_Cl33
       +Y[js31]*rate_eval.S31__n_S30
       +Y[jar34]*rate_eval.Ar34__He4_S30
       +rho*Y[jhe4]*Y[jsi26]*rate_eval.He4_Si26__S30
       +rho*Y[jp]*Y[jp29]*rate_eval.p_P29__S30
       +rho*Y[jp]*Y[jp30]*rate_eval.p_P30__n_S30
       +rho*Y[jp]*Y[jcl33]*rate_eval.p_Cl33__He4_S30
       )

    dYdt[js31] = (
       -Y[js31]*rate_eval.S31__P31__weak__wc12
       -Y[js31]*rate_eval.S31__n_S30
       -Y[js31]*rate_eval.S31__p_P30
       -rho*Y[jn]*Y[js31]*rate_eval.n_S31__S32
       -rho*Y[jn]*Y[js31]*rate_eval.n_S31__p_P31
       -rho*Y[jn]*Y[js31]*rate_eval.n_S31__He4_Si28
       -rho*Y[jn]*Y[js31]*rate_eval.n_S31__C12_Ne20
       -rho*Y[jn]*Y[js31]*rate_eval.n_S31__O16_O16
       -rho*Y[jhe4]*Y[js31]*rate_eval.He4_S31__n_Ar34
       -rho*Y[jhe4]*Y[js31]*rate_eval.He4_S31__p_Cl34
       +Y[js32]*rate_eval.S32__n_S31
       +rho*Y[jp]*Y[jp30]*rate_eval.p_P30__S31
       +rho*Y[jn]*Y[js30]*rate_eval.n_S30__S31
       +5.00000000000000e-01*rho*Y[jo16]**2*rate_eval.O16_O16__n_S31
       +rho*Y[jc12]*Y[jne20]*rate_eval.C12_Ne20__n_S31
       +rho*Y[jhe4]*Y[jsi28]*rate_eval.He4_Si28__n_S31
       +rho*Y[jp]*Y[jp31]*rate_eval.p_P31__n_S31
       +rho*Y[jp]*Y[jcl34]*rate_eval.p_Cl34__He4_S31
       +rho*Y[jn]*Y[jar34]*rate_eval.n_Ar34__He4_S31
       )

    dYdt[js32] = (
       -Y[js32]*rate_eval.S32__n_S31
       -Y[js32]*rate_eval.S32__p_P31
       -Y[js32]*rate_eval.S32__He4_Si28
       -rho*Y[jn]*Y[js32]*rate_eval.n_S32__S33
       -rho*Y[jp]*Y[js32]*rate_eval.p_S32__Cl33
       -rho*Y[jhe4]*Y[js32]*rate_eval.He4_S32__Ar36
       -rho*Y[jn]*Y[js32]*rate_eval.n_S32__He4_Si29
       -rho*Y[jp]*Y[js32]*rate_eval.p_S32__He4_P29
       -rho*Y[jhe4]*Y[js32]*rate_eval.He4_S32__p_Cl35
       +Y[js33]*rate_eval.S33__n_S32
       +Y[jcl33]*rate_eval.Cl33__p_S32
       +Y[jar36]*rate_eval.Ar36__He4_S32
       +rho*Y[jhe4]*Y[jsi28]*rate_eval.He4_Si28__S32
       +rho*Y[jp]*Y[jp31]*rate_eval.p_P31__S32
       +rho*Y[jn]*Y[js31]*rate_eval.n_S31__S32
       +rho*Y[jhe4]*Y[jsi29]*rate_eval.He4_Si29__n_S32
       +rho*Y[jhe4]*Y[jp29]*rate_eval.He4_P29__p_S32
       +rho*Y[jp]*Y[jcl35]*rate_eval.p_Cl35__He4_S32
       )

    dYdt[js33] = (
       -Y[js33]*rate_eval.S33__n_S32
       -Y[js33]*rate_eval.S33__He4_Si29
       -rho*Y[jp]*Y[js33]*rate_eval.p_S33__Cl34
       -rho*Y[jhe4]*Y[js33]*rate_eval.He4_S33__Ar37
       -rho*Y[jn]*Y[js33]*rate_eval.n_S33__He4_Si30
       -rho*Y[jp]*Y[js33]*rate_eval.p_S33__n_Cl33
       -rho*Y[jp]*Y[js33]*rate_eval.p_S33__He4_P30
       -rho*Y[jhe4]*Y[js33]*rate_eval.He4_S33__n_Ar36
       +Y[jcl33]*rate_eval.Cl33__S33__weak__wc12
       +Y[jcl34]*rate_eval.Cl34__p_S33
       +Y[jar37]*rate_eval.Ar37__He4_S33
       +rho*Y[jhe4]*Y[jsi29]*rate_eval.He4_Si29__S33
       +rho*Y[jn]*Y[js32]*rate_eval.n_S32__S33
       +rho*Y[jhe4]*Y[jsi30]*rate_eval.He4_Si30__n_S33
       +rho*Y[jhe4]*Y[jp30]*rate_eval.He4_P30__p_S33
       +rho*Y[jn]*Y[jcl33]*rate_eval.n_Cl33__p_S33
       +rho*Y[jn]*Y[jar36]*rate_eval.n_Ar36__He4_S33
       )

    dYdt[jcl33] = (
       -Y[jcl33]*rate_eval.Cl33__S33__weak__wc12
       -Y[jcl33]*rate_eval.Cl33__p_S32
       -Y[jcl33]*rate_eval.Cl33__He4_P29
       -rho*Y[jn]*Y[jcl33]*rate_eval.n_Cl33__Cl34
       -rho*Y[jp]*Y[jcl33]*rate_eval.p_Cl33__Ar34
       -rho*Y[jn]*Y[jcl33]*rate_eval.n_Cl33__p_S33
       -rho*Y[jn]*Y[jcl33]*rate_eval.n_Cl33__He4_P30
       -rho*Y[jp]*Y[jcl33]*rate_eval.p_Cl33__He4_S30
       -rho*Y[jhe4]*Y[jcl33]*rate_eval.He4_Cl33__p_Ar36
       +Y[jcl34]*rate_eval.Cl34__n_Cl33
       +Y[jar34]*rate_eval.Ar34__p_Cl33
       +rho*Y[jhe4]*Y[jp29]*rate_eval.He4_P29__Cl33
       +rho*Y[jp]*Y[js32]*rate_eval.p_S32__Cl33
       +rho*Y[jhe4]*Y[jp30]*rate_eval.He4_P30__n_Cl33
       +rho*Y[jhe4]*Y[js30]*rate_eval.He4_S30__p_Cl33
       +rho*Y[jp]*Y[js33]*rate_eval.p_S33__n_Cl33
       +rho*Y[jp]*Y[jar36]*rate_eval.p_Ar36__He4_Cl33
       )

    dYdt[jcl34] = (
       -Y[jcl34]*rate_eval.Cl34__n_Cl33
       -Y[jcl34]*rate_eval.Cl34__p_S33
       -Y[jcl34]*rate_eval.Cl34__He4_P30
       -rho*Y[jn]*Y[jcl34]*rate_eval.n_Cl34__Cl35
       -rho*Y[jn]*Y[jcl34]*rate_eval.n_Cl34__He4_P31
       -rho*Y[jp]*Y[jcl34]*rate_eval.p_Cl34__n_Ar34
       -rho*Y[jp]*Y[jcl34]*rate_eval.p_Cl34__He4_S31
       -rho*Y[jhe4]*Y[jcl34]*rate_eval.He4_Cl34__p_Ar37
       +Y[jar34]*rate_eval.Ar34__Cl34__weak__wc12
       +Y[jcl35]*rate_eval.Cl35__n_Cl34
       +rho*Y[jhe4]*Y[jp30]*rate_eval.He4_P30__Cl34
       +rho*Y[jp]*Y[js33]*rate_eval.p_S33__Cl34
       +rho*Y[jn]*Y[jcl33]*rate_eval.n_Cl33__Cl34
       +rho*Y[jhe4]*Y[jp31]*rate_eval.He4_P31__n_Cl34
       +rho*Y[jhe4]*Y[js31]*rate_eval.He4_S31__p_Cl34
       +rho*Y[jn]*Y[jar34]*rate_eval.n_Ar34__p_Cl34
       +rho*Y[jp]*Y[jar37]*rate_eval.p_Ar37__He4_Cl34
       )

    dYdt[jcl35] = (
       -Y[jcl35]*rate_eval.Cl35__n_Cl34
       -Y[jcl35]*rate_eval.Cl35__He4_P31
       -rho*Y[jp]*Y[jcl35]*rate_eval.p_Cl35__Ar36
       -rho*Y[jhe4]*Y[jcl35]*rate_eval.He4_Cl35__K39
       -rho*Y[jp]*Y[jcl35]*rate_eval.p_Cl35__He4_S32
       -rho*Y[jhe4]*Y[jcl35]*rate_eval.He4_Cl35__p_Ar38
       +Y[jar36]*rate_eval.Ar36__p_Cl35
       +Y[jk39]*rate_eval.K39__He4_Cl35
       +rho*Y[jhe4]*Y[jp31]*rate_eval.He4_P31__Cl35
       +rho*Y[jn]*Y[jcl34]*rate_eval.n_Cl34__Cl35
       +rho*Y[jhe4]*Y[js32]*rate_eval.He4_S32__p_Cl35
       +rho*Y[jp]*Y[jar38]*rate_eval.p_Ar38__He4_Cl35
       )

    dYdt[jar34] = (
       -Y[jar34]*rate_eval.Ar34__Cl34__weak__wc12
       -Y[jar34]*rate_eval.Ar34__p_Cl33
       -Y[jar34]*rate_eval.Ar34__He4_S30
       -rho*Y[jn]*Y[jar34]*rate_eval.n_Ar34__p_Cl34
       -rho*Y[jn]*Y[jar34]*rate_eval.n_Ar34__He4_S31
       +rho*Y[jhe4]*Y[js30]*rate_eval.He4_S30__Ar34
       +rho*Y[jp]*Y[jcl33]*rate_eval.p_Cl33__Ar34
       +rho*Y[jhe4]*Y[js31]*rate_eval.He4_S31__n_Ar34
       +rho*Y[jp]*Y[jcl34]*rate_eval.p_Cl34__n_Ar34
       )

    dYdt[jar36] = (
       -Y[jar36]*rate_eval.Ar36__p_Cl35
       -Y[jar36]*rate_eval.Ar36__He4_S32
       -rho*Y[jn]*Y[jar36]*rate_eval.n_Ar36__Ar37
       -rho*Y[jhe4]*Y[jar36]*rate_eval.He4_Ar36__Ca40
       -rho*Y[jn]*Y[jar36]*rate_eval.n_Ar36__He4_S33
       -rho*Y[jp]*Y[jar36]*rate_eval.p_Ar36__He4_Cl33
       -rho*Y[jhe4]*Y[jar36]*rate_eval.He4_Ar36__p_K39
       +Y[jar37]*rate_eval.Ar37__n_Ar36
       +Y[jca40]*rate_eval.Ca40__He4_Ar36
       +rho*Y[jhe4]*Y[js32]*rate_eval.He4_S32__Ar36
       +rho*Y[jp]*Y[jcl35]*rate_eval.p_Cl35__Ar36
       +rho*Y[jhe4]*Y[js33]*rate_eval.He4_S33__n_Ar36
       +rho*Y[jhe4]*Y[jcl33]*rate_eval.He4_Cl33__p_Ar36
       +rho*Y[jp]*Y[jk39]*rate_eval.p_K39__He4_Ar36
       )

    dYdt[jar37] = (
       -Y[jar37]*rate_eval.Ar37__n_Ar36
       -Y[jar37]*rate_eval.Ar37__He4_S33
       -rho*Y[jn]*Y[jar37]*rate_eval.n_Ar37__Ar38
       -rho*Y[jp]*Y[jar37]*rate_eval.p_Ar37__He4_Cl34
       -rho*Y[jhe4]*Y[jar37]*rate_eval.He4_Ar37__n_Ca40
       +Y[jar38]*rate_eval.Ar38__n_Ar37
       +rho*Y[jhe4]*Y[js33]*rate_eval.He4_S33__Ar37
       +rho*Y[jn]*Y[jar36]*rate_eval.n_Ar36__Ar37
       +rho*Y[jhe4]*Y[jcl34]*rate_eval.He4_Cl34__p_Ar37
       +rho*Y[jn]*Y[jca40]*rate_eval.n_Ca40__He4_Ar37
       )

    dYdt[jar38] = (
       -Y[jar38]*rate_eval.Ar38__n_Ar37
       -rho*Y[jn]*Y[jar38]*rate_eval.n_Ar38__Ar39
       -rho*Y[jp]*Y[jar38]*rate_eval.p_Ar38__K39
       -rho*Y[jp]*Y[jar38]*rate_eval.p_Ar38__He4_Cl35
       +Y[jar39]*rate_eval.Ar39__n_Ar38
       +Y[jk39]*rate_eval.K39__p_Ar38
       +rho*Y[jn]*Y[jar37]*rate_eval.n_Ar37__Ar38
       +rho*Y[jhe4]*Y[jcl35]*rate_eval.He4_Cl35__p_Ar38
       )

    dYdt[jar39] = (
       -Y[jar39]*rate_eval.Ar39__K39__weak__wc12
       -Y[jar39]*rate_eval.Ar39__n_Ar38
       -rho*Y[jp]*Y[jar39]*rate_eval.p_Ar39__n_K39
       +rho*Y[jn]*Y[jar38]*rate_eval.n_Ar38__Ar39
       +rho*Y[jn]*Y[jk39]*rate_eval.n_K39__p_Ar39
       )

    dYdt[jk39] = (
       -Y[jk39]*rate_eval.K39__p_Ar38
       -Y[jk39]*rate_eval.K39__He4_Cl35
       -rho*Y[jp]*Y[jk39]*rate_eval.p_K39__Ca40
       -rho*Y[jhe4]*Y[jk39]*rate_eval.He4_K39__Sc43
       -rho*Y[jn]*Y[jk39]*rate_eval.n_K39__p_Ar39
       -rho*Y[jp]*Y[jk39]*rate_eval.p_K39__He4_Ar36
       +Y[jar39]*rate_eval.Ar39__K39__weak__wc12
       +Y[jca40]*rate_eval.Ca40__p_K39
       +Y[jsc43]*rate_eval.Sc43__He4_K39
       +rho*Y[jhe4]*Y[jcl35]*rate_eval.He4_Cl35__K39
       +rho*Y[jp]*Y[jar38]*rate_eval.p_Ar38__K39
       +rho*Y[jhe4]*Y[jar36]*rate_eval.He4_Ar36__p_K39
       +rho*Y[jp]*Y[jar39]*rate_eval.p_Ar39__n_K39
       )

    dYdt[jca40] = (
       -Y[jca40]*rate_eval.Ca40__p_K39
       -Y[jca40]*rate_eval.Ca40__He4_Ar36
       -rho*Y[jhe4]*Y[jca40]*rate_eval.He4_Ca40__Ti44
       -rho*Y[jn]*Y[jca40]*rate_eval.n_Ca40__He4_Ar37
       -rho*Y[jhe4]*Y[jca40]*rate_eval.He4_Ca40__p_Sc43
       +Y[jti44]*rate_eval.Ti44__He4_Ca40
       +rho*Y[jhe4]*Y[jar36]*rate_eval.He4_Ar36__Ca40
       +rho*Y[jp]*Y[jk39]*rate_eval.p_K39__Ca40
       +rho*Y[jhe4]*Y[jar37]*rate_eval.He4_Ar37__n_Ca40
       +rho*Y[jp]*Y[jsc43]*rate_eval.p_Sc43__He4_Ca40
       )

    dYdt[jsc43] = (
       -Y[jsc43]*rate_eval.Sc43__He4_K39
       -rho*Y[jp]*Y[jsc43]*rate_eval.p_Sc43__Ti44
       -rho*Y[jhe4]*Y[jsc43]*rate_eval.He4_Sc43__V47
       -rho*Y[jp]*Y[jsc43]*rate_eval.p_Sc43__He4_Ca40
       +Y[jti44]*rate_eval.Ti44__p_Sc43
       +Y[jv47]*rate_eval.V47__He4_Sc43
       +rho*Y[jhe4]*Y[jk39]*rate_eval.He4_K39__Sc43
       +rho*Y[jhe4]*Y[jca40]*rate_eval.He4_Ca40__p_Sc43
       )

    dYdt[jti44] = (
       -Y[jti44]*rate_eval.Ti44__p_Sc43
       -Y[jti44]*rate_eval.Ti44__He4_Ca40
       -rho*Y[jhe4]*Y[jti44]*rate_eval.He4_Ti44__Cr48
       -rho*Y[jhe4]*Y[jti44]*rate_eval.He4_Ti44__p_V47
       +Y[jcr48]*rate_eval.Cr48__He4_Ti44
       +rho*Y[jhe4]*Y[jca40]*rate_eval.He4_Ca40__Ti44
       +rho*Y[jp]*Y[jsc43]*rate_eval.p_Sc43__Ti44
       +rho*Y[jp]*Y[jv47]*rate_eval.p_V47__He4_Ti44
       )

    dYdt[jv47] = (
       -Y[jv47]*rate_eval.V47__He4_Sc43
       -rho*Y[jp]*Y[jv47]*rate_eval.p_V47__Cr48
       -rho*Y[jhe4]*Y[jv47]*rate_eval.He4_V47__Mn51
       -rho*Y[jp]*Y[jv47]*rate_eval.p_V47__He4_Ti44
       +Y[jcr48]*rate_eval.Cr48__p_V47
       +Y[jmn51]*rate_eval.Mn51__He4_V47
       +rho*Y[jhe4]*Y[jsc43]*rate_eval.He4_Sc43__V47
       +rho*Y[jhe4]*Y[jti44]*rate_eval.He4_Ti44__p_V47
       )

    dYdt[jcr48] = (
       -Y[jcr48]*rate_eval.Cr48__p_V47
       -Y[jcr48]*rate_eval.Cr48__He4_Ti44
       -rho*Y[jhe4]*Y[jcr48]*rate_eval.He4_Cr48__Fe52
       -rho*Y[jhe4]*Y[jcr48]*rate_eval.He4_Cr48__p_Mn51
       +Y[jfe52]*rate_eval.Fe52__He4_Cr48
       +rho*Y[jhe4]*Y[jti44]*rate_eval.He4_Ti44__Cr48
       +rho*Y[jp]*Y[jv47]*rate_eval.p_V47__Cr48
       +rho*Y[jp]*Y[jmn51]*rate_eval.p_Mn51__He4_Cr48
       )

    dYdt[jmn51] = (
       -Y[jmn51]*rate_eval.Mn51__He4_V47
       -rho*Y[jp]*Y[jmn51]*rate_eval.p_Mn51__Fe52
       -rho*Y[jhe4]*Y[jmn51]*rate_eval.He4_Mn51__Co55
       -rho*Y[jp]*Y[jmn51]*rate_eval.p_Mn51__He4_Cr48
       +Y[jfe52]*rate_eval.Fe52__p_Mn51
       +Y[jco55]*rate_eval.Co55__He4_Mn51
       +rho*Y[jhe4]*Y[jv47]*rate_eval.He4_V47__Mn51
       +rho*Y[jhe4]*Y[jcr48]*rate_eval.He4_Cr48__p_Mn51
       )

    dYdt[jfe52] = (
       -Y[jfe52]*rate_eval.Fe52__p_Mn51
       -Y[jfe52]*rate_eval.Fe52__He4_Cr48
       -rho*Y[jhe4]*Y[jfe52]*rate_eval.He4_Fe52__Ni56
       -rho*Y[jhe4]*Y[jfe52]*rate_eval.He4_Fe52__p_Co55
       +Y[jni56]*rate_eval.Ni56__He4_Fe52
       +rho*Y[jhe4]*Y[jcr48]*rate_eval.He4_Cr48__Fe52
       +rho*Y[jp]*Y[jmn51]*rate_eval.p_Mn51__Fe52
       +rho*Y[jp]*Y[jco55]*rate_eval.p_Co55__He4_Fe52
       )

    dYdt[jfe55] = (
       -rho*Y[jhe4]*Y[jfe55]*rate_eval.He4_Fe55__Ni59
       -rho*Y[jp]*Y[jfe55]*rate_eval.p_Fe55__n_Co55
       -rho*Y[jhe4]*Y[jfe55]*rate_eval.He4_Fe55__n_Ni58
       +Y[jco55]*rate_eval.Co55__Fe55__weak__wc12
       +Y[jni59]*rate_eval.Ni59__He4_Fe55
       +rho*Y[jn]*Y[jco55]*rate_eval.n_Co55__p_Fe55
       +rho*Y[jn]*Y[jni58]*rate_eval.n_Ni58__He4_Fe55
       )

    dYdt[jco55] = (
       -Y[jco55]*rate_eval.Co55__Fe55__weak__wc12
       -Y[jco55]*rate_eval.Co55__He4_Mn51
       -rho*Y[jp]*Y[jco55]*rate_eval.p_Co55__Ni56
       -rho*Y[jn]*Y[jco55]*rate_eval.n_Co55__p_Fe55
       -rho*Y[jp]*Y[jco55]*rate_eval.p_Co55__He4_Fe52
       -rho*Y[jhe4]*Y[jco55]*rate_eval.He4_Co55__p_Ni58
       +Y[jni56]*rate_eval.Ni56__p_Co55
       +rho*Y[jhe4]*Y[jmn51]*rate_eval.He4_Mn51__Co55
       +rho*Y[jhe4]*Y[jfe52]*rate_eval.He4_Fe52__p_Co55
       +rho*Y[jp]*Y[jfe55]*rate_eval.p_Fe55__n_Co55
       +rho*Y[jp]*Y[jni58]*rate_eval.p_Ni58__He4_Co55
       )

    dYdt[jni56] = (
       -Y[jni56]*rate_eval.Ni56__p_Co55
       -Y[jni56]*rate_eval.Ni56__He4_Fe52
       +rho*Y[jhe4]*Y[jfe52]*rate_eval.He4_Fe52__Ni56
       +rho*Y[jp]*Y[jco55]*rate_eval.p_Co55__Ni56
       )

    dYdt[jni58] = (
       -rho*Y[jn]*Y[jni58]*rate_eval.n_Ni58__Ni59
       -rho*Y[jn]*Y[jni58]*rate_eval.n_Ni58__He4_Fe55
       -rho*Y[jp]*Y[jni58]*rate_eval.p_Ni58__He4_Co55
       +Y[jni59]*rate_eval.Ni59__n_Ni58
       +rho*Y[jhe4]*Y[jfe55]*rate_eval.He4_Fe55__n_Ni58
       +rho*Y[jhe4]*Y[jco55]*rate_eval.He4_Co55__p_Ni58
       )

    dYdt[jni59] = (
       -Y[jni59]*rate_eval.Ni59__n_Ni58
       -Y[jni59]*rate_eval.Ni59__He4_Fe55
       +rho*Y[jhe4]*Y[jfe55]*rate_eval.He4_Fe55__Ni59
       +rho*Y[jn]*Y[jni58]*rate_eval.n_Ni58__Ni59
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
    N13__C13__weak__wc12(rate_eval, tf)
    O15__N15__weak__wc12(rate_eval, tf)
    Na22__Ne22__weak__wc12(rate_eval, tf)
    Mg23__Na23__weak__wc12(rate_eval, tf)
    Al25__Mg25__weak__wc12(rate_eval, tf)
    Al26__Mg26__weak__wc12(rate_eval, tf)
    Si26__Al26__weak__wc12(rate_eval, tf)
    P29__Si29__weak__wc12(rate_eval, tf)
    P30__Si30__weak__wc12(rate_eval, tf)
    S30__P30__weak__wc12(rate_eval, tf)
    S31__P31__weak__wc12(rate_eval, tf)
    Cl33__S33__weak__wc12(rate_eval, tf)
    Ar34__Cl34__weak__wc12(rate_eval, tf)
    Ar39__K39__weak__wc12(rate_eval, tf)
    Co55__Fe55__weak__wc12(rate_eval, tf)
    C12__p_B11(rate_eval, tf)
    C13__n_C12(rate_eval, tf)
    N13__p_C12(rate_eval, tf)
    N14__n_N13(rate_eval, tf)
    N14__p_C13(rate_eval, tf)
    N15__n_N14(rate_eval, tf)
    O15__p_N14(rate_eval, tf)
    O16__n_O15(rate_eval, tf)
    O16__p_N15(rate_eval, tf)
    O16__He4_C12(rate_eval, tf)
    O17__n_O16(rate_eval, tf)
    F18__p_O17(rate_eval, tf)
    F18__He4_N14(rate_eval, tf)
    Ne19__p_F18(rate_eval, tf)
    Ne19__He4_O15(rate_eval, tf)
    Ne20__n_Ne19(rate_eval, tf)
    Ne20__He4_O16(rate_eval, tf)
    Ne21__n_Ne20(rate_eval, tf)
    Ne21__He4_O17(rate_eval, tf)
    Ne22__n_Ne21(rate_eval, tf)
    Na22__p_Ne21(rate_eval, tf)
    Na22__He4_F18(rate_eval, tf)
    Na23__n_Na22(rate_eval, tf)
    Na23__p_Ne22(rate_eval, tf)
    Mg23__p_Na22(rate_eval, tf)
    Mg23__He4_Ne19(rate_eval, tf)
    Mg24__n_Mg23(rate_eval, tf)
    Mg24__p_Na23(rate_eval, tf)
    Mg24__He4_Ne20(rate_eval, tf)
    Mg25__n_Mg24(rate_eval, tf)
    Mg25__He4_Ne21(rate_eval, tf)
    Mg26__n_Mg25(rate_eval, tf)
    Mg26__He4_Ne22(rate_eval, tf)
    Al25__p_Mg24(rate_eval, tf)
    Al26__n_Al25(rate_eval, tf)
    Al26__p_Mg25(rate_eval, tf)
    Al26__He4_Na22(rate_eval, tf)
    Al27__n_Al26(rate_eval, tf)
    Al27__p_Mg26(rate_eval, tf)
    Al27__He4_Na23(rate_eval, tf)
    Si26__p_Al25(rate_eval, tf)
    Si28__p_Al27(rate_eval, tf)
    Si28__He4_Mg24(rate_eval, tf)
    Si29__n_Si28(rate_eval, tf)
    Si29__He4_Mg25(rate_eval, tf)
    Si30__n_Si29(rate_eval, tf)
    Si30__He4_Mg26(rate_eval, tf)
    P29__p_Si28(rate_eval, tf)
    P29__He4_Al25(rate_eval, tf)
    P30__n_P29(rate_eval, tf)
    P30__p_Si29(rate_eval, tf)
    P30__He4_Al26(rate_eval, tf)
    P31__n_P30(rate_eval, tf)
    P31__p_Si30(rate_eval, tf)
    P31__He4_Al27(rate_eval, tf)
    S30__p_P29(rate_eval, tf)
    S30__He4_Si26(rate_eval, tf)
    S31__n_S30(rate_eval, tf)
    S31__p_P30(rate_eval, tf)
    S32__n_S31(rate_eval, tf)
    S32__p_P31(rate_eval, tf)
    S32__He4_Si28(rate_eval, tf)
    S33__n_S32(rate_eval, tf)
    S33__He4_Si29(rate_eval, tf)
    Cl33__p_S32(rate_eval, tf)
    Cl33__He4_P29(rate_eval, tf)
    Cl34__n_Cl33(rate_eval, tf)
    Cl34__p_S33(rate_eval, tf)
    Cl34__He4_P30(rate_eval, tf)
    Cl35__n_Cl34(rate_eval, tf)
    Cl35__He4_P31(rate_eval, tf)
    Ar34__p_Cl33(rate_eval, tf)
    Ar34__He4_S30(rate_eval, tf)
    Ar36__p_Cl35(rate_eval, tf)
    Ar36__He4_S32(rate_eval, tf)
    Ar37__n_Ar36(rate_eval, tf)
    Ar37__He4_S33(rate_eval, tf)
    Ar38__n_Ar37(rate_eval, tf)
    Ar39__n_Ar38(rate_eval, tf)
    K39__p_Ar38(rate_eval, tf)
    K39__He4_Cl35(rate_eval, tf)
    Ca40__p_K39(rate_eval, tf)
    Ca40__He4_Ar36(rate_eval, tf)
    Sc43__He4_K39(rate_eval, tf)
    Ti44__p_Sc43(rate_eval, tf)
    Ti44__He4_Ca40(rate_eval, tf)
    V47__He4_Sc43(rate_eval, tf)
    Cr48__p_V47(rate_eval, tf)
    Cr48__He4_Ti44(rate_eval, tf)
    Mn51__He4_V47(rate_eval, tf)
    Fe52__p_Mn51(rate_eval, tf)
    Fe52__He4_Cr48(rate_eval, tf)
    Co55__He4_Mn51(rate_eval, tf)
    Ni56__p_Co55(rate_eval, tf)
    Ni56__He4_Fe52(rate_eval, tf)
    Ni59__n_Ni58(rate_eval, tf)
    Ni59__He4_Fe55(rate_eval, tf)
    C12__He4_He4_He4(rate_eval, tf)
    p_B11__C12(rate_eval, tf)
    n_C12__C13(rate_eval, tf)
    p_C12__N13(rate_eval, tf)
    He4_C12__O16(rate_eval, tf)
    p_C13__N14(rate_eval, tf)
    n_N13__N14(rate_eval, tf)
    n_N14__N15(rate_eval, tf)
    p_N14__O15(rate_eval, tf)
    He4_N14__F18(rate_eval, tf)
    p_N15__O16(rate_eval, tf)
    n_O15__O16(rate_eval, tf)
    He4_O15__Ne19(rate_eval, tf)
    n_O16__O17(rate_eval, tf)
    He4_O16__Ne20(rate_eval, tf)
    p_O17__F18(rate_eval, tf)
    He4_O17__Ne21(rate_eval, tf)
    p_F18__Ne19(rate_eval, tf)
    He4_F18__Na22(rate_eval, tf)
    n_Ne19__Ne20(rate_eval, tf)
    He4_Ne19__Mg23(rate_eval, tf)
    n_Ne20__Ne21(rate_eval, tf)
    He4_Ne20__Mg24(rate_eval, tf)
    n_Ne21__Ne22(rate_eval, tf)
    p_Ne21__Na22(rate_eval, tf)
    He4_Ne21__Mg25(rate_eval, tf)
    p_Ne22__Na23(rate_eval, tf)
    He4_Ne22__Mg26(rate_eval, tf)
    n_Na22__Na23(rate_eval, tf)
    p_Na22__Mg23(rate_eval, tf)
    He4_Na22__Al26(rate_eval, tf)
    p_Na23__Mg24(rate_eval, tf)
    He4_Na23__Al27(rate_eval, tf)
    n_Mg23__Mg24(rate_eval, tf)
    n_Mg24__Mg25(rate_eval, tf)
    p_Mg24__Al25(rate_eval, tf)
    He4_Mg24__Si28(rate_eval, tf)
    n_Mg25__Mg26(rate_eval, tf)
    p_Mg25__Al26(rate_eval, tf)
    He4_Mg25__Si29(rate_eval, tf)
    p_Mg26__Al27(rate_eval, tf)
    He4_Mg26__Si30(rate_eval, tf)
    n_Al25__Al26(rate_eval, tf)
    p_Al25__Si26(rate_eval, tf)
    He4_Al25__P29(rate_eval, tf)
    n_Al26__Al27(rate_eval, tf)
    He4_Al26__P30(rate_eval, tf)
    p_Al27__Si28(rate_eval, tf)
    He4_Al27__P31(rate_eval, tf)
    He4_Si26__S30(rate_eval, tf)
    n_Si28__Si29(rate_eval, tf)
    p_Si28__P29(rate_eval, tf)
    He4_Si28__S32(rate_eval, tf)
    n_Si29__Si30(rate_eval, tf)
    p_Si29__P30(rate_eval, tf)
    He4_Si29__S33(rate_eval, tf)
    p_Si30__P31(rate_eval, tf)
    n_P29__P30(rate_eval, tf)
    p_P29__S30(rate_eval, tf)
    He4_P29__Cl33(rate_eval, tf)
    n_P30__P31(rate_eval, tf)
    p_P30__S31(rate_eval, tf)
    He4_P30__Cl34(rate_eval, tf)
    p_P31__S32(rate_eval, tf)
    He4_P31__Cl35(rate_eval, tf)
    n_S30__S31(rate_eval, tf)
    He4_S30__Ar34(rate_eval, tf)
    n_S31__S32(rate_eval, tf)
    n_S32__S33(rate_eval, tf)
    p_S32__Cl33(rate_eval, tf)
    He4_S32__Ar36(rate_eval, tf)
    p_S33__Cl34(rate_eval, tf)
    He4_S33__Ar37(rate_eval, tf)
    n_Cl33__Cl34(rate_eval, tf)
    p_Cl33__Ar34(rate_eval, tf)
    n_Cl34__Cl35(rate_eval, tf)
    p_Cl35__Ar36(rate_eval, tf)
    He4_Cl35__K39(rate_eval, tf)
    n_Ar36__Ar37(rate_eval, tf)
    He4_Ar36__Ca40(rate_eval, tf)
    n_Ar37__Ar38(rate_eval, tf)
    n_Ar38__Ar39(rate_eval, tf)
    p_Ar38__K39(rate_eval, tf)
    p_K39__Ca40(rate_eval, tf)
    He4_K39__Sc43(rate_eval, tf)
    He4_Ca40__Ti44(rate_eval, tf)
    p_Sc43__Ti44(rate_eval, tf)
    He4_Sc43__V47(rate_eval, tf)
    He4_Ti44__Cr48(rate_eval, tf)
    p_V47__Cr48(rate_eval, tf)
    He4_V47__Mn51(rate_eval, tf)
    He4_Cr48__Fe52(rate_eval, tf)
    p_Mn51__Fe52(rate_eval, tf)
    He4_Mn51__Co55(rate_eval, tf)
    He4_Fe52__Ni56(rate_eval, tf)
    He4_Fe55__Ni59(rate_eval, tf)
    p_Co55__Ni56(rate_eval, tf)
    n_Ni58__Ni59(rate_eval, tf)
    He4_B11__n_N14(rate_eval, tf)
    He4_C12__n_O15(rate_eval, tf)
    He4_C12__p_N15(rate_eval, tf)
    C12_C12__n_Mg23(rate_eval, tf)
    C12_C12__p_Na23(rate_eval, tf)
    C12_C12__He4_Ne20(rate_eval, tf)
    p_C13__n_N13(rate_eval, tf)
    He4_C13__n_O16(rate_eval, tf)
    n_N13__p_C13(rate_eval, tf)
    He4_N13__p_O16(rate_eval, tf)
    n_N14__He4_B11(rate_eval, tf)
    He4_N14__p_O17(rate_eval, tf)
    p_N15__n_O15(rate_eval, tf)
    p_N15__He4_C12(rate_eval, tf)
    He4_N15__n_F18(rate_eval, tf)
    n_O15__p_N15(rate_eval, tf)
    n_O15__He4_C12(rate_eval, tf)
    He4_O15__p_F18(rate_eval, tf)
    n_O16__He4_C13(rate_eval, tf)
    p_O16__He4_N13(rate_eval, tf)
    He4_O16__n_Ne19(rate_eval, tf)
    C12_O16__p_Al27(rate_eval, tf)
    C12_O16__He4_Mg24(rate_eval, tf)
    O16_O16__n_S31(rate_eval, tf)
    O16_O16__p_P31(rate_eval, tf)
    O16_O16__He4_Si28(rate_eval, tf)
    p_O17__He4_N14(rate_eval, tf)
    He4_O17__n_Ne20(rate_eval, tf)
    n_F18__He4_N15(rate_eval, tf)
    p_F18__He4_O15(rate_eval, tf)
    He4_F18__p_Ne21(rate_eval, tf)
    n_Ne19__He4_O16(rate_eval, tf)
    He4_Ne19__p_Na22(rate_eval, tf)
    n_Ne20__He4_O17(rate_eval, tf)
    He4_Ne20__n_Mg23(rate_eval, tf)
    He4_Ne20__p_Na23(rate_eval, tf)
    He4_Ne20__C12_C12(rate_eval, tf)
    C12_Ne20__n_S31(rate_eval, tf)
    C12_Ne20__p_P31(rate_eval, tf)
    C12_Ne20__He4_Si28(rate_eval, tf)
    p_Ne21__He4_F18(rate_eval, tf)
    He4_Ne21__n_Mg24(rate_eval, tf)
    p_Ne22__n_Na22(rate_eval, tf)
    He4_Ne22__n_Mg25(rate_eval, tf)
    n_Na22__p_Ne22(rate_eval, tf)
    p_Na22__He4_Ne19(rate_eval, tf)
    He4_Na22__n_Al25(rate_eval, tf)
    He4_Na22__p_Mg25(rate_eval, tf)
    p_Na23__n_Mg23(rate_eval, tf)
    p_Na23__He4_Ne20(rate_eval, tf)
    p_Na23__C12_C12(rate_eval, tf)
    He4_Na23__n_Al26(rate_eval, tf)
    He4_Na23__p_Mg26(rate_eval, tf)
    n_Mg23__p_Na23(rate_eval, tf)
    n_Mg23__He4_Ne20(rate_eval, tf)
    n_Mg23__C12_C12(rate_eval, tf)
    He4_Mg23__n_Si26(rate_eval, tf)
    He4_Mg23__p_Al26(rate_eval, tf)
    n_Mg24__He4_Ne21(rate_eval, tf)
    He4_Mg24__p_Al27(rate_eval, tf)
    He4_Mg24__C12_O16(rate_eval, tf)
    n_Mg25__He4_Ne22(rate_eval, tf)
    p_Mg25__n_Al25(rate_eval, tf)
    p_Mg25__He4_Na22(rate_eval, tf)
    He4_Mg25__n_Si28(rate_eval, tf)
    p_Mg26__n_Al26(rate_eval, tf)
    p_Mg26__He4_Na23(rate_eval, tf)
    He4_Mg26__n_Si29(rate_eval, tf)
    n_Al25__p_Mg25(rate_eval, tf)
    n_Al25__He4_Na22(rate_eval, tf)
    He4_Al25__p_Si28(rate_eval, tf)
    n_Al26__p_Mg26(rate_eval, tf)
    n_Al26__He4_Na23(rate_eval, tf)
    p_Al26__n_Si26(rate_eval, tf)
    p_Al26__He4_Mg23(rate_eval, tf)
    He4_Al26__n_P29(rate_eval, tf)
    He4_Al26__p_Si29(rate_eval, tf)
    p_Al27__He4_Mg24(rate_eval, tf)
    p_Al27__C12_O16(rate_eval, tf)
    He4_Al27__n_P30(rate_eval, tf)
    He4_Al27__p_Si30(rate_eval, tf)
    n_Si26__p_Al26(rate_eval, tf)
    n_Si26__He4_Mg23(rate_eval, tf)
    He4_Si26__p_P29(rate_eval, tf)
    n_Si28__He4_Mg25(rate_eval, tf)
    p_Si28__He4_Al25(rate_eval, tf)
    He4_Si28__n_S31(rate_eval, tf)
    He4_Si28__p_P31(rate_eval, tf)
    He4_Si28__C12_Ne20(rate_eval, tf)
    He4_Si28__O16_O16(rate_eval, tf)
    n_Si29__He4_Mg26(rate_eval, tf)
    p_Si29__n_P29(rate_eval, tf)
    p_Si29__He4_Al26(rate_eval, tf)
    He4_Si29__n_S32(rate_eval, tf)
    p_Si30__n_P30(rate_eval, tf)
    p_Si30__He4_Al27(rate_eval, tf)
    He4_Si30__n_S33(rate_eval, tf)
    n_P29__p_Si29(rate_eval, tf)
    n_P29__He4_Al26(rate_eval, tf)
    p_P29__He4_Si26(rate_eval, tf)
    He4_P29__p_S32(rate_eval, tf)
    n_P30__p_Si30(rate_eval, tf)
    n_P30__He4_Al27(rate_eval, tf)
    p_P30__n_S30(rate_eval, tf)
    He4_P30__n_Cl33(rate_eval, tf)
    He4_P30__p_S33(rate_eval, tf)
    p_P31__n_S31(rate_eval, tf)
    p_P31__He4_Si28(rate_eval, tf)
    p_P31__C12_Ne20(rate_eval, tf)
    p_P31__O16_O16(rate_eval, tf)
    He4_P31__n_Cl34(rate_eval, tf)
    n_S30__p_P30(rate_eval, tf)
    He4_S30__p_Cl33(rate_eval, tf)
    n_S31__p_P31(rate_eval, tf)
    n_S31__He4_Si28(rate_eval, tf)
    n_S31__C12_Ne20(rate_eval, tf)
    n_S31__O16_O16(rate_eval, tf)
    He4_S31__n_Ar34(rate_eval, tf)
    He4_S31__p_Cl34(rate_eval, tf)
    n_S32__He4_Si29(rate_eval, tf)
    p_S32__He4_P29(rate_eval, tf)
    He4_S32__p_Cl35(rate_eval, tf)
    n_S33__He4_Si30(rate_eval, tf)
    p_S33__n_Cl33(rate_eval, tf)
    p_S33__He4_P30(rate_eval, tf)
    He4_S33__n_Ar36(rate_eval, tf)
    n_Cl33__p_S33(rate_eval, tf)
    n_Cl33__He4_P30(rate_eval, tf)
    p_Cl33__He4_S30(rate_eval, tf)
    He4_Cl33__p_Ar36(rate_eval, tf)
    n_Cl34__He4_P31(rate_eval, tf)
    p_Cl34__n_Ar34(rate_eval, tf)
    p_Cl34__He4_S31(rate_eval, tf)
    He4_Cl34__p_Ar37(rate_eval, tf)
    p_Cl35__He4_S32(rate_eval, tf)
    He4_Cl35__p_Ar38(rate_eval, tf)
    n_Ar34__p_Cl34(rate_eval, tf)
    n_Ar34__He4_S31(rate_eval, tf)
    n_Ar36__He4_S33(rate_eval, tf)
    p_Ar36__He4_Cl33(rate_eval, tf)
    He4_Ar36__p_K39(rate_eval, tf)
    p_Ar37__He4_Cl34(rate_eval, tf)
    He4_Ar37__n_Ca40(rate_eval, tf)
    p_Ar38__He4_Cl35(rate_eval, tf)
    p_Ar39__n_K39(rate_eval, tf)
    n_K39__p_Ar39(rate_eval, tf)
    p_K39__He4_Ar36(rate_eval, tf)
    n_Ca40__He4_Ar37(rate_eval, tf)
    He4_Ca40__p_Sc43(rate_eval, tf)
    p_Sc43__He4_Ca40(rate_eval, tf)
    He4_Ti44__p_V47(rate_eval, tf)
    p_V47__He4_Ti44(rate_eval, tf)
    He4_Cr48__p_Mn51(rate_eval, tf)
    p_Mn51__He4_Cr48(rate_eval, tf)
    He4_Fe52__p_Co55(rate_eval, tf)
    p_Fe55__n_Co55(rate_eval, tf)
    He4_Fe55__n_Ni58(rate_eval, tf)
    n_Co55__p_Fe55(rate_eval, tf)
    p_Co55__He4_Fe52(rate_eval, tf)
    He4_Co55__p_Ni58(rate_eval, tf)
    n_Ni58__He4_Fe55(rate_eval, tf)
    p_Ni58__He4_Co55(rate_eval, tf)
    p_B11__He4_He4_He4(rate_eval, tf)
    He4_He4_He4__C12(rate_eval, tf)
    He4_He4_He4__p_B11(rate_eval, tf)

    if screen_func is not None:
        plasma_state = PlasmaState(T, rho, Y, Z)

        scn_fac = ScreenFactors(1, 1, 5, 11)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_B11__C12 *= scor
        rate_eval.p_B11__He4_He4_He4 *= scor

        scn_fac = ScreenFactors(1, 1, 6, 12)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_C12__N13 *= scor

        scn_fac = ScreenFactors(2, 4, 6, 12)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_C12__O16 *= scor
        rate_eval.He4_C12__n_O15 *= scor
        rate_eval.He4_C12__p_N15 *= scor

        scn_fac = ScreenFactors(1, 1, 6, 13)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_C13__N14 *= scor
        rate_eval.p_C13__n_N13 *= scor

        scn_fac = ScreenFactors(1, 1, 7, 14)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_N14__O15 *= scor

        scn_fac = ScreenFactors(2, 4, 7, 14)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_N14__F18 *= scor
        rate_eval.He4_N14__p_O17 *= scor

        scn_fac = ScreenFactors(1, 1, 7, 15)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_N15__O16 *= scor
        rate_eval.p_N15__n_O15 *= scor
        rate_eval.p_N15__He4_C12 *= scor

        scn_fac = ScreenFactors(2, 4, 8, 15)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_O15__Ne19 *= scor
        rate_eval.He4_O15__p_F18 *= scor

        scn_fac = ScreenFactors(2, 4, 8, 16)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_O16__Ne20 *= scor
        rate_eval.He4_O16__n_Ne19 *= scor

        scn_fac = ScreenFactors(1, 1, 8, 17)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_O17__F18 *= scor
        rate_eval.p_O17__He4_N14 *= scor

        scn_fac = ScreenFactors(2, 4, 8, 17)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_O17__Ne21 *= scor
        rate_eval.He4_O17__n_Ne20 *= scor

        scn_fac = ScreenFactors(1, 1, 9, 18)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_F18__Ne19 *= scor
        rate_eval.p_F18__He4_O15 *= scor

        scn_fac = ScreenFactors(2, 4, 9, 18)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_F18__Na22 *= scor
        rate_eval.He4_F18__p_Ne21 *= scor

        scn_fac = ScreenFactors(2, 4, 10, 19)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Ne19__Mg23 *= scor
        rate_eval.He4_Ne19__p_Na22 *= scor

        scn_fac = ScreenFactors(2, 4, 10, 20)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Ne20__Mg24 *= scor
        rate_eval.He4_Ne20__n_Mg23 *= scor
        rate_eval.He4_Ne20__p_Na23 *= scor
        rate_eval.He4_Ne20__C12_C12 *= scor

        scn_fac = ScreenFactors(1, 1, 10, 21)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_Ne21__Na22 *= scor
        rate_eval.p_Ne21__He4_F18 *= scor

        scn_fac = ScreenFactors(2, 4, 10, 21)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Ne21__Mg25 *= scor
        rate_eval.He4_Ne21__n_Mg24 *= scor

        scn_fac = ScreenFactors(1, 1, 10, 22)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_Ne22__Na23 *= scor
        rate_eval.p_Ne22__n_Na22 *= scor

        scn_fac = ScreenFactors(2, 4, 10, 22)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Ne22__Mg26 *= scor
        rate_eval.He4_Ne22__n_Mg25 *= scor

        scn_fac = ScreenFactors(1, 1, 11, 22)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_Na22__Mg23 *= scor
        rate_eval.p_Na22__He4_Ne19 *= scor

        scn_fac = ScreenFactors(2, 4, 11, 22)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Na22__Al26 *= scor
        rate_eval.He4_Na22__n_Al25 *= scor
        rate_eval.He4_Na22__p_Mg25 *= scor

        scn_fac = ScreenFactors(1, 1, 11, 23)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_Na23__Mg24 *= scor
        rate_eval.p_Na23__n_Mg23 *= scor
        rate_eval.p_Na23__He4_Ne20 *= scor
        rate_eval.p_Na23__C12_C12 *= scor

        scn_fac = ScreenFactors(2, 4, 11, 23)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Na23__Al27 *= scor
        rate_eval.He4_Na23__n_Al26 *= scor
        rate_eval.He4_Na23__p_Mg26 *= scor

        scn_fac = ScreenFactors(1, 1, 12, 24)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_Mg24__Al25 *= scor

        scn_fac = ScreenFactors(2, 4, 12, 24)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Mg24__Si28 *= scor
        rate_eval.He4_Mg24__p_Al27 *= scor
        rate_eval.He4_Mg24__C12_O16 *= scor

        scn_fac = ScreenFactors(1, 1, 12, 25)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_Mg25__Al26 *= scor
        rate_eval.p_Mg25__n_Al25 *= scor
        rate_eval.p_Mg25__He4_Na22 *= scor

        scn_fac = ScreenFactors(2, 4, 12, 25)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Mg25__Si29 *= scor
        rate_eval.He4_Mg25__n_Si28 *= scor

        scn_fac = ScreenFactors(1, 1, 12, 26)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_Mg26__Al27 *= scor
        rate_eval.p_Mg26__n_Al26 *= scor
        rate_eval.p_Mg26__He4_Na23 *= scor

        scn_fac = ScreenFactors(2, 4, 12, 26)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Mg26__Si30 *= scor
        rate_eval.He4_Mg26__n_Si29 *= scor

        scn_fac = ScreenFactors(1, 1, 13, 25)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_Al25__Si26 *= scor

        scn_fac = ScreenFactors(2, 4, 13, 25)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Al25__P29 *= scor
        rate_eval.He4_Al25__p_Si28 *= scor

        scn_fac = ScreenFactors(2, 4, 13, 26)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Al26__P30 *= scor
        rate_eval.He4_Al26__n_P29 *= scor
        rate_eval.He4_Al26__p_Si29 *= scor

        scn_fac = ScreenFactors(1, 1, 13, 27)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_Al27__Si28 *= scor
        rate_eval.p_Al27__He4_Mg24 *= scor
        rate_eval.p_Al27__C12_O16 *= scor

        scn_fac = ScreenFactors(2, 4, 13, 27)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Al27__P31 *= scor
        rate_eval.He4_Al27__n_P30 *= scor
        rate_eval.He4_Al27__p_Si30 *= scor

        scn_fac = ScreenFactors(2, 4, 14, 26)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Si26__S30 *= scor
        rate_eval.He4_Si26__p_P29 *= scor

        scn_fac = ScreenFactors(1, 1, 14, 28)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_Si28__P29 *= scor
        rate_eval.p_Si28__He4_Al25 *= scor

        scn_fac = ScreenFactors(2, 4, 14, 28)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Si28__S32 *= scor
        rate_eval.He4_Si28__n_S31 *= scor
        rate_eval.He4_Si28__p_P31 *= scor
        rate_eval.He4_Si28__C12_Ne20 *= scor
        rate_eval.He4_Si28__O16_O16 *= scor

        scn_fac = ScreenFactors(1, 1, 14, 29)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_Si29__P30 *= scor
        rate_eval.p_Si29__n_P29 *= scor
        rate_eval.p_Si29__He4_Al26 *= scor

        scn_fac = ScreenFactors(2, 4, 14, 29)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Si29__S33 *= scor
        rate_eval.He4_Si29__n_S32 *= scor

        scn_fac = ScreenFactors(1, 1, 14, 30)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_Si30__P31 *= scor
        rate_eval.p_Si30__n_P30 *= scor
        rate_eval.p_Si30__He4_Al27 *= scor

        scn_fac = ScreenFactors(1, 1, 15, 29)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_P29__S30 *= scor
        rate_eval.p_P29__He4_Si26 *= scor

        scn_fac = ScreenFactors(2, 4, 15, 29)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_P29__Cl33 *= scor
        rate_eval.He4_P29__p_S32 *= scor

        scn_fac = ScreenFactors(1, 1, 15, 30)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_P30__S31 *= scor
        rate_eval.p_P30__n_S30 *= scor

        scn_fac = ScreenFactors(2, 4, 15, 30)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_P30__Cl34 *= scor
        rate_eval.He4_P30__n_Cl33 *= scor
        rate_eval.He4_P30__p_S33 *= scor

        scn_fac = ScreenFactors(1, 1, 15, 31)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_P31__S32 *= scor
        rate_eval.p_P31__n_S31 *= scor
        rate_eval.p_P31__He4_Si28 *= scor
        rate_eval.p_P31__C12_Ne20 *= scor
        rate_eval.p_P31__O16_O16 *= scor

        scn_fac = ScreenFactors(2, 4, 15, 31)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_P31__Cl35 *= scor
        rate_eval.He4_P31__n_Cl34 *= scor

        scn_fac = ScreenFactors(2, 4, 16, 30)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_S30__Ar34 *= scor
        rate_eval.He4_S30__p_Cl33 *= scor

        scn_fac = ScreenFactors(1, 1, 16, 32)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_S32__Cl33 *= scor
        rate_eval.p_S32__He4_P29 *= scor

        scn_fac = ScreenFactors(2, 4, 16, 32)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_S32__Ar36 *= scor
        rate_eval.He4_S32__p_Cl35 *= scor

        scn_fac = ScreenFactors(1, 1, 16, 33)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_S33__Cl34 *= scor
        rate_eval.p_S33__n_Cl33 *= scor
        rate_eval.p_S33__He4_P30 *= scor

        scn_fac = ScreenFactors(2, 4, 16, 33)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_S33__Ar37 *= scor
        rate_eval.He4_S33__n_Ar36 *= scor

        scn_fac = ScreenFactors(1, 1, 17, 33)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_Cl33__Ar34 *= scor
        rate_eval.p_Cl33__He4_S30 *= scor

        scn_fac = ScreenFactors(1, 1, 17, 35)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_Cl35__Ar36 *= scor
        rate_eval.p_Cl35__He4_S32 *= scor

        scn_fac = ScreenFactors(2, 4, 17, 35)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Cl35__K39 *= scor
        rate_eval.He4_Cl35__p_Ar38 *= scor

        scn_fac = ScreenFactors(2, 4, 18, 36)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Ar36__Ca40 *= scor
        rate_eval.He4_Ar36__p_K39 *= scor

        scn_fac = ScreenFactors(1, 1, 18, 38)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_Ar38__K39 *= scor
        rate_eval.p_Ar38__He4_Cl35 *= scor

        scn_fac = ScreenFactors(1, 1, 19, 39)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_K39__Ca40 *= scor
        rate_eval.p_K39__He4_Ar36 *= scor

        scn_fac = ScreenFactors(2, 4, 19, 39)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_K39__Sc43 *= scor

        scn_fac = ScreenFactors(2, 4, 20, 40)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Ca40__Ti44 *= scor
        rate_eval.He4_Ca40__p_Sc43 *= scor

        scn_fac = ScreenFactors(1, 1, 21, 43)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_Sc43__Ti44 *= scor
        rate_eval.p_Sc43__He4_Ca40 *= scor

        scn_fac = ScreenFactors(2, 4, 21, 43)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Sc43__V47 *= scor

        scn_fac = ScreenFactors(2, 4, 22, 44)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Ti44__Cr48 *= scor
        rate_eval.He4_Ti44__p_V47 *= scor

        scn_fac = ScreenFactors(1, 1, 23, 47)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_V47__Cr48 *= scor
        rate_eval.p_V47__He4_Ti44 *= scor

        scn_fac = ScreenFactors(2, 4, 23, 47)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_V47__Mn51 *= scor

        scn_fac = ScreenFactors(2, 4, 24, 48)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Cr48__Fe52 *= scor
        rate_eval.He4_Cr48__p_Mn51 *= scor

        scn_fac = ScreenFactors(1, 1, 25, 51)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_Mn51__Fe52 *= scor
        rate_eval.p_Mn51__He4_Cr48 *= scor

        scn_fac = ScreenFactors(2, 4, 25, 51)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Mn51__Co55 *= scor

        scn_fac = ScreenFactors(2, 4, 26, 52)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Fe52__Ni56 *= scor
        rate_eval.He4_Fe52__p_Co55 *= scor

        scn_fac = ScreenFactors(2, 4, 26, 55)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Fe55__Ni59 *= scor
        rate_eval.He4_Fe55__n_Ni58 *= scor

        scn_fac = ScreenFactors(1, 1, 27, 55)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_Co55__Ni56 *= scor
        rate_eval.p_Co55__He4_Fe52 *= scor

        scn_fac = ScreenFactors(2, 4, 5, 11)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_B11__n_N14 *= scor

        scn_fac = ScreenFactors(6, 12, 6, 12)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.C12_C12__n_Mg23 *= scor
        rate_eval.C12_C12__p_Na23 *= scor
        rate_eval.C12_C12__He4_Ne20 *= scor

        scn_fac = ScreenFactors(2, 4, 6, 13)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_C13__n_O16 *= scor

        scn_fac = ScreenFactors(2, 4, 7, 13)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_N13__p_O16 *= scor

        scn_fac = ScreenFactors(2, 4, 7, 15)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_N15__n_F18 *= scor

        scn_fac = ScreenFactors(1, 1, 8, 16)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_O16__He4_N13 *= scor

        scn_fac = ScreenFactors(6, 12, 8, 16)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.C12_O16__p_Al27 *= scor
        rate_eval.C12_O16__He4_Mg24 *= scor

        scn_fac = ScreenFactors(8, 16, 8, 16)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.O16_O16__n_S31 *= scor
        rate_eval.O16_O16__p_P31 *= scor
        rate_eval.O16_O16__He4_Si28 *= scor

        scn_fac = ScreenFactors(6, 12, 10, 20)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.C12_Ne20__n_S31 *= scor
        rate_eval.C12_Ne20__p_P31 *= scor
        rate_eval.C12_Ne20__He4_Si28 *= scor

        scn_fac = ScreenFactors(2, 4, 12, 23)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Mg23__n_Si26 *= scor
        rate_eval.He4_Mg23__p_Al26 *= scor

        scn_fac = ScreenFactors(1, 1, 13, 26)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_Al26__n_Si26 *= scor
        rate_eval.p_Al26__He4_Mg23 *= scor

        scn_fac = ScreenFactors(2, 4, 14, 30)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Si30__n_S33 *= scor

        scn_fac = ScreenFactors(2, 4, 16, 31)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_S31__n_Ar34 *= scor
        rate_eval.He4_S31__p_Cl34 *= scor

        scn_fac = ScreenFactors(2, 4, 17, 33)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Cl33__p_Ar36 *= scor

        scn_fac = ScreenFactors(1, 1, 17, 34)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_Cl34__n_Ar34 *= scor
        rate_eval.p_Cl34__He4_S31 *= scor

        scn_fac = ScreenFactors(2, 4, 17, 34)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Cl34__p_Ar37 *= scor

        scn_fac = ScreenFactors(1, 1, 18, 36)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_Ar36__He4_Cl33 *= scor

        scn_fac = ScreenFactors(1, 1, 18, 37)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_Ar37__He4_Cl34 *= scor

        scn_fac = ScreenFactors(2, 4, 18, 37)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Ar37__n_Ca40 *= scor

        scn_fac = ScreenFactors(1, 1, 18, 39)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_Ar39__n_K39 *= scor

        scn_fac = ScreenFactors(1, 1, 26, 55)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_Fe55__n_Co55 *= scor

        scn_fac = ScreenFactors(2, 4, 27, 55)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.He4_Co55__p_Ni58 *= scor

        scn_fac = ScreenFactors(1, 1, 28, 58)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_Ni58__He4_Co55 *= scor

        scn_fac = ScreenFactors(2, 4, 2, 4)
        scor = screen_func(plasma_state, scn_fac)
        scn_fac2 = ScreenFactors(2, 4, 4, 8)
        scor2 = screen_func(plasma_state, scn_fac2)
        rate_eval.He4_He4_He4__C12 *= scor * scor2
        rate_eval.He4_He4_He4__p_B11 *= scor * scor2

    jac = np.zeros((nnuc, nnuc), dtype=np.float64)

    jac[jn, jn] = (
       -rate_eval.n__p__weak__wc12
       -rho*Y[jc12]*rate_eval.n_C12__C13
       -rho*Y[jn13]*rate_eval.n_N13__N14
       -rho*Y[jn14]*rate_eval.n_N14__N15
       -rho*Y[jo15]*rate_eval.n_O15__O16
       -rho*Y[jo16]*rate_eval.n_O16__O17
       -rho*Y[jne19]*rate_eval.n_Ne19__Ne20
       -rho*Y[jne20]*rate_eval.n_Ne20__Ne21
       -rho*Y[jne21]*rate_eval.n_Ne21__Ne22
       -rho*Y[jna22]*rate_eval.n_Na22__Na23
       -rho*Y[jmg23]*rate_eval.n_Mg23__Mg24
       -rho*Y[jmg24]*rate_eval.n_Mg24__Mg25
       -rho*Y[jmg25]*rate_eval.n_Mg25__Mg26
       -rho*Y[jal25]*rate_eval.n_Al25__Al26
       -rho*Y[jal26]*rate_eval.n_Al26__Al27
       -rho*Y[jsi28]*rate_eval.n_Si28__Si29
       -rho*Y[jsi29]*rate_eval.n_Si29__Si30
       -rho*Y[jp29]*rate_eval.n_P29__P30
       -rho*Y[jp30]*rate_eval.n_P30__P31
       -rho*Y[js30]*rate_eval.n_S30__S31
       -rho*Y[js31]*rate_eval.n_S31__S32
       -rho*Y[js32]*rate_eval.n_S32__S33
       -rho*Y[jcl33]*rate_eval.n_Cl33__Cl34
       -rho*Y[jcl34]*rate_eval.n_Cl34__Cl35
       -rho*Y[jar36]*rate_eval.n_Ar36__Ar37
       -rho*Y[jar37]*rate_eval.n_Ar37__Ar38
       -rho*Y[jar38]*rate_eval.n_Ar38__Ar39
       -rho*Y[jni58]*rate_eval.n_Ni58__Ni59
       -rho*Y[jn13]*rate_eval.n_N13__p_C13
       -rho*Y[jn14]*rate_eval.n_N14__He4_B11
       -rho*Y[jo15]*rate_eval.n_O15__p_N15
       -rho*Y[jo15]*rate_eval.n_O15__He4_C12
       -rho*Y[jo16]*rate_eval.n_O16__He4_C13
       -rho*Y[jf18]*rate_eval.n_F18__He4_N15
       -rho*Y[jne19]*rate_eval.n_Ne19__He4_O16
       -rho*Y[jne20]*rate_eval.n_Ne20__He4_O17
       -rho*Y[jna22]*rate_eval.n_Na22__p_Ne22
       -rho*Y[jmg23]*rate_eval.n_Mg23__p_Na23
       -rho*Y[jmg23]*rate_eval.n_Mg23__He4_Ne20
       -rho*Y[jmg23]*rate_eval.n_Mg23__C12_C12
       -rho*Y[jmg24]*rate_eval.n_Mg24__He4_Ne21
       -rho*Y[jmg25]*rate_eval.n_Mg25__He4_Ne22
       -rho*Y[jal25]*rate_eval.n_Al25__p_Mg25
       -rho*Y[jal25]*rate_eval.n_Al25__He4_Na22
       -rho*Y[jal26]*rate_eval.n_Al26__p_Mg26
       -rho*Y[jal26]*rate_eval.n_Al26__He4_Na23
       -rho*Y[jsi26]*rate_eval.n_Si26__p_Al26
       -rho*Y[jsi26]*rate_eval.n_Si26__He4_Mg23
       -rho*Y[jsi28]*rate_eval.n_Si28__He4_Mg25
       -rho*Y[jsi29]*rate_eval.n_Si29__He4_Mg26
       -rho*Y[jp29]*rate_eval.n_P29__p_Si29
       -rho*Y[jp29]*rate_eval.n_P29__He4_Al26
       -rho*Y[jp30]*rate_eval.n_P30__p_Si30
       -rho*Y[jp30]*rate_eval.n_P30__He4_Al27
       -rho*Y[js30]*rate_eval.n_S30__p_P30
       -rho*Y[js31]*rate_eval.n_S31__p_P31
       -rho*Y[js31]*rate_eval.n_S31__He4_Si28
       -rho*Y[js31]*rate_eval.n_S31__C12_Ne20
       -rho*Y[js31]*rate_eval.n_S31__O16_O16
       -rho*Y[js32]*rate_eval.n_S32__He4_Si29
       -rho*Y[js33]*rate_eval.n_S33__He4_Si30
       -rho*Y[jcl33]*rate_eval.n_Cl33__p_S33
       -rho*Y[jcl33]*rate_eval.n_Cl33__He4_P30
       -rho*Y[jcl34]*rate_eval.n_Cl34__He4_P31
       -rho*Y[jar34]*rate_eval.n_Ar34__p_Cl34
       -rho*Y[jar34]*rate_eval.n_Ar34__He4_S31
       -rho*Y[jar36]*rate_eval.n_Ar36__He4_S33
       -rho*Y[jk39]*rate_eval.n_K39__p_Ar39
       -rho*Y[jca40]*rate_eval.n_Ca40__He4_Ar37
       -rho*Y[jco55]*rate_eval.n_Co55__p_Fe55
       -rho*Y[jni58]*rate_eval.n_Ni58__He4_Fe55
       )

    jac[jn, jp] = (
       +rho*Y[jc13]*rate_eval.p_C13__n_N13
       +rho*Y[jn15]*rate_eval.p_N15__n_O15
       +rho*Y[jne22]*rate_eval.p_Ne22__n_Na22
       +rho*Y[jna23]*rate_eval.p_Na23__n_Mg23
       +rho*Y[jmg25]*rate_eval.p_Mg25__n_Al25
       +rho*Y[jmg26]*rate_eval.p_Mg26__n_Al26
       +rho*Y[jal26]*rate_eval.p_Al26__n_Si26
       +rho*Y[jsi29]*rate_eval.p_Si29__n_P29
       +rho*Y[jsi30]*rate_eval.p_Si30__n_P30
       +rho*Y[jp30]*rate_eval.p_P30__n_S30
       +rho*Y[jp31]*rate_eval.p_P31__n_S31
       +rho*Y[js33]*rate_eval.p_S33__n_Cl33
       +rho*Y[jcl34]*rate_eval.p_Cl34__n_Ar34
       +rho*Y[jar39]*rate_eval.p_Ar39__n_K39
       +rho*Y[jfe55]*rate_eval.p_Fe55__n_Co55
       )

    jac[jn, jhe4] = (
       +rho*Y[jb11]*rate_eval.He4_B11__n_N14
       +rho*Y[jc12]*rate_eval.He4_C12__n_O15
       +rho*Y[jc13]*rate_eval.He4_C13__n_O16
       +rho*Y[jn15]*rate_eval.He4_N15__n_F18
       +rho*Y[jo16]*rate_eval.He4_O16__n_Ne19
       +rho*Y[jo17]*rate_eval.He4_O17__n_Ne20
       +rho*Y[jne20]*rate_eval.He4_Ne20__n_Mg23
       +rho*Y[jne21]*rate_eval.He4_Ne21__n_Mg24
       +rho*Y[jne22]*rate_eval.He4_Ne22__n_Mg25
       +rho*Y[jna22]*rate_eval.He4_Na22__n_Al25
       +rho*Y[jna23]*rate_eval.He4_Na23__n_Al26
       +rho*Y[jmg23]*rate_eval.He4_Mg23__n_Si26
       +rho*Y[jmg25]*rate_eval.He4_Mg25__n_Si28
       +rho*Y[jmg26]*rate_eval.He4_Mg26__n_Si29
       +rho*Y[jal26]*rate_eval.He4_Al26__n_P29
       +rho*Y[jal27]*rate_eval.He4_Al27__n_P30
       +rho*Y[jsi28]*rate_eval.He4_Si28__n_S31
       +rho*Y[jsi29]*rate_eval.He4_Si29__n_S32
       +rho*Y[jsi30]*rate_eval.He4_Si30__n_S33
       +rho*Y[jp30]*rate_eval.He4_P30__n_Cl33
       +rho*Y[jp31]*rate_eval.He4_P31__n_Cl34
       +rho*Y[js31]*rate_eval.He4_S31__n_Ar34
       +rho*Y[js33]*rate_eval.He4_S33__n_Ar36
       +rho*Y[jar37]*rate_eval.He4_Ar37__n_Ca40
       +rho*Y[jfe55]*rate_eval.He4_Fe55__n_Ni58
       )

    jac[jn, jb11] = (
       +rho*Y[jhe4]*rate_eval.He4_B11__n_N14
       )

    jac[jn, jc12] = (
       -rho*Y[jn]*rate_eval.n_C12__C13
       +rho*Y[jhe4]*rate_eval.He4_C12__n_O15
       +5.00000000000000e-01*rho*2*Y[jc12]*rate_eval.C12_C12__n_Mg23
       +rho*Y[jne20]*rate_eval.C12_Ne20__n_S31
       )

    jac[jn, jc13] = (
       +rate_eval.C13__n_C12
       +rho*Y[jp]*rate_eval.p_C13__n_N13
       +rho*Y[jhe4]*rate_eval.He4_C13__n_O16
       )

    jac[jn, jn13] = (
       -rho*Y[jn]*rate_eval.n_N13__N14
       -rho*Y[jn]*rate_eval.n_N13__p_C13
       )

    jac[jn, jn14] = (
       -rho*Y[jn]*rate_eval.n_N14__N15
       -rho*Y[jn]*rate_eval.n_N14__He4_B11
       +rate_eval.N14__n_N13
       )

    jac[jn, jn15] = (
       +rate_eval.N15__n_N14
       +rho*Y[jp]*rate_eval.p_N15__n_O15
       +rho*Y[jhe4]*rate_eval.He4_N15__n_F18
       )

    jac[jn, jo15] = (
       -rho*Y[jn]*rate_eval.n_O15__O16
       -rho*Y[jn]*rate_eval.n_O15__p_N15
       -rho*Y[jn]*rate_eval.n_O15__He4_C12
       )

    jac[jn, jo16] = (
       -rho*Y[jn]*rate_eval.n_O16__O17
       -rho*Y[jn]*rate_eval.n_O16__He4_C13
       +rate_eval.O16__n_O15
       +rho*Y[jhe4]*rate_eval.He4_O16__n_Ne19
       +5.00000000000000e-01*rho*2*Y[jo16]*rate_eval.O16_O16__n_S31
       )

    jac[jn, jo17] = (
       +rate_eval.O17__n_O16
       +rho*Y[jhe4]*rate_eval.He4_O17__n_Ne20
       )

    jac[jn, jf18] = (
       -rho*Y[jn]*rate_eval.n_F18__He4_N15
       )

    jac[jn, jne19] = (
       -rho*Y[jn]*rate_eval.n_Ne19__Ne20
       -rho*Y[jn]*rate_eval.n_Ne19__He4_O16
       )

    jac[jn, jne20] = (
       -rho*Y[jn]*rate_eval.n_Ne20__Ne21
       -rho*Y[jn]*rate_eval.n_Ne20__He4_O17
       +rate_eval.Ne20__n_Ne19
       +rho*Y[jhe4]*rate_eval.He4_Ne20__n_Mg23
       +rho*Y[jc12]*rate_eval.C12_Ne20__n_S31
       )

    jac[jn, jne21] = (
       -rho*Y[jn]*rate_eval.n_Ne21__Ne22
       +rate_eval.Ne21__n_Ne20
       +rho*Y[jhe4]*rate_eval.He4_Ne21__n_Mg24
       )

    jac[jn, jne22] = (
       +rate_eval.Ne22__n_Ne21
       +rho*Y[jp]*rate_eval.p_Ne22__n_Na22
       +rho*Y[jhe4]*rate_eval.He4_Ne22__n_Mg25
       )

    jac[jn, jna22] = (
       -rho*Y[jn]*rate_eval.n_Na22__Na23
       -rho*Y[jn]*rate_eval.n_Na22__p_Ne22
       +rho*Y[jhe4]*rate_eval.He4_Na22__n_Al25
       )

    jac[jn, jna23] = (
       +rate_eval.Na23__n_Na22
       +rho*Y[jp]*rate_eval.p_Na23__n_Mg23
       +rho*Y[jhe4]*rate_eval.He4_Na23__n_Al26
       )

    jac[jn, jmg23] = (
       -rho*Y[jn]*rate_eval.n_Mg23__Mg24
       -rho*Y[jn]*rate_eval.n_Mg23__p_Na23
       -rho*Y[jn]*rate_eval.n_Mg23__He4_Ne20
       -rho*Y[jn]*rate_eval.n_Mg23__C12_C12
       +rho*Y[jhe4]*rate_eval.He4_Mg23__n_Si26
       )

    jac[jn, jmg24] = (
       -rho*Y[jn]*rate_eval.n_Mg24__Mg25
       -rho*Y[jn]*rate_eval.n_Mg24__He4_Ne21
       +rate_eval.Mg24__n_Mg23
       )

    jac[jn, jmg25] = (
       -rho*Y[jn]*rate_eval.n_Mg25__Mg26
       -rho*Y[jn]*rate_eval.n_Mg25__He4_Ne22
       +rate_eval.Mg25__n_Mg24
       +rho*Y[jp]*rate_eval.p_Mg25__n_Al25
       +rho*Y[jhe4]*rate_eval.He4_Mg25__n_Si28
       )

    jac[jn, jmg26] = (
       +rate_eval.Mg26__n_Mg25
       +rho*Y[jp]*rate_eval.p_Mg26__n_Al26
       +rho*Y[jhe4]*rate_eval.He4_Mg26__n_Si29
       )

    jac[jn, jal25] = (
       -rho*Y[jn]*rate_eval.n_Al25__Al26
       -rho*Y[jn]*rate_eval.n_Al25__p_Mg25
       -rho*Y[jn]*rate_eval.n_Al25__He4_Na22
       )

    jac[jn, jal26] = (
       -rho*Y[jn]*rate_eval.n_Al26__Al27
       -rho*Y[jn]*rate_eval.n_Al26__p_Mg26
       -rho*Y[jn]*rate_eval.n_Al26__He4_Na23
       +rate_eval.Al26__n_Al25
       +rho*Y[jp]*rate_eval.p_Al26__n_Si26
       +rho*Y[jhe4]*rate_eval.He4_Al26__n_P29
       )

    jac[jn, jal27] = (
       +rate_eval.Al27__n_Al26
       +rho*Y[jhe4]*rate_eval.He4_Al27__n_P30
       )

    jac[jn, jsi26] = (
       -rho*Y[jn]*rate_eval.n_Si26__p_Al26
       -rho*Y[jn]*rate_eval.n_Si26__He4_Mg23
       )

    jac[jn, jsi28] = (
       -rho*Y[jn]*rate_eval.n_Si28__Si29
       -rho*Y[jn]*rate_eval.n_Si28__He4_Mg25
       +rho*Y[jhe4]*rate_eval.He4_Si28__n_S31
       )

    jac[jn, jsi29] = (
       -rho*Y[jn]*rate_eval.n_Si29__Si30
       -rho*Y[jn]*rate_eval.n_Si29__He4_Mg26
       +rate_eval.Si29__n_Si28
       +rho*Y[jp]*rate_eval.p_Si29__n_P29
       +rho*Y[jhe4]*rate_eval.He4_Si29__n_S32
       )

    jac[jn, jsi30] = (
       +rate_eval.Si30__n_Si29
       +rho*Y[jp]*rate_eval.p_Si30__n_P30
       +rho*Y[jhe4]*rate_eval.He4_Si30__n_S33
       )

    jac[jn, jp29] = (
       -rho*Y[jn]*rate_eval.n_P29__P30
       -rho*Y[jn]*rate_eval.n_P29__p_Si29
       -rho*Y[jn]*rate_eval.n_P29__He4_Al26
       )

    jac[jn, jp30] = (
       -rho*Y[jn]*rate_eval.n_P30__P31
       -rho*Y[jn]*rate_eval.n_P30__p_Si30
       -rho*Y[jn]*rate_eval.n_P30__He4_Al27
       +rate_eval.P30__n_P29
       +rho*Y[jp]*rate_eval.p_P30__n_S30
       +rho*Y[jhe4]*rate_eval.He4_P30__n_Cl33
       )

    jac[jn, jp31] = (
       +rate_eval.P31__n_P30
       +rho*Y[jp]*rate_eval.p_P31__n_S31
       +rho*Y[jhe4]*rate_eval.He4_P31__n_Cl34
       )

    jac[jn, js30] = (
       -rho*Y[jn]*rate_eval.n_S30__S31
       -rho*Y[jn]*rate_eval.n_S30__p_P30
       )

    jac[jn, js31] = (
       -rho*Y[jn]*rate_eval.n_S31__S32
       -rho*Y[jn]*rate_eval.n_S31__p_P31
       -rho*Y[jn]*rate_eval.n_S31__He4_Si28
       -rho*Y[jn]*rate_eval.n_S31__C12_Ne20
       -rho*Y[jn]*rate_eval.n_S31__O16_O16
       +rate_eval.S31__n_S30
       +rho*Y[jhe4]*rate_eval.He4_S31__n_Ar34
       )

    jac[jn, js32] = (
       -rho*Y[jn]*rate_eval.n_S32__S33
       -rho*Y[jn]*rate_eval.n_S32__He4_Si29
       +rate_eval.S32__n_S31
       )

    jac[jn, js33] = (
       -rho*Y[jn]*rate_eval.n_S33__He4_Si30
       +rate_eval.S33__n_S32
       +rho*Y[jp]*rate_eval.p_S33__n_Cl33
       +rho*Y[jhe4]*rate_eval.He4_S33__n_Ar36
       )

    jac[jn, jcl33] = (
       -rho*Y[jn]*rate_eval.n_Cl33__Cl34
       -rho*Y[jn]*rate_eval.n_Cl33__p_S33
       -rho*Y[jn]*rate_eval.n_Cl33__He4_P30
       )

    jac[jn, jcl34] = (
       -rho*Y[jn]*rate_eval.n_Cl34__Cl35
       -rho*Y[jn]*rate_eval.n_Cl34__He4_P31
       +rate_eval.Cl34__n_Cl33
       +rho*Y[jp]*rate_eval.p_Cl34__n_Ar34
       )

    jac[jn, jcl35] = (
       +rate_eval.Cl35__n_Cl34
       )

    jac[jn, jar34] = (
       -rho*Y[jn]*rate_eval.n_Ar34__p_Cl34
       -rho*Y[jn]*rate_eval.n_Ar34__He4_S31
       )

    jac[jn, jar36] = (
       -rho*Y[jn]*rate_eval.n_Ar36__Ar37
       -rho*Y[jn]*rate_eval.n_Ar36__He4_S33
       )

    jac[jn, jar37] = (
       -rho*Y[jn]*rate_eval.n_Ar37__Ar38
       +rate_eval.Ar37__n_Ar36
       +rho*Y[jhe4]*rate_eval.He4_Ar37__n_Ca40
       )

    jac[jn, jar38] = (
       -rho*Y[jn]*rate_eval.n_Ar38__Ar39
       +rate_eval.Ar38__n_Ar37
       )

    jac[jn, jar39] = (
       +rate_eval.Ar39__n_Ar38
       +rho*Y[jp]*rate_eval.p_Ar39__n_K39
       )

    jac[jn, jk39] = (
       -rho*Y[jn]*rate_eval.n_K39__p_Ar39
       )

    jac[jn, jca40] = (
       -rho*Y[jn]*rate_eval.n_Ca40__He4_Ar37
       )

    jac[jn, jfe55] = (
       +rho*Y[jp]*rate_eval.p_Fe55__n_Co55
       +rho*Y[jhe4]*rate_eval.He4_Fe55__n_Ni58
       )

    jac[jn, jco55] = (
       -rho*Y[jn]*rate_eval.n_Co55__p_Fe55
       )

    jac[jn, jni58] = (
       -rho*Y[jn]*rate_eval.n_Ni58__Ni59
       -rho*Y[jn]*rate_eval.n_Ni58__He4_Fe55
       )

    jac[jn, jni59] = (
       +rate_eval.Ni59__n_Ni58
       )

    jac[jp, jn] = (
       +rate_eval.n__p__weak__wc12
       +rho*Y[jn13]*rate_eval.n_N13__p_C13
       +rho*Y[jo15]*rate_eval.n_O15__p_N15
       +rho*Y[jna22]*rate_eval.n_Na22__p_Ne22
       +rho*Y[jmg23]*rate_eval.n_Mg23__p_Na23
       +rho*Y[jal25]*rate_eval.n_Al25__p_Mg25
       +rho*Y[jal26]*rate_eval.n_Al26__p_Mg26
       +rho*Y[jsi26]*rate_eval.n_Si26__p_Al26
       +rho*Y[jp29]*rate_eval.n_P29__p_Si29
       +rho*Y[jp30]*rate_eval.n_P30__p_Si30
       +rho*Y[js30]*rate_eval.n_S30__p_P30
       +rho*Y[js31]*rate_eval.n_S31__p_P31
       +rho*Y[jcl33]*rate_eval.n_Cl33__p_S33
       +rho*Y[jar34]*rate_eval.n_Ar34__p_Cl34
       +rho*Y[jk39]*rate_eval.n_K39__p_Ar39
       +rho*Y[jco55]*rate_eval.n_Co55__p_Fe55
       )

    jac[jp, jp] = (
       -rho*Y[jb11]*rate_eval.p_B11__C12
       -rho*Y[jc12]*rate_eval.p_C12__N13
       -rho*Y[jc13]*rate_eval.p_C13__N14
       -rho*Y[jn14]*rate_eval.p_N14__O15
       -rho*Y[jn15]*rate_eval.p_N15__O16
       -rho*Y[jo17]*rate_eval.p_O17__F18
       -rho*Y[jf18]*rate_eval.p_F18__Ne19
       -rho*Y[jne21]*rate_eval.p_Ne21__Na22
       -rho*Y[jne22]*rate_eval.p_Ne22__Na23
       -rho*Y[jna22]*rate_eval.p_Na22__Mg23
       -rho*Y[jna23]*rate_eval.p_Na23__Mg24
       -rho*Y[jmg24]*rate_eval.p_Mg24__Al25
       -rho*Y[jmg25]*rate_eval.p_Mg25__Al26
       -rho*Y[jmg26]*rate_eval.p_Mg26__Al27
       -rho*Y[jal25]*rate_eval.p_Al25__Si26
       -rho*Y[jal27]*rate_eval.p_Al27__Si28
       -rho*Y[jsi28]*rate_eval.p_Si28__P29
       -rho*Y[jsi29]*rate_eval.p_Si29__P30
       -rho*Y[jsi30]*rate_eval.p_Si30__P31
       -rho*Y[jp29]*rate_eval.p_P29__S30
       -rho*Y[jp30]*rate_eval.p_P30__S31
       -rho*Y[jp31]*rate_eval.p_P31__S32
       -rho*Y[js32]*rate_eval.p_S32__Cl33
       -rho*Y[js33]*rate_eval.p_S33__Cl34
       -rho*Y[jcl33]*rate_eval.p_Cl33__Ar34
       -rho*Y[jcl35]*rate_eval.p_Cl35__Ar36
       -rho*Y[jar38]*rate_eval.p_Ar38__K39
       -rho*Y[jk39]*rate_eval.p_K39__Ca40
       -rho*Y[jsc43]*rate_eval.p_Sc43__Ti44
       -rho*Y[jv47]*rate_eval.p_V47__Cr48
       -rho*Y[jmn51]*rate_eval.p_Mn51__Fe52
       -rho*Y[jco55]*rate_eval.p_Co55__Ni56
       -rho*Y[jc13]*rate_eval.p_C13__n_N13
       -rho*Y[jn15]*rate_eval.p_N15__n_O15
       -rho*Y[jn15]*rate_eval.p_N15__He4_C12
       -rho*Y[jo16]*rate_eval.p_O16__He4_N13
       -rho*Y[jo17]*rate_eval.p_O17__He4_N14
       -rho*Y[jf18]*rate_eval.p_F18__He4_O15
       -rho*Y[jne21]*rate_eval.p_Ne21__He4_F18
       -rho*Y[jne22]*rate_eval.p_Ne22__n_Na22
       -rho*Y[jna22]*rate_eval.p_Na22__He4_Ne19
       -rho*Y[jna23]*rate_eval.p_Na23__n_Mg23
       -rho*Y[jna23]*rate_eval.p_Na23__He4_Ne20
       -rho*Y[jna23]*rate_eval.p_Na23__C12_C12
       -rho*Y[jmg25]*rate_eval.p_Mg25__n_Al25
       -rho*Y[jmg25]*rate_eval.p_Mg25__He4_Na22
       -rho*Y[jmg26]*rate_eval.p_Mg26__n_Al26
       -rho*Y[jmg26]*rate_eval.p_Mg26__He4_Na23
       -rho*Y[jal26]*rate_eval.p_Al26__n_Si26
       -rho*Y[jal26]*rate_eval.p_Al26__He4_Mg23
       -rho*Y[jal27]*rate_eval.p_Al27__He4_Mg24
       -rho*Y[jal27]*rate_eval.p_Al27__C12_O16
       -rho*Y[jsi28]*rate_eval.p_Si28__He4_Al25
       -rho*Y[jsi29]*rate_eval.p_Si29__n_P29
       -rho*Y[jsi29]*rate_eval.p_Si29__He4_Al26
       -rho*Y[jsi30]*rate_eval.p_Si30__n_P30
       -rho*Y[jsi30]*rate_eval.p_Si30__He4_Al27
       -rho*Y[jp29]*rate_eval.p_P29__He4_Si26
       -rho*Y[jp30]*rate_eval.p_P30__n_S30
       -rho*Y[jp31]*rate_eval.p_P31__n_S31
       -rho*Y[jp31]*rate_eval.p_P31__He4_Si28
       -rho*Y[jp31]*rate_eval.p_P31__C12_Ne20
       -rho*Y[jp31]*rate_eval.p_P31__O16_O16
       -rho*Y[js32]*rate_eval.p_S32__He4_P29
       -rho*Y[js33]*rate_eval.p_S33__n_Cl33
       -rho*Y[js33]*rate_eval.p_S33__He4_P30
       -rho*Y[jcl33]*rate_eval.p_Cl33__He4_S30
       -rho*Y[jcl34]*rate_eval.p_Cl34__n_Ar34
       -rho*Y[jcl34]*rate_eval.p_Cl34__He4_S31
       -rho*Y[jcl35]*rate_eval.p_Cl35__He4_S32
       -rho*Y[jar36]*rate_eval.p_Ar36__He4_Cl33
       -rho*Y[jar37]*rate_eval.p_Ar37__He4_Cl34
       -rho*Y[jar38]*rate_eval.p_Ar38__He4_Cl35
       -rho*Y[jar39]*rate_eval.p_Ar39__n_K39
       -rho*Y[jk39]*rate_eval.p_K39__He4_Ar36
       -rho*Y[jsc43]*rate_eval.p_Sc43__He4_Ca40
       -rho*Y[jv47]*rate_eval.p_V47__He4_Ti44
       -rho*Y[jmn51]*rate_eval.p_Mn51__He4_Cr48
       -rho*Y[jfe55]*rate_eval.p_Fe55__n_Co55
       -rho*Y[jco55]*rate_eval.p_Co55__He4_Fe52
       -rho*Y[jni58]*rate_eval.p_Ni58__He4_Co55
       -rho*Y[jb11]*rate_eval.p_B11__He4_He4_He4
       )

    jac[jp, jhe4] = (
       +rho*Y[jc12]*rate_eval.He4_C12__p_N15
       +rho*Y[jn13]*rate_eval.He4_N13__p_O16
       +rho*Y[jn14]*rate_eval.He4_N14__p_O17
       +rho*Y[jo15]*rate_eval.He4_O15__p_F18
       +rho*Y[jf18]*rate_eval.He4_F18__p_Ne21
       +rho*Y[jne19]*rate_eval.He4_Ne19__p_Na22
       +rho*Y[jne20]*rate_eval.He4_Ne20__p_Na23
       +rho*Y[jna22]*rate_eval.He4_Na22__p_Mg25
       +rho*Y[jna23]*rate_eval.He4_Na23__p_Mg26
       +rho*Y[jmg23]*rate_eval.He4_Mg23__p_Al26
       +rho*Y[jmg24]*rate_eval.He4_Mg24__p_Al27
       +rho*Y[jal25]*rate_eval.He4_Al25__p_Si28
       +rho*Y[jal26]*rate_eval.He4_Al26__p_Si29
       +rho*Y[jal27]*rate_eval.He4_Al27__p_Si30
       +rho*Y[jsi26]*rate_eval.He4_Si26__p_P29
       +rho*Y[jsi28]*rate_eval.He4_Si28__p_P31
       +rho*Y[jp29]*rate_eval.He4_P29__p_S32
       +rho*Y[jp30]*rate_eval.He4_P30__p_S33
       +rho*Y[js30]*rate_eval.He4_S30__p_Cl33
       +rho*Y[js31]*rate_eval.He4_S31__p_Cl34
       +rho*Y[js32]*rate_eval.He4_S32__p_Cl35
       +rho*Y[jcl33]*rate_eval.He4_Cl33__p_Ar36
       +rho*Y[jcl34]*rate_eval.He4_Cl34__p_Ar37
       +rho*Y[jcl35]*rate_eval.He4_Cl35__p_Ar38
       +rho*Y[jar36]*rate_eval.He4_Ar36__p_K39
       +rho*Y[jca40]*rate_eval.He4_Ca40__p_Sc43
       +rho*Y[jti44]*rate_eval.He4_Ti44__p_V47
       +rho*Y[jcr48]*rate_eval.He4_Cr48__p_Mn51
       +rho*Y[jfe52]*rate_eval.He4_Fe52__p_Co55
       +rho*Y[jco55]*rate_eval.He4_Co55__p_Ni58
       +1.66666666666667e-01*rho**2*3*Y[jhe4]**2*rate_eval.He4_He4_He4__p_B11
       )

    jac[jp, jb11] = (
       -rho*Y[jp]*rate_eval.p_B11__C12
       -rho*Y[jp]*rate_eval.p_B11__He4_He4_He4
       )

    jac[jp, jc12] = (
       -rho*Y[jp]*rate_eval.p_C12__N13
       +rate_eval.C12__p_B11
       +rho*Y[jhe4]*rate_eval.He4_C12__p_N15
       +5.00000000000000e-01*rho*2*Y[jc12]*rate_eval.C12_C12__p_Na23
       +rho*Y[jo16]*rate_eval.C12_O16__p_Al27
       +rho*Y[jne20]*rate_eval.C12_Ne20__p_P31
       )

    jac[jp, jc13] = (
       -rho*Y[jp]*rate_eval.p_C13__N14
       -rho*Y[jp]*rate_eval.p_C13__n_N13
       )

    jac[jp, jn13] = (
       +rate_eval.N13__p_C12
       +rho*Y[jn]*rate_eval.n_N13__p_C13
       +rho*Y[jhe4]*rate_eval.He4_N13__p_O16
       )

    jac[jp, jn14] = (
       -rho*Y[jp]*rate_eval.p_N14__O15
       +rate_eval.N14__p_C13
       +rho*Y[jhe4]*rate_eval.He4_N14__p_O17
       )

    jac[jp, jn15] = (
       -rho*Y[jp]*rate_eval.p_N15__O16
       -rho*Y[jp]*rate_eval.p_N15__n_O15
       -rho*Y[jp]*rate_eval.p_N15__He4_C12
       )

    jac[jp, jo15] = (
       +rate_eval.O15__p_N14
       +rho*Y[jn]*rate_eval.n_O15__p_N15
       +rho*Y[jhe4]*rate_eval.He4_O15__p_F18
       )

    jac[jp, jo16] = (
       -rho*Y[jp]*rate_eval.p_O16__He4_N13
       +rate_eval.O16__p_N15
       +rho*Y[jc12]*rate_eval.C12_O16__p_Al27
       +5.00000000000000e-01*rho*2*Y[jo16]*rate_eval.O16_O16__p_P31
       )

    jac[jp, jo17] = (
       -rho*Y[jp]*rate_eval.p_O17__F18
       -rho*Y[jp]*rate_eval.p_O17__He4_N14
       )

    jac[jp, jf18] = (
       -rho*Y[jp]*rate_eval.p_F18__Ne19
       -rho*Y[jp]*rate_eval.p_F18__He4_O15
       +rate_eval.F18__p_O17
       +rho*Y[jhe4]*rate_eval.He4_F18__p_Ne21
       )

    jac[jp, jne19] = (
       +rate_eval.Ne19__p_F18
       +rho*Y[jhe4]*rate_eval.He4_Ne19__p_Na22
       )

    jac[jp, jne20] = (
       +rho*Y[jhe4]*rate_eval.He4_Ne20__p_Na23
       +rho*Y[jc12]*rate_eval.C12_Ne20__p_P31
       )

    jac[jp, jne21] = (
       -rho*Y[jp]*rate_eval.p_Ne21__Na22
       -rho*Y[jp]*rate_eval.p_Ne21__He4_F18
       )

    jac[jp, jne22] = (
       -rho*Y[jp]*rate_eval.p_Ne22__Na23
       -rho*Y[jp]*rate_eval.p_Ne22__n_Na22
       )

    jac[jp, jna22] = (
       -rho*Y[jp]*rate_eval.p_Na22__Mg23
       -rho*Y[jp]*rate_eval.p_Na22__He4_Ne19
       +rate_eval.Na22__p_Ne21
       +rho*Y[jn]*rate_eval.n_Na22__p_Ne22
       +rho*Y[jhe4]*rate_eval.He4_Na22__p_Mg25
       )

    jac[jp, jna23] = (
       -rho*Y[jp]*rate_eval.p_Na23__Mg24
       -rho*Y[jp]*rate_eval.p_Na23__n_Mg23
       -rho*Y[jp]*rate_eval.p_Na23__He4_Ne20
       -rho*Y[jp]*rate_eval.p_Na23__C12_C12
       +rate_eval.Na23__p_Ne22
       +rho*Y[jhe4]*rate_eval.He4_Na23__p_Mg26
       )

    jac[jp, jmg23] = (
       +rate_eval.Mg23__p_Na22
       +rho*Y[jn]*rate_eval.n_Mg23__p_Na23
       +rho*Y[jhe4]*rate_eval.He4_Mg23__p_Al26
       )

    jac[jp, jmg24] = (
       -rho*Y[jp]*rate_eval.p_Mg24__Al25
       +rate_eval.Mg24__p_Na23
       +rho*Y[jhe4]*rate_eval.He4_Mg24__p_Al27
       )

    jac[jp, jmg25] = (
       -rho*Y[jp]*rate_eval.p_Mg25__Al26
       -rho*Y[jp]*rate_eval.p_Mg25__n_Al25
       -rho*Y[jp]*rate_eval.p_Mg25__He4_Na22
       )

    jac[jp, jmg26] = (
       -rho*Y[jp]*rate_eval.p_Mg26__Al27
       -rho*Y[jp]*rate_eval.p_Mg26__n_Al26
       -rho*Y[jp]*rate_eval.p_Mg26__He4_Na23
       )

    jac[jp, jal25] = (
       -rho*Y[jp]*rate_eval.p_Al25__Si26
       +rate_eval.Al25__p_Mg24
       +rho*Y[jn]*rate_eval.n_Al25__p_Mg25
       +rho*Y[jhe4]*rate_eval.He4_Al25__p_Si28
       )

    jac[jp, jal26] = (
       -rho*Y[jp]*rate_eval.p_Al26__n_Si26
       -rho*Y[jp]*rate_eval.p_Al26__He4_Mg23
       +rate_eval.Al26__p_Mg25
       +rho*Y[jn]*rate_eval.n_Al26__p_Mg26
       +rho*Y[jhe4]*rate_eval.He4_Al26__p_Si29
       )

    jac[jp, jal27] = (
       -rho*Y[jp]*rate_eval.p_Al27__Si28
       -rho*Y[jp]*rate_eval.p_Al27__He4_Mg24
       -rho*Y[jp]*rate_eval.p_Al27__C12_O16
       +rate_eval.Al27__p_Mg26
       +rho*Y[jhe4]*rate_eval.He4_Al27__p_Si30
       )

    jac[jp, jsi26] = (
       +rate_eval.Si26__p_Al25
       +rho*Y[jn]*rate_eval.n_Si26__p_Al26
       +rho*Y[jhe4]*rate_eval.He4_Si26__p_P29
       )

    jac[jp, jsi28] = (
       -rho*Y[jp]*rate_eval.p_Si28__P29
       -rho*Y[jp]*rate_eval.p_Si28__He4_Al25
       +rate_eval.Si28__p_Al27
       +rho*Y[jhe4]*rate_eval.He4_Si28__p_P31
       )

    jac[jp, jsi29] = (
       -rho*Y[jp]*rate_eval.p_Si29__P30
       -rho*Y[jp]*rate_eval.p_Si29__n_P29
       -rho*Y[jp]*rate_eval.p_Si29__He4_Al26
       )

    jac[jp, jsi30] = (
       -rho*Y[jp]*rate_eval.p_Si30__P31
       -rho*Y[jp]*rate_eval.p_Si30__n_P30
       -rho*Y[jp]*rate_eval.p_Si30__He4_Al27
       )

    jac[jp, jp29] = (
       -rho*Y[jp]*rate_eval.p_P29__S30
       -rho*Y[jp]*rate_eval.p_P29__He4_Si26
       +rate_eval.P29__p_Si28
       +rho*Y[jn]*rate_eval.n_P29__p_Si29
       +rho*Y[jhe4]*rate_eval.He4_P29__p_S32
       )

    jac[jp, jp30] = (
       -rho*Y[jp]*rate_eval.p_P30__S31
       -rho*Y[jp]*rate_eval.p_P30__n_S30
       +rate_eval.P30__p_Si29
       +rho*Y[jn]*rate_eval.n_P30__p_Si30
       +rho*Y[jhe4]*rate_eval.He4_P30__p_S33
       )

    jac[jp, jp31] = (
       -rho*Y[jp]*rate_eval.p_P31__S32
       -rho*Y[jp]*rate_eval.p_P31__n_S31
       -rho*Y[jp]*rate_eval.p_P31__He4_Si28
       -rho*Y[jp]*rate_eval.p_P31__C12_Ne20
       -rho*Y[jp]*rate_eval.p_P31__O16_O16
       +rate_eval.P31__p_Si30
       )

    jac[jp, js30] = (
       +rate_eval.S30__p_P29
       +rho*Y[jn]*rate_eval.n_S30__p_P30
       +rho*Y[jhe4]*rate_eval.He4_S30__p_Cl33
       )

    jac[jp, js31] = (
       +rate_eval.S31__p_P30
       +rho*Y[jn]*rate_eval.n_S31__p_P31
       +rho*Y[jhe4]*rate_eval.He4_S31__p_Cl34
       )

    jac[jp, js32] = (
       -rho*Y[jp]*rate_eval.p_S32__Cl33
       -rho*Y[jp]*rate_eval.p_S32__He4_P29
       +rate_eval.S32__p_P31
       +rho*Y[jhe4]*rate_eval.He4_S32__p_Cl35
       )

    jac[jp, js33] = (
       -rho*Y[jp]*rate_eval.p_S33__Cl34
       -rho*Y[jp]*rate_eval.p_S33__n_Cl33
       -rho*Y[jp]*rate_eval.p_S33__He4_P30
       )

    jac[jp, jcl33] = (
       -rho*Y[jp]*rate_eval.p_Cl33__Ar34
       -rho*Y[jp]*rate_eval.p_Cl33__He4_S30
       +rate_eval.Cl33__p_S32
       +rho*Y[jn]*rate_eval.n_Cl33__p_S33
       +rho*Y[jhe4]*rate_eval.He4_Cl33__p_Ar36
       )

    jac[jp, jcl34] = (
       -rho*Y[jp]*rate_eval.p_Cl34__n_Ar34
       -rho*Y[jp]*rate_eval.p_Cl34__He4_S31
       +rate_eval.Cl34__p_S33
       +rho*Y[jhe4]*rate_eval.He4_Cl34__p_Ar37
       )

    jac[jp, jcl35] = (
       -rho*Y[jp]*rate_eval.p_Cl35__Ar36
       -rho*Y[jp]*rate_eval.p_Cl35__He4_S32
       +rho*Y[jhe4]*rate_eval.He4_Cl35__p_Ar38
       )

    jac[jp, jar34] = (
       +rate_eval.Ar34__p_Cl33
       +rho*Y[jn]*rate_eval.n_Ar34__p_Cl34
       )

    jac[jp, jar36] = (
       -rho*Y[jp]*rate_eval.p_Ar36__He4_Cl33
       +rate_eval.Ar36__p_Cl35
       +rho*Y[jhe4]*rate_eval.He4_Ar36__p_K39
       )

    jac[jp, jar37] = (
       -rho*Y[jp]*rate_eval.p_Ar37__He4_Cl34
       )

    jac[jp, jar38] = (
       -rho*Y[jp]*rate_eval.p_Ar38__K39
       -rho*Y[jp]*rate_eval.p_Ar38__He4_Cl35
       )

    jac[jp, jar39] = (
       -rho*Y[jp]*rate_eval.p_Ar39__n_K39
       )

    jac[jp, jk39] = (
       -rho*Y[jp]*rate_eval.p_K39__Ca40
       -rho*Y[jp]*rate_eval.p_K39__He4_Ar36
       +rate_eval.K39__p_Ar38
       +rho*Y[jn]*rate_eval.n_K39__p_Ar39
       )

    jac[jp, jca40] = (
       +rate_eval.Ca40__p_K39
       +rho*Y[jhe4]*rate_eval.He4_Ca40__p_Sc43
       )

    jac[jp, jsc43] = (
       -rho*Y[jp]*rate_eval.p_Sc43__Ti44
       -rho*Y[jp]*rate_eval.p_Sc43__He4_Ca40
       )

    jac[jp, jti44] = (
       +rate_eval.Ti44__p_Sc43
       +rho*Y[jhe4]*rate_eval.He4_Ti44__p_V47
       )

    jac[jp, jv47] = (
       -rho*Y[jp]*rate_eval.p_V47__Cr48
       -rho*Y[jp]*rate_eval.p_V47__He4_Ti44
       )

    jac[jp, jcr48] = (
       +rate_eval.Cr48__p_V47
       +rho*Y[jhe4]*rate_eval.He4_Cr48__p_Mn51
       )

    jac[jp, jmn51] = (
       -rho*Y[jp]*rate_eval.p_Mn51__Fe52
       -rho*Y[jp]*rate_eval.p_Mn51__He4_Cr48
       )

    jac[jp, jfe52] = (
       +rate_eval.Fe52__p_Mn51
       +rho*Y[jhe4]*rate_eval.He4_Fe52__p_Co55
       )

    jac[jp, jfe55] = (
       -rho*Y[jp]*rate_eval.p_Fe55__n_Co55
       )

    jac[jp, jco55] = (
       -rho*Y[jp]*rate_eval.p_Co55__Ni56
       -rho*Y[jp]*rate_eval.p_Co55__He4_Fe52
       +rho*Y[jn]*rate_eval.n_Co55__p_Fe55
       +rho*Y[jhe4]*rate_eval.He4_Co55__p_Ni58
       )

    jac[jp, jni56] = (
       +rate_eval.Ni56__p_Co55
       )

    jac[jp, jni58] = (
       -rho*Y[jp]*rate_eval.p_Ni58__He4_Co55
       )

    jac[jhe4, jn] = (
       +rho*Y[jn14]*rate_eval.n_N14__He4_B11
       +rho*Y[jo15]*rate_eval.n_O15__He4_C12
       +rho*Y[jo16]*rate_eval.n_O16__He4_C13
       +rho*Y[jf18]*rate_eval.n_F18__He4_N15
       +rho*Y[jne19]*rate_eval.n_Ne19__He4_O16
       +rho*Y[jne20]*rate_eval.n_Ne20__He4_O17
       +rho*Y[jmg23]*rate_eval.n_Mg23__He4_Ne20
       +rho*Y[jmg24]*rate_eval.n_Mg24__He4_Ne21
       +rho*Y[jmg25]*rate_eval.n_Mg25__He4_Ne22
       +rho*Y[jal25]*rate_eval.n_Al25__He4_Na22
       +rho*Y[jal26]*rate_eval.n_Al26__He4_Na23
       +rho*Y[jsi26]*rate_eval.n_Si26__He4_Mg23
       +rho*Y[jsi28]*rate_eval.n_Si28__He4_Mg25
       +rho*Y[jsi29]*rate_eval.n_Si29__He4_Mg26
       +rho*Y[jp29]*rate_eval.n_P29__He4_Al26
       +rho*Y[jp30]*rate_eval.n_P30__He4_Al27
       +rho*Y[js31]*rate_eval.n_S31__He4_Si28
       +rho*Y[js32]*rate_eval.n_S32__He4_Si29
       +rho*Y[js33]*rate_eval.n_S33__He4_Si30
       +rho*Y[jcl33]*rate_eval.n_Cl33__He4_P30
       +rho*Y[jcl34]*rate_eval.n_Cl34__He4_P31
       +rho*Y[jar34]*rate_eval.n_Ar34__He4_S31
       +rho*Y[jar36]*rate_eval.n_Ar36__He4_S33
       +rho*Y[jca40]*rate_eval.n_Ca40__He4_Ar37
       +rho*Y[jni58]*rate_eval.n_Ni58__He4_Fe55
       )

    jac[jhe4, jp] = (
       +rho*Y[jn15]*rate_eval.p_N15__He4_C12
       +rho*Y[jo16]*rate_eval.p_O16__He4_N13
       +rho*Y[jo17]*rate_eval.p_O17__He4_N14
       +rho*Y[jf18]*rate_eval.p_F18__He4_O15
       +rho*Y[jne21]*rate_eval.p_Ne21__He4_F18
       +rho*Y[jna22]*rate_eval.p_Na22__He4_Ne19
       +rho*Y[jna23]*rate_eval.p_Na23__He4_Ne20
       +rho*Y[jmg25]*rate_eval.p_Mg25__He4_Na22
       +rho*Y[jmg26]*rate_eval.p_Mg26__He4_Na23
       +rho*Y[jal26]*rate_eval.p_Al26__He4_Mg23
       +rho*Y[jal27]*rate_eval.p_Al27__He4_Mg24
       +rho*Y[jsi28]*rate_eval.p_Si28__He4_Al25
       +rho*Y[jsi29]*rate_eval.p_Si29__He4_Al26
       +rho*Y[jsi30]*rate_eval.p_Si30__He4_Al27
       +rho*Y[jp29]*rate_eval.p_P29__He4_Si26
       +rho*Y[jp31]*rate_eval.p_P31__He4_Si28
       +rho*Y[js32]*rate_eval.p_S32__He4_P29
       +rho*Y[js33]*rate_eval.p_S33__He4_P30
       +rho*Y[jcl33]*rate_eval.p_Cl33__He4_S30
       +rho*Y[jcl34]*rate_eval.p_Cl34__He4_S31
       +rho*Y[jcl35]*rate_eval.p_Cl35__He4_S32
       +rho*Y[jar36]*rate_eval.p_Ar36__He4_Cl33
       +rho*Y[jar37]*rate_eval.p_Ar37__He4_Cl34
       +rho*Y[jar38]*rate_eval.p_Ar38__He4_Cl35
       +rho*Y[jk39]*rate_eval.p_K39__He4_Ar36
       +rho*Y[jsc43]*rate_eval.p_Sc43__He4_Ca40
       +rho*Y[jv47]*rate_eval.p_V47__He4_Ti44
       +rho*Y[jmn51]*rate_eval.p_Mn51__He4_Cr48
       +rho*Y[jco55]*rate_eval.p_Co55__He4_Fe52
       +rho*Y[jni58]*rate_eval.p_Ni58__He4_Co55
       +3*rho*Y[jb11]*rate_eval.p_B11__He4_He4_He4
       )

    jac[jhe4, jhe4] = (
       -rho*Y[jc12]*rate_eval.He4_C12__O16
       -rho*Y[jn14]*rate_eval.He4_N14__F18
       -rho*Y[jo15]*rate_eval.He4_O15__Ne19
       -rho*Y[jo16]*rate_eval.He4_O16__Ne20
       -rho*Y[jo17]*rate_eval.He4_O17__Ne21
       -rho*Y[jf18]*rate_eval.He4_F18__Na22
       -rho*Y[jne19]*rate_eval.He4_Ne19__Mg23
       -rho*Y[jne20]*rate_eval.He4_Ne20__Mg24
       -rho*Y[jne21]*rate_eval.He4_Ne21__Mg25
       -rho*Y[jne22]*rate_eval.He4_Ne22__Mg26
       -rho*Y[jna22]*rate_eval.He4_Na22__Al26
       -rho*Y[jna23]*rate_eval.He4_Na23__Al27
       -rho*Y[jmg24]*rate_eval.He4_Mg24__Si28
       -rho*Y[jmg25]*rate_eval.He4_Mg25__Si29
       -rho*Y[jmg26]*rate_eval.He4_Mg26__Si30
       -rho*Y[jal25]*rate_eval.He4_Al25__P29
       -rho*Y[jal26]*rate_eval.He4_Al26__P30
       -rho*Y[jal27]*rate_eval.He4_Al27__P31
       -rho*Y[jsi26]*rate_eval.He4_Si26__S30
       -rho*Y[jsi28]*rate_eval.He4_Si28__S32
       -rho*Y[jsi29]*rate_eval.He4_Si29__S33
       -rho*Y[jp29]*rate_eval.He4_P29__Cl33
       -rho*Y[jp30]*rate_eval.He4_P30__Cl34
       -rho*Y[jp31]*rate_eval.He4_P31__Cl35
       -rho*Y[js30]*rate_eval.He4_S30__Ar34
       -rho*Y[js32]*rate_eval.He4_S32__Ar36
       -rho*Y[js33]*rate_eval.He4_S33__Ar37
       -rho*Y[jcl35]*rate_eval.He4_Cl35__K39
       -rho*Y[jar36]*rate_eval.He4_Ar36__Ca40
       -rho*Y[jk39]*rate_eval.He4_K39__Sc43
       -rho*Y[jca40]*rate_eval.He4_Ca40__Ti44
       -rho*Y[jsc43]*rate_eval.He4_Sc43__V47
       -rho*Y[jti44]*rate_eval.He4_Ti44__Cr48
       -rho*Y[jv47]*rate_eval.He4_V47__Mn51
       -rho*Y[jcr48]*rate_eval.He4_Cr48__Fe52
       -rho*Y[jmn51]*rate_eval.He4_Mn51__Co55
       -rho*Y[jfe52]*rate_eval.He4_Fe52__Ni56
       -rho*Y[jfe55]*rate_eval.He4_Fe55__Ni59
       -rho*Y[jb11]*rate_eval.He4_B11__n_N14
       -rho*Y[jc12]*rate_eval.He4_C12__n_O15
       -rho*Y[jc12]*rate_eval.He4_C12__p_N15
       -rho*Y[jc13]*rate_eval.He4_C13__n_O16
       -rho*Y[jn13]*rate_eval.He4_N13__p_O16
       -rho*Y[jn14]*rate_eval.He4_N14__p_O17
       -rho*Y[jn15]*rate_eval.He4_N15__n_F18
       -rho*Y[jo15]*rate_eval.He4_O15__p_F18
       -rho*Y[jo16]*rate_eval.He4_O16__n_Ne19
       -rho*Y[jo17]*rate_eval.He4_O17__n_Ne20
       -rho*Y[jf18]*rate_eval.He4_F18__p_Ne21
       -rho*Y[jne19]*rate_eval.He4_Ne19__p_Na22
       -rho*Y[jne20]*rate_eval.He4_Ne20__n_Mg23
       -rho*Y[jne20]*rate_eval.He4_Ne20__p_Na23
       -rho*Y[jne20]*rate_eval.He4_Ne20__C12_C12
       -rho*Y[jne21]*rate_eval.He4_Ne21__n_Mg24
       -rho*Y[jne22]*rate_eval.He4_Ne22__n_Mg25
       -rho*Y[jna22]*rate_eval.He4_Na22__n_Al25
       -rho*Y[jna22]*rate_eval.He4_Na22__p_Mg25
       -rho*Y[jna23]*rate_eval.He4_Na23__n_Al26
       -rho*Y[jna23]*rate_eval.He4_Na23__p_Mg26
       -rho*Y[jmg23]*rate_eval.He4_Mg23__n_Si26
       -rho*Y[jmg23]*rate_eval.He4_Mg23__p_Al26
       -rho*Y[jmg24]*rate_eval.He4_Mg24__p_Al27
       -rho*Y[jmg24]*rate_eval.He4_Mg24__C12_O16
       -rho*Y[jmg25]*rate_eval.He4_Mg25__n_Si28
       -rho*Y[jmg26]*rate_eval.He4_Mg26__n_Si29
       -rho*Y[jal25]*rate_eval.He4_Al25__p_Si28
       -rho*Y[jal26]*rate_eval.He4_Al26__n_P29
       -rho*Y[jal26]*rate_eval.He4_Al26__p_Si29
       -rho*Y[jal27]*rate_eval.He4_Al27__n_P30
       -rho*Y[jal27]*rate_eval.He4_Al27__p_Si30
       -rho*Y[jsi26]*rate_eval.He4_Si26__p_P29
       -rho*Y[jsi28]*rate_eval.He4_Si28__n_S31
       -rho*Y[jsi28]*rate_eval.He4_Si28__p_P31
       -rho*Y[jsi28]*rate_eval.He4_Si28__C12_Ne20
       -rho*Y[jsi28]*rate_eval.He4_Si28__O16_O16
       -rho*Y[jsi29]*rate_eval.He4_Si29__n_S32
       -rho*Y[jsi30]*rate_eval.He4_Si30__n_S33
       -rho*Y[jp29]*rate_eval.He4_P29__p_S32
       -rho*Y[jp30]*rate_eval.He4_P30__n_Cl33
       -rho*Y[jp30]*rate_eval.He4_P30__p_S33
       -rho*Y[jp31]*rate_eval.He4_P31__n_Cl34
       -rho*Y[js30]*rate_eval.He4_S30__p_Cl33
       -rho*Y[js31]*rate_eval.He4_S31__n_Ar34
       -rho*Y[js31]*rate_eval.He4_S31__p_Cl34
       -rho*Y[js32]*rate_eval.He4_S32__p_Cl35
       -rho*Y[js33]*rate_eval.He4_S33__n_Ar36
       -rho*Y[jcl33]*rate_eval.He4_Cl33__p_Ar36
       -rho*Y[jcl34]*rate_eval.He4_Cl34__p_Ar37
       -rho*Y[jcl35]*rate_eval.He4_Cl35__p_Ar38
       -rho*Y[jar36]*rate_eval.He4_Ar36__p_K39
       -rho*Y[jar37]*rate_eval.He4_Ar37__n_Ca40
       -rho*Y[jca40]*rate_eval.He4_Ca40__p_Sc43
       -rho*Y[jti44]*rate_eval.He4_Ti44__p_V47
       -rho*Y[jcr48]*rate_eval.He4_Cr48__p_Mn51
       -rho*Y[jfe52]*rate_eval.He4_Fe52__p_Co55
       -rho*Y[jfe55]*rate_eval.He4_Fe55__n_Ni58
       -rho*Y[jco55]*rate_eval.He4_Co55__p_Ni58
       -3*1.66666666666667e-01*rho**2*3*Y[jhe4]**2*rate_eval.He4_He4_He4__C12
       -3*1.66666666666667e-01*rho**2*3*Y[jhe4]**2*rate_eval.He4_He4_He4__p_B11
       )

    jac[jhe4, jb11] = (
       -rho*Y[jhe4]*rate_eval.He4_B11__n_N14
       +3*rho*Y[jp]*rate_eval.p_B11__He4_He4_He4
       )

    jac[jhe4, jc12] = (
       -rho*Y[jhe4]*rate_eval.He4_C12__O16
       -rho*Y[jhe4]*rate_eval.He4_C12__n_O15
       -rho*Y[jhe4]*rate_eval.He4_C12__p_N15
       +3*rate_eval.C12__He4_He4_He4
       +5.00000000000000e-01*rho*2*Y[jc12]*rate_eval.C12_C12__He4_Ne20
       +rho*Y[jo16]*rate_eval.C12_O16__He4_Mg24
       +rho*Y[jne20]*rate_eval.C12_Ne20__He4_Si28
       )

    jac[jhe4, jc13] = (
       -rho*Y[jhe4]*rate_eval.He4_C13__n_O16
       )

    jac[jhe4, jn13] = (
       -rho*Y[jhe4]*rate_eval.He4_N13__p_O16
       )

    jac[jhe4, jn14] = (
       -rho*Y[jhe4]*rate_eval.He4_N14__F18
       -rho*Y[jhe4]*rate_eval.He4_N14__p_O17
       +rho*Y[jn]*rate_eval.n_N14__He4_B11
       )

    jac[jhe4, jn15] = (
       -rho*Y[jhe4]*rate_eval.He4_N15__n_F18
       +rho*Y[jp]*rate_eval.p_N15__He4_C12
       )

    jac[jhe4, jo15] = (
       -rho*Y[jhe4]*rate_eval.He4_O15__Ne19
       -rho*Y[jhe4]*rate_eval.He4_O15__p_F18
       +rho*Y[jn]*rate_eval.n_O15__He4_C12
       )

    jac[jhe4, jo16] = (
       -rho*Y[jhe4]*rate_eval.He4_O16__Ne20
       -rho*Y[jhe4]*rate_eval.He4_O16__n_Ne19
       +rate_eval.O16__He4_C12
       +rho*Y[jn]*rate_eval.n_O16__He4_C13
       +rho*Y[jp]*rate_eval.p_O16__He4_N13
       +rho*Y[jc12]*rate_eval.C12_O16__He4_Mg24
       +5.00000000000000e-01*rho*2*Y[jo16]*rate_eval.O16_O16__He4_Si28
       )

    jac[jhe4, jo17] = (
       -rho*Y[jhe4]*rate_eval.He4_O17__Ne21
       -rho*Y[jhe4]*rate_eval.He4_O17__n_Ne20
       +rho*Y[jp]*rate_eval.p_O17__He4_N14
       )

    jac[jhe4, jf18] = (
       -rho*Y[jhe4]*rate_eval.He4_F18__Na22
       -rho*Y[jhe4]*rate_eval.He4_F18__p_Ne21
       +rate_eval.F18__He4_N14
       +rho*Y[jn]*rate_eval.n_F18__He4_N15
       +rho*Y[jp]*rate_eval.p_F18__He4_O15
       )

    jac[jhe4, jne19] = (
       -rho*Y[jhe4]*rate_eval.He4_Ne19__Mg23
       -rho*Y[jhe4]*rate_eval.He4_Ne19__p_Na22
       +rate_eval.Ne19__He4_O15
       +rho*Y[jn]*rate_eval.n_Ne19__He4_O16
       )

    jac[jhe4, jne20] = (
       -rho*Y[jhe4]*rate_eval.He4_Ne20__Mg24
       -rho*Y[jhe4]*rate_eval.He4_Ne20__n_Mg23
       -rho*Y[jhe4]*rate_eval.He4_Ne20__p_Na23
       -rho*Y[jhe4]*rate_eval.He4_Ne20__C12_C12
       +rate_eval.Ne20__He4_O16
       +rho*Y[jn]*rate_eval.n_Ne20__He4_O17
       +rho*Y[jc12]*rate_eval.C12_Ne20__He4_Si28
       )

    jac[jhe4, jne21] = (
       -rho*Y[jhe4]*rate_eval.He4_Ne21__Mg25
       -rho*Y[jhe4]*rate_eval.He4_Ne21__n_Mg24
       +rate_eval.Ne21__He4_O17
       +rho*Y[jp]*rate_eval.p_Ne21__He4_F18
       )

    jac[jhe4, jne22] = (
       -rho*Y[jhe4]*rate_eval.He4_Ne22__Mg26
       -rho*Y[jhe4]*rate_eval.He4_Ne22__n_Mg25
       )

    jac[jhe4, jna22] = (
       -rho*Y[jhe4]*rate_eval.He4_Na22__Al26
       -rho*Y[jhe4]*rate_eval.He4_Na22__n_Al25
       -rho*Y[jhe4]*rate_eval.He4_Na22__p_Mg25
       +rate_eval.Na22__He4_F18
       +rho*Y[jp]*rate_eval.p_Na22__He4_Ne19
       )

    jac[jhe4, jna23] = (
       -rho*Y[jhe4]*rate_eval.He4_Na23__Al27
       -rho*Y[jhe4]*rate_eval.He4_Na23__n_Al26
       -rho*Y[jhe4]*rate_eval.He4_Na23__p_Mg26
       +rho*Y[jp]*rate_eval.p_Na23__He4_Ne20
       )

    jac[jhe4, jmg23] = (
       -rho*Y[jhe4]*rate_eval.He4_Mg23__n_Si26
       -rho*Y[jhe4]*rate_eval.He4_Mg23__p_Al26
       +rate_eval.Mg23__He4_Ne19
       +rho*Y[jn]*rate_eval.n_Mg23__He4_Ne20
       )

    jac[jhe4, jmg24] = (
       -rho*Y[jhe4]*rate_eval.He4_Mg24__Si28
       -rho*Y[jhe4]*rate_eval.He4_Mg24__p_Al27
       -rho*Y[jhe4]*rate_eval.He4_Mg24__C12_O16
       +rate_eval.Mg24__He4_Ne20
       +rho*Y[jn]*rate_eval.n_Mg24__He4_Ne21
       )

    jac[jhe4, jmg25] = (
       -rho*Y[jhe4]*rate_eval.He4_Mg25__Si29
       -rho*Y[jhe4]*rate_eval.He4_Mg25__n_Si28
       +rate_eval.Mg25__He4_Ne21
       +rho*Y[jn]*rate_eval.n_Mg25__He4_Ne22
       +rho*Y[jp]*rate_eval.p_Mg25__He4_Na22
       )

    jac[jhe4, jmg26] = (
       -rho*Y[jhe4]*rate_eval.He4_Mg26__Si30
       -rho*Y[jhe4]*rate_eval.He4_Mg26__n_Si29
       +rate_eval.Mg26__He4_Ne22
       +rho*Y[jp]*rate_eval.p_Mg26__He4_Na23
       )

    jac[jhe4, jal25] = (
       -rho*Y[jhe4]*rate_eval.He4_Al25__P29
       -rho*Y[jhe4]*rate_eval.He4_Al25__p_Si28
       +rho*Y[jn]*rate_eval.n_Al25__He4_Na22
       )

    jac[jhe4, jal26] = (
       -rho*Y[jhe4]*rate_eval.He4_Al26__P30
       -rho*Y[jhe4]*rate_eval.He4_Al26__n_P29
       -rho*Y[jhe4]*rate_eval.He4_Al26__p_Si29
       +rate_eval.Al26__He4_Na22
       +rho*Y[jn]*rate_eval.n_Al26__He4_Na23
       +rho*Y[jp]*rate_eval.p_Al26__He4_Mg23
       )

    jac[jhe4, jal27] = (
       -rho*Y[jhe4]*rate_eval.He4_Al27__P31
       -rho*Y[jhe4]*rate_eval.He4_Al27__n_P30
       -rho*Y[jhe4]*rate_eval.He4_Al27__p_Si30
       +rate_eval.Al27__He4_Na23
       +rho*Y[jp]*rate_eval.p_Al27__He4_Mg24
       )

    jac[jhe4, jsi26] = (
       -rho*Y[jhe4]*rate_eval.He4_Si26__S30
       -rho*Y[jhe4]*rate_eval.He4_Si26__p_P29
       +rho*Y[jn]*rate_eval.n_Si26__He4_Mg23
       )

    jac[jhe4, jsi28] = (
       -rho*Y[jhe4]*rate_eval.He4_Si28__S32
       -rho*Y[jhe4]*rate_eval.He4_Si28__n_S31
       -rho*Y[jhe4]*rate_eval.He4_Si28__p_P31
       -rho*Y[jhe4]*rate_eval.He4_Si28__C12_Ne20
       -rho*Y[jhe4]*rate_eval.He4_Si28__O16_O16
       +rate_eval.Si28__He4_Mg24
       +rho*Y[jn]*rate_eval.n_Si28__He4_Mg25
       +rho*Y[jp]*rate_eval.p_Si28__He4_Al25
       )

    jac[jhe4, jsi29] = (
       -rho*Y[jhe4]*rate_eval.He4_Si29__S33
       -rho*Y[jhe4]*rate_eval.He4_Si29__n_S32
       +rate_eval.Si29__He4_Mg25
       +rho*Y[jn]*rate_eval.n_Si29__He4_Mg26
       +rho*Y[jp]*rate_eval.p_Si29__He4_Al26
       )

    jac[jhe4, jsi30] = (
       -rho*Y[jhe4]*rate_eval.He4_Si30__n_S33
       +rate_eval.Si30__He4_Mg26
       +rho*Y[jp]*rate_eval.p_Si30__He4_Al27
       )

    jac[jhe4, jp29] = (
       -rho*Y[jhe4]*rate_eval.He4_P29__Cl33
       -rho*Y[jhe4]*rate_eval.He4_P29__p_S32
       +rate_eval.P29__He4_Al25
       +rho*Y[jn]*rate_eval.n_P29__He4_Al26
       +rho*Y[jp]*rate_eval.p_P29__He4_Si26
       )

    jac[jhe4, jp30] = (
       -rho*Y[jhe4]*rate_eval.He4_P30__Cl34
       -rho*Y[jhe4]*rate_eval.He4_P30__n_Cl33
       -rho*Y[jhe4]*rate_eval.He4_P30__p_S33
       +rate_eval.P30__He4_Al26
       +rho*Y[jn]*rate_eval.n_P30__He4_Al27
       )

    jac[jhe4, jp31] = (
       -rho*Y[jhe4]*rate_eval.He4_P31__Cl35
       -rho*Y[jhe4]*rate_eval.He4_P31__n_Cl34
       +rate_eval.P31__He4_Al27
       +rho*Y[jp]*rate_eval.p_P31__He4_Si28
       )

    jac[jhe4, js30] = (
       -rho*Y[jhe4]*rate_eval.He4_S30__Ar34
       -rho*Y[jhe4]*rate_eval.He4_S30__p_Cl33
       +rate_eval.S30__He4_Si26
       )

    jac[jhe4, js31] = (
       -rho*Y[jhe4]*rate_eval.He4_S31__n_Ar34
       -rho*Y[jhe4]*rate_eval.He4_S31__p_Cl34
       +rho*Y[jn]*rate_eval.n_S31__He4_Si28
       )

    jac[jhe4, js32] = (
       -rho*Y[jhe4]*rate_eval.He4_S32__Ar36
       -rho*Y[jhe4]*rate_eval.He4_S32__p_Cl35
       +rate_eval.S32__He4_Si28
       +rho*Y[jn]*rate_eval.n_S32__He4_Si29
       +rho*Y[jp]*rate_eval.p_S32__He4_P29
       )

    jac[jhe4, js33] = (
       -rho*Y[jhe4]*rate_eval.He4_S33__Ar37
       -rho*Y[jhe4]*rate_eval.He4_S33__n_Ar36
       +rate_eval.S33__He4_Si29
       +rho*Y[jn]*rate_eval.n_S33__He4_Si30
       +rho*Y[jp]*rate_eval.p_S33__He4_P30
       )

    jac[jhe4, jcl33] = (
       -rho*Y[jhe4]*rate_eval.He4_Cl33__p_Ar36
       +rate_eval.Cl33__He4_P29
       +rho*Y[jn]*rate_eval.n_Cl33__He4_P30
       +rho*Y[jp]*rate_eval.p_Cl33__He4_S30
       )

    jac[jhe4, jcl34] = (
       -rho*Y[jhe4]*rate_eval.He4_Cl34__p_Ar37
       +rate_eval.Cl34__He4_P30
       +rho*Y[jn]*rate_eval.n_Cl34__He4_P31
       +rho*Y[jp]*rate_eval.p_Cl34__He4_S31
       )

    jac[jhe4, jcl35] = (
       -rho*Y[jhe4]*rate_eval.He4_Cl35__K39
       -rho*Y[jhe4]*rate_eval.He4_Cl35__p_Ar38
       +rate_eval.Cl35__He4_P31
       +rho*Y[jp]*rate_eval.p_Cl35__He4_S32
       )

    jac[jhe4, jar34] = (
       +rate_eval.Ar34__He4_S30
       +rho*Y[jn]*rate_eval.n_Ar34__He4_S31
       )

    jac[jhe4, jar36] = (
       -rho*Y[jhe4]*rate_eval.He4_Ar36__Ca40
       -rho*Y[jhe4]*rate_eval.He4_Ar36__p_K39
       +rate_eval.Ar36__He4_S32
       +rho*Y[jn]*rate_eval.n_Ar36__He4_S33
       +rho*Y[jp]*rate_eval.p_Ar36__He4_Cl33
       )

    jac[jhe4, jar37] = (
       -rho*Y[jhe4]*rate_eval.He4_Ar37__n_Ca40
       +rate_eval.Ar37__He4_S33
       +rho*Y[jp]*rate_eval.p_Ar37__He4_Cl34
       )

    jac[jhe4, jar38] = (
       +rho*Y[jp]*rate_eval.p_Ar38__He4_Cl35
       )

    jac[jhe4, jk39] = (
       -rho*Y[jhe4]*rate_eval.He4_K39__Sc43
       +rate_eval.K39__He4_Cl35
       +rho*Y[jp]*rate_eval.p_K39__He4_Ar36
       )

    jac[jhe4, jca40] = (
       -rho*Y[jhe4]*rate_eval.He4_Ca40__Ti44
       -rho*Y[jhe4]*rate_eval.He4_Ca40__p_Sc43
       +rate_eval.Ca40__He4_Ar36
       +rho*Y[jn]*rate_eval.n_Ca40__He4_Ar37
       )

    jac[jhe4, jsc43] = (
       -rho*Y[jhe4]*rate_eval.He4_Sc43__V47
       +rate_eval.Sc43__He4_K39
       +rho*Y[jp]*rate_eval.p_Sc43__He4_Ca40
       )

    jac[jhe4, jti44] = (
       -rho*Y[jhe4]*rate_eval.He4_Ti44__Cr48
       -rho*Y[jhe4]*rate_eval.He4_Ti44__p_V47
       +rate_eval.Ti44__He4_Ca40
       )

    jac[jhe4, jv47] = (
       -rho*Y[jhe4]*rate_eval.He4_V47__Mn51
       +rate_eval.V47__He4_Sc43
       +rho*Y[jp]*rate_eval.p_V47__He4_Ti44
       )

    jac[jhe4, jcr48] = (
       -rho*Y[jhe4]*rate_eval.He4_Cr48__Fe52
       -rho*Y[jhe4]*rate_eval.He4_Cr48__p_Mn51
       +rate_eval.Cr48__He4_Ti44
       )

    jac[jhe4, jmn51] = (
       -rho*Y[jhe4]*rate_eval.He4_Mn51__Co55
       +rate_eval.Mn51__He4_V47
       +rho*Y[jp]*rate_eval.p_Mn51__He4_Cr48
       )

    jac[jhe4, jfe52] = (
       -rho*Y[jhe4]*rate_eval.He4_Fe52__Ni56
       -rho*Y[jhe4]*rate_eval.He4_Fe52__p_Co55
       +rate_eval.Fe52__He4_Cr48
       )

    jac[jhe4, jfe55] = (
       -rho*Y[jhe4]*rate_eval.He4_Fe55__Ni59
       -rho*Y[jhe4]*rate_eval.He4_Fe55__n_Ni58
       )

    jac[jhe4, jco55] = (
       -rho*Y[jhe4]*rate_eval.He4_Co55__p_Ni58
       +rate_eval.Co55__He4_Mn51
       +rho*Y[jp]*rate_eval.p_Co55__He4_Fe52
       )

    jac[jhe4, jni56] = (
       +rate_eval.Ni56__He4_Fe52
       )

    jac[jhe4, jni58] = (
       +rho*Y[jn]*rate_eval.n_Ni58__He4_Fe55
       +rho*Y[jp]*rate_eval.p_Ni58__He4_Co55
       )

    jac[jhe4, jni59] = (
       +rate_eval.Ni59__He4_Fe55
       )

    jac[jb11, jn] = (
       +rho*Y[jn14]*rate_eval.n_N14__He4_B11
       )

    jac[jb11, jp] = (
       -rho*Y[jb11]*rate_eval.p_B11__C12
       -rho*Y[jb11]*rate_eval.p_B11__He4_He4_He4
       )

    jac[jb11, jhe4] = (
       -rho*Y[jb11]*rate_eval.He4_B11__n_N14
       +1.66666666666667e-01*rho**2*3*Y[jhe4]**2*rate_eval.He4_He4_He4__p_B11
       )

    jac[jb11, jb11] = (
       -rho*Y[jp]*rate_eval.p_B11__C12
       -rho*Y[jhe4]*rate_eval.He4_B11__n_N14
       -rho*Y[jp]*rate_eval.p_B11__He4_He4_He4
       )

    jac[jb11, jc12] = (
       +rate_eval.C12__p_B11
       )

    jac[jb11, jn14] = (
       +rho*Y[jn]*rate_eval.n_N14__He4_B11
       )

    jac[jc12, jn] = (
       -rho*Y[jc12]*rate_eval.n_C12__C13
       +rho*Y[jo15]*rate_eval.n_O15__He4_C12
       +2*rho*Y[jmg23]*rate_eval.n_Mg23__C12_C12
       +rho*Y[js31]*rate_eval.n_S31__C12_Ne20
       )

    jac[jc12, jp] = (
       -rho*Y[jc12]*rate_eval.p_C12__N13
       +rho*Y[jb11]*rate_eval.p_B11__C12
       +rho*Y[jn15]*rate_eval.p_N15__He4_C12
       +2*rho*Y[jna23]*rate_eval.p_Na23__C12_C12
       +rho*Y[jal27]*rate_eval.p_Al27__C12_O16
       +rho*Y[jp31]*rate_eval.p_P31__C12_Ne20
       )

    jac[jc12, jhe4] = (
       -rho*Y[jc12]*rate_eval.He4_C12__O16
       -rho*Y[jc12]*rate_eval.He4_C12__n_O15
       -rho*Y[jc12]*rate_eval.He4_C12__p_N15
       +2*rho*Y[jne20]*rate_eval.He4_Ne20__C12_C12
       +rho*Y[jmg24]*rate_eval.He4_Mg24__C12_O16
       +rho*Y[jsi28]*rate_eval.He4_Si28__C12_Ne20
       +1.66666666666667e-01*rho**2*3*Y[jhe4]**2*rate_eval.He4_He4_He4__C12
       )

    jac[jc12, jb11] = (
       +rho*Y[jp]*rate_eval.p_B11__C12
       )

    jac[jc12, jc12] = (
       -rate_eval.C12__p_B11
       -rate_eval.C12__He4_He4_He4
       -rho*Y[jn]*rate_eval.n_C12__C13
       -rho*Y[jp]*rate_eval.p_C12__N13
       -rho*Y[jhe4]*rate_eval.He4_C12__O16
       -rho*Y[jhe4]*rate_eval.He4_C12__n_O15
       -rho*Y[jhe4]*rate_eval.He4_C12__p_N15
       -2*5.00000000000000e-01*rho*2*Y[jc12]*rate_eval.C12_C12__n_Mg23
       -2*5.00000000000000e-01*rho*2*Y[jc12]*rate_eval.C12_C12__p_Na23
       -2*5.00000000000000e-01*rho*2*Y[jc12]*rate_eval.C12_C12__He4_Ne20
       -rho*Y[jo16]*rate_eval.C12_O16__p_Al27
       -rho*Y[jo16]*rate_eval.C12_O16__He4_Mg24
       -rho*Y[jne20]*rate_eval.C12_Ne20__n_S31
       -rho*Y[jne20]*rate_eval.C12_Ne20__p_P31
       -rho*Y[jne20]*rate_eval.C12_Ne20__He4_Si28
       )

    jac[jc12, jc13] = (
       +rate_eval.C13__n_C12
       )

    jac[jc12, jn13] = (
       +rate_eval.N13__p_C12
       )

    jac[jc12, jn15] = (
       +rho*Y[jp]*rate_eval.p_N15__He4_C12
       )

    jac[jc12, jo15] = (
       +rho*Y[jn]*rate_eval.n_O15__He4_C12
       )

    jac[jc12, jo16] = (
       -rho*Y[jc12]*rate_eval.C12_O16__p_Al27
       -rho*Y[jc12]*rate_eval.C12_O16__He4_Mg24
       +rate_eval.O16__He4_C12
       )

    jac[jc12, jne20] = (
       -rho*Y[jc12]*rate_eval.C12_Ne20__n_S31
       -rho*Y[jc12]*rate_eval.C12_Ne20__p_P31
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

    jac[jc12, jsi28] = (
       +rho*Y[jhe4]*rate_eval.He4_Si28__C12_Ne20
       )

    jac[jc12, jp31] = (
       +rho*Y[jp]*rate_eval.p_P31__C12_Ne20
       )

    jac[jc12, js31] = (
       +rho*Y[jn]*rate_eval.n_S31__C12_Ne20
       )

    jac[jc13, jn] = (
       +rho*Y[jc12]*rate_eval.n_C12__C13
       +rho*Y[jn13]*rate_eval.n_N13__p_C13
       +rho*Y[jo16]*rate_eval.n_O16__He4_C13
       )

    jac[jc13, jp] = (
       -rho*Y[jc13]*rate_eval.p_C13__N14
       -rho*Y[jc13]*rate_eval.p_C13__n_N13
       )

    jac[jc13, jhe4] = (
       -rho*Y[jc13]*rate_eval.He4_C13__n_O16
       )

    jac[jc13, jc12] = (
       +rho*Y[jn]*rate_eval.n_C12__C13
       )

    jac[jc13, jc13] = (
       -rate_eval.C13__n_C12
       -rho*Y[jp]*rate_eval.p_C13__N14
       -rho*Y[jp]*rate_eval.p_C13__n_N13
       -rho*Y[jhe4]*rate_eval.He4_C13__n_O16
       )

    jac[jc13, jn13] = (
       +rate_eval.N13__C13__weak__wc12
       +rho*Y[jn]*rate_eval.n_N13__p_C13
       )

    jac[jc13, jn14] = (
       +rate_eval.N14__p_C13
       )

    jac[jc13, jo16] = (
       +rho*Y[jn]*rate_eval.n_O16__He4_C13
       )

    jac[jn13, jn] = (
       -rho*Y[jn13]*rate_eval.n_N13__N14
       -rho*Y[jn13]*rate_eval.n_N13__p_C13
       )

    jac[jn13, jp] = (
       +rho*Y[jc12]*rate_eval.p_C12__N13
       +rho*Y[jc13]*rate_eval.p_C13__n_N13
       +rho*Y[jo16]*rate_eval.p_O16__He4_N13
       )

    jac[jn13, jhe4] = (
       -rho*Y[jn13]*rate_eval.He4_N13__p_O16
       )

    jac[jn13, jc12] = (
       +rho*Y[jp]*rate_eval.p_C12__N13
       )

    jac[jn13, jc13] = (
       +rho*Y[jp]*rate_eval.p_C13__n_N13
       )

    jac[jn13, jn13] = (
       -rate_eval.N13__C13__weak__wc12
       -rate_eval.N13__p_C12
       -rho*Y[jn]*rate_eval.n_N13__N14
       -rho*Y[jn]*rate_eval.n_N13__p_C13
       -rho*Y[jhe4]*rate_eval.He4_N13__p_O16
       )

    jac[jn13, jn14] = (
       +rate_eval.N14__n_N13
       )

    jac[jn13, jo16] = (
       +rho*Y[jp]*rate_eval.p_O16__He4_N13
       )

    jac[jn14, jn] = (
       -rho*Y[jn14]*rate_eval.n_N14__N15
       -rho*Y[jn14]*rate_eval.n_N14__He4_B11
       +rho*Y[jn13]*rate_eval.n_N13__N14
       )

    jac[jn14, jp] = (
       -rho*Y[jn14]*rate_eval.p_N14__O15
       +rho*Y[jc13]*rate_eval.p_C13__N14
       +rho*Y[jo17]*rate_eval.p_O17__He4_N14
       )

    jac[jn14, jhe4] = (
       -rho*Y[jn14]*rate_eval.He4_N14__F18
       -rho*Y[jn14]*rate_eval.He4_N14__p_O17
       +rho*Y[jb11]*rate_eval.He4_B11__n_N14
       )

    jac[jn14, jb11] = (
       +rho*Y[jhe4]*rate_eval.He4_B11__n_N14
       )

    jac[jn14, jc13] = (
       +rho*Y[jp]*rate_eval.p_C13__N14
       )

    jac[jn14, jn13] = (
       +rho*Y[jn]*rate_eval.n_N13__N14
       )

    jac[jn14, jn14] = (
       -rate_eval.N14__n_N13
       -rate_eval.N14__p_C13
       -rho*Y[jn]*rate_eval.n_N14__N15
       -rho*Y[jp]*rate_eval.p_N14__O15
       -rho*Y[jhe4]*rate_eval.He4_N14__F18
       -rho*Y[jn]*rate_eval.n_N14__He4_B11
       -rho*Y[jhe4]*rate_eval.He4_N14__p_O17
       )

    jac[jn14, jn15] = (
       +rate_eval.N15__n_N14
       )

    jac[jn14, jo15] = (
       +rate_eval.O15__p_N14
       )

    jac[jn14, jo17] = (
       +rho*Y[jp]*rate_eval.p_O17__He4_N14
       )

    jac[jn14, jf18] = (
       +rate_eval.F18__He4_N14
       )

    jac[jn15, jn] = (
       +rho*Y[jn14]*rate_eval.n_N14__N15
       +rho*Y[jo15]*rate_eval.n_O15__p_N15
       +rho*Y[jf18]*rate_eval.n_F18__He4_N15
       )

    jac[jn15, jp] = (
       -rho*Y[jn15]*rate_eval.p_N15__O16
       -rho*Y[jn15]*rate_eval.p_N15__n_O15
       -rho*Y[jn15]*rate_eval.p_N15__He4_C12
       )

    jac[jn15, jhe4] = (
       -rho*Y[jn15]*rate_eval.He4_N15__n_F18
       +rho*Y[jc12]*rate_eval.He4_C12__p_N15
       )

    jac[jn15, jc12] = (
       +rho*Y[jhe4]*rate_eval.He4_C12__p_N15
       )

    jac[jn15, jn14] = (
       +rho*Y[jn]*rate_eval.n_N14__N15
       )

    jac[jn15, jn15] = (
       -rate_eval.N15__n_N14
       -rho*Y[jp]*rate_eval.p_N15__O16
       -rho*Y[jp]*rate_eval.p_N15__n_O15
       -rho*Y[jp]*rate_eval.p_N15__He4_C12
       -rho*Y[jhe4]*rate_eval.He4_N15__n_F18
       )

    jac[jn15, jo15] = (
       +rate_eval.O15__N15__weak__wc12
       +rho*Y[jn]*rate_eval.n_O15__p_N15
       )

    jac[jn15, jo16] = (
       +rate_eval.O16__p_N15
       )

    jac[jn15, jf18] = (
       +rho*Y[jn]*rate_eval.n_F18__He4_N15
       )

    jac[jo15, jn] = (
       -rho*Y[jo15]*rate_eval.n_O15__O16
       -rho*Y[jo15]*rate_eval.n_O15__p_N15
       -rho*Y[jo15]*rate_eval.n_O15__He4_C12
       )

    jac[jo15, jp] = (
       +rho*Y[jn14]*rate_eval.p_N14__O15
       +rho*Y[jn15]*rate_eval.p_N15__n_O15
       +rho*Y[jf18]*rate_eval.p_F18__He4_O15
       )

    jac[jo15, jhe4] = (
       -rho*Y[jo15]*rate_eval.He4_O15__Ne19
       -rho*Y[jo15]*rate_eval.He4_O15__p_F18
       +rho*Y[jc12]*rate_eval.He4_C12__n_O15
       )

    jac[jo15, jc12] = (
       +rho*Y[jhe4]*rate_eval.He4_C12__n_O15
       )

    jac[jo15, jn14] = (
       +rho*Y[jp]*rate_eval.p_N14__O15
       )

    jac[jo15, jn15] = (
       +rho*Y[jp]*rate_eval.p_N15__n_O15
       )

    jac[jo15, jo15] = (
       -rate_eval.O15__N15__weak__wc12
       -rate_eval.O15__p_N14
       -rho*Y[jn]*rate_eval.n_O15__O16
       -rho*Y[jhe4]*rate_eval.He4_O15__Ne19
       -rho*Y[jn]*rate_eval.n_O15__p_N15
       -rho*Y[jn]*rate_eval.n_O15__He4_C12
       -rho*Y[jhe4]*rate_eval.He4_O15__p_F18
       )

    jac[jo15, jo16] = (
       +rate_eval.O16__n_O15
       )

    jac[jo15, jf18] = (
       +rho*Y[jp]*rate_eval.p_F18__He4_O15
       )

    jac[jo15, jne19] = (
       +rate_eval.Ne19__He4_O15
       )

    jac[jo16, jn] = (
       -rho*Y[jo16]*rate_eval.n_O16__O17
       -rho*Y[jo16]*rate_eval.n_O16__He4_C13
       +rho*Y[jo15]*rate_eval.n_O15__O16
       +rho*Y[jne19]*rate_eval.n_Ne19__He4_O16
       +2*rho*Y[js31]*rate_eval.n_S31__O16_O16
       )

    jac[jo16, jp] = (
       -rho*Y[jo16]*rate_eval.p_O16__He4_N13
       +rho*Y[jn15]*rate_eval.p_N15__O16
       +rho*Y[jal27]*rate_eval.p_Al27__C12_O16
       +2*rho*Y[jp31]*rate_eval.p_P31__O16_O16
       )

    jac[jo16, jhe4] = (
       -rho*Y[jo16]*rate_eval.He4_O16__Ne20
       -rho*Y[jo16]*rate_eval.He4_O16__n_Ne19
       +rho*Y[jc12]*rate_eval.He4_C12__O16
       +rho*Y[jc13]*rate_eval.He4_C13__n_O16
       +rho*Y[jn13]*rate_eval.He4_N13__p_O16
       +rho*Y[jmg24]*rate_eval.He4_Mg24__C12_O16
       +2*rho*Y[jsi28]*rate_eval.He4_Si28__O16_O16
       )

    jac[jo16, jc12] = (
       -rho*Y[jo16]*rate_eval.C12_O16__p_Al27
       -rho*Y[jo16]*rate_eval.C12_O16__He4_Mg24
       +rho*Y[jhe4]*rate_eval.He4_C12__O16
       )

    jac[jo16, jc13] = (
       +rho*Y[jhe4]*rate_eval.He4_C13__n_O16
       )

    jac[jo16, jn13] = (
       +rho*Y[jhe4]*rate_eval.He4_N13__p_O16
       )

    jac[jo16, jn15] = (
       +rho*Y[jp]*rate_eval.p_N15__O16
       )

    jac[jo16, jo15] = (
       +rho*Y[jn]*rate_eval.n_O15__O16
       )

    jac[jo16, jo16] = (
       -rate_eval.O16__n_O15
       -rate_eval.O16__p_N15
       -rate_eval.O16__He4_C12
       -rho*Y[jn]*rate_eval.n_O16__O17
       -rho*Y[jhe4]*rate_eval.He4_O16__Ne20
       -rho*Y[jn]*rate_eval.n_O16__He4_C13
       -rho*Y[jp]*rate_eval.p_O16__He4_N13
       -rho*Y[jhe4]*rate_eval.He4_O16__n_Ne19
       -rho*Y[jc12]*rate_eval.C12_O16__p_Al27
       -rho*Y[jc12]*rate_eval.C12_O16__He4_Mg24
       -2*5.00000000000000e-01*rho*2*Y[jo16]*rate_eval.O16_O16__n_S31
       -2*5.00000000000000e-01*rho*2*Y[jo16]*rate_eval.O16_O16__p_P31
       -2*5.00000000000000e-01*rho*2*Y[jo16]*rate_eval.O16_O16__He4_Si28
       )

    jac[jo16, jo17] = (
       +rate_eval.O17__n_O16
       )

    jac[jo16, jne19] = (
       +rho*Y[jn]*rate_eval.n_Ne19__He4_O16
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

    jac[jo16, jsi28] = (
       +2*rho*Y[jhe4]*rate_eval.He4_Si28__O16_O16
       )

    jac[jo16, jp31] = (
       +2*rho*Y[jp]*rate_eval.p_P31__O16_O16
       )

    jac[jo16, js31] = (
       +2*rho*Y[jn]*rate_eval.n_S31__O16_O16
       )

    jac[jo17, jn] = (
       +rho*Y[jo16]*rate_eval.n_O16__O17
       +rho*Y[jne20]*rate_eval.n_Ne20__He4_O17
       )

    jac[jo17, jp] = (
       -rho*Y[jo17]*rate_eval.p_O17__F18
       -rho*Y[jo17]*rate_eval.p_O17__He4_N14
       )

    jac[jo17, jhe4] = (
       -rho*Y[jo17]*rate_eval.He4_O17__Ne21
       -rho*Y[jo17]*rate_eval.He4_O17__n_Ne20
       +rho*Y[jn14]*rate_eval.He4_N14__p_O17
       )

    jac[jo17, jn14] = (
       +rho*Y[jhe4]*rate_eval.He4_N14__p_O17
       )

    jac[jo17, jo16] = (
       +rho*Y[jn]*rate_eval.n_O16__O17
       )

    jac[jo17, jo17] = (
       -rate_eval.O17__n_O16
       -rho*Y[jp]*rate_eval.p_O17__F18
       -rho*Y[jhe4]*rate_eval.He4_O17__Ne21
       -rho*Y[jp]*rate_eval.p_O17__He4_N14
       -rho*Y[jhe4]*rate_eval.He4_O17__n_Ne20
       )

    jac[jo17, jf18] = (
       +rate_eval.F18__p_O17
       )

    jac[jo17, jne20] = (
       +rho*Y[jn]*rate_eval.n_Ne20__He4_O17
       )

    jac[jo17, jne21] = (
       +rate_eval.Ne21__He4_O17
       )

    jac[jf18, jn] = (
       -rho*Y[jf18]*rate_eval.n_F18__He4_N15
       )

    jac[jf18, jp] = (
       -rho*Y[jf18]*rate_eval.p_F18__Ne19
       -rho*Y[jf18]*rate_eval.p_F18__He4_O15
       +rho*Y[jo17]*rate_eval.p_O17__F18
       +rho*Y[jne21]*rate_eval.p_Ne21__He4_F18
       )

    jac[jf18, jhe4] = (
       -rho*Y[jf18]*rate_eval.He4_F18__Na22
       -rho*Y[jf18]*rate_eval.He4_F18__p_Ne21
       +rho*Y[jn14]*rate_eval.He4_N14__F18
       +rho*Y[jn15]*rate_eval.He4_N15__n_F18
       +rho*Y[jo15]*rate_eval.He4_O15__p_F18
       )

    jac[jf18, jn14] = (
       +rho*Y[jhe4]*rate_eval.He4_N14__F18
       )

    jac[jf18, jn15] = (
       +rho*Y[jhe4]*rate_eval.He4_N15__n_F18
       )

    jac[jf18, jo15] = (
       +rho*Y[jhe4]*rate_eval.He4_O15__p_F18
       )

    jac[jf18, jo17] = (
       +rho*Y[jp]*rate_eval.p_O17__F18
       )

    jac[jf18, jf18] = (
       -rate_eval.F18__p_O17
       -rate_eval.F18__He4_N14
       -rho*Y[jp]*rate_eval.p_F18__Ne19
       -rho*Y[jhe4]*rate_eval.He4_F18__Na22
       -rho*Y[jn]*rate_eval.n_F18__He4_N15
       -rho*Y[jp]*rate_eval.p_F18__He4_O15
       -rho*Y[jhe4]*rate_eval.He4_F18__p_Ne21
       )

    jac[jf18, jne19] = (
       +rate_eval.Ne19__p_F18
       )

    jac[jf18, jne21] = (
       +rho*Y[jp]*rate_eval.p_Ne21__He4_F18
       )

    jac[jf18, jna22] = (
       +rate_eval.Na22__He4_F18
       )

    jac[jne19, jn] = (
       -rho*Y[jne19]*rate_eval.n_Ne19__Ne20
       -rho*Y[jne19]*rate_eval.n_Ne19__He4_O16
       )

    jac[jne19, jp] = (
       +rho*Y[jf18]*rate_eval.p_F18__Ne19
       +rho*Y[jna22]*rate_eval.p_Na22__He4_Ne19
       )

    jac[jne19, jhe4] = (
       -rho*Y[jne19]*rate_eval.He4_Ne19__Mg23
       -rho*Y[jne19]*rate_eval.He4_Ne19__p_Na22
       +rho*Y[jo15]*rate_eval.He4_O15__Ne19
       +rho*Y[jo16]*rate_eval.He4_O16__n_Ne19
       )

    jac[jne19, jo15] = (
       +rho*Y[jhe4]*rate_eval.He4_O15__Ne19
       )

    jac[jne19, jo16] = (
       +rho*Y[jhe4]*rate_eval.He4_O16__n_Ne19
       )

    jac[jne19, jf18] = (
       +rho*Y[jp]*rate_eval.p_F18__Ne19
       )

    jac[jne19, jne19] = (
       -rate_eval.Ne19__p_F18
       -rate_eval.Ne19__He4_O15
       -rho*Y[jn]*rate_eval.n_Ne19__Ne20
       -rho*Y[jhe4]*rate_eval.He4_Ne19__Mg23
       -rho*Y[jn]*rate_eval.n_Ne19__He4_O16
       -rho*Y[jhe4]*rate_eval.He4_Ne19__p_Na22
       )

    jac[jne19, jne20] = (
       +rate_eval.Ne20__n_Ne19
       )

    jac[jne19, jna22] = (
       +rho*Y[jp]*rate_eval.p_Na22__He4_Ne19
       )

    jac[jne19, jmg23] = (
       +rate_eval.Mg23__He4_Ne19
       )

    jac[jne20, jn] = (
       -rho*Y[jne20]*rate_eval.n_Ne20__Ne21
       -rho*Y[jne20]*rate_eval.n_Ne20__He4_O17
       +rho*Y[jne19]*rate_eval.n_Ne19__Ne20
       +rho*Y[jmg23]*rate_eval.n_Mg23__He4_Ne20
       +rho*Y[js31]*rate_eval.n_S31__C12_Ne20
       )

    jac[jne20, jp] = (
       +rho*Y[jna23]*rate_eval.p_Na23__He4_Ne20
       +rho*Y[jp31]*rate_eval.p_P31__C12_Ne20
       )

    jac[jne20, jhe4] = (
       -rho*Y[jne20]*rate_eval.He4_Ne20__Mg24
       -rho*Y[jne20]*rate_eval.He4_Ne20__n_Mg23
       -rho*Y[jne20]*rate_eval.He4_Ne20__p_Na23
       -rho*Y[jne20]*rate_eval.He4_Ne20__C12_C12
       +rho*Y[jo16]*rate_eval.He4_O16__Ne20
       +rho*Y[jo17]*rate_eval.He4_O17__n_Ne20
       +rho*Y[jsi28]*rate_eval.He4_Si28__C12_Ne20
       )

    jac[jne20, jc12] = (
       -rho*Y[jne20]*rate_eval.C12_Ne20__n_S31
       -rho*Y[jne20]*rate_eval.C12_Ne20__p_P31
       -rho*Y[jne20]*rate_eval.C12_Ne20__He4_Si28
       +5.00000000000000e-01*rho*2*Y[jc12]*rate_eval.C12_C12__He4_Ne20
       )

    jac[jne20, jo16] = (
       +rho*Y[jhe4]*rate_eval.He4_O16__Ne20
       )

    jac[jne20, jo17] = (
       +rho*Y[jhe4]*rate_eval.He4_O17__n_Ne20
       )

    jac[jne20, jne19] = (
       +rho*Y[jn]*rate_eval.n_Ne19__Ne20
       )

    jac[jne20, jne20] = (
       -rate_eval.Ne20__n_Ne19
       -rate_eval.Ne20__He4_O16
       -rho*Y[jn]*rate_eval.n_Ne20__Ne21
       -rho*Y[jhe4]*rate_eval.He4_Ne20__Mg24
       -rho*Y[jn]*rate_eval.n_Ne20__He4_O17
       -rho*Y[jhe4]*rate_eval.He4_Ne20__n_Mg23
       -rho*Y[jhe4]*rate_eval.He4_Ne20__p_Na23
       -rho*Y[jhe4]*rate_eval.He4_Ne20__C12_C12
       -rho*Y[jc12]*rate_eval.C12_Ne20__n_S31
       -rho*Y[jc12]*rate_eval.C12_Ne20__p_P31
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

    jac[jne20, jp31] = (
       +rho*Y[jp]*rate_eval.p_P31__C12_Ne20
       )

    jac[jne20, js31] = (
       +rho*Y[jn]*rate_eval.n_S31__C12_Ne20
       )

    jac[jne21, jn] = (
       -rho*Y[jne21]*rate_eval.n_Ne21__Ne22
       +rho*Y[jne20]*rate_eval.n_Ne20__Ne21
       +rho*Y[jmg24]*rate_eval.n_Mg24__He4_Ne21
       )

    jac[jne21, jp] = (
       -rho*Y[jne21]*rate_eval.p_Ne21__Na22
       -rho*Y[jne21]*rate_eval.p_Ne21__He4_F18
       )

    jac[jne21, jhe4] = (
       -rho*Y[jne21]*rate_eval.He4_Ne21__Mg25
       -rho*Y[jne21]*rate_eval.He4_Ne21__n_Mg24
       +rho*Y[jo17]*rate_eval.He4_O17__Ne21
       +rho*Y[jf18]*rate_eval.He4_F18__p_Ne21
       )

    jac[jne21, jo17] = (
       +rho*Y[jhe4]*rate_eval.He4_O17__Ne21
       )

    jac[jne21, jf18] = (
       +rho*Y[jhe4]*rate_eval.He4_F18__p_Ne21
       )

    jac[jne21, jne20] = (
       +rho*Y[jn]*rate_eval.n_Ne20__Ne21
       )

    jac[jne21, jne21] = (
       -rate_eval.Ne21__n_Ne20
       -rate_eval.Ne21__He4_O17
       -rho*Y[jn]*rate_eval.n_Ne21__Ne22
       -rho*Y[jp]*rate_eval.p_Ne21__Na22
       -rho*Y[jhe4]*rate_eval.He4_Ne21__Mg25
       -rho*Y[jp]*rate_eval.p_Ne21__He4_F18
       -rho*Y[jhe4]*rate_eval.He4_Ne21__n_Mg24
       )

    jac[jne21, jne22] = (
       +rate_eval.Ne22__n_Ne21
       )

    jac[jne21, jna22] = (
       +rate_eval.Na22__p_Ne21
       )

    jac[jne21, jmg24] = (
       +rho*Y[jn]*rate_eval.n_Mg24__He4_Ne21
       )

    jac[jne21, jmg25] = (
       +rate_eval.Mg25__He4_Ne21
       )

    jac[jne22, jn] = (
       +rho*Y[jne21]*rate_eval.n_Ne21__Ne22
       +rho*Y[jna22]*rate_eval.n_Na22__p_Ne22
       +rho*Y[jmg25]*rate_eval.n_Mg25__He4_Ne22
       )

    jac[jne22, jp] = (
       -rho*Y[jne22]*rate_eval.p_Ne22__Na23
       -rho*Y[jne22]*rate_eval.p_Ne22__n_Na22
       )

    jac[jne22, jhe4] = (
       -rho*Y[jne22]*rate_eval.He4_Ne22__Mg26
       -rho*Y[jne22]*rate_eval.He4_Ne22__n_Mg25
       )

    jac[jne22, jne21] = (
       +rho*Y[jn]*rate_eval.n_Ne21__Ne22
       )

    jac[jne22, jne22] = (
       -rate_eval.Ne22__n_Ne21
       -rho*Y[jp]*rate_eval.p_Ne22__Na23
       -rho*Y[jhe4]*rate_eval.He4_Ne22__Mg26
       -rho*Y[jp]*rate_eval.p_Ne22__n_Na22
       -rho*Y[jhe4]*rate_eval.He4_Ne22__n_Mg25
       )

    jac[jne22, jna22] = (
       +rate_eval.Na22__Ne22__weak__wc12
       +rho*Y[jn]*rate_eval.n_Na22__p_Ne22
       )

    jac[jne22, jna23] = (
       +rate_eval.Na23__p_Ne22
       )

    jac[jne22, jmg25] = (
       +rho*Y[jn]*rate_eval.n_Mg25__He4_Ne22
       )

    jac[jne22, jmg26] = (
       +rate_eval.Mg26__He4_Ne22
       )

    jac[jna22, jn] = (
       -rho*Y[jna22]*rate_eval.n_Na22__Na23
       -rho*Y[jna22]*rate_eval.n_Na22__p_Ne22
       +rho*Y[jal25]*rate_eval.n_Al25__He4_Na22
       )

    jac[jna22, jp] = (
       -rho*Y[jna22]*rate_eval.p_Na22__Mg23
       -rho*Y[jna22]*rate_eval.p_Na22__He4_Ne19
       +rho*Y[jne21]*rate_eval.p_Ne21__Na22
       +rho*Y[jne22]*rate_eval.p_Ne22__n_Na22
       +rho*Y[jmg25]*rate_eval.p_Mg25__He4_Na22
       )

    jac[jna22, jhe4] = (
       -rho*Y[jna22]*rate_eval.He4_Na22__Al26
       -rho*Y[jna22]*rate_eval.He4_Na22__n_Al25
       -rho*Y[jna22]*rate_eval.He4_Na22__p_Mg25
       +rho*Y[jf18]*rate_eval.He4_F18__Na22
       +rho*Y[jne19]*rate_eval.He4_Ne19__p_Na22
       )

    jac[jna22, jf18] = (
       +rho*Y[jhe4]*rate_eval.He4_F18__Na22
       )

    jac[jna22, jne19] = (
       +rho*Y[jhe4]*rate_eval.He4_Ne19__p_Na22
       )

    jac[jna22, jne21] = (
       +rho*Y[jp]*rate_eval.p_Ne21__Na22
       )

    jac[jna22, jne22] = (
       +rho*Y[jp]*rate_eval.p_Ne22__n_Na22
       )

    jac[jna22, jna22] = (
       -rate_eval.Na22__Ne22__weak__wc12
       -rate_eval.Na22__p_Ne21
       -rate_eval.Na22__He4_F18
       -rho*Y[jn]*rate_eval.n_Na22__Na23
       -rho*Y[jp]*rate_eval.p_Na22__Mg23
       -rho*Y[jhe4]*rate_eval.He4_Na22__Al26
       -rho*Y[jn]*rate_eval.n_Na22__p_Ne22
       -rho*Y[jp]*rate_eval.p_Na22__He4_Ne19
       -rho*Y[jhe4]*rate_eval.He4_Na22__n_Al25
       -rho*Y[jhe4]*rate_eval.He4_Na22__p_Mg25
       )

    jac[jna22, jna23] = (
       +rate_eval.Na23__n_Na22
       )

    jac[jna22, jmg23] = (
       +rate_eval.Mg23__p_Na22
       )

    jac[jna22, jmg25] = (
       +rho*Y[jp]*rate_eval.p_Mg25__He4_Na22
       )

    jac[jna22, jal25] = (
       +rho*Y[jn]*rate_eval.n_Al25__He4_Na22
       )

    jac[jna22, jal26] = (
       +rate_eval.Al26__He4_Na22
       )

    jac[jna23, jn] = (
       +rho*Y[jna22]*rate_eval.n_Na22__Na23
       +rho*Y[jmg23]*rate_eval.n_Mg23__p_Na23
       +rho*Y[jal26]*rate_eval.n_Al26__He4_Na23
       )

    jac[jna23, jp] = (
       -rho*Y[jna23]*rate_eval.p_Na23__Mg24
       -rho*Y[jna23]*rate_eval.p_Na23__n_Mg23
       -rho*Y[jna23]*rate_eval.p_Na23__He4_Ne20
       -rho*Y[jna23]*rate_eval.p_Na23__C12_C12
       +rho*Y[jne22]*rate_eval.p_Ne22__Na23
       +rho*Y[jmg26]*rate_eval.p_Mg26__He4_Na23
       )

    jac[jna23, jhe4] = (
       -rho*Y[jna23]*rate_eval.He4_Na23__Al27
       -rho*Y[jna23]*rate_eval.He4_Na23__n_Al26
       -rho*Y[jna23]*rate_eval.He4_Na23__p_Mg26
       +rho*Y[jne20]*rate_eval.He4_Ne20__p_Na23
       )

    jac[jna23, jc12] = (
       +5.00000000000000e-01*rho*2*Y[jc12]*rate_eval.C12_C12__p_Na23
       )

    jac[jna23, jne20] = (
       +rho*Y[jhe4]*rate_eval.He4_Ne20__p_Na23
       )

    jac[jna23, jne22] = (
       +rho*Y[jp]*rate_eval.p_Ne22__Na23
       )

    jac[jna23, jna22] = (
       +rho*Y[jn]*rate_eval.n_Na22__Na23
       )

    jac[jna23, jna23] = (
       -rate_eval.Na23__n_Na22
       -rate_eval.Na23__p_Ne22
       -rho*Y[jp]*rate_eval.p_Na23__Mg24
       -rho*Y[jhe4]*rate_eval.He4_Na23__Al27
       -rho*Y[jp]*rate_eval.p_Na23__n_Mg23
       -rho*Y[jp]*rate_eval.p_Na23__He4_Ne20
       -rho*Y[jp]*rate_eval.p_Na23__C12_C12
       -rho*Y[jhe4]*rate_eval.He4_Na23__n_Al26
       -rho*Y[jhe4]*rate_eval.He4_Na23__p_Mg26
       )

    jac[jna23, jmg23] = (
       +rate_eval.Mg23__Na23__weak__wc12
       +rho*Y[jn]*rate_eval.n_Mg23__p_Na23
       )

    jac[jna23, jmg24] = (
       +rate_eval.Mg24__p_Na23
       )

    jac[jna23, jmg26] = (
       +rho*Y[jp]*rate_eval.p_Mg26__He4_Na23
       )

    jac[jna23, jal26] = (
       +rho*Y[jn]*rate_eval.n_Al26__He4_Na23
       )

    jac[jna23, jal27] = (
       +rate_eval.Al27__He4_Na23
       )

    jac[jmg23, jn] = (
       -rho*Y[jmg23]*rate_eval.n_Mg23__Mg24
       -rho*Y[jmg23]*rate_eval.n_Mg23__p_Na23
       -rho*Y[jmg23]*rate_eval.n_Mg23__He4_Ne20
       -rho*Y[jmg23]*rate_eval.n_Mg23__C12_C12
       +rho*Y[jsi26]*rate_eval.n_Si26__He4_Mg23
       )

    jac[jmg23, jp] = (
       +rho*Y[jna22]*rate_eval.p_Na22__Mg23
       +rho*Y[jna23]*rate_eval.p_Na23__n_Mg23
       +rho*Y[jal26]*rate_eval.p_Al26__He4_Mg23
       )

    jac[jmg23, jhe4] = (
       -rho*Y[jmg23]*rate_eval.He4_Mg23__n_Si26
       -rho*Y[jmg23]*rate_eval.He4_Mg23__p_Al26
       +rho*Y[jne19]*rate_eval.He4_Ne19__Mg23
       +rho*Y[jne20]*rate_eval.He4_Ne20__n_Mg23
       )

    jac[jmg23, jc12] = (
       +5.00000000000000e-01*rho*2*Y[jc12]*rate_eval.C12_C12__n_Mg23
       )

    jac[jmg23, jne19] = (
       +rho*Y[jhe4]*rate_eval.He4_Ne19__Mg23
       )

    jac[jmg23, jne20] = (
       +rho*Y[jhe4]*rate_eval.He4_Ne20__n_Mg23
       )

    jac[jmg23, jna22] = (
       +rho*Y[jp]*rate_eval.p_Na22__Mg23
       )

    jac[jmg23, jna23] = (
       +rho*Y[jp]*rate_eval.p_Na23__n_Mg23
       )

    jac[jmg23, jmg23] = (
       -rate_eval.Mg23__Na23__weak__wc12
       -rate_eval.Mg23__p_Na22
       -rate_eval.Mg23__He4_Ne19
       -rho*Y[jn]*rate_eval.n_Mg23__Mg24
       -rho*Y[jn]*rate_eval.n_Mg23__p_Na23
       -rho*Y[jn]*rate_eval.n_Mg23__He4_Ne20
       -rho*Y[jn]*rate_eval.n_Mg23__C12_C12
       -rho*Y[jhe4]*rate_eval.He4_Mg23__n_Si26
       -rho*Y[jhe4]*rate_eval.He4_Mg23__p_Al26
       )

    jac[jmg23, jmg24] = (
       +rate_eval.Mg24__n_Mg23
       )

    jac[jmg23, jal26] = (
       +rho*Y[jp]*rate_eval.p_Al26__He4_Mg23
       )

    jac[jmg23, jsi26] = (
       +rho*Y[jn]*rate_eval.n_Si26__He4_Mg23
       )

    jac[jmg24, jn] = (
       -rho*Y[jmg24]*rate_eval.n_Mg24__Mg25
       -rho*Y[jmg24]*rate_eval.n_Mg24__He4_Ne21
       +rho*Y[jmg23]*rate_eval.n_Mg23__Mg24
       )

    jac[jmg24, jp] = (
       -rho*Y[jmg24]*rate_eval.p_Mg24__Al25
       +rho*Y[jna23]*rate_eval.p_Na23__Mg24
       +rho*Y[jal27]*rate_eval.p_Al27__He4_Mg24
       )

    jac[jmg24, jhe4] = (
       -rho*Y[jmg24]*rate_eval.He4_Mg24__Si28
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
       -rho*Y[jn]*rate_eval.n_Mg24__Mg25
       -rho*Y[jp]*rate_eval.p_Mg24__Al25
       -rho*Y[jhe4]*rate_eval.He4_Mg24__Si28
       -rho*Y[jn]*rate_eval.n_Mg24__He4_Ne21
       -rho*Y[jhe4]*rate_eval.He4_Mg24__p_Al27
       -rho*Y[jhe4]*rate_eval.He4_Mg24__C12_O16
       )

    jac[jmg24, jmg25] = (
       +rate_eval.Mg25__n_Mg24
       )

    jac[jmg24, jal25] = (
       +rate_eval.Al25__p_Mg24
       )

    jac[jmg24, jal27] = (
       +rho*Y[jp]*rate_eval.p_Al27__He4_Mg24
       )

    jac[jmg24, jsi28] = (
       +rate_eval.Si28__He4_Mg24
       )

    jac[jmg25, jn] = (
       -rho*Y[jmg25]*rate_eval.n_Mg25__Mg26
       -rho*Y[jmg25]*rate_eval.n_Mg25__He4_Ne22
       +rho*Y[jmg24]*rate_eval.n_Mg24__Mg25
       +rho*Y[jal25]*rate_eval.n_Al25__p_Mg25
       +rho*Y[jsi28]*rate_eval.n_Si28__He4_Mg25
       )

    jac[jmg25, jp] = (
       -rho*Y[jmg25]*rate_eval.p_Mg25__Al26
       -rho*Y[jmg25]*rate_eval.p_Mg25__n_Al25
       -rho*Y[jmg25]*rate_eval.p_Mg25__He4_Na22
       )

    jac[jmg25, jhe4] = (
       -rho*Y[jmg25]*rate_eval.He4_Mg25__Si29
       -rho*Y[jmg25]*rate_eval.He4_Mg25__n_Si28
       +rho*Y[jne21]*rate_eval.He4_Ne21__Mg25
       +rho*Y[jne22]*rate_eval.He4_Ne22__n_Mg25
       +rho*Y[jna22]*rate_eval.He4_Na22__p_Mg25
       )

    jac[jmg25, jne21] = (
       +rho*Y[jhe4]*rate_eval.He4_Ne21__Mg25
       )

    jac[jmg25, jne22] = (
       +rho*Y[jhe4]*rate_eval.He4_Ne22__n_Mg25
       )

    jac[jmg25, jna22] = (
       +rho*Y[jhe4]*rate_eval.He4_Na22__p_Mg25
       )

    jac[jmg25, jmg24] = (
       +rho*Y[jn]*rate_eval.n_Mg24__Mg25
       )

    jac[jmg25, jmg25] = (
       -rate_eval.Mg25__n_Mg24
       -rate_eval.Mg25__He4_Ne21
       -rho*Y[jn]*rate_eval.n_Mg25__Mg26
       -rho*Y[jp]*rate_eval.p_Mg25__Al26
       -rho*Y[jhe4]*rate_eval.He4_Mg25__Si29
       -rho*Y[jn]*rate_eval.n_Mg25__He4_Ne22
       -rho*Y[jp]*rate_eval.p_Mg25__n_Al25
       -rho*Y[jp]*rate_eval.p_Mg25__He4_Na22
       -rho*Y[jhe4]*rate_eval.He4_Mg25__n_Si28
       )

    jac[jmg25, jmg26] = (
       +rate_eval.Mg26__n_Mg25
       )

    jac[jmg25, jal25] = (
       +rate_eval.Al25__Mg25__weak__wc12
       +rho*Y[jn]*rate_eval.n_Al25__p_Mg25
       )

    jac[jmg25, jal26] = (
       +rate_eval.Al26__p_Mg25
       )

    jac[jmg25, jsi28] = (
       +rho*Y[jn]*rate_eval.n_Si28__He4_Mg25
       )

    jac[jmg25, jsi29] = (
       +rate_eval.Si29__He4_Mg25
       )

    jac[jmg26, jn] = (
       +rho*Y[jmg25]*rate_eval.n_Mg25__Mg26
       +rho*Y[jal26]*rate_eval.n_Al26__p_Mg26
       +rho*Y[jsi29]*rate_eval.n_Si29__He4_Mg26
       )

    jac[jmg26, jp] = (
       -rho*Y[jmg26]*rate_eval.p_Mg26__Al27
       -rho*Y[jmg26]*rate_eval.p_Mg26__n_Al26
       -rho*Y[jmg26]*rate_eval.p_Mg26__He4_Na23
       )

    jac[jmg26, jhe4] = (
       -rho*Y[jmg26]*rate_eval.He4_Mg26__Si30
       -rho*Y[jmg26]*rate_eval.He4_Mg26__n_Si29
       +rho*Y[jne22]*rate_eval.He4_Ne22__Mg26
       +rho*Y[jna23]*rate_eval.He4_Na23__p_Mg26
       )

    jac[jmg26, jne22] = (
       +rho*Y[jhe4]*rate_eval.He4_Ne22__Mg26
       )

    jac[jmg26, jna23] = (
       +rho*Y[jhe4]*rate_eval.He4_Na23__p_Mg26
       )

    jac[jmg26, jmg25] = (
       +rho*Y[jn]*rate_eval.n_Mg25__Mg26
       )

    jac[jmg26, jmg26] = (
       -rate_eval.Mg26__n_Mg25
       -rate_eval.Mg26__He4_Ne22
       -rho*Y[jp]*rate_eval.p_Mg26__Al27
       -rho*Y[jhe4]*rate_eval.He4_Mg26__Si30
       -rho*Y[jp]*rate_eval.p_Mg26__n_Al26
       -rho*Y[jp]*rate_eval.p_Mg26__He4_Na23
       -rho*Y[jhe4]*rate_eval.He4_Mg26__n_Si29
       )

    jac[jmg26, jal26] = (
       +rate_eval.Al26__Mg26__weak__wc12
       +rho*Y[jn]*rate_eval.n_Al26__p_Mg26
       )

    jac[jmg26, jal27] = (
       +rate_eval.Al27__p_Mg26
       )

    jac[jmg26, jsi29] = (
       +rho*Y[jn]*rate_eval.n_Si29__He4_Mg26
       )

    jac[jmg26, jsi30] = (
       +rate_eval.Si30__He4_Mg26
       )

    jac[jal25, jn] = (
       -rho*Y[jal25]*rate_eval.n_Al25__Al26
       -rho*Y[jal25]*rate_eval.n_Al25__p_Mg25
       -rho*Y[jal25]*rate_eval.n_Al25__He4_Na22
       )

    jac[jal25, jp] = (
       -rho*Y[jal25]*rate_eval.p_Al25__Si26
       +rho*Y[jmg24]*rate_eval.p_Mg24__Al25
       +rho*Y[jmg25]*rate_eval.p_Mg25__n_Al25
       +rho*Y[jsi28]*rate_eval.p_Si28__He4_Al25
       )

    jac[jal25, jhe4] = (
       -rho*Y[jal25]*rate_eval.He4_Al25__P29
       -rho*Y[jal25]*rate_eval.He4_Al25__p_Si28
       +rho*Y[jna22]*rate_eval.He4_Na22__n_Al25
       )

    jac[jal25, jna22] = (
       +rho*Y[jhe4]*rate_eval.He4_Na22__n_Al25
       )

    jac[jal25, jmg24] = (
       +rho*Y[jp]*rate_eval.p_Mg24__Al25
       )

    jac[jal25, jmg25] = (
       +rho*Y[jp]*rate_eval.p_Mg25__n_Al25
       )

    jac[jal25, jal25] = (
       -rate_eval.Al25__Mg25__weak__wc12
       -rate_eval.Al25__p_Mg24
       -rho*Y[jn]*rate_eval.n_Al25__Al26
       -rho*Y[jp]*rate_eval.p_Al25__Si26
       -rho*Y[jhe4]*rate_eval.He4_Al25__P29
       -rho*Y[jn]*rate_eval.n_Al25__p_Mg25
       -rho*Y[jn]*rate_eval.n_Al25__He4_Na22
       -rho*Y[jhe4]*rate_eval.He4_Al25__p_Si28
       )

    jac[jal25, jal26] = (
       +rate_eval.Al26__n_Al25
       )

    jac[jal25, jsi26] = (
       +rate_eval.Si26__p_Al25
       )

    jac[jal25, jsi28] = (
       +rho*Y[jp]*rate_eval.p_Si28__He4_Al25
       )

    jac[jal25, jp29] = (
       +rate_eval.P29__He4_Al25
       )

    jac[jal26, jn] = (
       -rho*Y[jal26]*rate_eval.n_Al26__Al27
       -rho*Y[jal26]*rate_eval.n_Al26__p_Mg26
       -rho*Y[jal26]*rate_eval.n_Al26__He4_Na23
       +rho*Y[jal25]*rate_eval.n_Al25__Al26
       +rho*Y[jsi26]*rate_eval.n_Si26__p_Al26
       +rho*Y[jp29]*rate_eval.n_P29__He4_Al26
       )

    jac[jal26, jp] = (
       -rho*Y[jal26]*rate_eval.p_Al26__n_Si26
       -rho*Y[jal26]*rate_eval.p_Al26__He4_Mg23
       +rho*Y[jmg25]*rate_eval.p_Mg25__Al26
       +rho*Y[jmg26]*rate_eval.p_Mg26__n_Al26
       +rho*Y[jsi29]*rate_eval.p_Si29__He4_Al26
       )

    jac[jal26, jhe4] = (
       -rho*Y[jal26]*rate_eval.He4_Al26__P30
       -rho*Y[jal26]*rate_eval.He4_Al26__n_P29
       -rho*Y[jal26]*rate_eval.He4_Al26__p_Si29
       +rho*Y[jna22]*rate_eval.He4_Na22__Al26
       +rho*Y[jna23]*rate_eval.He4_Na23__n_Al26
       +rho*Y[jmg23]*rate_eval.He4_Mg23__p_Al26
       )

    jac[jal26, jna22] = (
       +rho*Y[jhe4]*rate_eval.He4_Na22__Al26
       )

    jac[jal26, jna23] = (
       +rho*Y[jhe4]*rate_eval.He4_Na23__n_Al26
       )

    jac[jal26, jmg23] = (
       +rho*Y[jhe4]*rate_eval.He4_Mg23__p_Al26
       )

    jac[jal26, jmg25] = (
       +rho*Y[jp]*rate_eval.p_Mg25__Al26
       )

    jac[jal26, jmg26] = (
       +rho*Y[jp]*rate_eval.p_Mg26__n_Al26
       )

    jac[jal26, jal25] = (
       +rho*Y[jn]*rate_eval.n_Al25__Al26
       )

    jac[jal26, jal26] = (
       -rate_eval.Al26__Mg26__weak__wc12
       -rate_eval.Al26__n_Al25
       -rate_eval.Al26__p_Mg25
       -rate_eval.Al26__He4_Na22
       -rho*Y[jn]*rate_eval.n_Al26__Al27
       -rho*Y[jhe4]*rate_eval.He4_Al26__P30
       -rho*Y[jn]*rate_eval.n_Al26__p_Mg26
       -rho*Y[jn]*rate_eval.n_Al26__He4_Na23
       -rho*Y[jp]*rate_eval.p_Al26__n_Si26
       -rho*Y[jp]*rate_eval.p_Al26__He4_Mg23
       -rho*Y[jhe4]*rate_eval.He4_Al26__n_P29
       -rho*Y[jhe4]*rate_eval.He4_Al26__p_Si29
       )

    jac[jal26, jal27] = (
       +rate_eval.Al27__n_Al26
       )

    jac[jal26, jsi26] = (
       +rate_eval.Si26__Al26__weak__wc12
       +rho*Y[jn]*rate_eval.n_Si26__p_Al26
       )

    jac[jal26, jsi29] = (
       +rho*Y[jp]*rate_eval.p_Si29__He4_Al26
       )

    jac[jal26, jp29] = (
       +rho*Y[jn]*rate_eval.n_P29__He4_Al26
       )

    jac[jal26, jp30] = (
       +rate_eval.P30__He4_Al26
       )

    jac[jal27, jn] = (
       +rho*Y[jal26]*rate_eval.n_Al26__Al27
       +rho*Y[jp30]*rate_eval.n_P30__He4_Al27
       )

    jac[jal27, jp] = (
       -rho*Y[jal27]*rate_eval.p_Al27__Si28
       -rho*Y[jal27]*rate_eval.p_Al27__He4_Mg24
       -rho*Y[jal27]*rate_eval.p_Al27__C12_O16
       +rho*Y[jmg26]*rate_eval.p_Mg26__Al27
       +rho*Y[jsi30]*rate_eval.p_Si30__He4_Al27
       )

    jac[jal27, jhe4] = (
       -rho*Y[jal27]*rate_eval.He4_Al27__P31
       -rho*Y[jal27]*rate_eval.He4_Al27__n_P30
       -rho*Y[jal27]*rate_eval.He4_Al27__p_Si30
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

    jac[jal27, jmg26] = (
       +rho*Y[jp]*rate_eval.p_Mg26__Al27
       )

    jac[jal27, jal26] = (
       +rho*Y[jn]*rate_eval.n_Al26__Al27
       )

    jac[jal27, jal27] = (
       -rate_eval.Al27__n_Al26
       -rate_eval.Al27__p_Mg26
       -rate_eval.Al27__He4_Na23
       -rho*Y[jp]*rate_eval.p_Al27__Si28
       -rho*Y[jhe4]*rate_eval.He4_Al27__P31
       -rho*Y[jp]*rate_eval.p_Al27__He4_Mg24
       -rho*Y[jp]*rate_eval.p_Al27__C12_O16
       -rho*Y[jhe4]*rate_eval.He4_Al27__n_P30
       -rho*Y[jhe4]*rate_eval.He4_Al27__p_Si30
       )

    jac[jal27, jsi28] = (
       +rate_eval.Si28__p_Al27
       )

    jac[jal27, jsi30] = (
       +rho*Y[jp]*rate_eval.p_Si30__He4_Al27
       )

    jac[jal27, jp30] = (
       +rho*Y[jn]*rate_eval.n_P30__He4_Al27
       )

    jac[jal27, jp31] = (
       +rate_eval.P31__He4_Al27
       )

    jac[jsi26, jn] = (
       -rho*Y[jsi26]*rate_eval.n_Si26__p_Al26
       -rho*Y[jsi26]*rate_eval.n_Si26__He4_Mg23
       )

    jac[jsi26, jp] = (
       +rho*Y[jal25]*rate_eval.p_Al25__Si26
       +rho*Y[jal26]*rate_eval.p_Al26__n_Si26
       +rho*Y[jp29]*rate_eval.p_P29__He4_Si26
       )

    jac[jsi26, jhe4] = (
       -rho*Y[jsi26]*rate_eval.He4_Si26__S30
       -rho*Y[jsi26]*rate_eval.He4_Si26__p_P29
       +rho*Y[jmg23]*rate_eval.He4_Mg23__n_Si26
       )

    jac[jsi26, jmg23] = (
       +rho*Y[jhe4]*rate_eval.He4_Mg23__n_Si26
       )

    jac[jsi26, jal25] = (
       +rho*Y[jp]*rate_eval.p_Al25__Si26
       )

    jac[jsi26, jal26] = (
       +rho*Y[jp]*rate_eval.p_Al26__n_Si26
       )

    jac[jsi26, jsi26] = (
       -rate_eval.Si26__Al26__weak__wc12
       -rate_eval.Si26__p_Al25
       -rho*Y[jhe4]*rate_eval.He4_Si26__S30
       -rho*Y[jn]*rate_eval.n_Si26__p_Al26
       -rho*Y[jn]*rate_eval.n_Si26__He4_Mg23
       -rho*Y[jhe4]*rate_eval.He4_Si26__p_P29
       )

    jac[jsi26, jp29] = (
       +rho*Y[jp]*rate_eval.p_P29__He4_Si26
       )

    jac[jsi26, js30] = (
       +rate_eval.S30__He4_Si26
       )

    jac[jsi28, jn] = (
       -rho*Y[jsi28]*rate_eval.n_Si28__Si29
       -rho*Y[jsi28]*rate_eval.n_Si28__He4_Mg25
       +rho*Y[js31]*rate_eval.n_S31__He4_Si28
       )

    jac[jsi28, jp] = (
       -rho*Y[jsi28]*rate_eval.p_Si28__P29
       -rho*Y[jsi28]*rate_eval.p_Si28__He4_Al25
       +rho*Y[jal27]*rate_eval.p_Al27__Si28
       +rho*Y[jp31]*rate_eval.p_P31__He4_Si28
       )

    jac[jsi28, jhe4] = (
       -rho*Y[jsi28]*rate_eval.He4_Si28__S32
       -rho*Y[jsi28]*rate_eval.He4_Si28__n_S31
       -rho*Y[jsi28]*rate_eval.He4_Si28__p_P31
       -rho*Y[jsi28]*rate_eval.He4_Si28__C12_Ne20
       -rho*Y[jsi28]*rate_eval.He4_Si28__O16_O16
       +rho*Y[jmg24]*rate_eval.He4_Mg24__Si28
       +rho*Y[jmg25]*rate_eval.He4_Mg25__n_Si28
       +rho*Y[jal25]*rate_eval.He4_Al25__p_Si28
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

    jac[jsi28, jmg25] = (
       +rho*Y[jhe4]*rate_eval.He4_Mg25__n_Si28
       )

    jac[jsi28, jal25] = (
       +rho*Y[jhe4]*rate_eval.He4_Al25__p_Si28
       )

    jac[jsi28, jal27] = (
       +rho*Y[jp]*rate_eval.p_Al27__Si28
       )

    jac[jsi28, jsi28] = (
       -rate_eval.Si28__p_Al27
       -rate_eval.Si28__He4_Mg24
       -rho*Y[jn]*rate_eval.n_Si28__Si29
       -rho*Y[jp]*rate_eval.p_Si28__P29
       -rho*Y[jhe4]*rate_eval.He4_Si28__S32
       -rho*Y[jn]*rate_eval.n_Si28__He4_Mg25
       -rho*Y[jp]*rate_eval.p_Si28__He4_Al25
       -rho*Y[jhe4]*rate_eval.He4_Si28__n_S31
       -rho*Y[jhe4]*rate_eval.He4_Si28__p_P31
       -rho*Y[jhe4]*rate_eval.He4_Si28__C12_Ne20
       -rho*Y[jhe4]*rate_eval.He4_Si28__O16_O16
       )

    jac[jsi28, jsi29] = (
       +rate_eval.Si29__n_Si28
       )

    jac[jsi28, jp29] = (
       +rate_eval.P29__p_Si28
       )

    jac[jsi28, jp31] = (
       +rho*Y[jp]*rate_eval.p_P31__He4_Si28
       )

    jac[jsi28, js31] = (
       +rho*Y[jn]*rate_eval.n_S31__He4_Si28
       )

    jac[jsi28, js32] = (
       +rate_eval.S32__He4_Si28
       )

    jac[jsi29, jn] = (
       -rho*Y[jsi29]*rate_eval.n_Si29__Si30
       -rho*Y[jsi29]*rate_eval.n_Si29__He4_Mg26
       +rho*Y[jsi28]*rate_eval.n_Si28__Si29
       +rho*Y[jp29]*rate_eval.n_P29__p_Si29
       +rho*Y[js32]*rate_eval.n_S32__He4_Si29
       )

    jac[jsi29, jp] = (
       -rho*Y[jsi29]*rate_eval.p_Si29__P30
       -rho*Y[jsi29]*rate_eval.p_Si29__n_P29
       -rho*Y[jsi29]*rate_eval.p_Si29__He4_Al26
       )

    jac[jsi29, jhe4] = (
       -rho*Y[jsi29]*rate_eval.He4_Si29__S33
       -rho*Y[jsi29]*rate_eval.He4_Si29__n_S32
       +rho*Y[jmg25]*rate_eval.He4_Mg25__Si29
       +rho*Y[jmg26]*rate_eval.He4_Mg26__n_Si29
       +rho*Y[jal26]*rate_eval.He4_Al26__p_Si29
       )

    jac[jsi29, jmg25] = (
       +rho*Y[jhe4]*rate_eval.He4_Mg25__Si29
       )

    jac[jsi29, jmg26] = (
       +rho*Y[jhe4]*rate_eval.He4_Mg26__n_Si29
       )

    jac[jsi29, jal26] = (
       +rho*Y[jhe4]*rate_eval.He4_Al26__p_Si29
       )

    jac[jsi29, jsi28] = (
       +rho*Y[jn]*rate_eval.n_Si28__Si29
       )

    jac[jsi29, jsi29] = (
       -rate_eval.Si29__n_Si28
       -rate_eval.Si29__He4_Mg25
       -rho*Y[jn]*rate_eval.n_Si29__Si30
       -rho*Y[jp]*rate_eval.p_Si29__P30
       -rho*Y[jhe4]*rate_eval.He4_Si29__S33
       -rho*Y[jn]*rate_eval.n_Si29__He4_Mg26
       -rho*Y[jp]*rate_eval.p_Si29__n_P29
       -rho*Y[jp]*rate_eval.p_Si29__He4_Al26
       -rho*Y[jhe4]*rate_eval.He4_Si29__n_S32
       )

    jac[jsi29, jsi30] = (
       +rate_eval.Si30__n_Si29
       )

    jac[jsi29, jp29] = (
       +rate_eval.P29__Si29__weak__wc12
       +rho*Y[jn]*rate_eval.n_P29__p_Si29
       )

    jac[jsi29, jp30] = (
       +rate_eval.P30__p_Si29
       )

    jac[jsi29, js32] = (
       +rho*Y[jn]*rate_eval.n_S32__He4_Si29
       )

    jac[jsi29, js33] = (
       +rate_eval.S33__He4_Si29
       )

    jac[jsi30, jn] = (
       +rho*Y[jsi29]*rate_eval.n_Si29__Si30
       +rho*Y[jp30]*rate_eval.n_P30__p_Si30
       +rho*Y[js33]*rate_eval.n_S33__He4_Si30
       )

    jac[jsi30, jp] = (
       -rho*Y[jsi30]*rate_eval.p_Si30__P31
       -rho*Y[jsi30]*rate_eval.p_Si30__n_P30
       -rho*Y[jsi30]*rate_eval.p_Si30__He4_Al27
       )

    jac[jsi30, jhe4] = (
       -rho*Y[jsi30]*rate_eval.He4_Si30__n_S33
       +rho*Y[jmg26]*rate_eval.He4_Mg26__Si30
       +rho*Y[jal27]*rate_eval.He4_Al27__p_Si30
       )

    jac[jsi30, jmg26] = (
       +rho*Y[jhe4]*rate_eval.He4_Mg26__Si30
       )

    jac[jsi30, jal27] = (
       +rho*Y[jhe4]*rate_eval.He4_Al27__p_Si30
       )

    jac[jsi30, jsi29] = (
       +rho*Y[jn]*rate_eval.n_Si29__Si30
       )

    jac[jsi30, jsi30] = (
       -rate_eval.Si30__n_Si29
       -rate_eval.Si30__He4_Mg26
       -rho*Y[jp]*rate_eval.p_Si30__P31
       -rho*Y[jp]*rate_eval.p_Si30__n_P30
       -rho*Y[jp]*rate_eval.p_Si30__He4_Al27
       -rho*Y[jhe4]*rate_eval.He4_Si30__n_S33
       )

    jac[jsi30, jp30] = (
       +rate_eval.P30__Si30__weak__wc12
       +rho*Y[jn]*rate_eval.n_P30__p_Si30
       )

    jac[jsi30, jp31] = (
       +rate_eval.P31__p_Si30
       )

    jac[jsi30, js33] = (
       +rho*Y[jn]*rate_eval.n_S33__He4_Si30
       )

    jac[jp29, jn] = (
       -rho*Y[jp29]*rate_eval.n_P29__P30
       -rho*Y[jp29]*rate_eval.n_P29__p_Si29
       -rho*Y[jp29]*rate_eval.n_P29__He4_Al26
       )

    jac[jp29, jp] = (
       -rho*Y[jp29]*rate_eval.p_P29__S30
       -rho*Y[jp29]*rate_eval.p_P29__He4_Si26
       +rho*Y[jsi28]*rate_eval.p_Si28__P29
       +rho*Y[jsi29]*rate_eval.p_Si29__n_P29
       +rho*Y[js32]*rate_eval.p_S32__He4_P29
       )

    jac[jp29, jhe4] = (
       -rho*Y[jp29]*rate_eval.He4_P29__Cl33
       -rho*Y[jp29]*rate_eval.He4_P29__p_S32
       +rho*Y[jal25]*rate_eval.He4_Al25__P29
       +rho*Y[jal26]*rate_eval.He4_Al26__n_P29
       +rho*Y[jsi26]*rate_eval.He4_Si26__p_P29
       )

    jac[jp29, jal25] = (
       +rho*Y[jhe4]*rate_eval.He4_Al25__P29
       )

    jac[jp29, jal26] = (
       +rho*Y[jhe4]*rate_eval.He4_Al26__n_P29
       )

    jac[jp29, jsi26] = (
       +rho*Y[jhe4]*rate_eval.He4_Si26__p_P29
       )

    jac[jp29, jsi28] = (
       +rho*Y[jp]*rate_eval.p_Si28__P29
       )

    jac[jp29, jsi29] = (
       +rho*Y[jp]*rate_eval.p_Si29__n_P29
       )

    jac[jp29, jp29] = (
       -rate_eval.P29__Si29__weak__wc12
       -rate_eval.P29__p_Si28
       -rate_eval.P29__He4_Al25
       -rho*Y[jn]*rate_eval.n_P29__P30
       -rho*Y[jp]*rate_eval.p_P29__S30
       -rho*Y[jhe4]*rate_eval.He4_P29__Cl33
       -rho*Y[jn]*rate_eval.n_P29__p_Si29
       -rho*Y[jn]*rate_eval.n_P29__He4_Al26
       -rho*Y[jp]*rate_eval.p_P29__He4_Si26
       -rho*Y[jhe4]*rate_eval.He4_P29__p_S32
       )

    jac[jp29, jp30] = (
       +rate_eval.P30__n_P29
       )

    jac[jp29, js30] = (
       +rate_eval.S30__p_P29
       )

    jac[jp29, js32] = (
       +rho*Y[jp]*rate_eval.p_S32__He4_P29
       )

    jac[jp29, jcl33] = (
       +rate_eval.Cl33__He4_P29
       )

    jac[jp30, jn] = (
       -rho*Y[jp30]*rate_eval.n_P30__P31
       -rho*Y[jp30]*rate_eval.n_P30__p_Si30
       -rho*Y[jp30]*rate_eval.n_P30__He4_Al27
       +rho*Y[jp29]*rate_eval.n_P29__P30
       +rho*Y[js30]*rate_eval.n_S30__p_P30
       +rho*Y[jcl33]*rate_eval.n_Cl33__He4_P30
       )

    jac[jp30, jp] = (
       -rho*Y[jp30]*rate_eval.p_P30__S31
       -rho*Y[jp30]*rate_eval.p_P30__n_S30
       +rho*Y[jsi29]*rate_eval.p_Si29__P30
       +rho*Y[jsi30]*rate_eval.p_Si30__n_P30
       +rho*Y[js33]*rate_eval.p_S33__He4_P30
       )

    jac[jp30, jhe4] = (
       -rho*Y[jp30]*rate_eval.He4_P30__Cl34
       -rho*Y[jp30]*rate_eval.He4_P30__n_Cl33
       -rho*Y[jp30]*rate_eval.He4_P30__p_S33
       +rho*Y[jal26]*rate_eval.He4_Al26__P30
       +rho*Y[jal27]*rate_eval.He4_Al27__n_P30
       )

    jac[jp30, jal26] = (
       +rho*Y[jhe4]*rate_eval.He4_Al26__P30
       )

    jac[jp30, jal27] = (
       +rho*Y[jhe4]*rate_eval.He4_Al27__n_P30
       )

    jac[jp30, jsi29] = (
       +rho*Y[jp]*rate_eval.p_Si29__P30
       )

    jac[jp30, jsi30] = (
       +rho*Y[jp]*rate_eval.p_Si30__n_P30
       )

    jac[jp30, jp29] = (
       +rho*Y[jn]*rate_eval.n_P29__P30
       )

    jac[jp30, jp30] = (
       -rate_eval.P30__Si30__weak__wc12
       -rate_eval.P30__n_P29
       -rate_eval.P30__p_Si29
       -rate_eval.P30__He4_Al26
       -rho*Y[jn]*rate_eval.n_P30__P31
       -rho*Y[jp]*rate_eval.p_P30__S31
       -rho*Y[jhe4]*rate_eval.He4_P30__Cl34
       -rho*Y[jn]*rate_eval.n_P30__p_Si30
       -rho*Y[jn]*rate_eval.n_P30__He4_Al27
       -rho*Y[jp]*rate_eval.p_P30__n_S30
       -rho*Y[jhe4]*rate_eval.He4_P30__n_Cl33
       -rho*Y[jhe4]*rate_eval.He4_P30__p_S33
       )

    jac[jp30, jp31] = (
       +rate_eval.P31__n_P30
       )

    jac[jp30, js30] = (
       +rate_eval.S30__P30__weak__wc12
       +rho*Y[jn]*rate_eval.n_S30__p_P30
       )

    jac[jp30, js31] = (
       +rate_eval.S31__p_P30
       )

    jac[jp30, js33] = (
       +rho*Y[jp]*rate_eval.p_S33__He4_P30
       )

    jac[jp30, jcl33] = (
       +rho*Y[jn]*rate_eval.n_Cl33__He4_P30
       )

    jac[jp30, jcl34] = (
       +rate_eval.Cl34__He4_P30
       )

    jac[jp31, jn] = (
       +rho*Y[jp30]*rate_eval.n_P30__P31
       +rho*Y[js31]*rate_eval.n_S31__p_P31
       +rho*Y[jcl34]*rate_eval.n_Cl34__He4_P31
       )

    jac[jp31, jp] = (
       -rho*Y[jp31]*rate_eval.p_P31__S32
       -rho*Y[jp31]*rate_eval.p_P31__n_S31
       -rho*Y[jp31]*rate_eval.p_P31__He4_Si28
       -rho*Y[jp31]*rate_eval.p_P31__C12_Ne20
       -rho*Y[jp31]*rate_eval.p_P31__O16_O16
       +rho*Y[jsi30]*rate_eval.p_Si30__P31
       )

    jac[jp31, jhe4] = (
       -rho*Y[jp31]*rate_eval.He4_P31__Cl35
       -rho*Y[jp31]*rate_eval.He4_P31__n_Cl34
       +rho*Y[jal27]*rate_eval.He4_Al27__P31
       +rho*Y[jsi28]*rate_eval.He4_Si28__p_P31
       )

    jac[jp31, jc12] = (
       +rho*Y[jne20]*rate_eval.C12_Ne20__p_P31
       )

    jac[jp31, jo16] = (
       +5.00000000000000e-01*rho*2*Y[jo16]*rate_eval.O16_O16__p_P31
       )

    jac[jp31, jne20] = (
       +rho*Y[jc12]*rate_eval.C12_Ne20__p_P31
       )

    jac[jp31, jal27] = (
       +rho*Y[jhe4]*rate_eval.He4_Al27__P31
       )

    jac[jp31, jsi28] = (
       +rho*Y[jhe4]*rate_eval.He4_Si28__p_P31
       )

    jac[jp31, jsi30] = (
       +rho*Y[jp]*rate_eval.p_Si30__P31
       )

    jac[jp31, jp30] = (
       +rho*Y[jn]*rate_eval.n_P30__P31
       )

    jac[jp31, jp31] = (
       -rate_eval.P31__n_P30
       -rate_eval.P31__p_Si30
       -rate_eval.P31__He4_Al27
       -rho*Y[jp]*rate_eval.p_P31__S32
       -rho*Y[jhe4]*rate_eval.He4_P31__Cl35
       -rho*Y[jp]*rate_eval.p_P31__n_S31
       -rho*Y[jp]*rate_eval.p_P31__He4_Si28
       -rho*Y[jp]*rate_eval.p_P31__C12_Ne20
       -rho*Y[jp]*rate_eval.p_P31__O16_O16
       -rho*Y[jhe4]*rate_eval.He4_P31__n_Cl34
       )

    jac[jp31, js31] = (
       +rate_eval.S31__P31__weak__wc12
       +rho*Y[jn]*rate_eval.n_S31__p_P31
       )

    jac[jp31, js32] = (
       +rate_eval.S32__p_P31
       )

    jac[jp31, jcl34] = (
       +rho*Y[jn]*rate_eval.n_Cl34__He4_P31
       )

    jac[jp31, jcl35] = (
       +rate_eval.Cl35__He4_P31
       )

    jac[js30, jn] = (
       -rho*Y[js30]*rate_eval.n_S30__S31
       -rho*Y[js30]*rate_eval.n_S30__p_P30
       )

    jac[js30, jp] = (
       +rho*Y[jp29]*rate_eval.p_P29__S30
       +rho*Y[jp30]*rate_eval.p_P30__n_S30
       +rho*Y[jcl33]*rate_eval.p_Cl33__He4_S30
       )

    jac[js30, jhe4] = (
       -rho*Y[js30]*rate_eval.He4_S30__Ar34
       -rho*Y[js30]*rate_eval.He4_S30__p_Cl33
       +rho*Y[jsi26]*rate_eval.He4_Si26__S30
       )

    jac[js30, jsi26] = (
       +rho*Y[jhe4]*rate_eval.He4_Si26__S30
       )

    jac[js30, jp29] = (
       +rho*Y[jp]*rate_eval.p_P29__S30
       )

    jac[js30, jp30] = (
       +rho*Y[jp]*rate_eval.p_P30__n_S30
       )

    jac[js30, js30] = (
       -rate_eval.S30__P30__weak__wc12
       -rate_eval.S30__p_P29
       -rate_eval.S30__He4_Si26
       -rho*Y[jn]*rate_eval.n_S30__S31
       -rho*Y[jhe4]*rate_eval.He4_S30__Ar34
       -rho*Y[jn]*rate_eval.n_S30__p_P30
       -rho*Y[jhe4]*rate_eval.He4_S30__p_Cl33
       )

    jac[js30, js31] = (
       +rate_eval.S31__n_S30
       )

    jac[js30, jcl33] = (
       +rho*Y[jp]*rate_eval.p_Cl33__He4_S30
       )

    jac[js30, jar34] = (
       +rate_eval.Ar34__He4_S30
       )

    jac[js31, jn] = (
       -rho*Y[js31]*rate_eval.n_S31__S32
       -rho*Y[js31]*rate_eval.n_S31__p_P31
       -rho*Y[js31]*rate_eval.n_S31__He4_Si28
       -rho*Y[js31]*rate_eval.n_S31__C12_Ne20
       -rho*Y[js31]*rate_eval.n_S31__O16_O16
       +rho*Y[js30]*rate_eval.n_S30__S31
       +rho*Y[jar34]*rate_eval.n_Ar34__He4_S31
       )

    jac[js31, jp] = (
       +rho*Y[jp30]*rate_eval.p_P30__S31
       +rho*Y[jp31]*rate_eval.p_P31__n_S31
       +rho*Y[jcl34]*rate_eval.p_Cl34__He4_S31
       )

    jac[js31, jhe4] = (
       -rho*Y[js31]*rate_eval.He4_S31__n_Ar34
       -rho*Y[js31]*rate_eval.He4_S31__p_Cl34
       +rho*Y[jsi28]*rate_eval.He4_Si28__n_S31
       )

    jac[js31, jc12] = (
       +rho*Y[jne20]*rate_eval.C12_Ne20__n_S31
       )

    jac[js31, jo16] = (
       +5.00000000000000e-01*rho*2*Y[jo16]*rate_eval.O16_O16__n_S31
       )

    jac[js31, jne20] = (
       +rho*Y[jc12]*rate_eval.C12_Ne20__n_S31
       )

    jac[js31, jsi28] = (
       +rho*Y[jhe4]*rate_eval.He4_Si28__n_S31
       )

    jac[js31, jp30] = (
       +rho*Y[jp]*rate_eval.p_P30__S31
       )

    jac[js31, jp31] = (
       +rho*Y[jp]*rate_eval.p_P31__n_S31
       )

    jac[js31, js30] = (
       +rho*Y[jn]*rate_eval.n_S30__S31
       )

    jac[js31, js31] = (
       -rate_eval.S31__P31__weak__wc12
       -rate_eval.S31__n_S30
       -rate_eval.S31__p_P30
       -rho*Y[jn]*rate_eval.n_S31__S32
       -rho*Y[jn]*rate_eval.n_S31__p_P31
       -rho*Y[jn]*rate_eval.n_S31__He4_Si28
       -rho*Y[jn]*rate_eval.n_S31__C12_Ne20
       -rho*Y[jn]*rate_eval.n_S31__O16_O16
       -rho*Y[jhe4]*rate_eval.He4_S31__n_Ar34
       -rho*Y[jhe4]*rate_eval.He4_S31__p_Cl34
       )

    jac[js31, js32] = (
       +rate_eval.S32__n_S31
       )

    jac[js31, jcl34] = (
       +rho*Y[jp]*rate_eval.p_Cl34__He4_S31
       )

    jac[js31, jar34] = (
       +rho*Y[jn]*rate_eval.n_Ar34__He4_S31
       )

    jac[js32, jn] = (
       -rho*Y[js32]*rate_eval.n_S32__S33
       -rho*Y[js32]*rate_eval.n_S32__He4_Si29
       +rho*Y[js31]*rate_eval.n_S31__S32
       )

    jac[js32, jp] = (
       -rho*Y[js32]*rate_eval.p_S32__Cl33
       -rho*Y[js32]*rate_eval.p_S32__He4_P29
       +rho*Y[jp31]*rate_eval.p_P31__S32
       +rho*Y[jcl35]*rate_eval.p_Cl35__He4_S32
       )

    jac[js32, jhe4] = (
       -rho*Y[js32]*rate_eval.He4_S32__Ar36
       -rho*Y[js32]*rate_eval.He4_S32__p_Cl35
       +rho*Y[jsi28]*rate_eval.He4_Si28__S32
       +rho*Y[jsi29]*rate_eval.He4_Si29__n_S32
       +rho*Y[jp29]*rate_eval.He4_P29__p_S32
       )

    jac[js32, jsi28] = (
       +rho*Y[jhe4]*rate_eval.He4_Si28__S32
       )

    jac[js32, jsi29] = (
       +rho*Y[jhe4]*rate_eval.He4_Si29__n_S32
       )

    jac[js32, jp29] = (
       +rho*Y[jhe4]*rate_eval.He4_P29__p_S32
       )

    jac[js32, jp31] = (
       +rho*Y[jp]*rate_eval.p_P31__S32
       )

    jac[js32, js31] = (
       +rho*Y[jn]*rate_eval.n_S31__S32
       )

    jac[js32, js32] = (
       -rate_eval.S32__n_S31
       -rate_eval.S32__p_P31
       -rate_eval.S32__He4_Si28
       -rho*Y[jn]*rate_eval.n_S32__S33
       -rho*Y[jp]*rate_eval.p_S32__Cl33
       -rho*Y[jhe4]*rate_eval.He4_S32__Ar36
       -rho*Y[jn]*rate_eval.n_S32__He4_Si29
       -rho*Y[jp]*rate_eval.p_S32__He4_P29
       -rho*Y[jhe4]*rate_eval.He4_S32__p_Cl35
       )

    jac[js32, js33] = (
       +rate_eval.S33__n_S32
       )

    jac[js32, jcl33] = (
       +rate_eval.Cl33__p_S32
       )

    jac[js32, jcl35] = (
       +rho*Y[jp]*rate_eval.p_Cl35__He4_S32
       )

    jac[js32, jar36] = (
       +rate_eval.Ar36__He4_S32
       )

    jac[js33, jn] = (
       -rho*Y[js33]*rate_eval.n_S33__He4_Si30
       +rho*Y[js32]*rate_eval.n_S32__S33
       +rho*Y[jcl33]*rate_eval.n_Cl33__p_S33
       +rho*Y[jar36]*rate_eval.n_Ar36__He4_S33
       )

    jac[js33, jp] = (
       -rho*Y[js33]*rate_eval.p_S33__Cl34
       -rho*Y[js33]*rate_eval.p_S33__n_Cl33
       -rho*Y[js33]*rate_eval.p_S33__He4_P30
       )

    jac[js33, jhe4] = (
       -rho*Y[js33]*rate_eval.He4_S33__Ar37
       -rho*Y[js33]*rate_eval.He4_S33__n_Ar36
       +rho*Y[jsi29]*rate_eval.He4_Si29__S33
       +rho*Y[jsi30]*rate_eval.He4_Si30__n_S33
       +rho*Y[jp30]*rate_eval.He4_P30__p_S33
       )

    jac[js33, jsi29] = (
       +rho*Y[jhe4]*rate_eval.He4_Si29__S33
       )

    jac[js33, jsi30] = (
       +rho*Y[jhe4]*rate_eval.He4_Si30__n_S33
       )

    jac[js33, jp30] = (
       +rho*Y[jhe4]*rate_eval.He4_P30__p_S33
       )

    jac[js33, js32] = (
       +rho*Y[jn]*rate_eval.n_S32__S33
       )

    jac[js33, js33] = (
       -rate_eval.S33__n_S32
       -rate_eval.S33__He4_Si29
       -rho*Y[jp]*rate_eval.p_S33__Cl34
       -rho*Y[jhe4]*rate_eval.He4_S33__Ar37
       -rho*Y[jn]*rate_eval.n_S33__He4_Si30
       -rho*Y[jp]*rate_eval.p_S33__n_Cl33
       -rho*Y[jp]*rate_eval.p_S33__He4_P30
       -rho*Y[jhe4]*rate_eval.He4_S33__n_Ar36
       )

    jac[js33, jcl33] = (
       +rate_eval.Cl33__S33__weak__wc12
       +rho*Y[jn]*rate_eval.n_Cl33__p_S33
       )

    jac[js33, jcl34] = (
       +rate_eval.Cl34__p_S33
       )

    jac[js33, jar36] = (
       +rho*Y[jn]*rate_eval.n_Ar36__He4_S33
       )

    jac[js33, jar37] = (
       +rate_eval.Ar37__He4_S33
       )

    jac[jcl33, jn] = (
       -rho*Y[jcl33]*rate_eval.n_Cl33__Cl34
       -rho*Y[jcl33]*rate_eval.n_Cl33__p_S33
       -rho*Y[jcl33]*rate_eval.n_Cl33__He4_P30
       )

    jac[jcl33, jp] = (
       -rho*Y[jcl33]*rate_eval.p_Cl33__Ar34
       -rho*Y[jcl33]*rate_eval.p_Cl33__He4_S30
       +rho*Y[js32]*rate_eval.p_S32__Cl33
       +rho*Y[js33]*rate_eval.p_S33__n_Cl33
       +rho*Y[jar36]*rate_eval.p_Ar36__He4_Cl33
       )

    jac[jcl33, jhe4] = (
       -rho*Y[jcl33]*rate_eval.He4_Cl33__p_Ar36
       +rho*Y[jp29]*rate_eval.He4_P29__Cl33
       +rho*Y[jp30]*rate_eval.He4_P30__n_Cl33
       +rho*Y[js30]*rate_eval.He4_S30__p_Cl33
       )

    jac[jcl33, jp29] = (
       +rho*Y[jhe4]*rate_eval.He4_P29__Cl33
       )

    jac[jcl33, jp30] = (
       +rho*Y[jhe4]*rate_eval.He4_P30__n_Cl33
       )

    jac[jcl33, js30] = (
       +rho*Y[jhe4]*rate_eval.He4_S30__p_Cl33
       )

    jac[jcl33, js32] = (
       +rho*Y[jp]*rate_eval.p_S32__Cl33
       )

    jac[jcl33, js33] = (
       +rho*Y[jp]*rate_eval.p_S33__n_Cl33
       )

    jac[jcl33, jcl33] = (
       -rate_eval.Cl33__S33__weak__wc12
       -rate_eval.Cl33__p_S32
       -rate_eval.Cl33__He4_P29
       -rho*Y[jn]*rate_eval.n_Cl33__Cl34
       -rho*Y[jp]*rate_eval.p_Cl33__Ar34
       -rho*Y[jn]*rate_eval.n_Cl33__p_S33
       -rho*Y[jn]*rate_eval.n_Cl33__He4_P30
       -rho*Y[jp]*rate_eval.p_Cl33__He4_S30
       -rho*Y[jhe4]*rate_eval.He4_Cl33__p_Ar36
       )

    jac[jcl33, jcl34] = (
       +rate_eval.Cl34__n_Cl33
       )

    jac[jcl33, jar34] = (
       +rate_eval.Ar34__p_Cl33
       )

    jac[jcl33, jar36] = (
       +rho*Y[jp]*rate_eval.p_Ar36__He4_Cl33
       )

    jac[jcl34, jn] = (
       -rho*Y[jcl34]*rate_eval.n_Cl34__Cl35
       -rho*Y[jcl34]*rate_eval.n_Cl34__He4_P31
       +rho*Y[jcl33]*rate_eval.n_Cl33__Cl34
       +rho*Y[jar34]*rate_eval.n_Ar34__p_Cl34
       )

    jac[jcl34, jp] = (
       -rho*Y[jcl34]*rate_eval.p_Cl34__n_Ar34
       -rho*Y[jcl34]*rate_eval.p_Cl34__He4_S31
       +rho*Y[js33]*rate_eval.p_S33__Cl34
       +rho*Y[jar37]*rate_eval.p_Ar37__He4_Cl34
       )

    jac[jcl34, jhe4] = (
       -rho*Y[jcl34]*rate_eval.He4_Cl34__p_Ar37
       +rho*Y[jp30]*rate_eval.He4_P30__Cl34
       +rho*Y[jp31]*rate_eval.He4_P31__n_Cl34
       +rho*Y[js31]*rate_eval.He4_S31__p_Cl34
       )

    jac[jcl34, jp30] = (
       +rho*Y[jhe4]*rate_eval.He4_P30__Cl34
       )

    jac[jcl34, jp31] = (
       +rho*Y[jhe4]*rate_eval.He4_P31__n_Cl34
       )

    jac[jcl34, js31] = (
       +rho*Y[jhe4]*rate_eval.He4_S31__p_Cl34
       )

    jac[jcl34, js33] = (
       +rho*Y[jp]*rate_eval.p_S33__Cl34
       )

    jac[jcl34, jcl33] = (
       +rho*Y[jn]*rate_eval.n_Cl33__Cl34
       )

    jac[jcl34, jcl34] = (
       -rate_eval.Cl34__n_Cl33
       -rate_eval.Cl34__p_S33
       -rate_eval.Cl34__He4_P30
       -rho*Y[jn]*rate_eval.n_Cl34__Cl35
       -rho*Y[jn]*rate_eval.n_Cl34__He4_P31
       -rho*Y[jp]*rate_eval.p_Cl34__n_Ar34
       -rho*Y[jp]*rate_eval.p_Cl34__He4_S31
       -rho*Y[jhe4]*rate_eval.He4_Cl34__p_Ar37
       )

    jac[jcl34, jcl35] = (
       +rate_eval.Cl35__n_Cl34
       )

    jac[jcl34, jar34] = (
       +rate_eval.Ar34__Cl34__weak__wc12
       +rho*Y[jn]*rate_eval.n_Ar34__p_Cl34
       )

    jac[jcl34, jar37] = (
       +rho*Y[jp]*rate_eval.p_Ar37__He4_Cl34
       )

    jac[jcl35, jn] = (
       +rho*Y[jcl34]*rate_eval.n_Cl34__Cl35
       )

    jac[jcl35, jp] = (
       -rho*Y[jcl35]*rate_eval.p_Cl35__Ar36
       -rho*Y[jcl35]*rate_eval.p_Cl35__He4_S32
       +rho*Y[jar38]*rate_eval.p_Ar38__He4_Cl35
       )

    jac[jcl35, jhe4] = (
       -rho*Y[jcl35]*rate_eval.He4_Cl35__K39
       -rho*Y[jcl35]*rate_eval.He4_Cl35__p_Ar38
       +rho*Y[jp31]*rate_eval.He4_P31__Cl35
       +rho*Y[js32]*rate_eval.He4_S32__p_Cl35
       )

    jac[jcl35, jp31] = (
       +rho*Y[jhe4]*rate_eval.He4_P31__Cl35
       )

    jac[jcl35, js32] = (
       +rho*Y[jhe4]*rate_eval.He4_S32__p_Cl35
       )

    jac[jcl35, jcl34] = (
       +rho*Y[jn]*rate_eval.n_Cl34__Cl35
       )

    jac[jcl35, jcl35] = (
       -rate_eval.Cl35__n_Cl34
       -rate_eval.Cl35__He4_P31
       -rho*Y[jp]*rate_eval.p_Cl35__Ar36
       -rho*Y[jhe4]*rate_eval.He4_Cl35__K39
       -rho*Y[jp]*rate_eval.p_Cl35__He4_S32
       -rho*Y[jhe4]*rate_eval.He4_Cl35__p_Ar38
       )

    jac[jcl35, jar36] = (
       +rate_eval.Ar36__p_Cl35
       )

    jac[jcl35, jar38] = (
       +rho*Y[jp]*rate_eval.p_Ar38__He4_Cl35
       )

    jac[jcl35, jk39] = (
       +rate_eval.K39__He4_Cl35
       )

    jac[jar34, jn] = (
       -rho*Y[jar34]*rate_eval.n_Ar34__p_Cl34
       -rho*Y[jar34]*rate_eval.n_Ar34__He4_S31
       )

    jac[jar34, jp] = (
       +rho*Y[jcl33]*rate_eval.p_Cl33__Ar34
       +rho*Y[jcl34]*rate_eval.p_Cl34__n_Ar34
       )

    jac[jar34, jhe4] = (
       +rho*Y[js30]*rate_eval.He4_S30__Ar34
       +rho*Y[js31]*rate_eval.He4_S31__n_Ar34
       )

    jac[jar34, js30] = (
       +rho*Y[jhe4]*rate_eval.He4_S30__Ar34
       )

    jac[jar34, js31] = (
       +rho*Y[jhe4]*rate_eval.He4_S31__n_Ar34
       )

    jac[jar34, jcl33] = (
       +rho*Y[jp]*rate_eval.p_Cl33__Ar34
       )

    jac[jar34, jcl34] = (
       +rho*Y[jp]*rate_eval.p_Cl34__n_Ar34
       )

    jac[jar34, jar34] = (
       -rate_eval.Ar34__Cl34__weak__wc12
       -rate_eval.Ar34__p_Cl33
       -rate_eval.Ar34__He4_S30
       -rho*Y[jn]*rate_eval.n_Ar34__p_Cl34
       -rho*Y[jn]*rate_eval.n_Ar34__He4_S31
       )

    jac[jar36, jn] = (
       -rho*Y[jar36]*rate_eval.n_Ar36__Ar37
       -rho*Y[jar36]*rate_eval.n_Ar36__He4_S33
       )

    jac[jar36, jp] = (
       -rho*Y[jar36]*rate_eval.p_Ar36__He4_Cl33
       +rho*Y[jcl35]*rate_eval.p_Cl35__Ar36
       +rho*Y[jk39]*rate_eval.p_K39__He4_Ar36
       )

    jac[jar36, jhe4] = (
       -rho*Y[jar36]*rate_eval.He4_Ar36__Ca40
       -rho*Y[jar36]*rate_eval.He4_Ar36__p_K39
       +rho*Y[js32]*rate_eval.He4_S32__Ar36
       +rho*Y[js33]*rate_eval.He4_S33__n_Ar36
       +rho*Y[jcl33]*rate_eval.He4_Cl33__p_Ar36
       )

    jac[jar36, js32] = (
       +rho*Y[jhe4]*rate_eval.He4_S32__Ar36
       )

    jac[jar36, js33] = (
       +rho*Y[jhe4]*rate_eval.He4_S33__n_Ar36
       )

    jac[jar36, jcl33] = (
       +rho*Y[jhe4]*rate_eval.He4_Cl33__p_Ar36
       )

    jac[jar36, jcl35] = (
       +rho*Y[jp]*rate_eval.p_Cl35__Ar36
       )

    jac[jar36, jar36] = (
       -rate_eval.Ar36__p_Cl35
       -rate_eval.Ar36__He4_S32
       -rho*Y[jn]*rate_eval.n_Ar36__Ar37
       -rho*Y[jhe4]*rate_eval.He4_Ar36__Ca40
       -rho*Y[jn]*rate_eval.n_Ar36__He4_S33
       -rho*Y[jp]*rate_eval.p_Ar36__He4_Cl33
       -rho*Y[jhe4]*rate_eval.He4_Ar36__p_K39
       )

    jac[jar36, jar37] = (
       +rate_eval.Ar37__n_Ar36
       )

    jac[jar36, jk39] = (
       +rho*Y[jp]*rate_eval.p_K39__He4_Ar36
       )

    jac[jar36, jca40] = (
       +rate_eval.Ca40__He4_Ar36
       )

    jac[jar37, jn] = (
       -rho*Y[jar37]*rate_eval.n_Ar37__Ar38
       +rho*Y[jar36]*rate_eval.n_Ar36__Ar37
       +rho*Y[jca40]*rate_eval.n_Ca40__He4_Ar37
       )

    jac[jar37, jp] = (
       -rho*Y[jar37]*rate_eval.p_Ar37__He4_Cl34
       )

    jac[jar37, jhe4] = (
       -rho*Y[jar37]*rate_eval.He4_Ar37__n_Ca40
       +rho*Y[js33]*rate_eval.He4_S33__Ar37
       +rho*Y[jcl34]*rate_eval.He4_Cl34__p_Ar37
       )

    jac[jar37, js33] = (
       +rho*Y[jhe4]*rate_eval.He4_S33__Ar37
       )

    jac[jar37, jcl34] = (
       +rho*Y[jhe4]*rate_eval.He4_Cl34__p_Ar37
       )

    jac[jar37, jar36] = (
       +rho*Y[jn]*rate_eval.n_Ar36__Ar37
       )

    jac[jar37, jar37] = (
       -rate_eval.Ar37__n_Ar36
       -rate_eval.Ar37__He4_S33
       -rho*Y[jn]*rate_eval.n_Ar37__Ar38
       -rho*Y[jp]*rate_eval.p_Ar37__He4_Cl34
       -rho*Y[jhe4]*rate_eval.He4_Ar37__n_Ca40
       )

    jac[jar37, jar38] = (
       +rate_eval.Ar38__n_Ar37
       )

    jac[jar37, jca40] = (
       +rho*Y[jn]*rate_eval.n_Ca40__He4_Ar37
       )

    jac[jar38, jn] = (
       -rho*Y[jar38]*rate_eval.n_Ar38__Ar39
       +rho*Y[jar37]*rate_eval.n_Ar37__Ar38
       )

    jac[jar38, jp] = (
       -rho*Y[jar38]*rate_eval.p_Ar38__K39
       -rho*Y[jar38]*rate_eval.p_Ar38__He4_Cl35
       )

    jac[jar38, jhe4] = (
       +rho*Y[jcl35]*rate_eval.He4_Cl35__p_Ar38
       )

    jac[jar38, jcl35] = (
       +rho*Y[jhe4]*rate_eval.He4_Cl35__p_Ar38
       )

    jac[jar38, jar37] = (
       +rho*Y[jn]*rate_eval.n_Ar37__Ar38
       )

    jac[jar38, jar38] = (
       -rate_eval.Ar38__n_Ar37
       -rho*Y[jn]*rate_eval.n_Ar38__Ar39
       -rho*Y[jp]*rate_eval.p_Ar38__K39
       -rho*Y[jp]*rate_eval.p_Ar38__He4_Cl35
       )

    jac[jar38, jar39] = (
       +rate_eval.Ar39__n_Ar38
       )

    jac[jar38, jk39] = (
       +rate_eval.K39__p_Ar38
       )

    jac[jar39, jn] = (
       +rho*Y[jar38]*rate_eval.n_Ar38__Ar39
       +rho*Y[jk39]*rate_eval.n_K39__p_Ar39
       )

    jac[jar39, jp] = (
       -rho*Y[jar39]*rate_eval.p_Ar39__n_K39
       )

    jac[jar39, jar38] = (
       +rho*Y[jn]*rate_eval.n_Ar38__Ar39
       )

    jac[jar39, jar39] = (
       -rate_eval.Ar39__K39__weak__wc12
       -rate_eval.Ar39__n_Ar38
       -rho*Y[jp]*rate_eval.p_Ar39__n_K39
       )

    jac[jar39, jk39] = (
       +rho*Y[jn]*rate_eval.n_K39__p_Ar39
       )

    jac[jk39, jn] = (
       -rho*Y[jk39]*rate_eval.n_K39__p_Ar39
       )

    jac[jk39, jp] = (
       -rho*Y[jk39]*rate_eval.p_K39__Ca40
       -rho*Y[jk39]*rate_eval.p_K39__He4_Ar36
       +rho*Y[jar38]*rate_eval.p_Ar38__K39
       +rho*Y[jar39]*rate_eval.p_Ar39__n_K39
       )

    jac[jk39, jhe4] = (
       -rho*Y[jk39]*rate_eval.He4_K39__Sc43
       +rho*Y[jcl35]*rate_eval.He4_Cl35__K39
       +rho*Y[jar36]*rate_eval.He4_Ar36__p_K39
       )

    jac[jk39, jcl35] = (
       +rho*Y[jhe4]*rate_eval.He4_Cl35__K39
       )

    jac[jk39, jar36] = (
       +rho*Y[jhe4]*rate_eval.He4_Ar36__p_K39
       )

    jac[jk39, jar38] = (
       +rho*Y[jp]*rate_eval.p_Ar38__K39
       )

    jac[jk39, jar39] = (
       +rate_eval.Ar39__K39__weak__wc12
       +rho*Y[jp]*rate_eval.p_Ar39__n_K39
       )

    jac[jk39, jk39] = (
       -rate_eval.K39__p_Ar38
       -rate_eval.K39__He4_Cl35
       -rho*Y[jp]*rate_eval.p_K39__Ca40
       -rho*Y[jhe4]*rate_eval.He4_K39__Sc43
       -rho*Y[jn]*rate_eval.n_K39__p_Ar39
       -rho*Y[jp]*rate_eval.p_K39__He4_Ar36
       )

    jac[jk39, jca40] = (
       +rate_eval.Ca40__p_K39
       )

    jac[jk39, jsc43] = (
       +rate_eval.Sc43__He4_K39
       )

    jac[jca40, jn] = (
       -rho*Y[jca40]*rate_eval.n_Ca40__He4_Ar37
       )

    jac[jca40, jp] = (
       +rho*Y[jk39]*rate_eval.p_K39__Ca40
       +rho*Y[jsc43]*rate_eval.p_Sc43__He4_Ca40
       )

    jac[jca40, jhe4] = (
       -rho*Y[jca40]*rate_eval.He4_Ca40__Ti44
       -rho*Y[jca40]*rate_eval.He4_Ca40__p_Sc43
       +rho*Y[jar36]*rate_eval.He4_Ar36__Ca40
       +rho*Y[jar37]*rate_eval.He4_Ar37__n_Ca40
       )

    jac[jca40, jar36] = (
       +rho*Y[jhe4]*rate_eval.He4_Ar36__Ca40
       )

    jac[jca40, jar37] = (
       +rho*Y[jhe4]*rate_eval.He4_Ar37__n_Ca40
       )

    jac[jca40, jk39] = (
       +rho*Y[jp]*rate_eval.p_K39__Ca40
       )

    jac[jca40, jca40] = (
       -rate_eval.Ca40__p_K39
       -rate_eval.Ca40__He4_Ar36
       -rho*Y[jhe4]*rate_eval.He4_Ca40__Ti44
       -rho*Y[jn]*rate_eval.n_Ca40__He4_Ar37
       -rho*Y[jhe4]*rate_eval.He4_Ca40__p_Sc43
       )

    jac[jca40, jsc43] = (
       +rho*Y[jp]*rate_eval.p_Sc43__He4_Ca40
       )

    jac[jca40, jti44] = (
       +rate_eval.Ti44__He4_Ca40
       )

    jac[jsc43, jp] = (
       -rho*Y[jsc43]*rate_eval.p_Sc43__Ti44
       -rho*Y[jsc43]*rate_eval.p_Sc43__He4_Ca40
       )

    jac[jsc43, jhe4] = (
       -rho*Y[jsc43]*rate_eval.He4_Sc43__V47
       +rho*Y[jk39]*rate_eval.He4_K39__Sc43
       +rho*Y[jca40]*rate_eval.He4_Ca40__p_Sc43
       )

    jac[jsc43, jk39] = (
       +rho*Y[jhe4]*rate_eval.He4_K39__Sc43
       )

    jac[jsc43, jca40] = (
       +rho*Y[jhe4]*rate_eval.He4_Ca40__p_Sc43
       )

    jac[jsc43, jsc43] = (
       -rate_eval.Sc43__He4_K39
       -rho*Y[jp]*rate_eval.p_Sc43__Ti44
       -rho*Y[jhe4]*rate_eval.He4_Sc43__V47
       -rho*Y[jp]*rate_eval.p_Sc43__He4_Ca40
       )

    jac[jsc43, jti44] = (
       +rate_eval.Ti44__p_Sc43
       )

    jac[jsc43, jv47] = (
       +rate_eval.V47__He4_Sc43
       )

    jac[jti44, jp] = (
       +rho*Y[jsc43]*rate_eval.p_Sc43__Ti44
       +rho*Y[jv47]*rate_eval.p_V47__He4_Ti44
       )

    jac[jti44, jhe4] = (
       -rho*Y[jti44]*rate_eval.He4_Ti44__Cr48
       -rho*Y[jti44]*rate_eval.He4_Ti44__p_V47
       +rho*Y[jca40]*rate_eval.He4_Ca40__Ti44
       )

    jac[jti44, jca40] = (
       +rho*Y[jhe4]*rate_eval.He4_Ca40__Ti44
       )

    jac[jti44, jsc43] = (
       +rho*Y[jp]*rate_eval.p_Sc43__Ti44
       )

    jac[jti44, jti44] = (
       -rate_eval.Ti44__p_Sc43
       -rate_eval.Ti44__He4_Ca40
       -rho*Y[jhe4]*rate_eval.He4_Ti44__Cr48
       -rho*Y[jhe4]*rate_eval.He4_Ti44__p_V47
       )

    jac[jti44, jv47] = (
       +rho*Y[jp]*rate_eval.p_V47__He4_Ti44
       )

    jac[jti44, jcr48] = (
       +rate_eval.Cr48__He4_Ti44
       )

    jac[jv47, jp] = (
       -rho*Y[jv47]*rate_eval.p_V47__Cr48
       -rho*Y[jv47]*rate_eval.p_V47__He4_Ti44
       )

    jac[jv47, jhe4] = (
       -rho*Y[jv47]*rate_eval.He4_V47__Mn51
       +rho*Y[jsc43]*rate_eval.He4_Sc43__V47
       +rho*Y[jti44]*rate_eval.He4_Ti44__p_V47
       )

    jac[jv47, jsc43] = (
       +rho*Y[jhe4]*rate_eval.He4_Sc43__V47
       )

    jac[jv47, jti44] = (
       +rho*Y[jhe4]*rate_eval.He4_Ti44__p_V47
       )

    jac[jv47, jv47] = (
       -rate_eval.V47__He4_Sc43
       -rho*Y[jp]*rate_eval.p_V47__Cr48
       -rho*Y[jhe4]*rate_eval.He4_V47__Mn51
       -rho*Y[jp]*rate_eval.p_V47__He4_Ti44
       )

    jac[jv47, jcr48] = (
       +rate_eval.Cr48__p_V47
       )

    jac[jv47, jmn51] = (
       +rate_eval.Mn51__He4_V47
       )

    jac[jcr48, jp] = (
       +rho*Y[jv47]*rate_eval.p_V47__Cr48
       +rho*Y[jmn51]*rate_eval.p_Mn51__He4_Cr48
       )

    jac[jcr48, jhe4] = (
       -rho*Y[jcr48]*rate_eval.He4_Cr48__Fe52
       -rho*Y[jcr48]*rate_eval.He4_Cr48__p_Mn51
       +rho*Y[jti44]*rate_eval.He4_Ti44__Cr48
       )

    jac[jcr48, jti44] = (
       +rho*Y[jhe4]*rate_eval.He4_Ti44__Cr48
       )

    jac[jcr48, jv47] = (
       +rho*Y[jp]*rate_eval.p_V47__Cr48
       )

    jac[jcr48, jcr48] = (
       -rate_eval.Cr48__p_V47
       -rate_eval.Cr48__He4_Ti44
       -rho*Y[jhe4]*rate_eval.He4_Cr48__Fe52
       -rho*Y[jhe4]*rate_eval.He4_Cr48__p_Mn51
       )

    jac[jcr48, jmn51] = (
       +rho*Y[jp]*rate_eval.p_Mn51__He4_Cr48
       )

    jac[jcr48, jfe52] = (
       +rate_eval.Fe52__He4_Cr48
       )

    jac[jmn51, jp] = (
       -rho*Y[jmn51]*rate_eval.p_Mn51__Fe52
       -rho*Y[jmn51]*rate_eval.p_Mn51__He4_Cr48
       )

    jac[jmn51, jhe4] = (
       -rho*Y[jmn51]*rate_eval.He4_Mn51__Co55
       +rho*Y[jv47]*rate_eval.He4_V47__Mn51
       +rho*Y[jcr48]*rate_eval.He4_Cr48__p_Mn51
       )

    jac[jmn51, jv47] = (
       +rho*Y[jhe4]*rate_eval.He4_V47__Mn51
       )

    jac[jmn51, jcr48] = (
       +rho*Y[jhe4]*rate_eval.He4_Cr48__p_Mn51
       )

    jac[jmn51, jmn51] = (
       -rate_eval.Mn51__He4_V47
       -rho*Y[jp]*rate_eval.p_Mn51__Fe52
       -rho*Y[jhe4]*rate_eval.He4_Mn51__Co55
       -rho*Y[jp]*rate_eval.p_Mn51__He4_Cr48
       )

    jac[jmn51, jfe52] = (
       +rate_eval.Fe52__p_Mn51
       )

    jac[jmn51, jco55] = (
       +rate_eval.Co55__He4_Mn51
       )

    jac[jfe52, jp] = (
       +rho*Y[jmn51]*rate_eval.p_Mn51__Fe52
       +rho*Y[jco55]*rate_eval.p_Co55__He4_Fe52
       )

    jac[jfe52, jhe4] = (
       -rho*Y[jfe52]*rate_eval.He4_Fe52__Ni56
       -rho*Y[jfe52]*rate_eval.He4_Fe52__p_Co55
       +rho*Y[jcr48]*rate_eval.He4_Cr48__Fe52
       )

    jac[jfe52, jcr48] = (
       +rho*Y[jhe4]*rate_eval.He4_Cr48__Fe52
       )

    jac[jfe52, jmn51] = (
       +rho*Y[jp]*rate_eval.p_Mn51__Fe52
       )

    jac[jfe52, jfe52] = (
       -rate_eval.Fe52__p_Mn51
       -rate_eval.Fe52__He4_Cr48
       -rho*Y[jhe4]*rate_eval.He4_Fe52__Ni56
       -rho*Y[jhe4]*rate_eval.He4_Fe52__p_Co55
       )

    jac[jfe52, jco55] = (
       +rho*Y[jp]*rate_eval.p_Co55__He4_Fe52
       )

    jac[jfe52, jni56] = (
       +rate_eval.Ni56__He4_Fe52
       )

    jac[jfe55, jn] = (
       +rho*Y[jco55]*rate_eval.n_Co55__p_Fe55
       +rho*Y[jni58]*rate_eval.n_Ni58__He4_Fe55
       )

    jac[jfe55, jp] = (
       -rho*Y[jfe55]*rate_eval.p_Fe55__n_Co55
       )

    jac[jfe55, jhe4] = (
       -rho*Y[jfe55]*rate_eval.He4_Fe55__Ni59
       -rho*Y[jfe55]*rate_eval.He4_Fe55__n_Ni58
       )

    jac[jfe55, jfe55] = (
       -rho*Y[jhe4]*rate_eval.He4_Fe55__Ni59
       -rho*Y[jp]*rate_eval.p_Fe55__n_Co55
       -rho*Y[jhe4]*rate_eval.He4_Fe55__n_Ni58
       )

    jac[jfe55, jco55] = (
       +rate_eval.Co55__Fe55__weak__wc12
       +rho*Y[jn]*rate_eval.n_Co55__p_Fe55
       )

    jac[jfe55, jni58] = (
       +rho*Y[jn]*rate_eval.n_Ni58__He4_Fe55
       )

    jac[jfe55, jni59] = (
       +rate_eval.Ni59__He4_Fe55
       )

    jac[jco55, jn] = (
       -rho*Y[jco55]*rate_eval.n_Co55__p_Fe55
       )

    jac[jco55, jp] = (
       -rho*Y[jco55]*rate_eval.p_Co55__Ni56
       -rho*Y[jco55]*rate_eval.p_Co55__He4_Fe52
       +rho*Y[jfe55]*rate_eval.p_Fe55__n_Co55
       +rho*Y[jni58]*rate_eval.p_Ni58__He4_Co55
       )

    jac[jco55, jhe4] = (
       -rho*Y[jco55]*rate_eval.He4_Co55__p_Ni58
       +rho*Y[jmn51]*rate_eval.He4_Mn51__Co55
       +rho*Y[jfe52]*rate_eval.He4_Fe52__p_Co55
       )

    jac[jco55, jmn51] = (
       +rho*Y[jhe4]*rate_eval.He4_Mn51__Co55
       )

    jac[jco55, jfe52] = (
       +rho*Y[jhe4]*rate_eval.He4_Fe52__p_Co55
       )

    jac[jco55, jfe55] = (
       +rho*Y[jp]*rate_eval.p_Fe55__n_Co55
       )

    jac[jco55, jco55] = (
       -rate_eval.Co55__Fe55__weak__wc12
       -rate_eval.Co55__He4_Mn51
       -rho*Y[jp]*rate_eval.p_Co55__Ni56
       -rho*Y[jn]*rate_eval.n_Co55__p_Fe55
       -rho*Y[jp]*rate_eval.p_Co55__He4_Fe52
       -rho*Y[jhe4]*rate_eval.He4_Co55__p_Ni58
       )

    jac[jco55, jni56] = (
       +rate_eval.Ni56__p_Co55
       )

    jac[jco55, jni58] = (
       +rho*Y[jp]*rate_eval.p_Ni58__He4_Co55
       )

    jac[jni56, jp] = (
       +rho*Y[jco55]*rate_eval.p_Co55__Ni56
       )

    jac[jni56, jhe4] = (
       +rho*Y[jfe52]*rate_eval.He4_Fe52__Ni56
       )

    jac[jni56, jfe52] = (
       +rho*Y[jhe4]*rate_eval.He4_Fe52__Ni56
       )

    jac[jni56, jco55] = (
       +rho*Y[jp]*rate_eval.p_Co55__Ni56
       )

    jac[jni56, jni56] = (
       -rate_eval.Ni56__p_Co55
       -rate_eval.Ni56__He4_Fe52
       )

    jac[jni58, jn] = (
       -rho*Y[jni58]*rate_eval.n_Ni58__Ni59
       -rho*Y[jni58]*rate_eval.n_Ni58__He4_Fe55
       )

    jac[jni58, jp] = (
       -rho*Y[jni58]*rate_eval.p_Ni58__He4_Co55
       )

    jac[jni58, jhe4] = (
       +rho*Y[jfe55]*rate_eval.He4_Fe55__n_Ni58
       +rho*Y[jco55]*rate_eval.He4_Co55__p_Ni58
       )

    jac[jni58, jfe55] = (
       +rho*Y[jhe4]*rate_eval.He4_Fe55__n_Ni58
       )

    jac[jni58, jco55] = (
       +rho*Y[jhe4]*rate_eval.He4_Co55__p_Ni58
       )

    jac[jni58, jni58] = (
       -rho*Y[jn]*rate_eval.n_Ni58__Ni59
       -rho*Y[jn]*rate_eval.n_Ni58__He4_Fe55
       -rho*Y[jp]*rate_eval.p_Ni58__He4_Co55
       )

    jac[jni58, jni59] = (
       +rate_eval.Ni59__n_Ni58
       )

    jac[jni59, jn] = (
       +rho*Y[jni58]*rate_eval.n_Ni58__Ni59
       )

    jac[jni59, jhe4] = (
       +rho*Y[jfe55]*rate_eval.He4_Fe55__Ni59
       )

    jac[jni59, jfe55] = (
       +rho*Y[jhe4]*rate_eval.He4_Fe55__Ni59
       )

    jac[jni59, jni58] = (
       +rho*Y[jn]*rate_eval.n_Ni58__Ni59
       )

    jac[jni59, jni59] = (
       -rate_eval.Ni59__n_Ni58
       -rate_eval.Ni59__He4_Fe55
       )

    return jac
