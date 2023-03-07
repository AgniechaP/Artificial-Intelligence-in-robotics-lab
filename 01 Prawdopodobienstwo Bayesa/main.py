#!/usr/bin/env python

"""code template"""

import numpy as np


def main():
    #Zadanie 1. - łączne prawdopodobieństwo jako tablica numpy
    # ([
    #    [[ tootache & cavity], [~toothache & cavity]],
    #    [[ tootache & ~cavity], [~toothache & ~cavity]] ])
    P = np.array([[[0.108, 0.012], [0.072, 0.008]],
                 [[0.016, 0.064], [0.144, 0.576]]])
    print("1. Tablica - prawdopodobienstwo laczne:\n", P)

    #Zadanie 2. P(Toothache)
    #axis 0 -> cavity, axis 1 -> toothache, axis 2 -> catch
    P_too = np.sum(P, axis=(0,2))
    print("P_too = ", P_too)

    #Zadanie 3. P(Cavity)
    P_cav = np.sum(P, axis=(1,2))
    print("P_cav = ",P_cav)

    #Zadanie 4. P(Toothache|Cavity)
    P_toothache_giv_cavity = np.sum(P, axis=2) / P_cav
    print("P_toothache_giv_cavity = \n", P_toothache_giv_cavity)

    #Zadanie 5. P(Cavity|toothache v catch)
    licznik = P_cav - P[:,1,1]
    mianownik = np.sum(P, axis=(0,1,2)) - np.sum(P[:,1,1], axis=0)
    P_cavity_giv_toothache_or_catch = licznik/mianownik
    print("P_cavity_giv_toothache_or_catch = \n", P_cavity_giv_toothache_or_catch)


if __name__ == '__main__':
    main()
