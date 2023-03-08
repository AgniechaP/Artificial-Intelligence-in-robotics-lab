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

    #Zadanie 2. P(Toothache) wykorzystac mozemy wzor na prawdopodobienstwo brzegowe, marginalizujemy to co niepotrzebne, suma odbywa się po zmiennych, których chcemy się pozbyć
    #axis 0 -> cavity, axis 1 -> toothache, axis 2 -> catch
    P_too = np.sum(P, axis=(0,2))
    print("P_too = ", P_too)

    # Zadanie 2. z zajęć:
    P_cav_too = np.sum(P,axis=-1)
    # P_too = np.sum(P_cav_too, axis=0)

    #Zadanie 3. z zajęć:
    # P_cav = np.sum(P_cav_too, axis=-1)

    #Zadanie 3. P(Cavity)
    P_cav = np.sum(P, axis=(1,2))
    print("P_cav = ",P_cav)

    #Zadanie 4. P(Toothache|Cavity)
    P_toothache_giv_cavity = np.sum(P, axis=2) / P_cav
    print("P_toothache_giv_cavity = \n", P_toothache_giv_cavity)

    #Zadanie 4. Z zajęć: z reguly iloczynu bierzemy czlon odpowiadajacy za prawdopodobienstwo warunkowe
    P_too_giv_cav_z_zajec = P_cav_too.T/P_cav #Dokonujemy transpozycji, bo liczylismy P(b|a) a we wzorze mamy P(a|b)
    print("P_too_giv_cav_z_zajec: ", P_too_giv_cav_z_zajec)

    #Zadanie 5. P(Cavity|toothache v catch)
    licznik = P_cav - P[:,1,1]
    mianownik = np.sum(P, axis=(0,1,2)) - np.sum(P[:,1,1], axis=0)
    P_cavity_giv_toothache_or_catch = licznik/mianownik
    print("P_cavity_giv_toothache_or_catch = \n", P_cavity_giv_toothache_or_catch)

    #Zadanie 6. Zaleznosc wielkosci tablicy od liczby zmiennych, gdzie zmienne sa binarne 0/1:
    #Wielkosc tablicy wyraza sie jako 2^liczba zmiennych. Tu 2^3 = 8 i jest to wielkosc tablicy

    #Zadanie 7. Przechowywanie tablicy dla 32 zmiennych zapisujac liczby jako 23 bitowy float
    bits = 32 * 2**32 #ok. 16 GB
    print("Rozmiar (bity): ", bits)

    #Zadanie 8. P(Cavity | Tootache, Catch)
    # (1) Regula iloczynu: P(a,b) = P(a|b)P(b)
    # (2) Regula Bayesa: P(b|a) = P(a|b)*P(b)/P(a)
    # Zatem z (2) P(Cav|Too,Cat)=P(Too,Cat|Cav)*P(Cav)/P(Too,Cat)
    # Z (1) P(Cav|Too,Cat)=P(Too,Cat,Cav)/P(Too,Cat)
    # Stąd P(Too,Cat,Cav)=P(Too,Cat|Cav)*P(Cav)

    P_too_cat_giv_cav = P/np.reshape(P_cav, (2, 1, 1))
    print("P_too_cat_giv_cav ", P_too_cat_giv_cav)

    P_too_cat = np.sum(P, axis=0)
    print("P_too_cat: ", P_too_cat)

    P_cav_giv_too_cat = P/P_too_cat
    print("P_cav_giv_too_cat ",P_cav_giv_too_cat)

    #Zadanie 9.


if __name__ == '__main__':
    main()
