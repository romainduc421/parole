from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import write
import numpy.fft as FFT

# Fonction qui ouvre un fichier wav et
# renvoie la fréquence d'échantillonnage, le signal et le nombre d'échantillons
# Etape 1. Ouverture du fichier wav
def ouvertureWav(filename = 'test_seg.wav'):
    fichier = filename
    frequence_enchantillonage, valeurs_signal = read(fichier)
    nb_echantillon = valeurs_signal.shape[0]
    duree_ms = 1000 * nb_echantillon / frequence_enchantillonage

    return frequence_enchantillonage, valeurs_signal, nb_echantillon

# Fonction qui renvoie une fenêtre de Hamming
# Etape 2. Fenetrage de Hamming
def fenetrageHamming(N):
    return 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(N) / (N - 1))

# Fonction qui renvoie le fenetrage de Hamming d'un signal de taille N
def fenetrageHammingSignal(signal, N):
    for i in range(len(signal)):
        signal[i] = signal[i] * fenetrageHamming(N)
    return signal

# Fonction qui renvoie un tableau de morceaux de 32ms
# Etape 2. Récupération de la fenêtre à l'instant i et de taille m
def getMorceau32ms(signal, m, N):
    nb_fenetres = int((len(signal) - N) / m) + 1
    m32ms = np.zeros((nb_fenetres, N))
    for i in range(nb_fenetres):
        debut_fenetre = i * m
        fin_fenetre = debut_fenetre + N
        m32ms[i] = signal[debut_fenetre:fin_fenetre]
    return m32ms

# Fonction qui reconstruit le signal
# Etape 2. Reconstitution du signal
def reconstructionSignal(morceau32ms, m, N, valeurs_signal):
    signal_modif = np.zeros(len(valeurs_signal))
    somme_hamming = np.zeros(len(valeurs_signal))
    for i in range(len(morceau32ms)):
        debut_fenetre = i * m
        fin_fenetre = debut_fenetre + N
        signal_modif[debut_fenetre:fin_fenetre] += morceau32ms[i]
        somme_hamming[debut_fenetre:fin_fenetre] += fenetrageHamming(N)

    # On remplace les 0 par 1 pour éviter les divisions par 0
    for i in range(len(somme_hamming)):
        if somme_hamming[i] == 0:
            somme_hamming[i] = 1

    signal_modif = signal_modif / somme_hamming
    return signal_modif, somme_hamming

# Fonction qui calcule le spectre d'amplitude sur une fenêtre de 32ms
# Etape 4. Spectre d'amplitude
# spectre_amplitude[k] = 20.log(|X_k(o)|)
def spectreAmplitude(spectre, fftsize):
    spectre_amplitude_log = 20 * np.log10(np.abs(spectre))
    spectre_amplitude = np.abs(spectre)
    return spectre_amplitude_log, spectre_amplitude

# Fonction qui calcule la transformée de Fourier inverse
# Etape 3. Calcul de la transformée de Fourier inverse
def fourierInverse(fourier):
    signal = []
    for i in range(len(fourier)):
        signal.append(np.real(FFT.ifft(fourier[i], 1024)))
    return signal

# Fonction qui calcule la transformée de Fourier
# Etape 3. Calcul de la transformée de Fourier
def transformerFourier(morceaux):
    fourier = []
    for i in range(len(morceaux)):
        fourier.append(FFT.fft(morceaux[i], 1024))
    return fourier

# Fonction qui calcule le spectre de phase
# Etape 6. Spectre de phase
def spectrePhase(fourier, fftsize):
    spectre_phase = np.angle(fourier)
    return spectre_phase

# Fonction qui calcule le spectre de puissance
# Etape 7. Reconstruction du spectre
def spectrereconstruction (spectre_amplitude, spectre_phase,fftsize):
    spectre_reconstruction = spectre_amplitude * np.exp(1j * spectre_phase)
    return spectre_reconstruction

# Fonction qui réalise la moyenne sur les 4-5 premiers spectres
# Etape 8. Estimation du spectre de bruit
def spectreAmplitudeBruit(spectre_amplitude, fftsize):
    spectre_amplitude_bruit = np.zeros(fftsize)
    for i in range(4):
        spectre_amplitude_bruit += spectre_amplitude[i]
    spectre_amplitude_bruit = spectre_amplitude_bruit / 4
    return spectre_amplitude_bruit

# Fonction qui réalise le débruitage
# Etape 9. Débruitage par soustraction spectrale
def spectreAmplitudeDebruitage(spectre_amplitude, spectre_amplitude_bruit, fftsize):
    alpha = 2
    beta = 1
    gamma = 0
    spectre_amplitude_debruitage = np.zeros(fftsize)
    for k in range(len(spectre_amplitude)):
        soustraction = ((spectre_amplitude[k] ** alpha) - beta*(spectre_amplitude_bruit[k] ** alpha))**1/alpha
        spectre_amplitude_debruitage[k] = soustraction if soustraction > 0 else gamma*spectre_amplitude_bruit[k]
    return spectre_amplitude_debruitage

def main():
    ## Etape 1. Ouverture du fichier wav
    # Récupération de la fréquence d'échantillonnage, du signal et du nombre d'échantillons
    frequence_enchantillonage, valeurs_signal, nb_echantillon = ouvertureWav()
    print("Fréquence d'échantillonnage : ", frequence_enchantillonage)
    print("Nombre d'échantillons : ", nb_echantillon)
    print("Signal : ", valeurs_signal)

    ## Etape 2. Fenetrage de Hamming
    # Variables de découpage (tout les 8ms et fenêtre de 32ms)
    m = 8 * frequence_enchantillonage // 1000
    N = 32 * frequence_enchantillonage // 1000
    morceau32ms = getMorceau32ms(valeurs_signal, m, N)
    print("Morceaux de 32ms : ", morceau32ms)
    # Fenêtre de Hamming
    morceau32ms = fenetrageHammingSignal(morceau32ms, N)

    ## Etape 3. Calcul de la transformée de Fourier
    fourier = transformerFourier(morceau32ms)

    ## Etape 4. Calcul du spectre d'amplitude
    # Calcul du spectre d'amplitude
    spectre_amplitude_log, spectre_amplitude = spectreAmplitude(fourier, 1024)
    # Transpose le tableau pour avoir les bonnes dimensions
    spectre_amplitude_log = spectre_amplitude_log.T

    ## Etape 5. Pause sur le debruitage
    plt.imshow(spectre_amplitude_log, aspect='auto')
    plt.show()

    ## Etape 6. Spectre de phase
    spectre_phase = spectrePhase(fourier, 1024)

    ## Etape 8.9. Traitement sur le spectre d'amplitude
    # Estimation du spectre d'amplitude du bruit
    spectre_amplitude_bruit = spectreAmplitudeBruit(spectre_amplitude, 1024)
    # Débruitage par soustraction spectrale
    spectre_amplitude = spectreAmplitudeDebruitage(spectre_amplitude, spectre_amplitude_bruit, 1024)

    ## Etape 7. Reconstruction du signal
    # Reconstruction du spectre
    spectre_reconstruction = spectrereconstruction(spectre_amplitude, spectre_phase, 1024)

    ## Etape 3. Calcul de la transformée de Fourier inverse
    signal = fourierInverse(spectre_reconstruction)
    print("Signal : ", signal)

    ## Reconstitution du signal
    signal_modif, somme_hamming = reconstructionSignal(signal, m, N, valeurs_signal)
    print ("Signal modifié : ", signal_modif)
    # Création du fichier wav
    write("resultat.wav", frequence_enchantillonage, np.int16(signal_modif))

if __name__ == "__main__":
    main()



