from scipy.io.wavfile import read, write
import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as fft
import math
import sys

#afficher le spectre audio du fichier wav
def printinitialspeechwav(rate, data):
    # Afficher les informations sur le fichier
    nb_echantillons = len(data)
    duree = nb_echantillons / float(rate) * 1000
    print("Fréquence d'échantillonnage : {} Hz".format(rate))
    print("Taille du fichier : {} échantillons ({} ms)".format(nb_echantillons, duree))

    # Afficher le signal
    plt.figure(1)
    plt.subplot(211)
    plt.plot(data, color='lightblue')
    plt.title("Signal audio")

#renvoyer une fenêtre de Hamming
def hammingwindowing(size):
    c1, c2 = 0.54, 0.46
    hamming = []
    for k in range(size):
        hamming.append( c1 - c2 * math.cos(2 * math.pi * k / (size - 1)) )
    # plt.figure(3)
    # plt.title("Hamming window")
    # plt.plot(hamming, color='green')
    return hamming

#fenêtrage du morceau de signal (32ms)
def windowing(signal, hamming):
    return [ signal[_]*hamming[_]  for _ in range (len(hamming)) ]
    


# calcul du spectre d'amplitude sur une fenêtre
# Etape 4. spectre d'amplitude
# spectre_amplitude[k] = 20.log(|X_k(o)|)
def amplitudespectrum(spectre, fftsize):
    res = np.abs(spectre)
    spec_affichage = 20 * np.log10(res)
    # plt.figure(4)
    # plt.title("Amplitude spectrum")
    # plt.plot(spec_affichage)
    return res, spec_affichage[0: int(fftsize/2)]

# calcul de la phase du spectre
# Etape 6.
def spectrumphase(fourier):
    return np.angle(fourier)

# reconstruction du signal
# reconstruire un spectre complexe à partir du module et de la phase
# Etape 7.
def spectrereconstruction(spectre_amplitude, spectre_phase):
    reconstruction = spectre_amplitude * np.exp(1j * spectre_phase)
    return reconstruction


# débruitage par soustraction spectrale
# soustrait au spectre d'amplitude le spectre de l'estimation du bruit
# Etape 9. Débruitage par soustraction spectrale
def spectralsubstraction(spectre_amplitude, bruit, alpha, beta, gamma):
    # alpha = 2 #to adjust
    # beta = 8 #to adjust entre 8 et 10
    # gamma = 0.2 #garder petit < 0.5 si supérieur à 1 on rajoute du bruit
    res = np.zeros(len(spectre_amplitude))
    y = beta * (bruit ** alpha)
    for i in range(len(spectre_amplitude)):
        soustraction =  (spectre_amplitude[i] ** alpha) - y
        res[i] = soustraction ** (1.0/alpha) if (soustraction) > 0 else gamma * bruit

    return res



# Reconstruction du signal
def signalreconstruction(signal, m, N, alpha=2.0, beta=8.0, gamma=0.2):
    signal_modif, somme_hamming = np.zeros(len(signal)), np.zeros(len(signal))
    tab_spectres = [] # pour le stockage des spectres d'amplitude
    hamming = hammingwindowing(N)
    fftsize = 1024

    # Q8- noise estimation
    noise, size = 0, 5 #valeur bruit, nombre de fenetres traitees : 4-5 premiers spectres

    for i in range(0, len(signal) - N, m):
        fenetre = np.array(windowing(signal[i:i+N], hamming), dtype=float)
        spectre = fft.fft(fenetre, fftsize)
        amplitude, ampli_aff = amplitudespectrum(spectre, fftsize)
        
        
        if i < m*size - m:
            noise += np.mean(amplitude)
        if i == m*size - m:
            noise /= size  # diviser la somme totale des amplitudes spectrales par le nombre de fenêtres (sous-signaux traités)
            
        tab_spectres.append(ampli_aff)
    
            
        phase = spectrumphase(spectre)
        # modification spectre d'amplitude
        amplitude = spectralsubstraction(amplitude, noise, alpha, beta, gamma)

        # reconstruction du signal
        spectre = spectrereconstruction(amplitude, phase)
        fenetre = np.real(fft.ifft(spectre, fftsize))

        signal_modif[i:i+N] += fenetre[0:N]
        somme_hamming[i:i+N] += hamming
    
    # 5 - Pause sur le débruitage
    np.transpose(tab_spectres)
    plt.figure(7)
    # affichage spectrogramme
    plt.imshow(tab_spectres,aspect='auto', origin='lower')
    signal_modif = signal_modif / somme_hamming
    return signal_modif, tab_spectres



def main(file):

    np.seterr(invalid='ignore')
    fs, signal = read("fichiers_bruit/"+file)
    printinitialspeechwav(fs, signal)
    m, N = 8*fs//1000, 32*fs//1000

    # pour test_seg, test_seg_bruit_10dB
    # on a : m = 176.4, N = 705.6
    signal_best_modif, _ = signalreconstruction( signal, m, N, 2, 8, 0.2)
    for alpha in range (1,6):
        for beta in range(7,11,1):
            for gamma in np.arange(0.1,0.6,0.1):
                signal_modif, _ = signalreconstruction( signal, m, N, alpha, beta, gamma)
                write("output/output_A{}_B{}_G{}_{}".format(file.rsplit('.', maxsplit=1)[0], alpha, beta, gamma), fs, np.int16( signal_modif ))
    plt.figure(1).set_figheight(10)
    plt.figure(1).set_figwidth(12)
    plt.subplot(212)
    plt.plot(signal_best_modif)
    plt.title("Denoise signal")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print("Usage : python3 {}".format(sys.argv[0]))
        sys.exit(1)
    else:
        main("test_seg_bruit_10dB.wav")
        #main("test_seg_bruit_0dB.wav")
        #main("signal_avec_bruit_0dB.wav")