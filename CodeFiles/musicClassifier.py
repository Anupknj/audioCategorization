import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
import os


filePath = os.path.dirname(__file__)
storedPath = filePath + "\\data\\trainVIz"
figureSize = (7,4)

# this function takes file as input and returns images(MFCC, spectrogram and spectrum) and stores in given path
def createDataRepresentation(file):
    # load audio file with Librosa
    signal, sample_rate = librosa.load(file, sr=22050)
    # FFT -> power spectrum
    # perform Fourier transform
    fft = np.fft.fft(signal)
    # # calculate abs values on complex numbers to get magnitude
    spectrum = np.abs(fft)
    # create frequency variable
    f = np.linspace(0, sample_rate, len(spectrum))
    # take half of the spectrum and frequency
    leftSpectrum = spectrum[:int(len(spectrum)/2)]
    leftF = f[:int(len(spectrum)/2)]

    # plot spectrum

    plt.figure(figsize=figureSize)
    plt.plot(leftF, leftSpectrum, alpha=0.4)
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.title("Power spectrum")
    fileName = file.split("\\").pop() + "Spectrum.jpg"
    plt.savefig(os.path.join(storedPath, fileName))  
    plt.show(block=False)
    plt.close()


    # STFT -> spectrogram
    hopLength = 512 # in num. of samples
    nFFT = 2048 # window in num. of samples
    # calculate duration hop length and window in seconds
    hopLength_duration = float(hopLength)/sample_rate
    nFFT_duration = float(nFFT)/sample_rate
    print("STFT hop length duration is: {}s".format(hopLength_duration))
    print("STFT window duration is: {}s".format(nFFT_duration))
    # perform stft
    stft = librosa.stft(signal, n_fft=nFFT, hop_length=hopLength)
    # calculate abs values on complex numbers to get magnitude
    spectrogram = np.abs(stft)
    # apply logarithm to cast amplitude to Decibels
    log_spectrogram = librosa.amplitude_to_db(spectrogram)


    #plot spectrogram

    plt.figure(figsize=figureSize)
    librosa.display.specshow(log_spectrogram, sr=sample_rate, hop_length=hopLength)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram (dB)")
    fileName = file.split("\\").pop() + "Spectrogram.jpg"

    plt.savefig(os.path.join(storedPath, fileName))  
    plt.show(block=False)
    plt.close()


    # MFCCs
    # extract 13 MFCCs

    MFCCs = librosa.feature.mfcc(signal, sample_rate, n_fft=nFFT, hop_length=hopLength, n_mfcc=13)

     # display MFCCs
    plt.figure(figsize=figureSize)
    librosa.display.specshow(MFCCs, sr=sample_rate, hop_length=hopLength)
    plt.xlabel("Time")
    plt.ylabel("MFCC coefficients")
    plt.colorbar()
    plt.title("MFCCs")
    fileName = file.split("\\").pop() + "MFCC.jpg"
    # show plots
    plt.savefig(os.path.join(storedPath, fileName))  
    plt.show(block=False)
    plt.close()


# iterates through files given path
def iterateFilesinFolder():
    path = r'.\data\train'
    folder = os.fsencode(path)
    for file in os.listdir(folder):
        print(os.listdir(folder).index(file))
        filename = os.fsdecode(file)
        if filename.endswith( ('mp3','wav') ): 
            createDataRepresentation(path+"\\"+filename) # above function is called passing the path of the file


iterateFilesinFolder()