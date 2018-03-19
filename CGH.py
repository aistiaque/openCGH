import numpy as np
import cv2
import matplotlib.pyplot as plt




def hologram(url):

    #make an image array from the given image

    img = cv2.imread(url, 0)            #importing the image in grayscale and turning it into an array

    #determine the dimension of the image

    M = img.shape[0]                    #Width of the image
    N = img.shape[1]                    #Height of the image




    #formation of the object wave

    objectFourier = np.fft.fft2(img)                                   #Far field amplitude of the object wave
    shiftObject = np.fft.fftshift(objectFourier)                       #shifting the whites to the center
    objectWave = np.abs(shiftObject) * np.abs(shiftObject)             #absolute value of the image

    magnitude_spectrum = 20 * np.log(objectWave)                       #obtain the magnitude spectrum






    #formation of the reference wave

    reference = np.zeros((M, N))                                        #creating an array and filling it with zeros

    T1 = M / 2                                                          #finding the mid point
    T1 = T1 + 1                                                         #finding the mid point
    T2 = N / 2
    T2 = T2 + 1

    reference[T1, T2] = 100                                              #illuminate a point in the reference array

    referenceFourier = np.fft.fft2(reference)                            #applying Fast Fourier on the reference wave array
    referenceShift = np.fft.fftshift(referenceFourier)
    referenceWave = np.abs(referenceShift) * np.abs(referenceShift)





    # adding both the waves

    superpositionWave = shiftObject + referenceShift                                  #superposition of the waves

    superpositionedFinalWave = np.abs(superpositionWave) * np.abs(superpositionWave)

    hologram = superpositionedFinalWave - objectWave                                  #final interference pattern




    # reconstruction
    reconstructedWave = np.fft.fft2(hologram)
    reconstructShift = np.fft.fftshift(reconstructedWave)
    finalReconstruction = np.abs(reconstructShift) * np.abs(reconstructShift)        #RECONSTRUCTED image







    # plotting
    plt.subplot(231), plt.imshow(img, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(232), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.subplot(233), plt.imshow(objectWave, cmap='hot')
    plt.title('Object'), plt.xticks([]), plt.yticks([])
    plt.subplot(234), plt.imshow(referenceWave, cmap='gray')
    plt.title('Ref'), plt.xticks([]), plt.yticks([])
    plt.subplot(235), plt.imshow(hologram, cmap='gray')
    plt.title('Without filtering'), plt.xticks([]), plt.yticks([])
    plt.subplot(236), plt.imshow(finalReconstruction, cmap='gray')
    plt.title('Reconstructed image'), plt.xticks([]), plt.yticks([])
    plt.show()

