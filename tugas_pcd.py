import cv2
import numpy as np
import matplotlib.pyplot as plt

print("Memproses Tugas 1: Segmentasi Sobel...")

image_path_pear = 'dataset/pear/r0_1.jpg' 

img_pear = cv2.imread(image_path_pear)
pear_loaded = False 

if img_pear is None:
    print(f"Error: Tidak bisa membaca gambar dari {image_path_pear}")
else:
    pear_loaded = True
    gray_pear = cv2.cvtColor(img_pear, cv2.COLOR_BGR2GRAY)

    sobel_x_pear = cv2.Sobel(gray_pear, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y_pear = cv2.Sobel(gray_pear, cv2.CV_64F, 0, 1, ksize=5)
    sobel_magnitude_pear = cv2.magnitude(sobel_x_pear, sobel_y_pear)
    sobel_edges_pear = cv2.convertScaleAbs(sobel_magnitude_pear)

    _, binary_edges_pear = cv2.threshold(sobel_edges_pear, 50, 255, cv2.THRESH_BINARY)

    contours_pear, _ = cv2.findContours(binary_edges_pear, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask_pear = np.zeros_like(gray_pear)
    cv2.drawContours(mask_pear, contours_pear, -1, (255), cv2.FILLED)
    final_segmented_pear = cv2.bitwise_and(img_pear, img_pear, mask=mask_pear)

    print("Selesai memproses Sobel.")


print("\nMemproses Tugas 2: Deteksi Tepi Canny...")

image_path_nanas = 'dataset/nanas/NanasMuda (20).jpg'

img_nanas = cv2.imread(image_path_nanas)
nanas_loaded = False 

if img_nanas is None:
    print(f"Error: Tidak bisa membaca gambar dari {image_path_nanas}")
else:
    nanas_loaded = True
    gray_nanas = cv2.cvtColor(img_nanas, cv2.COLOR_BGR2GRAY)

    blurred_nanas = cv2.GaussianBlur(gray_nanas, (5, 5), 0)

    canny_edges_low = cv2.Canny(blurred_nanas, 30, 90)
    canny_edges_high = cv2.Canny(blurred_nanas, 100, 200)
    
    print("Selesai memproses Canny.")


print("\nMenampilkan semua hasil...")

plt.figure(figsize=(16, 12)) 

if pear_loaded:
    plt.subplot(3, 4, 1)
    plt.title('Pear Asli')
    plt.imshow(cv2.cvtColor(img_pear, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(3, 4, 2)
    plt.title('Pear Grayscale')
    plt.imshow(gray_pear, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 4, 3)
    plt.title('Pear Tepi Sobel')
    plt.imshow(sobel_edges_pear, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 4, 4)
    plt.title('Pear Tepi Biner')
    plt.imshow(binary_edges_pear, cmap='gray')
    plt.axis('off')
    
    plt.subplot(3, 4, 5)
    plt.title('Pear Mask Segmentasi')
    plt.imshow(mask_pear, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 4, 6)
    plt.title('Pear Hasil Segmentasi')
    plt.imshow(cv2.cvtColor(final_segmented_pear, cv2.COLOR_BGR2RGB))
    plt.axis('off')

if nanas_loaded:
    plt.subplot(3, 4, 9)
    plt.title('Nanas Asli')
    plt.imshow(cv2.cvtColor(img_nanas, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(3, 4, 10)
    plt.title('Nanas Blurred')
    plt.imshow(blurred_nanas, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 4, 11)
    plt.title('Nanas Canny (30, 90)')
    plt.imshow(canny_edges_low, cmap='gray')
    plt.axis('off')
    
    plt.subplot(3, 4, 12)
    plt.title('Nanas Canny (100, 200)')
    plt.imshow(canny_edges_high, cmap='gray')
    plt.axis('off')

plt.tight_layout() 
plt.show()

print("\nSemua proses selesai.")