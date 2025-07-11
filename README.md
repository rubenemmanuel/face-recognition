# faceRecognition


a. Proses Penyimpanan Gambar Wajah Beserta Atribut Identitas
Proses pertama yang dilakukan oleh program adalah memuat data wajah yang dikenal (known faces) dari sebuah direktori khusus yang disebut dataset. Direktori ini berisi file gambar wajah yang akan digunakan sebagai data referensi. Program membaca setiap file dalam direktori ini dengan langkah-langkah berikut:
Membaca Gambar dari Direktori
Program membaca file gambar dengan format .jpeg menggunakan pustaka OpenCV (cv2.imread). Setiap file gambar diambil berdasarkan nama file-nya, di mana nama file tanpa ekstensi akan digunakan sebagai identitas atau label wajah. Misalnya, jika ada file bernama john_doe.jpeg, maka nama john_doe akan disimpan sebagai label identitas wajah tersebut.
Konversi Warna Gambar
Setelah gambar berhasil dimuat, langkah selanjutnya adalah mengubah format warna dari BGR (format default OpenCV) menjadi RGB. Konversi ini diperlukan karena pustaka face_recognition hanya bekerja dengan format warna RGB. Proses konversi dilakukan menggunakan fungsi cv2.cvtColor.
Ekstraksi Encoding Wajah
Encoding wajah adalah representasi numerik dari fitur-fitur unik pada wajah, seperti jarak antara mata, panjang hidung, bentuk mulut, dan lain-lain. Encoding ini dihasilkan menggunakan fungsi face_recognition.face_encodings. Jika wajah ditemukan dalam gambar, encoding wajah disimpan ke dalam daftar known_encodings, sedangkan nama file tanpa ekstensi disimpan ke dalam daftar known_names. Jika wajah tidak ditemukan dalam gambar, program memberikan peringatan dan melewati file tersebut.
Output Akhir Proses
Pada akhir proses ini, kita memiliki dua data penting:
known_encodings: Sebuah daftar yang berisi encoding wajah dalam bentuk array numerik.
known_names: Sebuah daftar yang berisi nama atau label yang sesuai dengan encoding wajah.
Contoh:
Input: File john_doe.jpeg.
Output:
Encoding wajah berupa array numerik.
Nama "john_doe" disimpan sebagai label identitas.


b. Proses Awal Sebelum (Pre-processing)
Setelah data wajah dikenal berhasil dimuat, program memasuki tahap pengolahan data real-time dari webcam. Pada tahap ini, setiap frame yang diambil dari kamera diproses sebelum masuk ke tahap pengenalan wajah. Langkah-langkah pre-processing meliputi:
Pengambilan Gambar Real-Time dari Webcam
Program menangkap gambar secara terus-menerus dari webcam menggunakan pustaka OpenCV (cv2.VideoCapture). Untuk memastikan proses ini berjalan tanpa gangguan, program memanfaatkan teknik threading. Sebuah thread khusus dibuat untuk membaca frame dari webcam dan menyimpannya dalam sebuah antrian (queue). Dengan cara ini, pengolahan gambar dapat dilakukan secara paralel tanpa menghambat proses pengambilan gambar.
Resize Ukuran Frame
Frame yang diambil dari webcam biasanya memiliki resolusi besar yang memerlukan komputasi tinggi. Oleh karena itu, ukuran frame diperkecil menjadi separuhnya (50%) menggunakan fungsi cv2.resize. Proses ini tidak hanya mempercepat pengolahan data, tetapi juga mengurangi beban pada CPU atau GPU.
Konversi Warna ke RGB
Setelah ukuran frame diperkecil, frame diubah dari format warna BGR ke RGB untuk memastikan kompatibilitas dengan pustaka face_recognition.
Output dari tahap ini adalah frame yang lebih kecil dalam format RGB, siap untuk diproses lebih lanjut.

c. Proses Pencarian Fitur atau Training
Tahap pencarian fitur wajah dilakukan dengan memanfaatkan pustaka face_recognition, yang menggunakan model deep learning berbasis Convolutional Neural Network (CNN). Langkah-langkah yang dilakukan adalah sebagai berikut:
Deteksi Lokasi Wajah
Program mendeteksi lokasi wajah dalam frame menggunakan fungsi face_recognition.face_locations. Fungsi ini mengembalikan koordinat bounding box berupa empat nilai (atas, kanan, bawah, kiri) yang menentukan area wajah dalam gambar.
Ekstraksi Encoding Wajah
Encoding wajah diekstrak dari area bounding box menggunakan fungsi face_recognition.face_encodings. Encoding ini adalah representasi numerik dari fitur unik wajah yang digunakan untuk membandingkan wajah baru dengan wajah yang dikenal sebelumnya.
Proses ini dilakukan sekali selama inisialisasi program untuk mempersiapkan data referensi dari dataset wajah yang dikenal.

d. Proses Input Gambar yang Akan Dikenali
Setiap frame yang telah melewati pre-processing digunakan sebagai input untuk pengenalan wajah. Tahap ini mencakup:
Deteksi Wajah dalam Frame
Program mendeteksi semua wajah dalam frame menggunakan fungsi face_recognition.face_locations. Setiap wajah yang terdeteksi akan memiliki lokasi bounding box.
Pencocokan Wajah
Encoding wajah yang ditemukan dalam frame dibandingkan dengan encoding wajah yang dikenal menggunakan fungsi face_recognition.face_distance. Jarak terpendek antara encoding menunjukkan kecocokan terbaik.

e. Proses Mendapatkan Hasil Gambar yang Dikenali
Berdasarkan hasil pencocokan wajah, program menentukan identitas dari setiap wajah yang terdeteksi. Langkah-langkahnya meliputi:
Penentuan Identitas
Jika jarak terpendek (dari fungsi face_distance) berada di bawah ambang batas (threshold) 0.6, program menganggap wajah cocok dengan salah satu wajah yang dikenal. Identitas atau label wajah tersebut kemudian ditampilkan.
Visualisasi pada Frame
Program menggambar kotak (bounding box) di sekitar wajah yang terdeteksi menggunakan fungsi cv2.rectangle. Label nama wajah ditampilkan di bawah kotak dengan menggunakan fungsi cv2.putText.

f. Analisis Keberhasilan dan Kegagalan
Keberhasilan
Wajah yang dikenali dengan baik memiliki posisi frontal (menghadap langsung ke kamera).
Kondisi pencahayaan yang memadai meningkatkan akurasi pengenalan.
Wajah yang sudah dikenal dalam dataset dikenali dengan tingkat keberhasilan tinggi selama kondisinya konsisten.
Kegagalan
Wajah yang miring, sebagian tertutup, atau terhalang objek sering kali gagal dikenali.
Gangguan seperti pencahayaan buruk atau bayangan dapat menyebabkan deteksi yang tidak akurat.
Wajah baru yang tidak ada dalam dataset selalu dikategorikan sebagai "Unknown."

g. Proses Tambahan
Threading untuk Efisiensi
Penggunaan threading memungkinkan pengambilan gambar dari webcam berjalan secara paralel dengan pengolahan data. Teknik ini meningkatkan efisiensi dan kecepatan sistem.
Threshold Jarak
Nilai ambang batas (threshold) 0.6 dipilih untuk menentukan kecocokan wajah. Nilai ini dapat disesuaikan untuk mencapai keseimbangan antara sensitivitas dan akurasi pengenalan.
Pengolahan Dataset Lebih Lanjut
Dataset wajah dapat diperluas dengan menambahkan variasi gambar untuk setiap individu, seperti gambar dengan berbagai sudut atau kondisi pencahayaan.