# Store Sales Prediction — PyTorch (GPU)

Proyek ini memprediksi *unit sales* untuk setiap kombinasi toko dan produk menggunakan model Neural Network berbasis **PyTorch dengan akselerasi GPU (CUDA)**. Dataset berasal dari kompetisi Kaggle *Store Sales – Time Series Forecasting*.

---

## 📁 Dataset

Dataset yang digunakan:

- `train.csv`  
- `test.csv`  
- `stores.csv`  
- `holidays_events.csv`  
- `oil.csv`

Path dataset disesuaikan dengan project


---

## ⚙️ Feature Engineering

Fitur waktu:
- `year`, `month`, `day`, `weekofyear`
- `is_weekend`

Fitur lag:
- `lag_1`, `lag_3`, `lag_7`, `lag_30`

Fitur rolling:
- `rolling_7`, `rolling_30`

Fitur kategori dari tabel stores:
- `store_type`
- `cluster`
- `city`

Fitur eksternal:
- `oil_price`
- `holiday_flags`

Semua fitur kategori di-*encode* menggunakan **LabelEncoder**.

---

## 🤖 Model: PyTorch Neural Network

Arsitektur model:

- Dense: input → 256, ReLU  
- Dense: 256 → 128, ReLU  
- Dropout (0.2)  
- Dense: 128 → 64, ReLU  
- Dense: 64 → 1  

Loss:
- `MSELoss` (di ruang log)

Optimizer:
- `Adam(lr=0.001)`

Device:
- Otomatis menggunakan **CUDA GPU** jika tersedia  
- Fall-back ke CPU jika tidak ada GPU  

---

## 🏋️ Training

Training memakai mini-batch dengan DataLoader.

- Backpropagation standar PyTorch
- Target `sales` ditransformasi dengan **log1p**
- Saat inferensi, hasil dikembalikan dengan **expm1**

Tiap epoch menampilkan:
- Train Loss  
- Validation Loss  

---

## 📊 Evaluasi

Digunakan metrik:
### **RMSLE (log-space)**
Karena nilai sales tidak boleh negatif dan bersifat *right-skewed*.

---

## 📤 Submission

Hasil prediksi akan disimpan sebagai: submission.csv

