# P_inno – Heart Rate Analysis using PPG (ESP32)

## 📌 Giới thiệu
P_inno là dự án nghiên cứu và phát triển hệ thống đo và phân tích nhịp tim
dựa trên tín hiệu **PPG (Photoplethysmography)**, sử dụng **ESP32 + MAX30102**.

Dự án hướng tới:
- Thu thập tín hiệu nhịp tim từ thiết bị đeo
- Lọc nhiễu và xử lý tín hiệu PPG
- Tính toán các chỉ số HR & HRV (MeanRR, SDNN, RMSSD)
- Ứng dụng trong theo dõi sức khoẻ thời gian thực

---

## 🧠 Công nghệ sử dụng
- ESP32
- MAX30102 (PPG Sensor)
- Python (NumPy, Pandas, Matplotlib)
- Jupyter Notebook

---

## 📂 Cấu trúc thư mục
```text
P_inno/
├── data/           # Data PPG (CSV)
├── notebooks/      # Jupyter notebooks
├── src/            # Signal processing & HRV
├── README.md
└── requirements.txt



-------LÊ_DUY_ANH----------