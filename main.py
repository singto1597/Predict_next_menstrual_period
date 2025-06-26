from datetime import datetime, timedelta
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ข้อมูลวันเริ่มรอบเดือน
raw_dates = [
    "2024-09-18", "2024-10-18", "2024-11-17", "2024-12-15",
    "2025-01-13", "2025-02-11", "2025-03-08", "2025-04-05",
    "2025-05-07", "2025-06-08"
]

# แปลงเป็นช่วงห่าง (จำนวนวัน)
dates = [datetime.strptime(d, "%Y-%m-%d") for d in raw_dates]
intervals = np.array([(dates[i] - dates[i - 1]).days for i in range(1, len(dates))])

# เตรียมข้อมูลสำหรับ LSTM
sequence_length = 7
X, y = [], []
for i in range(len(intervals) - sequence_length):
    X.append(intervals[i:i + sequence_length])
    y.append(intervals[i + sequence_length])
X = np.array(X).reshape((-1, sequence_length, 1))
y = np.array(y)

# สร้างโมเดล LSTM
model = Sequential([
    LSTM(50, activation='relu', input_shape=(sequence_length, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# เทรน
model.fit(X, y, epochs=300, verbose=0)



# ทำนายรอบถัดไปจาก 3 รอบล่าสุด
last_seq = intervals[-7:].reshape((1, sequence_length, 1))
predicted = model.predict(last_seq, verbose=0)
predicted_interval = round(predicted[0][0])

# แสดงผล
last_date = dates[-1]
next_date = last_date + timedelta(days=predicted_interval)

print(f"รอบถัดไปน่าจะมาประมาณวันที่: {next_date.strftime('%Y-%m-%d')} ({predicted_interval} วันจากรอบล่าสุด)")


