from datetime import datetime, timedelta
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
# อ่านข้อมูลจากไฟล์
with open("period_data.txt", "r") as file:
    raw_dates = [line.strip() for line in file.readlines() if line.strip()]


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

choise = input("ต้องการเทรนหรือใช้โมเดลที่มีอยู่แล้ว? (t/use): ").strip().lower()

if choise == "use":
    from tensorflow.keras.models import load_model
    model = load_model("period_predictor_model_2.h5")
else:
    print("กำลังเทรนโมเดล...")
    # สร้างโมเดล LSTM
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(sequence_length, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    # เทรน
    model.fit(X, y, epochs=4000, verbose=1)
    print("โมเดลเทรนเสร็จสิ้น")
    model.save("period_predictor_model_2.h5")

last_seq = intervals[-sequence_length:].reshape((1, sequence_length, 1))
predicted = model.predict(last_seq, verbose=0)
predicted_interval = round(predicted[0][0])

# แสดงผล
last_date = dates[-1]
next_date = last_date + timedelta(days=predicted_interval)

print(f"รอบถัดไปน่าจะมาประมาณวันที่: {next_date.strftime('%Y-%m-%d')} ({predicted_interval} วันจากรอบล่าสุด)")

