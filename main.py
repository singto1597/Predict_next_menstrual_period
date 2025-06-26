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


import plotly.graph_objects as go
predicted_date = next_date

# สร้างแกน X และ Y สำหรับกราฟ
x_labels = [d.strftime("%Y-%m-%d") for d in dates[1:]]  # ตั้งแต่รอบที่ 2 เป็นต้นไป
x_labels.append(predicted_date.strftime("%Y-%m-%d"))

y_values = intervals + [predicted_interval]
bar_colors = ['skyblue'] * len(intervals) + ['red']

# สร้างกราฟ
fig = go.Figure(data=[
    go.Bar(
        x=x_labels,
        y=y_values,
        marker_color=bar_colors,
        text=[f"{y} วัน" for y in y_values],
        textposition='outside'
    )
])

fig.update_layout(
    title="จำนวนวันห่างของรอบเดือนแต่ละรอบ (แท่งแดง = ค่าที่ AI ทำนาย)",
    xaxis_title="วันที่เริ่มรอบเดือน",
    yaxis_title="จำนวนวันห่างจากรอบก่อนหน้า",
    xaxis=dict(tickangle=45),
    bargap=0.2,
    height=500,
    margin=dict(l=40, r=40, t=80, b=100)
)

fig.show()