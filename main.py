from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor

# อ่านข้อมูลจากไฟล์
with open("period_data.txt", "r") as file:
    raw_dates = [line.strip() for line in file.readlines() if line.strip()]

# แปลงเป็น datetime
dates = [datetime.strptime(d, "%Y-%m-%d") for d in raw_dates]

# คำนวณความห่างแต่ละรอบ
intervals = [(dates[i] - dates[i - 1]).days for i in range(1, len(dates))]

# เตรียมข้อมูลฝึก AI (ใช้ 2 รอบก่อนหน้า เพื่อทำนายรอบถัดไป)
X, y = [], []
for i in range(2, len(intervals)):
    X.append([intervals[i - 2], intervals[i - 1]])
    y.append(intervals[i])

# เทรนโมเดล
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# ทำนายจาก 2 รอบล่าสุด
predicted_interval = model.predict([intervals[-2:]])[0]
predicted_interval = round(predicted_interval)

# คำนวณวันที่น่าจะมาเมนส์รอบหน้า
next_date = dates[-1] + timedelta(days=predicted_interval)

# แสดงผล
print(f"รอบถัดไปน่าจะมาในวันที่: {next_date.strftime('%Y-%m-%d')} (ประมาณ {predicted_interval} วันจากรอบล่าสุด)")
