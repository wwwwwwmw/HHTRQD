# Hướng dẫn chạy chương trình HHTRQD (AHP + AI + Streamlit)

Tài liệu này hướng dẫn chạy project từ đầu đến cuối trên **Windows**, ưu tiên dùng **Terminal**.

---

## 0) Yêu cầu trước khi chạy

### Phần mềm cần có

- **Python 3.10+** (khuyến nghị 3.11/3.12). Kiểm tra:

```powershell
py --version
```

- **PostgreSQL 13+** (có thể dùng 15/16). Kiểm tra nhanh bằng `psql` (nếu có):

```powershell
psql --version
```

### Mở terminal đúng thư mục

Mở PowerShell (hoặc CMD), rồi `cd` vào thư mục dự án:

```powershell
cd D:\Documents\HHTRQD
```

---

## 1) Tạo môi trường và cài dependencies

### Cách 1 (khuyến nghị): dùng script có sẵn (không cần Activate.ps1)

Chạy:

```bat
install_deps.cmd
```

Script này sẽ:
- Tạo virtualenv tại `.venv` nếu chưa có
- Cài dependencies từ `requirements.txt`

### Cách 2: chạy tay bằng terminal

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## 2) Chuẩn bị PostgreSQL (bắt buộc)

Ứng dụng dùng PostgreSQL làm nguồn dữ liệu.

### 2.1) Tạo database

Mở terminal (hoặc pgAdmin), tạo DB (ví dụ `CARS_DB`). Nếu bạn có `psql`:

```powershell
psql -U postgres -h localhost -p 5432
```

Trong cửa sổ `psql`, chạy:

```sql
CREATE DATABASE "CARS_DB";
```

Thoát `psql`:

```sql
\q
```

### 2.2) Cấu hình biến môi trường kết nối DB

Project hỗ trợ đọc file `.env` trong thư mục dự án. Hiện có sẵn file `.env` mẫu như sau:

- `DB_HOST=localhost`
- `DB_NAME=CARS_DB`
- `DB_USER=postgres`
- `DB_PASS=123456`
- `DB_PORT=5432`

Hãy chỉnh lại cho đúng với máy bạn (đặc biệt `DB_PASS`).

Tuỳ chọn: có thể dùng một biến duy nhất `DATABASE_URL` (khuyến nghị khi deploy):

```powershell
$env:DATABASE_URL = "postgresql+psycopg2://USER:PASSWORD@HOST:5432/DBNAME"
```

Lưu ý: nếu bạn chạy bằng `.cmd` thì app vẫn đọc `.env`, nên thường không cần set `$env:` thủ công.

### 2.3) Tài khoản admin mặc định

Khi chạy lần đầu, app sẽ tự tạo bảng và seed dữ liệu/tiêu chí.
Có thể cấu hình admin mặc định bằng env:

- `ADMIN_USERNAME` (mặc định: `admin`)
- `ADMIN_PASSWORD` (mặc định: `admin123`)

---

## 3) Chạy ứng dụng Streamlit

### Cách 1 (khuyến nghị): dùng script có sẵn

```bat
run_app.cmd
```

Mặc định chạy ở cổng `8501`.

### Cách 2: chạy bằng streamlit trực tiếp

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.port 8501
```

Sau khi chạy, mở trình duyệt:

- http://localhost:8501

---

## 4) Luồng sử dụng trong app

### 4.1) Đăng nhập (tuỳ chọn)

- Guest vẫn dùng được trang “Gợi ý xe”.
- User: có thêm trang “Lịch sử”.
- Admin: có thêm “Dashboard”, “Thêm dữ liệu”, “Tiêu chí”…

Tài khoản demo (nếu DB đã seed đúng như README):
- `admin / admin123`
- `a / 123456`

### 4.2) Thêm dữ liệu xe (Admin)

Vào các trang admin để import/thêm dữ liệu vào DB.
Nếu DB chưa có dữ liệu xe thì trang “Gợi ý xe” sẽ báo trống.

### 4.3) Cấu hình tiêu chí (Admin)

Trong “Tiêu chí”:
- `kind=ahp`: tiêu chí tính AHP, có `direction` (benefit/cost)
- `kind=filter`: tiêu chí lọc đầu vào

### 4.4) Gợi ý xe (AHP + AI)

Trong “Gợi ý xe”:
1) Chọn preset mục tiêu sử dụng
2) Thiết lập bộ lọc
3) (Tuỳ chọn) bật AI nâng cao: bảo dưỡng / rủi ro sửa chữa
4) Chấm điểm ưu tiên tiêu chí (1–9)
5) Nhấn “Tìm xe phù hợp”

Kết quả:
- Tab “Top đề xuất”
- Tab “Danh sách sau chấm điểm”
- Khu vực “So sánh xe”: chọn tối đa 4 xe và xem bảng so sánh; giá trị tốt hơn được tô xanh.

---

## 5) Troubleshooting nhanh

### Lỗi không kết nối được PostgreSQL

- Kiểm tra PostgreSQL service đang chạy.
- Kiểm tra `.env` đúng host/port/user/pass/dbname.
- Nếu password có ký tự đặc biệt, thử đổi sang password đơn giản để test.

### Cổng 8501 bị chiếm

Chạy port khác:

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.port 8502
```

### Cài package bị lỗi

- Nâng pip:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

- Cài lại requirements:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## 6) Lệnh tổng hợp (copy/paste)

### Chạy nhanh nhất

```bat
cd D:\Documents\HHTRQD
install_deps.cmd
run_app.cmd
```

### Chạy tay (PowerShell)

```powershell
cd D:\Documents\HHTRQD
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.port 8501
```
