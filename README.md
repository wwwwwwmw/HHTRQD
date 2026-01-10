# HHTRQD

## Chạy demo Streamlit

Ứng dụng web minh họa quy trình AHP + AI nằm trong `streamlit_app.py`. Để chạy:

### Chạy nhanh trên Windows (không cần Activate.ps1)

```bat
install_deps.cmd
run_app.cmd
```

Hoặc chạy trực tiếp bằng Python trong `.venv`:

```bat
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.port 8501
```

Ứng dụng hiện dùng **PostgreSQL** làm nguồn dữ liệu (thay cho `cars.csv`), có đăng nhập với 2 role:

- **Guest (chưa đăng nhập)**: vẫn dùng trang “Gợi ý xe” bình thường.
- **User**: dùng “Gợi ý xe” + có trang “Lịch sử” (lưu các lần đề xuất).
- **Admin**: có thêm “Dashboard”, “Thêm dữ liệu”, “Tiêu chí” (thêm/xoá/bật-tắt tiêu chí AHP và tiêu chí lọc).

### 1) Cài dependencies

```bash
pip install -r requirements.txt
```

### 2) Cấu hình PostgreSQL

Thiết lập biến môi trường (khuyến nghị dùng `DATABASE_URL`):

- `DATABASE_URL=postgresql+psycopg2://USER:PASSWORD@HOST:5432/DBNAME`

Hoặc cấu hình từng phần:

- `POSTGRES_HOST`, `POSTGRES_PORT` (mặc định 5432), `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`

Hoặc dùng file `.env` (đã hỗ trợ tự đọc):

- `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASS`

Tuỳ chọn tạo admin mặc định (nếu DB chưa có user):

- `ADMIN_USERNAME` (mặc định: `admin`)
- `ADMIN_PASSWORD` (mặc định: `admin123`)

Khi chạy lần đầu, app sẽ tự tạo bảng và seed tiêu chí mặc định.
