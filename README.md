# HHTRQD

## Chạy demo Streamlit

Ứng dụng web minh họa quy trình AHP + AI nằm trong `streamlit_app.py`. Để chạy:

```bash
pip install -r requirements.txt  # nếu đã có danh sách phụ thuộc
# hoặc cài đặt tối thiểu
pip install streamlit pandas numpy scikit-learn pyarrow

streamlit run streamlit_app.py
```

Ứng dụng sẽ tải dữ liệu `cars.csv` (hoặc file CSV bạn upload), huấn luyện RandomForest, tính AHP (ma trận, trọng số, λ_max, CI, CR) và hiển thị bảng xếp hạng đề xuất theo thời gian thực.
