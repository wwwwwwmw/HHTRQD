from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DEFAULT_PATH = PROJECT_ROOT / "cars.csv"

AHP_CRITERIA: list[str] = [
    "price",
    "mileage",
    "year",
    "accidents_or_damage",
    "one_owner",
    "driver_rating",
    "seller_rating",
    "mpg",
    "price_drop",
]

CRITERIA_LABELS: dict[str, str] = {
    "price": "Giá bán",
    "mileage": "Số dặm đã chạy",
    "year": "Năm sản xuất",
    "accidents_or_damage": "Tai nạn/Hư hại",
    "one_owner": "Số chủ sở hữu",
    "driver_rating": "Đánh giá người lái",
    "seller_rating": "Đánh giá người bán",
    "mpg": "Tiết kiệm nhiên liệu",
    "price_drop": "Mức giảm giá",
}

DISPLAY_COLUMN_LABELS: dict[str, str] = {
    "rank": "Thứ hạng",
    "manufacturer": "Hãng",
    "model": "Mẫu xe",
    "year": "Năm",
    "price": "Giá (USD)",
    "mileage": "Số dặm",
    "mpg": "Tiết kiệm nhiên liệu",
    "fuel_type": "Loại nhiên liệu",
    "engine": "Động cơ",
    "accidents_or_damage": "Tai nạn/Hư hại",
    "one_owner": "Số chủ sở hữu",
    "driver_rating": "Điểm người lái",
    "seller_rating": "Điểm người bán",
    "price_drop": "Giảm giá (%)",
    "ahp_score": "Điểm phù hợp",
    "risk_pred": "Rủi ro (AI)",
    "risk_level": "Đánh giá rủi ro",
    "maintenance_cost_pred": "Ước tính bảo dưỡng (USD/năm)",
    "repair_risk_pred": "Nguy cơ sửa chữa (%)",
    "recommendation": "Đề xuất",
}

DEFAULT_SCORES: dict[str, int] = {
    "price": 8,
    "mileage": 7,
    "year": 6,
    "accidents_or_damage": 9,
    "one_owner": 5,
    "driver_rating": 7,
    "seller_rating": 5,
    "mpg": 4,
    "price_drop": 4,
}

BENEFIT_CRITERIA = ["year", "one_owner", "driver_rating", "seller_rating", "mpg", "price_drop"]
COST_CRITERIA = ["price", "mileage", "accidents_or_damage"]

RISK_TEXT = {0: "Thấp", 1: "Cao"}

PRESET_SCORES: dict[str, dict[str, int]] = {
    "Cân bằng chung": {**DEFAULT_SCORES},
    "Gia đình an toàn": {
        "price": 5,
        "mileage": 6,
        "year": 7,
        "accidents_or_damage": 9,
        "one_owner": 8,
        "driver_rating": 9,
        "seller_rating": 7,
        "mpg": 4,
        "price_drop": 4,
    },
    "Tiết kiệm chi phí": {
        "price": 9,
        "mileage": 8,
        "year": 5,
        "accidents_or_damage": 6,
        "one_owner": 5,
        "driver_rating": 6,
        "seller_rating": 5,
        "mpg": 8,
        "price_drop": 7,
    },
    "Xe mới sang trọng": {
        "price": 6,
        "mileage": 5,
        "year": 9,
        "accidents_or_damage": 7,
        "one_owner": 7,
        "driver_rating": 8,
        "seller_rating": 8,
        "mpg": 4,
        "price_drop": 6,
    },
}
