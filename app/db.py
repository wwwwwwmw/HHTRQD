from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    func,
    select,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


Role = Literal["admin", "user"]
CriterionKind = Literal["ahp", "filter"]
CriterionDirection = Literal["benefit", "cost", "none"]


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False, default="user")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class Car(Base):
    __tablename__ = "cars"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    manufacturer: Mapped[str] = mapped_column(String(128), nullable=False, default="Unknown")
    model: Mapped[str] = mapped_column(String(256), nullable=False, default="Unknown")
    year: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    mileage: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    price: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    mpg: Mapped[str] = mapped_column(String(64), nullable=False, default="Unknown")
    fuel_type: Mapped[str] = mapped_column(String(64), nullable=False, default="Unknown")
    engine: Mapped[str] = mapped_column(Text, nullable=False, default="Unknown")

    accidents_or_damage: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    one_owner: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    driver_rating: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    seller_rating: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    price_drop: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class Criterion(Base):
    __tablename__ = "criteria"

    __table_args__ = (
        UniqueConstraint("kind", "key", name="criteria_kind_key_unique"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String(64), nullable=False)
    label: Mapped[str] = mapped_column(String(128), nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)  # ahp | filter
    direction: Mapped[str] = mapped_column(String(16), nullable=False, default="none")  # benefit | cost | none
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class Recommendation(Base):
    __tablename__ = "recommendations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    inputs: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    results: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


@dataclass(frozen=True)
class DbConfig:
    url: str


def _load_dotenv_if_present() -> None:
    """Load .env key=value into os.environ if not already set.

    This keeps setup simple on Windows (no Activate.ps1 needed) and avoids
    adding an extra dependency like python-dotenv.
    """

    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return

    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except Exception:
        # If .env parsing fails, fall back to normal os.environ.
        return


def get_database_url() -> str | None:
    _load_dotenv_if_present()

    # Prefer DATABASE_URL; fallback to POSTGRES_* / DB_* pieces
    url = os.environ.get("DATABASE_URL")
    if url:
        return url

    # Support user's .env naming
    host = os.environ.get("POSTGRES_HOST") or os.environ.get("DB_HOST")
    db = os.environ.get("POSTGRES_DB") or os.environ.get("DB_NAME")
    user = os.environ.get("POSTGRES_USER") or os.environ.get("DB_USER")
    password = os.environ.get("POSTGRES_PASSWORD") or os.environ.get("DB_PASS")
    port = os.environ.get("POSTGRES_PORT") or os.environ.get("DB_PORT") or "5432"

    if not (host and db and user and password):
        return None

    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"


def create_db_engine() -> Engine:
    url = get_database_url()
    if not url:
        raise RuntimeError(
            "Chưa cấu hình PostgreSQL. Hãy đặt DATABASE_URL hoặc POSTGRES_HOST/DB/USER/PASSWORD, hoặc DB_HOST/DB_NAME/DB_USER/DB_PASS trong .env."
        )

    # pool_pre_ping helps recover from stale connections
    return create_engine(url, pool_pre_ping=True, future=True)


def init_db(engine: Engine) -> None:
    Base.metadata.create_all(engine)
    _migrate_criteria_unique_constraint(engine)
    _migrate_cars_column_types(engine)


def _migrate_cars_column_types(engine: Engine) -> None:
    """Auto-migrate cars text column sizes.

    CSV sources often contain long engine strings (e.g. full marketing names)
    which exceed the original VARCHAR(64) limit.
    """
    try:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE cars ALTER COLUMN manufacturer TYPE VARCHAR(128)"))
            conn.execute(text("ALTER TABLE cars ALTER COLUMN model TYPE VARCHAR(256)"))
            conn.execute(text("ALTER TABLE cars ALTER COLUMN mpg TYPE VARCHAR(64)"))
            conn.execute(text("ALTER TABLE cars ALTER COLUMN fuel_type TYPE VARCHAR(64)"))
            conn.execute(text("ALTER TABLE cars ALTER COLUMN engine TYPE TEXT"))
    except Exception:
        return


def _migrate_criteria_unique_constraint(engine: Engine) -> None:
    """Auto-migrate older schema where criteria.key was unique.

    We now allow the same key for different kinds (e.g. year for both AHP and filter),
    so uniqueness must be on (kind, key).
    """
    try:
        with engine.begin() as conn:
            # Drop legacy unique constraint (name observed from Postgres error).
            conn.execute(text("ALTER TABLE criteria DROP CONSTRAINT IF EXISTS criteria_key_key"))
            # Ensure composite unique exists.
            conn.execute(
                text(
                    "CREATE UNIQUE INDEX IF NOT EXISTS criteria_kind_key_uidx ON criteria (kind, key)"
                )
            )
    except Exception:
        # If anything goes wrong (permissions, different DB, etc.), keep app running.
        return


def healthcheck(engine: Engine) -> tuple[bool, str]:
    try:
        with engine.connect() as conn:
            conn.execute(select(1))
        return True, "OK"
    except OperationalError as exc:
        return False, str(exc)
    except Exception as exc:
        return False, str(exc)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def read_cars_df(engine: Engine) -> pd.DataFrame:
    with Session(engine) as session:
        rows = session.execute(select(Car)).scalars().all()

    if not rows:
        return pd.DataFrame(
            columns=[
                "id",
                "manufacturer",
                "model",
                "year",
                "mileage",
                "price",
                "mpg",
                "fuel_type",
                "engine",
                "accidents_or_damage",
                "one_owner",
                "driver_rating",
                "seller_rating",
                "price_drop",
            ]
        )

    data = [
        {
            "id": r.id,
            "manufacturer": r.manufacturer,
            "model": r.model,
            "year": r.year,
            "mileage": float(r.mileage),
            "price": float(r.price),
            "mpg": r.mpg,
            "fuel_type": r.fuel_type,
            "engine": r.engine,
            "accidents_or_damage": float(r.accidents_or_damage),
            "one_owner": float(r.one_owner),
            "driver_rating": float(r.driver_rating),
            "seller_rating": float(r.seller_rating),
            "price_drop": float(r.price_drop),
        }
        for r in rows
    ]
    return pd.DataFrame(data)


def upsert_default_criteria(engine: Engine, defaults: list[dict[str, Any]]) -> None:
    with Session(engine) as session:
        existing_pairs = {
            (c.kind, c.key)
            for c in session.execute(select(Criterion)).scalars().all()
        }
        for item in defaults:
            key = str(item["key"])
            kind = str(item.get("kind", "ahp"))
            pair = (kind, key)
            if pair in existing_pairs:
                continue
            session.add(
                Criterion(
                    key=key,
                    label=str(item.get("label", key)),
                    kind=kind,
                    direction=str(item.get("direction", "none")),
                    enabled=bool(item.get("enabled", True)),
                )
            )
            existing_pairs.add(pair)
        session.commit()


def list_criteria(engine: Engine, kind: str | None = None, enabled_only: bool = False) -> list[Criterion]:
    stmt = select(Criterion)
    if kind:
        stmt = stmt.where(Criterion.kind == kind)
    if enabled_only:
        stmt = stmt.where(Criterion.enabled.is_(True))
    stmt = stmt.order_by(Criterion.id.asc())
    with Session(engine) as session:
        return session.execute(stmt).scalars().all()


def delete_criterion(engine: Engine, criterion_id: int) -> None:
    with Session(engine) as session:
        obj = session.get(Criterion, criterion_id)
        if obj is None:
            return
        session.delete(obj)
        session.commit()


def create_criterion(
    engine: Engine,
    key: str,
    label: str,
    kind: CriterionKind,
    direction: CriterionDirection = "none",
    enabled: bool = True,
) -> None:
    with Session(engine) as session:
        session.add(
            Criterion(
                key=str(key),
                label=str(label),
                kind=str(kind),
                direction=str(direction),
                enabled=bool(enabled),
            )
        )
        session.commit()


def update_criterion(
    engine: Engine,
    criterion_id: int,
    *,
    label: str | None = None,
    direction: CriterionDirection | None = None,
    enabled: bool | None = None,
) -> None:
    with Session(engine) as session:
        obj = session.get(Criterion, int(criterion_id))
        if obj is None:
            return
        if label is not None:
            obj.label = str(label)
        if direction is not None:
            obj.direction = str(direction)
        if enabled is not None:
            obj.enabled = bool(enabled)
        session.commit()


def create_user(engine: Engine, username: str, password_hash: str, role: Role) -> None:
    with Session(engine) as session:
        session.add(User(username=username, password_hash=password_hash, role=role))
        session.commit()


def list_users(engine: Engine, limit: int = 200) -> list[User]:
    stmt = select(User).order_by(User.id.asc()).limit(int(limit))
    with Session(engine) as session:
        return session.execute(stmt).scalars().all()


def get_user_by_id(engine: Engine, user_id: int) -> User | None:
    with Session(engine) as session:
        return session.get(User, int(user_id))


def update_user(
    engine: Engine,
    user_id: int,
    *,
    username: str | None = None,
    role: Role | None = None,
    password_hash: str | None = None,
) -> None:
    with Session(engine) as session:
        obj = session.get(User, int(user_id))
        if obj is None:
            return
        if username is not None and username.strip():
            obj.username = username.strip()
        if role is not None:
            obj.role = str(role)
        if password_hash is not None and password_hash:
            obj.password_hash = password_hash
        session.commit()


def delete_user(engine: Engine, user_id: int) -> None:
    with Session(engine) as session:
        obj = session.get(User, int(user_id))
        if obj is None:
            return
        session.delete(obj)
        session.commit()


def get_user_by_username(engine: Engine, username: str) -> User | None:
    with Session(engine) as session:
        return session.execute(select(User).where(User.username == username)).scalar_one_or_none()


def count_rows(engine: Engine) -> dict[str, int]:
    with Session(engine) as session:
        users = int(session.scalar(select(func.count()).select_from(User)) or 0)
        cars = int(session.scalar(select(func.count()).select_from(Car)) or 0)
        recs = int(session.scalar(select(func.count()).select_from(Recommendation)) or 0)
    return {"users": users, "cars": cars, "recommendations": recs}


def insert_car(engine: Engine, payload: dict[str, Any]) -> None:
    allowed = {
        "manufacturer",
        "model",
        "year",
        "mileage",
        "price",
        "mpg",
        "fuel_type",
        "engine",
        "accidents_or_damage",
        "one_owner",
        "driver_rating",
        "seller_rating",
        "price_drop",
    }
    clean = {k: payload[k] for k in payload.keys() if k in allowed}
    with Session(engine) as session:
        session.add(Car(**clean))
        session.commit()


def _car_fingerprint(payload: dict[str, Any]) -> str:
    keys = [
        "manufacturer",
        "model",
        "year",
        "mileage",
        "price",
        "mpg",
        "fuel_type",
        "engine",
        "accidents_or_damage",
        "one_owner",
        "driver_rating",
        "seller_rating",
        "price_drop",
    ]

    parts: list[str] = []
    for k in keys:
        v = payload.get(k)
        if v is None:
            parts.append("")
            continue
        if isinstance(v, str):
            parts.append(v.strip().lower())
        elif isinstance(v, (float, int)):
            # round to reduce tiny float differences when importing from CSV
            if isinstance(v, float):
                parts.append(f"{v:.4f}")
            else:
                parts.append(str(int(v)))
        else:
            parts.append(str(v).strip().lower())

    raw = "|".join(parts).encode("utf-8")
    return sha1(raw).hexdigest()


def bulk_insert_cars_from_dataframe(
    engine: Engine,
    df: pd.DataFrame,
    *,
    skip_duplicates: bool = True,
    chunk_size: int = 500,
) -> tuple[int, int]:
    """Insert cars from a dataframe. Returns (inserted, skipped)."""

    if df is None or df.empty:
        return 0, 0

    allowed = {
        "manufacturer",
        "model",
        "year",
        "mileage",
        "price",
        "mpg",
        "fuel_type",
        "engine",
        "accidents_or_damage",
        "one_owner",
        "driver_rating",
        "seller_rating",
        "price_drop",
    }

    def _truncate(value: Any, max_len: int | None) -> Any:
        if max_len is None:
            return value
        if value is None:
            return value
        if isinstance(value, str) and len(value) > max_len:
            return value[:max_len]
        return value

    # Build existing fingerprints once
    existing_fps: set[str] = set()
    if skip_duplicates:
        with Session(engine) as session:
            rows = session.execute(
                select(
                    Car.manufacturer,
                    Car.model,
                    Car.year,
                    Car.mileage,
                    Car.price,
                    Car.mpg,
                    Car.fuel_type,
                    Car.engine,
                    Car.accidents_or_damage,
                    Car.one_owner,
                    Car.driver_rating,
                    Car.seller_rating,
                    Car.price_drop,
                )
            ).all()
        for r in rows:
            payload = {
                "manufacturer": r[0],
                "model": r[1],
                "year": int(r[2] or 0),
                "mileage": float(r[3] or 0.0),
                "price": float(r[4] or 0.0),
                "mpg": r[5],
                "fuel_type": r[6],
                "engine": r[7],
                "accidents_or_damage": float(r[8] or 0.0),
                "one_owner": float(r[9] or 0.0),
                "driver_rating": float(r[10] or 0.0),
                "seller_rating": float(r[11] or 0.0),
                "price_drop": float(r[12] or 0.0),
            }
            existing_fps.add(_car_fingerprint(payload))

    inserted = 0
    skipped = 0
    batch: list[Car] = []

    for _, row in df.iterrows():
        payload = {k: row[k] for k in df.columns if k in allowed}

        # Defaults
        payload.setdefault("manufacturer", "Unknown")
        payload.setdefault("model", "Unknown")
        payload.setdefault("mpg", "Unknown")
        payload.setdefault("fuel_type", "Unknown")
        payload.setdefault("engine", "Unknown")

        # Defensive truncation (helps if DB wasn't migrated yet)
        payload["manufacturer"] = _truncate(str(payload.get("manufacturer", "Unknown")), 128)
        payload["model"] = _truncate(str(payload.get("model", "Unknown")), 256)
        payload["mpg"] = _truncate(str(payload.get("mpg", "Unknown")), 64)
        payload["fuel_type"] = _truncate(str(payload.get("fuel_type", "Unknown")), 64)
        # engine is TEXT after migration; keep a generous cap anyway
        payload["engine"] = _truncate(str(payload.get("engine", "Unknown")), 2000)

        # Cast
        try:
            payload["year"] = int(payload.get("year", 0) or 0)
        except Exception:
            payload["year"] = 0
        for fk in [
            "mileage",
            "price",
            "accidents_or_damage",
            "one_owner",
            "driver_rating",
            "seller_rating",
            "price_drop",
        ]:
            try:
                payload[fk] = float(payload.get(fk, 0.0) or 0.0)
            except Exception:
                payload[fk] = 0.0

        if skip_duplicates:
            fp = _car_fingerprint(payload)
            if fp in existing_fps:
                skipped += 1
                continue
            existing_fps.add(fp)

        batch.append(Car(**payload))
        if len(batch) >= int(chunk_size):
            with Session(engine) as session:
                session.add_all(batch)
                session.commit()
            inserted += len(batch)
            batch = []

    if batch:
        with Session(engine) as session:
            session.add_all(batch)
            session.commit()
        inserted += len(batch)

    return inserted, skipped


def delete_car(engine: Engine, car_id: int) -> None:
    with Session(engine) as session:
        obj = session.get(Car, car_id)
        if obj is None:
            return
        session.delete(obj)
        session.commit()


def save_recommendation(engine: Engine, user_id: int, inputs: dict[str, Any], results: dict[str, Any]) -> None:
    with Session(engine) as session:
        session.add(Recommendation(user_id=user_id, inputs=inputs, results=results))
        session.commit()


def list_recommendations(engine: Engine, user_id: int, limit: int = 50) -> list[Recommendation]:
    stmt = (
        select(Recommendation)
        .where(Recommendation.user_id == user_id)
        .order_by(Recommendation.created_at.desc())
        .limit(int(limit))
    )
    with Session(engine) as session:
        return session.execute(stmt).scalars().all()
