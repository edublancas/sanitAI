import json
from pathlib import Path
import click

from sqlalchemy import (
    String,
    create_engine,
    Column,
    Integer,
    Boolean,
    ForeignKey,
    DateTime,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session, relationship
from sqlalchemy.orm import sessionmaker
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from datetime import datetime

from settings import DB_URI


class Base(DeclarativeBase):
    pass


class User(UserMixin, Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(120), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(256), nullable=False)
    is_admin: Mapped[bool] = mapped_column(default=False)

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)


class DetectionRule(Base):
    __tablename__ = "detection_rules"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    pattern = Column(String(1000), nullable=False)
    explanation = Column(String(1000))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    samples = relationship(
        "RuleSample", back_populates="rule", cascade="all, delete-orphan"
    )


class RuleSample(Base):
    __tablename__ = "rule_samples"

    id = Column(Integer, primary_key=True)
    rule_id = Column(Integer, ForeignKey("detection_rules.id"))
    sample_text = Column(String(1000), nullable=False)
    trigger_alert = Column(Boolean, nullable=False)
    explanation = Column(String(1000))
    rule = relationship("DetectionRule", back_populates="samples")


# Create engine and session factory
engine = create_engine(DB_URI)
SessionLocal = sessionmaker(bind=engine)


def init_db():
    Base.metadata.create_all(engine)

    # Create admin user if it doesn't exist
    with SessionLocal() as session:
        admin = session.query(User).filter_by(email="admin@example.com").first()
        if not admin:
            admin = User(email="admin@example.com", is_admin=True)
            admin.set_password("admin123")
            session.add(admin)
            session.commit()


def load_rules(load_all: bool = False):
    """Load detection rules from JSON files in rules/ directory.

    Parameters
    ----------
    load_all : bool, default=False
        If True, loads all JSON files in the rules directory.
        If False, excludes files that begin with 'test-'.

    Notes
    -----
    Removes existing rules and samples from database, then loads rules from
    JSON files in the rules/ directory.
    """
    print("Loading detection rules from rules/ directory...")

    with SessionLocal() as session:
        # Delete existing rules and samples
        print("Deleting existing rules and samples from database...")
        session.query(RuleSample).delete()
        session.query(DetectionRule).delete()

        # Load rules from JSON files
        rules_dir = Path("rules")
        if not rules_dir.exists():
            print("No rules directory found. Skipping rule loading.")
            return

        rule_files = list(rules_dir.glob("*.json"))
        if not load_all:
            rule_files = [f for f in rule_files if not f.name.startswith("test-")]

        print(f"Found {len(rule_files)} rule files to load")

        for rule_file in rule_files:
            print(f"Loading rule from {rule_file.name}...")
            rule_data = json.loads(rule_file.read_text())

            # Create rule
            rule = DetectionRule(
                name=rule_data["name"],
                pattern=rule_data["rule"],
                explanation=rule_data["explanation"],
            )
            session.add(rule)

            # Create samples
            test_cases = rule_data["test_cases"]
            print(f"Adding {len(test_cases)} test cases for rule {rule_data['name']}")
            for test_case in test_cases:
                sample = RuleSample(
                    rule=rule,
                    sample_text=test_case["text"],
                    trigger_alert=test_case["should_trigger_alert"],
                    explanation=test_case["explanation"],
                )
                session.add(sample)

        session.commit()
        print("Successfully loaded all rules and test cases")


def delete_rules():
    """Delete all detection rules and samples from the database."""
    print("Deleting all rules and samples from database...")
    with SessionLocal() as session:
        session.query(RuleSample).delete()
        session.query(DetectionRule).delete()
        session.commit()
    print("Successfully deleted all rules and samples")


@click.group()
def cli():
    """CLI for managing detection rules."""
    pass


@cli.command()
@click.option(
    "--all",
    "-a",
    "load_all",
    is_flag=True,
    help="Load all rules including test files",
)
def load(load_all: bool):
    """Load detection rules from JSON files in rules/ directory."""
    load_rules(load_all)


@cli.command()
def delete():
    """Delete all detection rules and samples from the database."""
    delete_rules()


@cli.command()
def db():
    """Create the database tables."""
    init_db()


if __name__ == "__main__":
    cli()
