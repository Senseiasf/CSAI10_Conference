from datetime import datetime
from typing import Any, Dict, List
from sqlalchemy import Column, String, Integer, Boolean, Text, DateTime, ForeignKey, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base, relationship
from pgvector.sqlalchemy import Vector

Base = declarative_base()


class DestinationModel(Base):
    __tablename__ = "destinations"
    
    id = Column(String(64), primary_key=True)
    name = Column(String(128), nullable=False)
    country = Column(String(64), nullable=False)
    price_min = Column(Integer, nullable=False, default=0)
    price_max = Column(Integer, nullable=False, default=0)
    avg_check = Column(Integer, nullable=False, default=0)
    vibe_description = Column(Text, nullable=False)
    climate_summary = Column(Text, nullable=False)
    # 384-dimensional vector representation of destination vibe for fast search
    embedding = Column(Vector(384), nullable=True)
    metadata_json = Column(JSONB, nullable=True)
    
    hotels = relationship("HotelModel", back_populates="destination", cascade="all, delete-orphan")
    risks = relationship("RiskAlertModel", back_populates="destination", cascade="all, delete-orphan")


class HotelModel(Base):
    __tablename__ = "hotels"
    
    id = Column(String(64), primary_key=True)
    destination_id = Column(String(64), ForeignKey("destinations.id"), nullable=False)
    name = Column(String(128), nullable=False)
    city = Column(String(64), nullable=False)
    stars = Column(Integer, default=4)
    category = Column(String(64), nullable=False)
    pet_friendly = Column(Boolean, default=False)
    cancellation_policy = Column(String(255), nullable=False)
    cancellation_type = Column(String(32), default="flexible")
    meals = Column(String(128), nullable=False)
    checkin_time = Column(String(10), default="15:00")
    checkout_time = Column(String(10), default="12:00")
    price_per_night_rub = Column(Integer, nullable=False)
    pet_surcharge_per_night_rub = Column(Integer, default=0, nullable=True)
    distance_to_sea_meters = Column(Integer, default=0)
    family_friendly = Column(Boolean, default=True)
    features = Column(JSONB, nullable=True)
    
    destination = relationship("DestinationModel", back_populates="hotels")


class RiskAlertModel(Base):
    __tablename__ = "risk_alerts"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    destination_id = Column(String(64), ForeignKey("destinations.id"), nullable=False)
    risk_type = Column(String(32), nullable=False)  # weather, geopolitics, logistics, nature, culture
    severity = Column(String(16), default="medium")  # low, medium, high
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    
    destination = relationship("DestinationModel", back_populates="risks")


class UserSessionModel(Base):
    __tablename__ = "user_sessions"
    
    user_id = Column(String(128), primary_key=True)  # TG user ID or web session UUID
    current_state = Column(String(64), default="INSPIRATION")
    profile = Column(JSONB, default=dict)  # preferences: budget, children, pets, dates
    current_cart = Column(JSONB, default=dict)  # current assembled trip
    trip_state = Column(JSONB, nullable=True)
    trip_quote = Column(JSONB, nullable=True)
    conversation_history = Column(JSONB, default=list)  # message list
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
