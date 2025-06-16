from dataclasses import dataclass, field
from typing import Dict, Any
from decimal import Decimal
from datetime import datetime, timezone
import json
from enum import Enum


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    CANCELLED = "cancelled"
    PARTIALLY_FILLED = "partially_filled"


@dataclass
class Market:
    id: str
    name: str
    current_price: Decimal
    volume: Decimal
    bid: Decimal
    ask: Decimal
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.current_price < 0:
            raise ValueError("Current price cannot be negative")
        if self.volume < 0:
            raise ValueError("Volume cannot be negative")
        if self.bid < 0:
            raise ValueError("Bid cannot be negative")
        if self.ask < 0:
            raise ValueError("Ask cannot be negative")
        if self.bid > self.ask:
            raise ValueError("Bid cannot be greater than ask")
    
    def spread(self) -> Decimal:
        return self.ask - self.bid
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "current_price": str(self.current_price),
            "volume": str(self.volume),
            "bid": str(self.bid),
            "ask": str(self.ask),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Market":
        return cls(
            id=data["id"],
            name=data["name"],
            current_price=Decimal(data["current_price"]),
            volume=Decimal(data["volume"]),
            bid=Decimal(data["bid"]),
            ask=Decimal(data["ask"]),
            metadata=data.get("metadata", {})
        )
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> "Market":
        return cls.from_dict(json.loads(json_str))


@dataclass
class Order:
    id: str
    market_id: str
    side: OrderSide
    price: Decimal
    size: Decimal
    status: OrderStatus
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    filled_size: Decimal = field(default=Decimal("0"))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.price <= 0:
            raise ValueError("Price must be positive")
        if self.size <= 0:
            raise ValueError("Size must be positive")
        if self.filled_size < 0:
            raise ValueError("Filled size cannot be negative")
        if self.filled_size > self.size:
            raise ValueError("Filled size cannot exceed order size")
    
    def remaining_size(self) -> Decimal:
        return self.size - self.filled_size
    
    def is_filled(self) -> bool:
        return self.filled_size == self.size
    
    def fill_percentage(self) -> Decimal:
        if self.size == 0:
            return Decimal("0")
        return (self.filled_size / self.size) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "market_id": self.market_id,
            "side": self.side.value,
            "price": str(self.price),
            "size": str(self.size),
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "filled_size": str(self.filled_size),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Order":
        return cls(
            id=data["id"],
            market_id=data["market_id"],
            side=OrderSide(data["side"]),
            price=Decimal(data["price"]),
            size=Decimal(data["size"]),
            status=OrderStatus(data["status"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            filled_size=Decimal(data["filled_size"]),
            metadata=data.get("metadata", {})
        )
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> "Order":
        return cls.from_dict(json.loads(json_str))


@dataclass
class Position:
    market_id: str
    size: Decimal
    entry_price: Decimal
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.entry_price <= 0:
            raise ValueError("Entry price must be positive")
    
    def current_value(self, current_price: Decimal) -> Decimal:
        return abs(self.size) * current_price
    
    def pnl(self, current_price: Decimal) -> Decimal:
        if self.size == 0:
            return Decimal("0")
        if self.size > 0:  # Long position
            return self.size * (current_price - self.entry_price)
        else:  # Short position
            return abs(self.size) * (self.entry_price - current_price)
    
    def pnl_percentage(self, current_price: Decimal) -> Decimal:
        if self.entry_price == 0:
            return Decimal("0")
        pnl_value = self.pnl(current_price)
        entry_value = abs(self.size) * self.entry_price
        return (pnl_value / entry_value) * 100
    
    def is_long(self) -> bool:
        return self.size > 0
    
    def is_short(self) -> bool:
        return self.size < 0
    
    def is_flat(self) -> bool:
        return self.size == 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "market_id": self.market_id,
            "size": str(self.size),
            "entry_price": str(self.entry_price),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Position":
        return cls(
            market_id=data["market_id"],
            size=Decimal(data["size"]),
            entry_price=Decimal(data["entry_price"]),
            metadata=data.get("metadata", {})
        )
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> "Position":
        return cls.from_dict(json.loads(json_str))