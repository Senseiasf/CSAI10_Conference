"""Referential integrity and coverage of the local logistics catalogue."""
from collections import Counter


def audit_catalog(data: dict) -> dict:
    destinations = data.get("destinations", [])
    hotels = [h["id"] for d in destinations for h in d.get("hotels", [])]
    airports = {a["code"] for a in data.get("origin_airports", [])}
    airports.update(a["code"] for d in destinations for a in d.get("airports", []))
    hotel_ids = set(hotels)
    flights = [f for d in destinations for f in d.get("flights", [])]
    transfers = [t for d in destinations for t in d.get("transfers", [])]
    errors = [f"Duplicate hotel: {h}" for h, count in Counter(hotels).items() if count > 1]
    for f in flights:
        if f["from"] not in airports or f["to"] not in airports:
            errors.append(f"Dangling flight: {f['flight_no']} {f['from']} -> {f['to']}")
    for t in transfers:
        if t["from"] not in airports | hotel_ids or t["to"] not in hotel_ids:
            errors.append(f"Dangling transfer: {t['from']} -> {t['to']}")
    pairs = {(f["from"], f["to"]) for f in flights}
    covered = {t["to"] for t in transfers if t["from"] in airports}
    return {
        "errors": errors, "destinations": len(destinations), "airports": len(airports),
        "hotels": len(hotels), "flights": len(flights), "transfers": len(transfers),
        "airport_transfers": sum(t["from"] in airports for t in transfers),
        "interhotel_transfers": sum(t["from"] in hotel_ids for t in transfers),
        "hotels_without_airport_transfer": sorted(hotel_ids - covered),
        "flights_without_reverse_pair": sum((f["to"], f["from"]) not in pairs for f in flights),
        "transfers_without_capacity": sum(t.get("capacity", t.get("max_passengers")) is None for t in transfers),
    }


def validate_catalog(data: dict) -> dict:
    report = audit_catalog(data)
    if report["errors"]:
        raise ValueError("Invalid logistics catalogue: " + "; ".join(report["errors"]))
    return report
