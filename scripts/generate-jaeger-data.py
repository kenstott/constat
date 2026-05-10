#!/usr/bin/env python3
"""Generate sample OTel trace data and send to Jaeger via OTLP HTTP.

No dependencies beyond requests (already in constat deps).
Sends raw OTLP JSON to Jaeger's OTLP HTTP endpoint (port 4318).

Services generated:
  - api-gateway: HTTP ingress (GET/POST/PUT/DELETE)
  - user-service: User CRUD operations
  - order-service: Order processing
  - payment-service: Payment processing with occasional failures
  - inventory-db: Database queries (PostgreSQL)
"""

import json
import random
import time
import uuid

import requests

OTLP_ENDPOINT = "http://localhost:4318/v1/traces"

# --- Service definitions ---

SERVICES = {
    "api-gateway": {
        "operations": [
            ("GET /api/users", "GET"),
            ("GET /api/users/{id}", "GET"),
            ("POST /api/users", "POST"),
            ("GET /api/orders", "GET"),
            ("POST /api/orders", "POST"),
            ("GET /api/orders/{id}", "GET"),
            ("PUT /api/orders/{id}/status", "PUT"),
            ("DELETE /api/orders/{id}", "DELETE"),
        ],
        "attributes": {"service.version": "2.1.0", "deployment.environment": "production"},
    },
    "user-service": {
        "operations": [
            ("UserService.GetUser", None),
            ("UserService.ListUsers", None),
            ("UserService.CreateUser", None),
            ("UserService.UpdateUser", None),
        ],
        "attributes": {"service.version": "1.5.2", "deployment.environment": "production"},
    },
    "order-service": {
        "operations": [
            ("OrderService.CreateOrder", None),
            ("OrderService.GetOrder", None),
            ("OrderService.ListOrders", None),
            ("OrderService.UpdateStatus", None),
            ("OrderService.CancelOrder", None),
        ],
        "attributes": {"service.version": "3.0.1", "deployment.environment": "production"},
    },
    "payment-service": {
        "operations": [
            ("PaymentService.ProcessPayment", None),
            ("PaymentService.RefundPayment", None),
            ("PaymentService.GetPaymentStatus", None),
        ],
        "attributes": {"service.version": "1.2.0", "deployment.environment": "production"},
    },
    "inventory-db": {
        "operations": [
            ("SELECT users", None),
            ("INSERT users", None),
            ("SELECT orders", None),
            ("INSERT orders", None),
            ("UPDATE orders", None),
            ("SELECT inventory", None),
            ("UPDATE inventory", None),
        ],
        "attributes": {"service.version": "1.0.0", "db.system": "postgresql"},
    },
}

# --- Trace generation flows ---

def _trace_id():
    return uuid.uuid4().hex

def _span_id():
    return uuid.uuid4().hex[:16]

def _now_ns():
    return int(time.time() * 1_000_000_000)

def _make_span(
    trace_id, span_id, parent_span_id, service, operation,
    start_ns, duration_ms, status_code=None, error=False,
    http_method=None, db_statement=None,
):
    """Build an OTLP span dict."""
    attributes = []

    if http_method:
        attributes.append({"key": "http.method", "value": {"stringValue": http_method}})
    if status_code is not None:
        attributes.append({"key": "http.status_code", "value": {"intValue": str(status_code)}})
    if db_statement:
        attributes.append({"key": "db.statement", "value": {"stringValue": db_statement}})
        attributes.append({"key": "db.system", "value": {"stringValue": "postgresql"}})
    if error:
        attributes.append({"key": "error", "value": {"boolValue": True}})

    span = {
        "traceId": trace_id,
        "spanId": span_id,
        "name": operation,
        "kind": 2 if http_method else 3,  # SERVER=2, CLIENT=3
        "startTimeUnixNano": str(start_ns),
        "endTimeUnixNano": str(start_ns + duration_ms * 1_000_000),
        "attributes": attributes,
        "status": {"code": 2 if error else 1},  # ERROR=2, OK=1
    }

    if parent_span_id:
        span["parentSpanId"] = parent_span_id

    return span, service


def _generate_user_list_flow(base_ns):
    """GET /api/users -> UserService.ListUsers -> SELECT users"""
    tid = _trace_id()
    gw_sid = _span_id()
    usr_sid = _span_id()
    db_sid = _span_id()

    return [
        _make_span(tid, gw_sid, None, "api-gateway", "GET /api/users",
                    base_ns, random.randint(20, 80), status_code=200, http_method="GET"),
        _make_span(tid, usr_sid, gw_sid, "user-service", "UserService.ListUsers",
                    base_ns + 2_000_000, random.randint(10, 50)),
        _make_span(tid, db_sid, usr_sid, "inventory-db", "SELECT users",
                    base_ns + 4_000_000, random.randint(2, 15), db_statement="SELECT * FROM users LIMIT 100"),
    ]


def _generate_create_order_flow(base_ns):
    """POST /api/orders -> OrderService.CreateOrder -> PaymentService.ProcessPayment + INSERT orders"""
    tid = _trace_id()
    gw_sid = _span_id()
    ord_sid = _span_id()
    pay_sid = _span_id()
    db_sid = _span_id()
    inv_sid = _span_id()

    # 15% chance of payment failure
    payment_fails = random.random() < 0.15
    error_status = 500 if payment_fails else 201

    return [
        _make_span(tid, gw_sid, None, "api-gateway", "POST /api/orders",
                    base_ns, random.randint(100, 300), status_code=error_status, http_method="POST",
                    error=payment_fails),
        _make_span(tid, ord_sid, gw_sid, "order-service", "OrderService.CreateOrder",
                    base_ns + 3_000_000, random.randint(50, 200), error=payment_fails),
        _make_span(tid, pay_sid, ord_sid, "payment-service", "PaymentService.ProcessPayment",
                    base_ns + 10_000_000, random.randint(30, 150), error=payment_fails),
        _make_span(tid, db_sid, ord_sid, "inventory-db", "INSERT orders",
                    base_ns + 60_000_000, random.randint(3, 20),
                    db_statement="INSERT INTO orders (user_id, total, status) VALUES ($1, $2, $3)"),
        _make_span(tid, inv_sid, ord_sid, "inventory-db", "UPDATE inventory",
                    base_ns + 70_000_000, random.randint(2, 10),
                    db_statement="UPDATE inventory SET quantity = quantity - $1 WHERE product_id = $2"),
    ]


def _generate_get_order_flow(base_ns):
    """GET /api/orders/{id} -> OrderService.GetOrder -> SELECT orders"""
    tid = _trace_id()
    gw_sid = _span_id()
    ord_sid = _span_id()
    db_sid = _span_id()

    return [
        _make_span(tid, gw_sid, None, "api-gateway", "GET /api/orders/{id}",
                    base_ns, random.randint(15, 60), status_code=200, http_method="GET"),
        _make_span(tid, ord_sid, gw_sid, "order-service", "OrderService.GetOrder",
                    base_ns + 2_000_000, random.randint(8, 40)),
        _make_span(tid, db_sid, ord_sid, "inventory-db", "SELECT orders",
                    base_ns + 4_000_000, random.randint(2, 12),
                    db_statement="SELECT * FROM orders WHERE id = $1"),
    ]


def _generate_update_status_flow(base_ns):
    """PUT /api/orders/{id}/status -> OrderService.UpdateStatus -> UPDATE orders"""
    tid = _trace_id()
    gw_sid = _span_id()
    ord_sid = _span_id()
    db_sid = _span_id()

    return [
        _make_span(tid, gw_sid, None, "api-gateway", "PUT /api/orders/{id}/status",
                    base_ns, random.randint(20, 70), status_code=200, http_method="PUT"),
        _make_span(tid, ord_sid, gw_sid, "order-service", "OrderService.UpdateStatus",
                    base_ns + 2_000_000, random.randint(10, 40)),
        _make_span(tid, db_sid, ord_sid, "inventory-db", "UPDATE orders",
                    base_ns + 4_000_000, random.randint(2, 15),
                    db_statement="UPDATE orders SET status = $1 WHERE id = $2"),
    ]


def _generate_create_user_flow(base_ns):
    """POST /api/users -> UserService.CreateUser -> INSERT users"""
    tid = _trace_id()
    gw_sid = _span_id()
    usr_sid = _span_id()
    db_sid = _span_id()

    return [
        _make_span(tid, gw_sid, None, "api-gateway", "POST /api/users",
                    base_ns, random.randint(25, 90), status_code=201, http_method="POST"),
        _make_span(tid, usr_sid, gw_sid, "user-service", "UserService.CreateUser",
                    base_ns + 3_000_000, random.randint(10, 50)),
        _make_span(tid, db_sid, usr_sid, "inventory-db", "INSERT users",
                    base_ns + 5_000_000, random.randint(3, 20),
                    db_statement="INSERT INTO users (name, email) VALUES ($1, $2)"),
    ]


FLOW_GENERATORS = [
    (_generate_user_list_flow, 3),      # weight
    (_generate_create_order_flow, 4),
    (_generate_get_order_flow, 3),
    (_generate_update_status_flow, 2),
    (_generate_create_user_flow, 2),
]


def generate_traces(count=100):
    """Generate `count` traces across all flow types."""
    # Build weighted list
    weighted = []
    for gen, weight in FLOW_GENERATORS:
        weighted.extend([gen] * weight)

    all_spans = []  # list of (span_dict, service_name)
    base_ns = _now_ns() - (count * 1_000_000_000)  # spread over last N seconds

    for i in range(count):
        gen = random.choice(weighted)
        flow_time = base_ns + i * random.randint(500_000_000, 2_000_000_000)
        spans = gen(flow_time)
        all_spans.extend(spans)

    return all_spans


def build_otlp_payload(spans_with_service):
    """Group spans by service and build OTLP JSON payload."""
    by_service = {}
    for span, svc in spans_with_service:
        by_service.setdefault(svc, []).append(span)

    resource_spans = []
    for svc, spans in by_service.items():
        svc_attrs = SERVICES[svc]["attributes"]
        resource_attrs = [{"key": "service.name", "value": {"stringValue": svc}}]
        for k, v in svc_attrs.items():
            resource_attrs.append({"key": k, "value": {"stringValue": v}})

        resource_spans.append({
            "resource": {"attributes": resource_attrs},
            "scopeSpans": [{
                "scope": {"name": "constat-sample-generator", "version": "1.0.0"},
                "spans": spans,
            }],
        })

    return {"resourceSpans": resource_spans}


def send_traces(payload):
    """Send OTLP JSON to Jaeger."""
    resp = requests.post(
        OTLP_ENDPOINT,
        json=payload,
        headers={"Content-Type": "application/json"},
    )
    resp.raise_for_status()
    return resp


def main():
    print("Generating 100 sample traces...")
    spans = generate_traces(100)
    print(f"  {len(spans)} total spans across 5 services")

    payload = build_otlp_payload(spans)
    send_traces(payload)
    print("  Sent to Jaeger via OTLP HTTP")

    # Give Jaeger a moment to index
    time.sleep(1)

    # Verify
    resp = requests.get("http://localhost:16686/api/services")
    services = resp.json().get("data", [])
    print(f"  Services visible in Jaeger: {services}")


if __name__ == "__main__":
    main()
