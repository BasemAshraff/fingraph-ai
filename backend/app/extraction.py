"""Entity, relationship, metric and transaction extraction for financial documents.

Rules run on every document, so the app works offline. When an LLM is configured, an extra
pass asks the model for additional entities and relations, and the results are merged.
Every fact keeps its page and the source line it came from, so the UI can show evidence.
"""
import json
import logging
import re
from statistics import median

from . import llm

log = logging.getLogger("fingraph.extraction")

NODE_TYPES = ("Company", "Vendor", "Invoice", "Transaction", "ReportingPeriod", "Document")

VENDOR_LINE = re.compile(r"(?im)^\s*(?:vendor|supplier|seller|issued by|from)\s*[:\-]\s*(.+?)\s*$")
COMPANY_LINE = re.compile(r"(?im)^\s*(?:customer|client|buyer|bill to|billed to|company|entity|to)\s*[:\-]\s*(.+?)\s*$")
# Invoice IDs must contain a digit, so ordinary words like "invoice template" are not captured.
INVOICE_NO = re.compile(r"(?i)\binvoice\s*(?:(?:no\.?|number|#)\s*[:#]?\s*)?((?=[A-Z0-9\-/]*\d)[A-Z0-9][A-Z0-9\-/]{2,})")
PERIOD = re.compile(
    r"(?i)\b(Q[1-4]\s*(?:FY)?\s*\d{4}|FY\s?\d{4}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})\b"
)
TRANSACTION = re.compile(r"(?m)^\s*(\d{4}-\d{2}-\d{2})\s*\|\s*([^|\n]+?)\s*\|\s*([^|\n]+?)\s*\|\s*(-?[\d,]+(?:\.\d+)?)\s*$")
METRIC_LABELS = [
    "total revenue", "revenue", "net income", "net profit", "gross profit", "operating expenses",
    "total expenses", "expenses", "ebitda", "subtotal", "tax", "total assets", "total liabilities",
    "cash", "amount due", "total",
]
METRIC = re.compile(
    r"(?im)^\s*(" + "|".join(re.escape(label) for label in METRIC_LABELS) + r")\s*[:\-]?\s*"
    r"(?:USD|EUR|GBP|EGP|\$|€|£)?\s*(\()?\s*(-?[\d,]+(?:\.\d+)?)\s*(\))?\s*$"
)


def node_id(node_type: str, name: str) -> str:
    return f"{node_type}:{' '.join(name.lower().split())}"


def _clean_name(raw: str) -> str:
    return raw.strip().strip("|").strip()


def _parse_number(text: str) -> float:
    return float(text.replace(",", ""))


def _period_label(raw: str) -> str:
    return " ".join(raw.split()).title().replace("Fy", "FY")


def extract_facts(doc_id: str, filename: str, pages: list[dict]) -> dict:
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    mentions: list[dict] = []
    metrics: list[dict] = []
    transactions: list[dict] = []

    def add_node(ntype: str, name: str, page: int | None, evidence: str | None) -> str:
        nid = node_id(ntype, name)
        nodes.setdefault(nid, {"label": name, "type": ntype})
        mentions.append({"node_id": nid, "page": page, "evidence": (evidence or "")[:240]})
        return nid

    def add_edge(source: str, target: str, relation: str, page, evidence):
        edges.append({"source": source, "target": target, "relation": relation, "page": page, "evidence": (evidence or "")[:240]})

    doc_node = add_node("Document", filename, None, None)
    vendor_ids, company_ids, period_ids, invoice_ids = [], [], [], []

    for page in pages:
        text, number = page["text"], page["page"]
        for match in VENDOR_LINE.finditer(text):
            vendor_ids.append(add_node("Vendor", _clean_name(match.group(1)), number, match.group(0)))
        for match in COMPANY_LINE.finditer(text):
            company_ids.append(add_node("Company", _clean_name(match.group(1)), number, match.group(0)))
        for match in INVOICE_NO.finditer(text):
            invoice_ids.append(add_node("Invoice", match.group(1).upper(), number, match.group(0)))
        for match in PERIOD.finditer(text):
            period_ids.append(add_node("ReportingPeriod", _period_label(match.group(1)), number, match.group(0)))
        for match in METRIC.finditer(text):
            value = _parse_number(match.group(3))
            if match.group(2) and match.group(4):
                value = -abs(value)
            metrics.append({"name": match.group(1).lower(), "value": value, "page": number, "evidence": match.group(0).strip()})
        for match in TRANSACTION.finditer(text):
            date, description, counterparty, amount = match.groups()
            transactions.append({
                "date": date, "description": description.strip(), "counterparty": counterparty.strip(),
                "amount": _parse_number(amount), "page": number,
            })
            txn_id = add_node("Transaction", f"{date} {description.strip()}", number, match.group(0))
            cp_id = add_node("Vendor", counterparty.strip(), number, match.group(0))
            add_edge(txn_id, cp_id, "WITH", number, match.group(0))

    # Relationships inside the document: invoices and the document itself link to parties and periods.
    primary_vendor = vendor_ids[0] if vendor_ids else None
    primary_company = company_ids[0] if company_ids else None
    primary_period = period_ids[0] if period_ids else None
    for inv in invoice_ids:
        if primary_vendor:
            add_edge(inv, primary_vendor, "ISSUED_BY", None, "Invoice issued by vendor on this document")
        if primary_company:
            add_edge(inv, primary_company, "BILLED_TO", None, "Invoice billed to company on this document")
        if primary_period:
            add_edge(inv, primary_period, "IN_PERIOD", None, "Invoice falls in this reporting period")
    if primary_period:
        add_edge(doc_node, primary_period, "REPORTS", None, "Document covers this reporting period")
    for vid in vendor_ids[:1] + company_ids[:1]:
        add_edge(doc_node, vid, "MENTIONS", None, "Party named on this document")

    # Optional LLM pass: adds entities and relations that rules cannot see (e.g. free-text narratives).
    if llm.llm_available():
        try:
            llm_nodes, llm_edges = _llm_extract(filename, pages)
            for nid, node in llm_nodes.items():
                nodes.setdefault(nid, node)
                mentions.append({"node_id": nid, "page": None, "evidence": "LLM extraction"})
            for edge in llm_edges:
                if edge["source"] in nodes and edge["target"] in nodes:
                    edges.append({**edge, "page": None, "evidence": edge.get("evidence", "LLM extraction")})
        except Exception as exc:  # the rule-based graph is still stored if the model call fails
            log.warning("LLM extraction failed for %s: %s", filename, exc)

    return {"nodes": nodes, "edges": edges, "mentions": mentions, "metrics": metrics, "transactions": transactions}


def _llm_extract(filename: str, pages: list[dict]):
    provider = llm.get_provider()
    text = "\n".join(p["text"] for p in pages)[:6000]
    prompt = (
        "Extract financial entities and relations from the document below. Reply with JSON only, shape: "
        '{"entities":[{"type":"Company|Vendor|Invoice|Transaction|ReportingPeriod","name":"..."}],'
        '"relations":[{"source":"Type:name","target":"Type:name","relation":"UPPER_SNAKE","evidence":"short quote"}]}. '
        "Use only facts stated in the text.\n\n" + text
    )
    raw = provider.chat([{"role": "user", "content": prompt}], json_mode=True)
    data = json.loads(raw)
    nodes, edges = {}, []
    for ent in data.get("entities", []):
        etype = ent.get("type")
        name = str(ent.get("name", "")).strip()
        if etype in NODE_TYPES and name:
            nodes[node_id(etype, name)] = {"label": name, "type": etype}
    for rel in data.get("relations", []):
        src, dst = rel.get("source", ""), rel.get("target", "")
        if ":" in src and ":" in dst:
            s_type, s_name = src.split(":", 1)
            d_type, d_name = dst.split(":", 1)
            s_id, d_id = node_id(s_type, s_name), node_id(d_type, d_name)
            if s_id in nodes or d_id in nodes:
                edges.append({"source": s_id, "target": d_id, "relation": str(rel.get("relation", "RELATED_TO")).upper(), "evidence": rel.get("evidence", "")})
    return nodes, edges


# ---------- analysis helpers used by the API ----------

def find_anomalies(documents: list[dict], metrics: list[dict], invoices: list[dict], transactions: list[dict]) -> list[dict]:
    """Rule-based checks. Each finding names the documents involved so it can be traced."""
    findings: list[dict] = []
    by_doc: dict[str, dict[str, float]] = {}
    for m in metrics:
        by_doc.setdefault(m["doc_id"], {})
        by_doc[m["doc_id"]].setdefault(m["name"], m["value"])
    names = {d["id"]: d["filename"] for d in documents}

    # 1. Subtotal + tax must equal total.
    for doc_id, values in by_doc.items():
        if {"subtotal", "tax", "total"} <= values.keys():
            expected = values["subtotal"] + values["tax"]
            if abs(expected - values["total"]) > 0.01:
                findings.append({
                    "severity": "high", "type": "total_mismatch", "documents": [names.get(doc_id, doc_id)],
                    "detail": f"Subtotal + tax = {expected:,.2f} but total = {values['total']:,.2f}.",
                })

    # 2. Net income larger than revenue is unusual and worth a check.
    for doc_id, values in by_doc.items():
        revenue = values.get("total revenue", values.get("revenue"))
        if revenue is not None and values.get("net income", 0) > revenue > 0:
            findings.append({
                "severity": "medium", "type": "income_exceeds_revenue", "documents": [names.get(doc_id, doc_id)],
                "detail": f"Net income ({values['net income']:,.2f}) is higher than revenue ({revenue:,.2f}).",
            })

    # 3. The same invoice number appearing in more than one document.
    seen: dict[str, set] = {}
    for inv in invoices:
        seen.setdefault(inv["label"], set()).add(inv["doc_id"])
    for label, doc_ids in seen.items():
        if len(doc_ids) > 1:
            findings.append({
                "severity": "medium", "type": "duplicate_invoice", "documents": [names.get(d, d) for d in doc_ids],
                "detail": f"Invoice {label} appears in {len(doc_ids)} documents. Check for duplicate billing.",
            })

    # 4. Outlier transactions using the median absolute deviation (robust to the outliers themselves).
    amounts = [t["amount"] for t in transactions if t["amount"] is not None]
    if len(amounts) >= 5:
        med = median(amounts)
        mad = median(abs(a - med) for a in amounts) or 1e-9
        for t in transactions:
            score = 0.6745 * (t["amount"] - med) / mad
            if abs(score) > 3.5:
                findings.append({
                    "severity": "medium", "type": "outlier_transaction", "documents": [names.get(t["doc_id"], t["doc_id"])],
                    "detail": f"{t['occurred_on']} '{t['description']}' amount {t['amount']:,.2f} is far from the median {med:,.2f}.",
                })

    # Period-over-period swings are deliberately not flagged here: documents can cover different companies
    # and periods, so the comparison is left to the explicit /api/compare endpoint.
    return findings


def compare_metric(metric: str, rows: list[dict], filenames: dict[str, str]) -> list[dict]:
    """rows: [{doc_id, value}] in the order given. Adds change vs the first row and vs the previous row."""
    out = []
    base = rows[0]["value"] if rows else None
    prev = None
    for row in rows:
        value = row["value"]
        entry = {"doc_id": row["doc_id"], "document": filenames.get(row["doc_id"], row["doc_id"]), "metric": metric, "value": value}
        if base not in (None, 0):
            entry["change_vs_first_pct"] = round((value - base) / abs(base) * 100, 2)
        if prev not in (None, 0):
            entry["change_vs_previous_pct"] = round((value - prev) / abs(prev) * 100, 2)
        prev = value
        out.append(entry)
    return out
