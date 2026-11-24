"""Generates synthetic financial documents for testing. No real or customer data is used.

Run from the project root:  python scripts/make_samples.py
Outputs go to samples/. The scanned files are image-only, so they exercise the OCR path.
"""
from pathlib import Path

from docx import Document
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

OUT = Path(__file__).resolve().parent.parent / "samples"
OUT.mkdir(exist_ok=True)


def invoice_pdf():
    c = canvas.Canvas(str(OUT / "invoice_nile_cloud_jan2026.pdf"), pagesize=A4)
    y = 800
    for line in [
        "INVOICE",
        "Invoice No: INV-2026-001",
        "Vendor: Nile Cloud Services",
        "Customer: Delta Retail Ltd",
        "Period: Q1 2026",
        "",
        "Monthly cloud hosting and support, January to March 2026.",
        "Payment terms: net 30 days from invoice date.",
        "",
        "Subtotal: 12,000.00 USD",
        "Tax: 1,800.00 USD",
        "Total: 13,800.00 USD",
        "",
        "Transactions",
        "2026-01-05 | Monthly hosting | Nile Cloud Services | 4,500.00",
        "2026-02-05 | Monthly hosting | Nile Cloud Services | 4,500.00",
        "2026-03-05 | Monthly hosting | Nile Cloud Services | 4,500.00",
        "2026-03-12 | Premium support | Nile Cloud Services | 4,500.00",
        "2026-03-20 | Data migration | Nile Cloud Services | 4,500.00",
        "2026-03-28 | Emergency restore | Nile Cloud Services | 95,000.00",
    ]:
        c.drawString(60, y, line)
        y -= 22
    c.save()


def financial_statement_pdf():
    c = canvas.Canvas(str(OUT / "annual_statement_delta_fy2025.pdf"), pagesize=A4)
    y = 800
    for line in [
        "Delta Retail Ltd - Annual Financial Statement",
        "Company: Delta Retail Ltd",
        "Reporting period: FY 2025",
        "",
        "Total revenue: 250,000.00",
        "Operating expenses: 190,000.00",
        "Gross profit: 160,000.00",
        "Net income: 32,500.00",
        "Total assets: 410,000.00",
        "Total liabilities: 180,000.00",
        "Cash: 45,000.00",
        "",
        "Revenue grew steadily through 2025, driven by online sales.",
        "Vendor payments in FY 2025 were mainly to Nile Cloud Services.",
    ]:
        c.drawString(60, y, line)
        y -= 22
    c.save()


def management_report_pdf():
    c = canvas.Canvas(str(OUT / "management_report_delta_q1_2026.pdf"), pagesize=A4)
    y = 800
    for line in [
        "Delta Retail Ltd - Management Report",
        "Company: Delta Retail Ltd",
        "Reporting period: Q1 2026",
        "",
        "Total revenue: 68,000.00",
        "Operating expenses: 51,000.00",
        "Net income: 9,800.00",
        "Cloud costs were billed under invoice INV-2026-001 and paid on time.",
        "",
        "Subtotal: 12,000.00",
        "Tax: 1,800.00",
        "Total: 15,000.00",
    ]:
        c.drawString(60, y, line)
        y -= 22
    c.save()


def contract_docx():
    doc = Document()
    doc.add_heading("Master Vendor Services Agreement", level=1)
    doc.add_paragraph("Vendor: Nile Cloud Services")
    doc.add_paragraph("Client: Delta Retail Ltd")
    doc.add_paragraph(
        "The vendor provides cloud hosting, support and data migration. Invoices are issued monthly "
        "and payable within 30 days. Late payments accrue interest at 1.5 percent per month."
    )
    doc.add_paragraph(
        "Either party may terminate with 60 days written notice. Emergency restore services are billed "
        "at the rate in Schedule A and must be approved in writing by the client."
    )
    table = doc.add_table(rows=3, cols=2)
    table.rows[0].cells[0].text, table.rows[0].cells[1].text = "Service", "Monthly fee (USD)"
    table.rows[1].cells[0].text, table.rows[1].cells[1].text = "Cloud hosting", "4,500.00"
    table.rows[2].cells[0].text, table.rows[2].cells[1].text = "Premium support", "1,200.00"
    doc.save(str(OUT / "vendor_agreement_nile_cloud.docx"))


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # older Pillow
        return ImageFont.load_default()


def scanned_invoice_png():
    img = Image.new("RGB", (1400, 900), "white")
    draw = ImageDraw.Draw(img)
    lines = [
        "SAHARA LOGISTICS",
        "Invoice No: INV-2026-014",
        "Customer: Delta Retail Ltd",
        "Period: March 2026",
        "Freight and warehousing services",
        "Total: 2,350.00 USD",
    ]
    for i, line in enumerate(lines):
        draw.text((80, 80 + i * 90), line, fill="black", font=_font(46))
    img.save(OUT / "scanned_invoice_sahara_mar2026.png")


def scanned_statement_pdf():
    img = Image.new("RGB", (1240, 1754), "white")  # A4 at about 150 dpi
    draw = ImageDraw.Draw(img)
    lines = [
        "Sahara Logistics - Quarterly Statement",
        "Company: Delta Retail Ltd",
        "Reporting period: Q1 2026",
        "Total revenue: 41,200.00",
        "Net income: 4,100.00",
    ]
    for i, line in enumerate(lines):
        draw.text((80, 120 + i * 80), line, fill="black", font=_font(40))
    img.save(OUT / "scanned_statement_sahara_q1.pdf", "PDF", resolution=150.0)


if __name__ == "__main__":
    invoice_pdf()
    financial_statement_pdf()
    management_report_pdf()
    contract_docx()
    scanned_invoice_png()
    scanned_statement_pdf()
    print("Generated synthetic samples in", OUT)
    for f in sorted(OUT.iterdir()):
        print(" -", f.name)
