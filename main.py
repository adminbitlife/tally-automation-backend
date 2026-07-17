from __future__ import annotations

import os
import tempfile
from typing import Annotated

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse

import uuid
import datetime
import xml.etree.ElementTree as ET
from io import BytesIO

from openai_service import extract_invoice_data

app = FastAPI(title="Image OCR Demo")
# app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={"request": request})


@app.post("/extract-text")
async def extract_text(file: Annotated[UploadFile, File(...)], purpose: Annotated[str, Form()] = "ocr"):
    if not file.filename:
        return JSONResponse(status_code=400, content={"error": "Please choose an image file."})

    suffix = os.path.splitext(file.filename)[1] or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        contents = await file.read()
        tmp.write(contents)
        temp_path = tmp.name

    try:
        if os.getenv("OPENAI_API_KEY"):
            invoice_json = extract_invoice_data(temp_path)
        else:
            invoice_json = {"error": "OPENAI_API_KEY is not set"}

        return {
            "invoice_json": invoice_json,
            "filename": file.filename,
            "purpose": purpose,
        }
    except Exception as exc:  # pragma: no cover - defensive path
        return JSONResponse(status_code=500, content={"error": str(exc)})
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _build_tally_xml(invoice: dict, company_name: str = "Byto Labs") -> bytes:
    # Minimal mapping to the requested Tally XML structure. This fills key fields.
    envelope = ET.Element('ENVELOPE')

    header = ET.SubElement(envelope, 'HEADER')
    ET.SubElement(header, 'TALLYREQUEST').text = 'Import Data'

    body = ET.SubElement(envelope, 'BODY')
    importdata = ET.SubElement(body, 'IMPORTDATA')

    requestdesc = ET.SubElement(importdata, 'REQUESTDESC')
    ET.SubElement(requestdesc, 'REPORTNAME').text = 'All Masters'
    staticvars = ET.SubElement(requestdesc, 'STATICVARIABLES')
    ET.SubElement(staticvars, 'SVCURRENTCOMPANY').text = company_name

    requestdata = ET.SubElement(importdata, 'REQUESTDATA')
    tallymessage = ET.SubElement(requestdata, 'TALLYMESSAGE')
    tallymessage.set('xmlns:UDF', 'TallyUDF')

    voucher = ET.SubElement(tallymessage, 'VOUCHER')
    guid = str(uuid.uuid4())
    voucher.set('REMOTEID', guid + '-00000001')
    voucher.set('VCHKEY', guid + ':00000001')
    voucher.set('VCHTYPE', 'Purchase')
    voucher.set('ACTION', 'Create')
    voucher.set('OBJVIEW', 'Invoice Voucher View')

    # Basic fields
    billing_addr_list = ET.SubElement(voucher, 'BASICBUYERADDRESS.LIST')
    addr_lines = invoice.get('seller', {}).get('address', '')
    if addr_lines:
        for line in str(addr_lines).split('\n'):
            ET.SubElement(billing_addr_list, 'BASICBUYERADDRESS').text = line

    ET.SubElement(voucher, 'OLDAUDITENTRYIDS.LIST', TYPE='Number')
    ET.SubElement(voucher, 'OLDAUDITENTRYIDS').text = '-1'

    # Dates: try to parse invoice_date else use today
    inv_date = invoice.get('invoice_date')
    try:
        dt = datetime.datetime.fromisoformat(inv_date)
    except Exception:
        dt = datetime.datetime.utcnow()
    datestr = dt.strftime('%Y%m%d')
    ET.SubElement(voucher, 'DATE').text = datestr
    ET.SubElement(voucher, 'REFERENCEDATE').text = datestr
    ET.SubElement(voucher, 'VCHSTATUSDATE').text = datestr

    ET.SubElement(voucher, 'GUID').text = guid + '-00000001'
    ET.SubElement(voucher, 'VOUCHERTYPENAME').text = 'Purchase'

    party_name = invoice.get('seller', {}).get('store_name') or invoice.get('seller', {}).get('owner_name') or invoice.get('buyer', {}).get('name') if invoice.get('buyer') else 'Unknown'
    ET.SubElement(voucher, 'PARTYNAME').text = party_name
    ET.SubElement(voucher, 'PARTYLEDGERNAME').text = 'Cash'
    ET.SubElement(voucher, 'VOUCHERNUMBER').text = '1'

    ET.SubElement(voucher, 'BASICBUYERNAME').text = company_name
    ET.SubElement(voucher, 'PARTYMAILINGNAME').text = party_name
    ET.SubElement(voucher, 'CONSIGNEEMAILINGNAME').text = company_name

    # Reference and totals
    ET.SubElement(voucher, 'REFERENCE').text = invoice.get('invoice_number', '') or ''
    totals = invoice.get('totals', {}) or {}
    total_amt = totals.get('total')
    if total_amt is None:
        # try compute from items
        items = invoice.get('items', []) or []
        try:
            total_amt = sum(float(i.get('amount', 0) or (float(i.get('rate',0)) * float(i.get('quantity',1)))) for i in items)
        except Exception:
            total_amt = 0
    ET.SubElement(voucher, 'BASICBASEPARTYNAME').text = 'Cash'
    ET.SubElement(voucher, 'NUMBERINGSTYLE').text = 'Auto Retain'

    # Inventory entries for each item
    for it in invoice.get('items', []) or []:
        inv_entry = ET.SubElement(voucher, 'ALLINVENTORYENTRIES.LIST')
        ET.SubElement(inv_entry, 'STOCKITEMNAME').text = it.get('description', 'Item')
        amount_val = it.get('amount') if it.get('amount') is not None else (it.get('rate', 0) * it.get('quantity', 1))
        ET.SubElement(inv_entry, 'AMOUNT').text = str(amount_val)

    # Ledger entries
    ledger = ET.SubElement(voucher, 'LEDGERENTRIES.LIST')
    ET.SubElement(ledger, 'LEDGERNAME').text = 'Cash'
    ET.SubElement(ledger, 'AMOUNT').text = str(total_amt if total_amt is not None else 0)

    # produce pretty XML bytes
    tree = ET.ElementTree(envelope)
    bio = BytesIO()
    tree.write(bio, encoding='utf-8', xml_declaration=True)
    return bio.getvalue()


@app.post('/generate-xml')
async def generate_xml_endpoint(payload: dict):
    # payload expected to contain `invoice_json` and optional `company_name`
    invoice = payload.get('invoice_json') if isinstance(payload, dict) else None
    if not invoice:
        return JSONResponse(status_code=400, content={'error': 'invoice_json is required in the payload'})

    company_name = payload.get('company_name', 'Byto Labs')
    xml_bytes = _build_tally_xml(invoice, company_name=company_name)

    filename = f"tally_import_{invoice.get('invoice_number','') or '1'}.xml"
    return StreamingResponse(BytesIO(xml_bytes), media_type='application/xml', headers={
        'Content-Disposition': f'attachment; filename="{filename}"'
    })
