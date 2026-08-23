"""Table extraction from PDF documents preserving column relationships."""

import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from person_b.extraction.normalization import normalize_whitespace
from person_b.models import (
    Evidence,
    EvidenceLocation,
    ExtractionLocationType,
)

try:
    import pdfplumber
except ImportError:
    pdfplumber = None


def normalize_pdf_cell(cell: Optional[str]) -> str:
    """Clean multiline PDF cell text and handle None values."""
    if cell is None:
        return ""
    # Replace internal newlines with space while collapsing whitespace
    cleaned = cell.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    return normalize_whitespace(cleaned)


def extract_pdf_tables(
    pdf_bytes_or_path: Union[bytes, str, Path],
    source_url: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Extract structured table matrices across all pages of a PDF document.

    Returns a dictionary containing:
      - tables: list of structured table dicts (with headers, rows, records, evidence, page_number, table_index)
      - total_pages: number of pages inspected
      - total_tables: total tables found
      - is_unreadable: boolean indicating if PDF yielded no usable text/tables
      - warnings: list of warning strings
    """
    if pdfplumber is None:
        return {
            "tables": [],
            "total_pages": 0,
            "total_tables": 0,
            "is_unreadable": True,
            "warnings": ["pdfplumber library is not installed."],
        }

    warnings: List[str] = []
    extracted_tables: List[Dict[str, Any]] = []
    all_evidence: List[Evidence] = []
    total_text_chars = 0
    total_pages = 0

    try:
        if isinstance(pdf_bytes_or_path, (str, Path)):
            pdf_file = pdfplumber.open(str(pdf_bytes_or_path))
        else:
            pdf_file = pdfplumber.open(io.BytesIO(pdf_bytes_or_path))

        with pdf_file as pdf:
            total_pages = len(pdf.pages)

            for page_idx, page in enumerate(pdf.pages):
                page_num = page_idx + 1
                page_text = page.extract_text() or ""
                total_text_chars += len(page_text.strip())

                raw_tables = page.extract_tables() or []

                for t_idx, raw_table in enumerate(raw_tables):
                    if not raw_table or len(raw_table) == 0:
                        continue

                    # Clean all cells
                    cleaned_grid: List[List[str]] = []
                    for row in raw_table:
                        cleaned_row = [normalize_pdf_cell(c) for c in row]
                        if any(c for c in cleaned_row):  # Skip completely empty rows
                            cleaned_grid.append(cleaned_row)

                    if not cleaned_grid or len(cleaned_grid) < 1:
                        continue

                    # Filter out tiny 1-cell / 1-column single-character artifact tables
                    if len(cleaned_grid) <= 2 and len(cleaned_grid[0]) <= 1 and sum(len(c) for r in cleaned_grid for c in r) < 10:
                        continue

                    headers = cleaned_grid[0]
                    data_rows = cleaned_grid[1:] if len(cleaned_grid) > 1 else cleaned_grid

                    table_records: List[Dict[str, Any]] = []
                    table_evidence: List[Evidence] = []

                    # Infer table title if row 0 has a title cell or from headers
                    table_title = headers[0] if headers and headers[0] else f"PDF Table (Page {page_num}, #{t_idx + 1})"

                    for r_idx, row in enumerate(data_rows):
                        rec: Dict[str, Any] = {}
                        row_label = row[0] if len(row) > 0 and row[0] else f"Row_{r_idx}"

                        for c_idx, cell in enumerate(row):
                            col_header = headers[c_idx] if c_idx < len(headers) and headers[c_idx] else f"col_{c_idx}"
                            rec[col_header] = cell

                            if cell:  # Only create evidence for non-empty cells
                                loc = EvidenceLocation(
                                    type=ExtractionLocationType.PDF_TABLE,
                                    table_name=table_title,
                                    page_number=page_num,
                                    row_index=r_idx,
                                    col_name=col_header,
                                    details={"table_index": t_idx, "row_label": row_label},
                                )
                                ev = Evidence(
                                    field=f"{row_label} -> {col_header}" if row_label != col_header else col_header,
                                    value=cell,
                                    source_url=source_url,
                                    evidence_text=f"{col_header} | {row_label} | {cell}",
                                    location=loc,
                                    confidence=0.95,
                                )
                                table_evidence.append(ev)
                                all_evidence.append(ev)

                        table_records.append(rec)

                    table_entry = {
                        "page_number": page_num,
                        "table_index": t_idx + 1,
                        "table_name": table_title,
                        "headers": headers,
                        "rows": data_rows,
                        "records": table_records,
                        "evidence": table_evidence,
                    }
                    extracted_tables.append(table_entry)

    except Exception as e:
        warnings.append(f"PDF table extraction failed: {str(e)}")

    # Check for image-only/unreadable PDF
    # If text is empty and no meaningful multi-cell tables found:
    is_unreadable = (total_text_chars == 0 and len(extracted_tables) == 0)

    if is_unreadable:
        warnings.append("PDF returned no extractable text or tables (scanned or image-only PDF).")

    return {
        "tables": extracted_tables,
        "total_pages": total_pages,
        "total_tables": len(extracted_tables),
        "evidence": all_evidence,
        "is_unreadable": is_unreadable,
        "warnings": warnings,
    }
