"""Deterministic pipe-delimited text table parser."""

import re
from typing import Any, Dict, List, Optional, Tuple

from person_b.extraction.normalization import normalize_whitespace
from person_b.models import (
    Evidence,
    EvidenceLocation,
    ExtractionLocationType,
)


def _is_table_separator_row(line: str) -> bool:
    """Check if line is a Markdown separator row like '--- | ---'."""
    cleaned = line.strip().replace(" ", "").replace("|", "")
    return len(cleaned) > 0 and all(c in "-:=" for c in cleaned)


def _split_table_row(line: str) -> List[str]:
    """Split a pipe-delimited table line into clean cell strings."""
    # Strip leading/trailing pipe if present
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]

    cells = [normalize_whitespace(c) for c in stripped.split("|")]
    return cells


def parse_text_tables(
    text: str,
    source_url: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Parse pipe-delimited text tables (| col1 | col2 |) from cleaned text.

    Returns a list of structured table dictionaries with:
      - table_name: detected title/heading
      - headers: list of column header strings
      - rows: list of cell string lists
      - records: list of dicts mapping column header -> cell value
      - evidence: list of Evidence objects for each cell
      - raw_text: verbatim text snippet of the table
      - line_range: (start_line, end_line)
    """
    if not text:
        return []

    lines = text.splitlines()
    tables: List[Dict[str, Any]] = []

    i = 0
    while i < len(lines):
        line = lines[i]
        if "|" in line:
            # Start of a potential table block
            start_line_idx = i
            table_lines: List[Tuple[int, str]] = []

            # Look backwards for a potential table title / heading
            table_title = None
            lookback = i - 1
            while lookback >= 0 and lookback >= i - 3:
                prev_line = lines[lookback].strip()
                if prev_line and "|" not in prev_line:
                    # Found a preceding heading line
                    table_title = prev_line
                    break
                lookback -= 1

            # Collect consecutive pipe-containing lines
            while i < len(lines) and ("|" in lines[i] or (lines[i].strip() == "" and i + 1 < len(lines) and "|" in lines[i + 1])):
                if "|" in lines[i]:
                    table_lines.append((i, lines[i]))
                i += 1

            end_line_idx = i - 1

            # Process collected lines
            parsed_rows: List[List[str]] = []
            for line_no, raw_line in table_lines:
                if _is_table_separator_row(raw_line):
                    continue
                cells = _split_table_row(raw_line)
                if any(c for c in cells):  # Non-empty row
                    parsed_rows.append(cells)

            if len(parsed_rows) >= 1:
                # Determine headers and data rows
                headers = parsed_rows[0]
                data_rows = parsed_rows[1:] if len(parsed_rows) > 1 else parsed_rows

                # If single row or headers look like data, adjust
                records: List[Dict[str, Any]] = []
                evidence_list: List[Evidence] = []

                # Build records
                for r_idx, row in enumerate(data_rows):
                    rec: Dict[str, Any] = {}
                    for c_idx, cell in enumerate(row):
                        header_name = headers[c_idx] if c_idx < len(headers) and headers[c_idx] else f"col_{c_idx}"
                        rec[header_name] = cell

                        # Create Evidence record for cell
                        loc = EvidenceLocation(
                            type=ExtractionLocationType.TEXT_TABLE,
                            table_name=table_title or (headers[0] if headers else "Table"),
                            row_index=r_idx,
                            col_name=header_name,
                        )
                        ev = Evidence(
                            field=header_name,
                            value=cell,
                            source_url=source_url,
                            evidence_text=f"{header_name} | {cell}" if header_name != cell else cell,
                            location=loc,
                            confidence=0.95,
                        )
                        evidence_list.append(ev)

                    records.append(rec)

                raw_chunk = "\n".join(l for _, l in table_lines)
                table_dict = {
                    "table_name": table_title or (headers[0] if headers else "Table"),
                    "headers": headers,
                    "rows": data_rows,
                    "records": records,
                    "evidence": evidence_list,
                    "raw_text": raw_chunk,
                    "line_range": (start_line_idx + 1, end_line_idx + 1),
                }
                tables.append(table_dict)
        else:
            i += 1

    return tables
