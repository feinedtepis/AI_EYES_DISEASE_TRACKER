"""Tiny reportlab helper used by evaluate.py to write the evaluation report PDF."""
from __future__ import annotations

import math

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

_ss = getSampleStyleSheet()
H1 = ParagraphStyle("h1", parent=_ss["Heading1"], fontSize=17, spaceAfter=8, textColor=colors.HexColor("#0b3d5c"))
H2 = ParagraphStyle("h2", parent=_ss["Heading2"], fontSize=12.5, spaceBefore=10, spaceAfter=5,
                    textColor=colors.HexColor("#0b3d5c"))
BODY = ParagraphStyle("b", parent=_ss["BodyText"], fontSize=9.5, leading=13, alignment=TA_LEFT)
SMALL = ParagraphStyle("s", parent=BODY, fontSize=8, leading=10.5, textColor=colors.HexColor("#444444"))
WARN = ParagraphStyle("w", parent=BODY, backColor=colors.HexColor("#fff4e5"), borderColor=colors.HexColor("#e8a33d"),
                      borderWidth=0.8, borderPadding=6, spaceBefore=4, spaceAfter=8)


def fmt(v, pct=False, nd=3):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return f"[{fmt(v[0], pct, nd)}, {fmt(v[1], pct, nd)}]"
    if isinstance(v, (int,)) and not isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{100 * v:.1f}%" if pct else f"{v:.{nd}f}"
    return str(v)


def table(rows, col_widths=None, header=True, font=7.8):
    t = Table(rows, colWidths=col_widths, repeatRows=1 if header else 0)
    style = [("FONTSIZE", (0, 0), (-1, -1), font), ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b8c4cc")),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 2.5),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b3d5c")),
                  ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
        for r in range(1, len(rows)):
            if r % 2 == 0:
                style.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#f2f6f8")))
    t.setStyle(TableStyle(style))
    return t


def image(path, width_cm=17):
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    return Image(str(path), width=width_cm * cm, height=width_cm * cm * h / w)


def build_pdf(path, story, title="EyeVision AI"):
    def _footer(c, d):
        c.saveState()
        c.setFont("Helvetica", 7)
        c.setFillColor(colors.HexColor("#777777"))
        c.drawString(1.8 * cm, 1.0 * cm, f"{title} — research/educational use only. Not a medical device.")
        c.drawRightString(A4[0] - 1.8 * cm, 1.0 * cm, f"Page {d.page}")
        c.restoreState()
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                            topMargin=1.6 * cm, bottomMargin=1.6 * cm, title=title)
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)


__all__ = ["H1", "H2", "BODY", "SMALL", "WARN", "fmt", "table", "image", "build_pdf", "Paragraph", "Spacer",
           "PageBreak"]
