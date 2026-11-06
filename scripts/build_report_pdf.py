#!/usr/bin/env python3
"""Build a polished PDF from docs/REPORT.md using ReportLab."""

from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import (
    BaseDocTemplate,
    CondPageBreak,
    HRFlowable,
    KeepTogether,
    LongTable,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "REPORT.md"
OUTPUT = ROOT / "output" / "pdf" / "Assignment_8_Optimization_Technical_Report.pdf"

NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#23689B")
TEAL = colors.HexColor("#2A7F8E")
PALE = colors.HexColor("#EAF2F7")
LIGHT = colors.HexColor("#F5F7F9")
INK = colors.HexColor("#202A33")
MUTED = colors.HexColor("#596773")
RULE = colors.HexColor("#C9D5DE")


def normalize(text: str) -> str:
    replacements = {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u00d7": "x",
        "\u2192": "->",
        "\u2265": ">=",
        "\u2264": "<=",
        "\u2248": "~",
        "\u00b1": "+/-",
        "\u00b2": "^2",
        "\u03bb": "lambda",
        "\u03b8": "theta",
        "\u03b1": "alpha",
        "\u03bc": "mu",
        "\u03a3": "sum",
        "\u2211": "sum",
        "\u2208": "in",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def inline_markup(text: str) -> str:
    text = normalize(text.strip())
    text = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", text)
    text = html.escape(text, quote=False)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r'<font name="Courier" color="#17324D">\1</font>', text)
    return text


class ReportDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str, **kwargs):
        super().__init__(filename, **kwargs)
        frame = self._make_frame()
        self.addPageTemplates(PageTemplate(id="main", frames=[frame], onPage=self._decorate_page))
        self._heading_index = 0

    def beforeDocument(self):
        self._heading_index = 0

    def _make_frame(self):
        from reportlab.platypus import Frame

        return Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            id="body",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        )

    def _decorate_page(self, canvas, doc):
        page = canvas.getPageNumber()
        canvas.saveState()
        width, height = A4
        if page == 1:
            canvas.setFillColor(NAVY)
            canvas.rect(0, 0, width, height, fill=1, stroke=0)
            canvas.setFillColor(TEAL)
            canvas.rect(0, 0, 12 * mm, height, fill=1, stroke=0)
        else:
            canvas.setStrokeColor(RULE)
            canvas.setLineWidth(0.5)
            canvas.line(self.leftMargin, height - 14 * mm, width - self.rightMargin, height - 14 * mm)
            canvas.setFont("Helvetica", 7.5)
            canvas.setFillColor(MUTED)
            canvas.drawString(self.leftMargin, height - 10.5 * mm, "ASSIGNMENT 8 - OPTIMIZE ANYTHING")
            canvas.drawRightString(width - self.rightMargin, height - 10.5 * mm, "Optimization Techniques")
            canvas.line(self.leftMargin, 13 * mm, width - self.rightMargin, 13 * mm)
            canvas.drawString(self.leftMargin, 8.5 * mm, "Technical Report")
            canvas.drawRightString(width - self.rightMargin, 8.5 * mm, f"Page {page}")
        canvas.restoreState()

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph):
            style = flowable.style.name
            if style in {"H1", "H2", "H3"}:
                level = 0 if style in {"H1", "H2"} else 1
                text = flowable.getPlainText()
                key = f"heading-{self._heading_index}"
                self._heading_index += 1
                self.canv.bookmarkPage(key)
                self.canv.addOutlineEntry(text, key, level=level, closed=False)
                self.notify("TOCEntry", (level, text, self.page, key))


def build_styles():
    base = getSampleStyleSheet()
    return {
        "Body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.1,
            leading=13.1,
            textColor=INK,
            spaceAfter=5.5,
            alignment=TA_LEFT,
            allowWidows=0,
            allowOrphans=0,
        ),
        "H1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=17,
            leading=20,
            textColor=NAVY,
            spaceBefore=12,
            spaceAfter=8,
            keepWithNext=True,
        ),
        "H2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12.2,
            leading=15,
            textColor=BLUE,
            spaceBefore=9,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "H3": ParagraphStyle(
            "H3",
            parent=base["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=10.2,
            leading=12.5,
            textColor=TEAL,
            spaceBefore=7,
            spaceAfter=3.5,
            keepWithNext=True,
        ),
        "Bullet": ParagraphStyle(
            "Bullet",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.9,
            leading=12.5,
            textColor=INK,
            leftIndent=12,
            firstLineIndent=-7,
            bulletIndent=3,
            spaceAfter=3,
        ),
        "Code": ParagraphStyle(
            "Code",
            parent=base["Code"],
            fontName="Courier",
            fontSize=7.5,
            leading=10.2,
            leftIndent=8,
            rightIndent=8,
            borderColor=RULE,
            borderWidth=0.5,
            borderPadding=7,
            backColor=LIGHT,
            textColor=colors.HexColor("#243746"),
            spaceBefore=3,
            spaceAfter=7,
        ),
        "Meta": ParagraphStyle(
            "Meta",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            textColor=colors.white,
            alignment=TA_CENTER,
            spaceAfter=2,
        ),
        "Title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=30,
            leading=35,
            textColor=colors.white,
            alignment=TA_CENTER,
            spaceAfter=12,
        ),
        "Subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=15,
            leading=19,
            textColor=colors.HexColor("#D8EEF2"),
            alignment=TA_CENTER,
            spaceAfter=18,
        ),
        "TOCTitle": ParagraphStyle(
            "TOCTitle",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            textColor=NAVY,
            spaceAfter=14,
        ),
    }


def table_widths(rows: list[list[str]], available: float) -> list[float]:
    columns = max(len(row) for row in rows)
    weights = []
    for col in range(columns):
        lengths = [min(42, max(5, len(re.sub(r"<[^>]+>", "", row[col])))) for row in rows if col < len(row)]
        weights.append(max(lengths) if lengths else 8)
    floor = available * (0.12 if columns <= 4 else 0.09)
    raw = [max(floor, available * weight / sum(weights)) for weight in weights]
    scale = available / sum(raw)
    return [value * scale for value in raw]


def make_table(rows: list[list[str]], styles, available_width: float):
    formatted = []
    for r_index, row in enumerate(rows):
        style = ParagraphStyle(
            f"Cell{r_index}",
            parent=styles["Body"],
            fontName="Helvetica-Bold" if r_index == 0 else "Helvetica",
            fontSize=7.0 if len(row) <= 5 else 6.2,
            leading=8.6 if len(row) <= 5 else 7.6,
            textColor=colors.white if r_index == 0 else INK,
            spaceAfter=0,
        )
        formatted.append([Paragraph(cell, style) for cell in row])
    table = LongTable(
        formatted,
        colWidths=table_widths(rows, available_width),
        repeatRows=1,
        splitByRow=1,
        hAlign="LEFT",
    )
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.35, RULE),
    ]
    for index in range(1, len(rows)):
        commands.append(("BACKGROUND", (0, index), (-1, index), colors.white if index % 2 else PALE))
    table.setStyle(TableStyle(commands))
    return table


def parse_markdown(text: str, styles, available_width: float):
    lines = text.splitlines()
    story = []
    paragraph: list[str] = []
    code: list[str] = []
    table_rows: list[list[str]] = []
    in_code = False

    def flush_paragraph():
        if paragraph:
            joined = " ".join(part.strip() for part in paragraph)
            story.append(Paragraph(inline_markup(joined), styles["Body"]))
            paragraph.clear()

    def flush_table():
        if table_rows:
            rows = table_rows[:]
            if len(rows) > 1 and all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in rows[1]):
                rows.pop(1)
            story.append(Spacer(1, 3))
            story.append(make_table(rows, styles, available_width))
            story.append(Spacer(1, 7))
            table_rows.clear()

    for raw in lines:
        line = raw.rstrip()
        if line.startswith("```"):
            flush_paragraph()
            flush_table()
            if in_code:
                escaped = html.escape(normalize("\n".join(code)), quote=False).replace("\n", "<br/>")
                story.append(Paragraph(escaped, styles["Code"]))
                code.clear()
            in_code = not in_code
            continue
        if in_code:
            code.append(line)
            continue
        if line.startswith("|") and line.endswith("|"):
            flush_paragraph()
            cells = [inline_markup(cell.strip()) for cell in line.strip("|").split("|")]
            table_rows.append(cells)
            continue
        flush_table()
        if not line.strip():
            flush_paragraph()
            continue
        heading = re.match(r"^(#{1,3})\s+(.*)$", line)
        if heading:
            flush_paragraph()
            level = len(heading.group(1))
            title = inline_markup(heading.group(2))
            if level == 1:
                story.append(CondPageBreak(35 * mm))
                story.append(Paragraph(title, styles["H1"]))
                story.append(HRFlowable(width="100%", thickness=1.2, color=TEAL, spaceAfter=7))
            elif level == 2:
                story.append(CondPageBreak(45 * mm))
                story.append(Paragraph(title, styles["H2"]))
            else:
                story.append(CondPageBreak(20 * mm))
                story.append(Paragraph(title, styles["H3"]))
            continue
        bullet = re.match(r"^[-*]\s+(.*)$", line)
        number = re.match(r"^(\d+)\.\s+(.*)$", line)
        if bullet:
            flush_paragraph()
            story.append(Paragraph(inline_markup(bullet.group(1)), styles["Bullet"], bulletText="-"))
        elif number:
            flush_paragraph()
            story.append(Paragraph(inline_markup(number.group(2)), styles["Bullet"], bulletText=f"{number.group(1)}."))
        elif line.startswith("**") and line.endswith("  "):
            flush_paragraph()
            story.append(Paragraph(inline_markup(line), styles["Body"]))
        else:
            paragraph.append(line)
    flush_paragraph()
    flush_table()
    return story


def build_pdf() -> Path:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    styles = build_styles()
    doc = ReportDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=19 * mm,
        bottomMargin=18 * mm,
        title="Assignment 8 - Optimize Anything",
        author="Optimization Techniques",
        subject="Technical report covering optimization experiment Tracks A-J",
    )

    source_text = SOURCE.read_text(encoding="utf-8")
    source_text = re.sub(r"\*\*Student name:\*\*.*\n", "", source_text)
    source_text = re.sub(r"\*\*(?:Student ID|Roll number):\*\*.*\n", "", source_text)
    source_text = re.sub(r"\*\*Course:\*\*.*\n", "", source_text)
    source_text = re.sub(r"\*\*Report date:\*\*.*\n", "", source_text)
    source_text = re.sub(r"\*\*Implementation period:\*\*.*\n", "", source_text)
    source_text = re.sub(r"\*\*Repository status:\*\*.*\n", "", source_text)
    source_text = re.sub(r"^# Assignment 8 Technical Report\n", "", source_text)
    source_text = re.sub(r"^## Optimize Anything.*?\n", "", source_text, count=1, flags=re.M)

    story = [
        Spacer(1, 45 * mm),
        Paragraph("ASSIGNMENT 8", styles["Meta"]),
        Spacer(1, 6 * mm),
        Paragraph("Optimize Anything", styles["Title"]),
        Paragraph("Technical Report", styles["Subtitle"]),
        HRFlowable(width="52%", thickness=1.5, color=colors.HexColor("#7BD3DE"), spaceBefore=4, spaceAfter=14),
        Paragraph("Discover a process. Formulate the problem.<br/>Apply optimization. Prove the improvement.", styles["Meta"]),
        Spacer(1, 16 * mm),
        Paragraph("<b>Ankita Sagar</b>", styles["Meta"]),
        Paragraph("Roll No: 241071063", styles["Meta"]),
        Spacer(1, 16 * mm),
        Paragraph("Optimization Techniques", styles["Meta"]),
        Paragraph("Tracks A-J | AI-assisted development | Reproducible experiments", styles["Meta"]),
        Spacer(1, 14 * mm),
        Paragraph("6 November 2026", styles["Meta"]),
        PageBreak(),
        Paragraph("Contents", styles["TOCTitle"]),
    ]
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle("TOC1", fontName="Helvetica-Bold", fontSize=9.3, leading=13, leftIndent=0, firstLineIndent=0, textColor=NAVY, spaceBefore=3),
        ParagraphStyle("TOC2", fontName="Helvetica", fontSize=8.4, leading=11, leftIndent=14, firstLineIndent=0, textColor=MUTED),
    ]
    story.extend([toc, PageBreak()])
    story.extend(parse_markdown(source_text, styles, doc.width))
    doc.multiBuild(story)
    return OUTPUT


if __name__ == "__main__":
    path = build_pdf()
    print(path)
