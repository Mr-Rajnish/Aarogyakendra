import io
import html
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


def _bullet_block(items: List[str]) -> str:
    if not items:
        return ""
    # ReportLab Paragraph supports basic HTML-like tags.
    return "<br/>".join([f"&#8226; {str(i)}" for i in items])


def _escape(s: Any) -> str:
    return html.escape("" if s is None else str(s))


def _draw_medical_icon(c: canvas.Canvas, x: float, y: float, size: float = 28.0, stroke_color=colors.HexColor("#0FA3B1")) -> None:
    """
    Draw a simple medical cross icon (used for watermark/branding).
    """
    c.saveState()
    c.setStrokeColor(stroke_color)
    c.setLineWidth(2.5)
    half = size / 2
    # vertical line
    c.line(x, y - half, x, y + half)
    # horizontal line
    c.line(x - half, y, x + half, y)
    c.restoreState()


def _on_first_page(c: canvas.Canvas, doc) -> None:
    width, height = A4

    brand_primary = colors.HexColor("#0FA3B1")  # teal
    brand_accent = colors.HexColor("#39FF14")  # neon green

    # Header background bar
    c.saveState()
    c.setFillColor(brand_primary)
    c.rect(0, height - 72, width, 72, stroke=0, fill=1)

    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(36, height - 46, "AarogyaKendra")

    c.setFont("Helvetica", 10)
    c.drawString(36, height - 62, "AI Healthcare Assistant Report (Not a Diagnosis)")

    # Accent line
    c.setStrokeColor(brand_accent)
    c.setLineWidth(3)
    c.line(36, height - 70, 220, height - 70)

    # Medical icon on left
    c.setStrokeColor(colors.white)
    _draw_medical_icon(c, 260, height - 46, size=26, stroke_color=colors.white)

    # Watermark image (medical symbol background) - if available.
    # The image is stored in the Cursor assets folder that you already generated/provided.
    watermark_path = r"C:\Users\rajni\.cursor\projects\c-Users-rajni-Desktop-AarogyaKendra\assets\c__Users_rajni_AppData_Roaming_Cursor_User_workspaceStorage_718ed68f3d3577cdadb8bea5cd8cf437_images_image-1d1d14fd-b6d9-440e-b772-e27a35bd26d9.png"
    if os.path.exists(watermark_path):
        try:
            # Try to apply translucency via Pillow (best-effort).
            try:
                from PIL import Image as PILImage  # type: ignore

                pil_img = PILImage.open(watermark_path).convert("RGBA")
                alpha = pil_img.split()[-1]
                # Reduce opacity to a faint watermark.
                alpha = alpha.point(lambda p: int(p * 0.08))
                pil_img.putalpha(alpha)
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                buf.seek(0)
                img_reader = ImageReader(buf)
            except Exception:
                img_reader = ImageReader(watermark_path)

            img_w = width * 0.42
            img_h = img_w
            # Draw centered-ish in the background.
            c.saveState()
            c.drawImage(img_reader, width * 0.53 - img_w / 2, height * 0.52 - img_h / 2, width=img_w, height=img_h, mask="auto")
            c.restoreState()
        except Exception:
            # Fallback to drawn cross icon watermark.
            c.setStrokeColor(colors.lightgrey)
            c.setLineWidth(3)
            _draw_medical_icon(c, width * 0.82, height * 0.55, size=90, stroke_color=colors.lightgrey)
    else:
        c.setStrokeColor(colors.lightgrey)
        c.setLineWidth(3)
        _draw_medical_icon(c, width * 0.82, height * 0.55, size=90, stroke_color=colors.lightgrey)
    c.restoreState()


def generate_health_report_pdf(
    patient_input: str,
    uploaded_report_summary: str,
    assessment: Dict[str, Any],
    hospitals: Optional[List[Dict[str, Any]]] = None,
    timestamp: Optional[datetime] = None,
) -> bytes:
    """
    Builds a structured PDF report and returns bytes for Streamlit download.
    """
    timestamp = timestamp or datetime.now()
    hospitals = hospitals or []

    diseases = assessment.get("diseases") or []
    precautions = assessment.get("precautions") or []
    lifestyle_changes = assessment.get("lifestyle_changes") or []
    medications = assessment.get("medications") or []
    medication_disclaimer = assessment.get("medication_disclaimer") or "Consult a doctor before taking any medication"

    meds_lines = []
    for m in medications:
        if isinstance(m, dict):
            name = m.get("name") or ""
            reason = m.get("reason") or ""
            if reason:
                meds_lines.append(f"{name} - {reason}".strip(" -"))
            else:
                meds_lines.append(str(name))
        else:
            meds_lines.append(str(m))

    styles = getSampleStyleSheet()
    normal = styles["BodyText"]
    heading = styles["Heading2"]
    title = styles["Title"]

    # Brand styles
    brand_primary = colors.HexColor("#0FA3B1")
    brand_accent = colors.HexColor("#39FF14")

    title_custom = ParagraphStyle(
        "TitleCustom",
        parent=title,
        textColor=brand_primary,
        fontSize=20,
        spaceAfter=10,
    )

    heading_custom = ParagraphStyle(
        "HeadingCustom",
        parent=heading,
        textColor=brand_primary,
        fontSize=14,
        leading=18,
        spaceBefore=8,
        spaceAfter=6,
    )

    label_style = ParagraphStyle(
        "LabelStyle",
        parent=normal,
        textColor=brand_primary,
        fontSize=10.5,
    )

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=90,  # room for header bar
        bottomMargin=36,
    )

    elements = []
    elements.append(Paragraph("Health Report Summary", title_custom))
    elements.append(
        Paragraph(f"Generated on: <b>{timestamp.strftime('%Y-%m-%d %H:%M:%S')}</b>", normal)
    )
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Disclaimer", heading_custom))
    elements.append(
        Paragraph(
            "This is an AI-based system and not a substitute for professional medical advice.",
            normal,
        )
    )
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Patient Input / Uploaded Report Summary", heading_custom))
    if patient_input:
        elements.append(Paragraph(f"<b>Symptoms:</b> {_escape(patient_input)}", normal))
        elements.append(Spacer(1, 10))
    if uploaded_report_summary:
        # Keep it compact. ReportLab will wrap.
        elements.append(Paragraph(uploaded_report_summary.replace("\n", "<br/>"), normal))
        elements.append(Spacer(1, 10))
    else:
        elements.append(Paragraph("No uploaded report summary provided.", normal))
    elements.append(Spacer(1, 18))

    elements.append(Paragraph("AI Impression (Not a Diagnosis)", heading_custom))
    if diseases:
        elements.append(Paragraph("<br/>".join([f"&#8226; {_escape(d)}" for d in diseases]), normal))
    else:
        elements.append(Paragraph("No diseases extracted from the AI output.", normal))
    elements.append(Spacer(1, 12))

    summary = (assessment.get("summary") or "").strip()
    if summary:
        elements.append(Paragraph(f"<i>{_escape(summary)}</i>", normal))
        elements.append(Spacer(1, 12))

    elements.append(Paragraph("Suggested Precautions", heading_custom))
    elements.append(Paragraph(_bullet_block(precautions) or "No precautions provided.", normal))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Lifestyle Changes", heading_custom))
    elements.append(Paragraph(_bullet_block(lifestyle_changes) or "No lifestyle changes provided.", normal))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Medication Notes (General Ideas)", heading_custom))
    elements.append(Paragraph(_bullet_block(meds_lines) or "No medication suggestions provided.", normal))
    elements.append(Spacer(1, 10))

    # Highlight medication disclaimer in a colored box
    disclaimer_para = Paragraph(f"<b>{_escape(medication_disclaimer)}</b>", normal)
    box_data = [[disclaimer_para]]
    box = Table(
        box_data,
        colWidths=[doc.width],
        hAlign="LEFT",
    )
    box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#E8F7F9")),
                ("BOX", (0, 0), (-1, -1), 1, brand_primary),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    elements.append(box)
    elements.append(Spacer(1, 18))

    elements.append(Paragraph("Recommended Nearby Hospitals", heading_custom))
    if hospitals:
        data: List[List[Any]] = []
        header = [
            Paragraph("<b>Hospital</b>", label_style),
            Paragraph("<b>Address</b>", label_style),
            Paragraph("<b>Distance</b>", label_style),
        ]
        data.append(header)

        for h in hospitals:
            name = _escape(h.get("name") or "Hospital")
            address = _escape(h.get("address") or "")
            dist = h.get("distance_km")
            dist_str = f"{dist:.1f} km" if isinstance(dist, (int, float)) else "N/A"
            data.append([Paragraph(f"<b>{name}</b>", normal), Paragraph(address.replace('\n', '<br/>'), normal), Paragraph(dist_str, normal)])

        table = Table(data, colWidths=[160, doc.width - 160 - 90, 90])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0FA3B1")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#B7DDE2")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3FBFC")]),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        elements.append(table)
    else:
        elements.append(Paragraph("Hospital recommendations were not provided.", normal))

    # IMPORTANT: page callbacks must be passed in build(), not constructor.
    doc.build(elements, onFirstPage=_on_first_page, onLaterPages=_on_first_page)
    return buffer.getvalue()

