from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from PIL import Image as PILImage
import io
import datetime
import numpy as np

def get_treatment_advice(label):
    """Generates medical suggestions based on AI diagnosis."""
    if label == "PNEUMONIA":
        return """<b>TREATMENT SUGGESTIONS & ADVICE:</b><br/>
        1. Consult a Pulmonologist immediately for clinical correlation.<br/>
        2. Start prescribed antibiotic/antiviral therapy as directed.<br/>
        3. Maintain high fluid intake and complete bed rest.<br/>
        4. Monitor oxygen levels (SpO2) using a pulse oximeter.<br/>
        5. Follow-up chest radiograph is recommended after 10-14 days."""
    else:
        return """<b>TREATMENT SUGGESTIONS & ADVICE:</b><br/>
        1. No immediate clinical evidence of acute pneumonia detected.<br/>
        2. If symptoms like cough or fever persist, consult a physician.<br/>
        3. Practice steam inhalation and stay hydrated.<br/>
        4. Monitor for any new respiratory distress symptoms."""

def generate_medical_pdf(p_name, p_age, p_id, label, conf, status, heatmap_img, history_df):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)
    styles = getSampleStyleSheet()
    elements = []

    # 1. HOSPITAL HEADER
    elements.append(Paragraph("<b>ST. JUDE'S GENERAL HOSPITAL - RADIOLOGY DEPT</b>", styles['Title']))
    elements.append(Paragraph("<center>123 Medical Center Drive, Metropolis, NY 10001</center>", styles['Normal']))
    elements.append(Spacer(1, 15))

    # 2. PATIENT DATA TABLE (Includes date from session)
    current_date = datetime.datetime.now().strftime("%B %d, %Y")
    data = [
        [f"Patient Name: {p_name}", f"Date: {current_date}"],
        [f"Patient Age: {p_age}", f"Accession No: RAD-{p_id}"],
        [f"Patient ID: {p_id}", f"Study: Chest X-ray, 1 View (PA)"]
    ]
    t = Table(data, colWidths=[250, 250])
    t.setStyle(TableStyle([('BOX', (0,0), (-1,-1), 1, colors.black), ('GRID', (0,0), (-1,-1), 0.5, colors.grey)]))
    elements.append(t)
    elements.append(Spacer(1, 20))

    # 3. AI FINDINGS
    elements.append(Paragraph(f"<b>AI ANALYSIS RESULT: {label}</b>", styles['Heading3']))
    elements.append(Paragraph(f"Analysis shows markers consistent with {label} with {conf*100:.1f}% confidence.", styles['Normal']))
    elements.append(Spacer(1, 15))

    # 4. HEATMAP IMAGE
    img_byte_arr = io.BytesIO()
    if isinstance(heatmap_img, np.ndarray):
        if heatmap_img.max() <= 1.0: heatmap_img = (heatmap_img * 255).astype(np.uint8)
        heatmap_pil = PILImage.fromarray(heatmap_img)
    else:
        heatmap_pil = heatmap_img
    heatmap_pil.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    elements.append(RLImage(img_byte_arr, width=300, height=300))
    elements.append(Spacer(1, 20))

    # 5. DYNAMIC TREATMENT SUGGESTIONS
    advice_text = get_treatment_advice(label)
    elements.append(Paragraph(advice_text, styles['Normal']))
    elements.append(Spacer(1, 30))

    # 6. SIGNATURE
    elements.append(Paragraph("<b>REPORTING RADIOLOGIST:</b>", styles['Normal']))
    elements.append(Paragraph("(Signed) MedVision AI Diagnostic Core", styles['Normal']))

    doc.build(elements)
    buffer.seek(0)
    return buffer