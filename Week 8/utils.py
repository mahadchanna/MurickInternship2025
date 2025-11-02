import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import io

def categorize_grade(grade):
    """Categorize grade into performance levels"""
    if grade >= 16:
        return 'Excellent'
    elif grade >= 14:
        return 'Good'
    elif grade >= 10:
        return 'Average'
    else:
        return 'Poor'

def get_risk_level(grade, failures, absences):
    """Determine student risk level"""
    if grade < 10 or failures > 2 or absences > 20:
        return 'High', '🔴'
    elif grade < 14 or failures > 0 or absences > 10:
        return 'Medium', '🟡'
    else:
        return 'Low', '🟢'

def create_radar_chart(student_data, features):
    """Create radar chart for student comparison"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=student_data,
        theta=features,
        fill='toself',
        name='Student Performance'
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 20])),
        showlegend=True
    )
    
    return fig

def generate_pdf_report(student_data, prediction, recommendations):
    """Generate PDF report for student"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()
    
    # Title
    title = Paragraph("<b>Student Performance Report</b>", styles['Title'])
    elements.append(title)
    elements.append(Spacer(1, 12))
    
    # Student Info
    info_text = f"""
    <b>Generated:</b> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}<br/>
    <b>Predicted Grade:</b> {prediction:.2f}/20<br/>
    <b>Status:</b> {'Pass' if prediction >= 10 else 'Fail'}
    """
    elements.append(Paragraph(info_text, styles['Normal']))
    elements.append(Spacer(1, 12))
    
    # Recommendations
    elements.append(Paragraph("<b>Recommendations:</b>", styles['Heading2']))
    for rec in recommendations:
        elements.append(Paragraph(f"• {rec}", styles['Normal']))
    
    doc.build(elements)
    buffer.seek(0)
    return buffer

def calculate_percentile(value, series):
    """Calculate percentile rank"""
    return (series < value).sum() / len(series) * 100
