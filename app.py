import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io

# PDF imports
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Churn Prediction & Retention Strategy",
    layout="wide",
    page_icon="📉",
)

# ---------------------------------------------------------
# Global custom CSS
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    /* Page background */
    .stApp {
        background: radial-gradient(circle at top left, #1f2933 0, #020617 45%, #020617 100%);
        color: #e5e7eb;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text",
                     "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Content container */
    .block-container {
        padding-top: 4.5rem !important;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
        max-width: 1180px;
    }

    /* Hero title */
    .app-title {
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        background: linear-gradient(90deg, #f97316, #facc15, #22c55e);
        -webkit-background-clip: text;
        color: transparent;
        margin-bottom: 0.25rem;
    }

    .app-subtitle {
        font-size: 0.98rem;
        color: #9ca3af;
        margin-bottom: 1.5rem;
    }

    /* Card base */
    .card {
        border-radius: 1.25rem;
        padding: 1.2rem 1.4rem;
        background: rgba(15, 23, 42, 0.92);
        border: 1px solid rgba(148, 163, 184, 0.18);
        box-shadow:
            0 18px 45px rgba(15, 23, 42, 0.75),
            0 0 0 1px rgba(15, 23, 42, 0.7);
        backdrop-filter: blur(18px);
    }

    /* Metric cards */
    .metric-card {
        border-radius: 1.1rem;
        padding: 1rem 1.2rem;
        background: rgba(15,23,42,0.9);
        border: 1px solid rgba(148,163,184,0.3);
    }

    .metric-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: #9ca3af;
        margin-bottom: 0.15rem;
    }

    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f9fafb;
        white-space: nowrap;
    }

    .metric-helper {
        font-size: 0.78rem;
        color: #9ca3af;
        margin-top: 0.1rem;
    }

    /* Risk badge */
    .risk-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .risk-low {
        background: rgba(22, 163, 74, 0.16);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.45);
    }
    .risk-medium {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.45);
    }
    .risk-high, .risk-very_high {
        background: rgba(220, 38, 38, 0.16);
        color: #fb7185;
        border: 1px solid rgba(248, 113, 113, 0.6);
    }

    /* Section titles */
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }

    .section-subtitle {
        font-size: 0.85rem;
        color: #9ca3af;
        margin-bottom: 0.8rem;
    }

    /* Recommended action text */
    .action-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: #9ca3af;
    }
    .action-pill {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        background: rgba(59, 130, 246, 0.15);
        border: 1px solid rgba(96, 165, 250, 0.5);
        color: #bfdbfe;
        font-size: 0.8rem;
        font-weight: 500;
        margin-top: 0.35rem;
        margin-bottom: 0.9rem;
    }

    /* Dataframe tweaks */
    .stDataFrame {
        border-radius: 0.85rem;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("churn_model.pkl")
    feature_cols = joblib.load("feature_columns.pkl")
    df = pd.read_csv("cleaned_telco.csv")
    return model, feature_cols, df


model, feature_cols, df = load_artifacts()


@st.cache_resource
def build_encoded_features(df, feature_cols):
    X_raw = df.drop(columns=["churn"], errors="ignore")
    X_enc = pd.get_dummies(X_raw, drop_first=True)
    X_enc = X_enc.reindex(columns=feature_cols, fill_value=0)
    return X_enc


X_enc = build_encoded_features(df, feature_cols)

# ---------------------------------------------------------
# Business logic: retention strategy
# ---------------------------------------------------------
def generate_retention_strategy(orig_row, churn_prob, df):
    strategy = {}

    # risk level buckets
    if churn_prob > 0.8:
        strategy["risk_level"] = "very_high"
    elif churn_prob > 0.6:
        strategy["risk_level"] = "high"
    elif churn_prob > 0.4:
        strategy["risk_level"] = "medium"
    else:
        strategy["risk_level"] = "low"

    # detect columns dynamically
    monthly_col = next(
        (c for c in df.columns if "monthly" in c.lower() and "charge" in c.lower()),
        None,
    )
    tenure_col = next((c for c in df.columns if "tenure" in c.lower()), None)
    satisfaction_col = next((c for c in df.columns if "satisfaction" in c.lower()), None)

    monthly_charge = float(orig_row[monthly_col]) if monthly_col is not None else None
    tenure = float(orig_row[tenure_col]) if tenure_col is not None else None
    satisfaction = float(orig_row[satisfaction_col]) if satisfaction_col is not None else None

    segments = []

    if monthly_charge is not None:
        high_price_threshold = np.percentile(df[monthly_col], 75)
        if monthly_charge >= high_price_threshold:
            segments.append("price_sensitive")

    if tenure is not None and tenure < 6:
        segments.append("new_customer")

    if satisfaction is not None and satisfaction <= 3:
        segments.append("unhappy_experience")

    if not segments:
        segments.append("general_risk")

    strategy["segments"] = segments

    if "price_sensitive" in segments:
        action = "offer_temporary_discount"
        message = (
            "We’ve reviewed your account and would like to offer you a limited-time discount on your plan "
            "for the next 3 months to help you get more value from the service."
        )
    elif "unhappy_experience" in segments:
        action = "priority_support_outreach"
        message = (
            "We noticed that your recent experience may not have fully met expectations. "
            "We’d like to assign a specialist to personally review your account and resolve any issues you’ve faced."
        )
    elif "new_customer" in segments:
        action = "onboarding_coaching"
        message = (
            "Since you recently joined us, we’d love to walk you through the key features that can help you the most. "
            "Here’s a quick-start guide and an option to book a 1:1 setup call."
        )
    else:
        action = "engagement_nudge"
        message = (
            "We’d like to share a personalized summary of features you’re not using yet and how they can help you "
            "get more value from your current plan."
        )

    strategy["action"] = action
    strategy["message"] = message

    return strategy

# ---------------------------------------------------------
# Premium PDF generator (with chart + "Created by Aron")
# ---------------------------------------------------------
def generate_premium_pdf(orig_row, churn_prob, strategy):
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=50,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()
    title = ParagraphStyle("title", parent=styles["Title"], fontSize=22, textColor="#111")
    section = ParagraphStyle("section", parent=styles["Heading2"], fontSize=14, textColor="#222")
    normal = ParagraphStyle("normal", parent=styles["BodyText"], fontSize=10, leading=14)

    elements = []

    # Title
    elements.append(Paragraph("CUSTOMER CHURN REPORT", title))
    elements.append(Spacer(1, 12))

    # Summary table
    summary = [
        ["Churn Probability", f"{churn_prob*100:.2f}%"],
        ["Risk Level", strategy["risk_level"].replace("_", " ").title()],
        ["Segments", ", ".join(strategy["segments"])],
    ]
    summary_table = Table(summary, colWidths=[180, 300])
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("BOX", (0, 0), (-1, -1), 1, colors.grey),
                ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.grey),
                ("FONT", (0, 0), (-1, -1), "Helvetica", 10),
            ]
        )
    )
    elements.append(summary_table)
    elements.append(Spacer(1, 20))

    # Chart
    elements.append(Paragraph("Churn Probability Chart", section))

    drawing = Drawing(300, 180)
    chart = VerticalBarChart()
    chart.x = 50
    chart.y = 30
    chart.height = 120
    chart.width = 200
    chart.data = [[churn_prob * 100]]
    chart.valueAxis.valueMin = 0
    chart.valueAxis.valueMax = 100
    chart.valueAxis.valueStep = 20
    chart.categoryAxis.categoryNames = ["Churn %"]

    if churn_prob > 0.6:
        chart.bars[0].fillColor = colors.red
    elif churn_prob > 0.4:
        chart.bars[0].fillColor = colors.orange
    else:
        chart.bars[0].fillColor = colors.green

    drawing.add(chart)
    elements.append(drawing)
    elements.append(Spacer(1, 20))

    # Recommended action
    elements.append(Paragraph("Recommended Action", section))
    elements.append(Paragraph(f"<b>{strategy['action'].replace('_',' ').title()}</b>", normal))
    elements.append(Paragraph(strategy["message"], normal))
    elements.append(Spacer(1, 20))

    # Customer details table
    elements.append(Paragraph("Customer Details", section))
    detail_rows = [[k.replace("_", " ").title(), str(v)] for k, v in orig_row.items()]
    details_table = Table(detail_rows, colWidths=[200, 280])
    details_table.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 1, colors.grey),
                ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.grey),
                ("FONT", (0, 0), (-1, -1), "Helvetica", 9),
            ]
        )
    )
    elements.append(details_table)
    elements.append(Spacer(1, 20))

    # Signature
    elements.append(Paragraph("<b>Created by Aron</b>", normal))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# ---------------------------------------------------------
# Render helpers
# ---------------------------------------------------------
def metric_card(label: str, value: str, helper: str = ""):
    # safer version (prevents ghost empty blocks)
    html = f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {f'<div class="metric-helper">{helper}</div>' if helper else ""}
        </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def risk_badge(level: str):
    cls = {
        "low": "risk-low",
        "medium": "risk-medium",
        "high": "risk-high",
        "very_high": "risk-very_high",
    }.get(level, "risk-low")
    label = level.replace("_", " ").title()
    st.markdown(
        f'<div class="risk-badge {cls}"><span>{label}</span></div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------
# Layout
# ---------------------------------------------------------

# Hero section
st.markdown(
    """
    <div>
        <div class="app-title">Customer Churn Prediction & Retention Strategy</div>
        <div class="app-subtitle">
            Score every customer for churn risk, understand why they might leave, and get
            a concrete retention play for your success or marketing team.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar selector
with st.sidebar:
    st.markdown("### 🔍 Select Customer")

    id_col = None
    for c in df.columns:
        if "customer_id" in c.lower() or c.lower() == "customerid":
            id_col = c
            break

    if id_col:
        ids = df[id_col].astype(str).tolist()
        selected_id = st.selectbox("Customer ID", options=ids)
        selected_idx = df[df[id_col].astype(str) == selected_id].index[0]
    else:
        selected_idx = st.selectbox("Row index", options=list(df.index))

    st.markdown("---")
    st.caption("Tip: use this sidebar to quickly jump between different customers.")

# Data for selected customer
row_for_model = X_enc.loc[[selected_idx]]
orig_row = df.loc[selected_idx]
churn_prob = float(model.predict_proba(row_for_model)[0, 1])
strategy = generate_retention_strategy(orig_row, churn_prob, df)

# 1) TOP ROW – metrics with spacing
m1, gap1, m2, gap2, m3 = st.columns([1, 0.2, 1, 0.2, 1])

with m1:
    metric_card("Churn probability", f"{churn_prob*100:,.2f}%")

with m2:
    metric_card("Risk level", strategy["risk_level"].replace("_", " ").title())

with m3:
    metric_card(
        "Segments",
        ", ".join(strategy["segments"]),
        helper="Behavioral drivers for this customer",
    )

st.markdown("<br/>", unsafe_allow_html=True)

# 2) SECOND ROW – left: Recommended Action, right: Risk Profile
left_col, right_col = st.columns([2, 1])

with left_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Recommended Action</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-subtitle'>Designed for your customer success / marketing team to execute.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='action-label'>Playbook</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='action-pill'>{strategy['action'].replace('_', ' ').title()}</div>",
        unsafe_allow_html=True,
    )
    st.write(strategy["message"])
    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Risk Profile</div>", unsafe_allow_html=True)
    risk_badge(strategy["risk_level"])
    st.markdown("<br/>", unsafe_allow_html=True)

    profile_fields = ["age", "tenure_months", "satisfaction_score", "country", "state"]
    for col in profile_fields:
        if col in df.columns:
            st.write(f"**{col.replace('_', ' ').title()}**: {orig_row[col]}")

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

# PDF download section
st.markdown("### 📄 Download PDF Report")
pdf_bytes = generate_premium_pdf(orig_row, churn_prob, strategy)
st.download_button(
    label="Download Report (PDF)",
    data=pdf_bytes,
    file_name=f"customer_{selected_idx}_churn_report.pdf",
    mime="application/pdf",
)

st.markdown("<br/>", unsafe_allow_html=True)

# 3) HIGH-RISK CUSTOMERS TABLE
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>🔥 High-Risk Customers Overview</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-subtitle'>Top customers ranked by churn probability — use this as a daily watchlist.</div>",
    unsafe_allow_html=True,
)

all_probs = model.predict_proba(X_enc)[:, 1]
risk_df = df.copy()
risk_df["churn_prob"] = all_probs
risk_df = risk_df.sort_values("churn_prob", ascending=False)

top_n = st.slider("How many customers to show", 5, 50, 20, key="top_n_slider")
display_cols = [c for c in ["customer_id", "age", "tenure_months", "satisfaction_score"] if c in risk_df.columns]
display_cols.append("churn_prob")

st.dataframe(
    risk_df[display_cols].head(top_n),
    use_container_width=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# 4) RAW RECORD
with st.expander("🔎 View raw customer record"):
    raw_df = (
        orig_row.astype(str)
        .to_frame(name="value")
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    st.dataframe(raw_df, use_container_width=True)
