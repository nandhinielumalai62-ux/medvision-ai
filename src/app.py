import streamlit as st
import sys
import os
import uuid
import datetime
import pandas as pd
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px 
from io import BytesIO

# --- 🛠️ STEP 1: SYSTEM PATH REPAIR ---
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from utils import prediction, gradcam, report, db

# --- 🎨 STEP 2: FUTURISTIC STYLING & CONFIG ---
st.set_page_config(page_title="MedVision AI Pro | Ultra", layout="wide", page_icon="🧬")

st.markdown("""
    <style>
    .main { background: #0e1117; }
    div[data-testid="stExpander"] { 
        background: rgba(255, 255, 255, 0.05); 
        backdrop-filter: blur(10px); 
        border: 1px solid rgba(255, 255, 255, 0.1); 
        border-radius: 15px; 
    }
    .stButton>button { 
        background: linear-gradient(45deg, #2e7d32, #1b5e20); 
        border: none; color: white; border-radius: 8px;
    }
    .stMetric { 
        background: #161b22; 
        border: 1px solid #30363d; 
        border-radius: 12px; 
        padding: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 🔐 CRITICAL: SESSION STATE INITIALIZATION ---
if 'hospital_registry' not in st.session_state:
    st.session_state['hospital_registry'] = pd.DataFrame(columns=[
        "Patient Name", "Age", "Gender", "DOB", "Phone", "Diagnosis", "Confidence (%)", "Risk Level", "CDSS Advice"
    ])
if 'msgs' not in st.session_state:
    st.session_state['msgs'] = []
if 'page' not in st.session_state:
    st.session_state.page = 'welcome'

db.init_db()

@st.cache_resource
def get_model():
    MODEL_PATH = os.path.join(ROOT_PATH, 'model', 'pneumonia_model.keras')
    return prediction.load_pneumonia_model(MODEL_PATH)

model = get_model()

# --- 🏠 STEP 3: NAVIGATION ---
if st.session_state.page == 'welcome':
    st.title("🧬 MedVision AI: Next-Gen Diagnostic Hub")
    st.subheader("Holographic Pulmonary Analytics & Clinical Decision Support")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("""
        ### 🚀 Advanced Features Enabled:
        * **Explainable AI (XAI):** Real-time Grad-CAM thermal mapping.
        * **3D Volumetric Analysis:** Signal intensity mapping for lung density.
        * **Silent Registration:** Automated data sync during AI inference.
        * **CDSS Advice:** AI-driven medical recommendations for clinicians.
        """)
        if st.button("🔓 AUTHORIZE CLINICIAN ACCESS"):
            st.session_state.page = 'login'
            st.rerun()
    with c2:
        st.info("💡 **System Status:** AI Core Online | Database Connected")
    st.stop()

if st.session_state.page == 'login':
    st.title("🔐 Clinical Gatekeeper")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            user = st.text_input("Username", placeholder="admin")
            pw = st.text_input("Password", type="password", placeholder="doctor123")
            if st.form_submit_button("UNLOCK DASHBOARD"):
                if user == "admin" and pw == "doctor123":
                    st.session_state.page = 'dashboard'
                    st.rerun()
                else:
                    st.error("Invalid Credentials")
    st.stop()

# --- 👨‍⚕️ STEP 4: SIDEBAR REGISTRY ---
with st.sidebar:
    st.title("👨‍⚕️ Clinical Registry")
    with st.expander("📝 Patient Enrollment", expanded=True):
        p_name = st.text_input("Full Name", placeholder="e.g. Preetha")
        p_age = st.number_input("Age", 1, 120, 25)
        p_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        p_phone = st.text_input("Contact Number", placeholder="e.g. 9856743120")
        p_id = st.text_input("Registry ID", value=str(uuid.uuid4())[:8].upper())
    
    if st.button("➕ REGISTER PATIENT"):
        new_row = {"Patient Name": p_name, "Age": p_age, "Gender": p_gender, "DOB": datetime.date.today().strftime("%d-%m-%Y"), "Phone": p_phone, "Diagnosis": "PENDING", "Confidence (%)": "0.0", "Risk Level": "N/A", "CDSS Advice": "N/A"}
        st.session_state['hospital_registry'] = pd.concat([st.session_state['hospital_registry'], pd.DataFrame([new_row])], ignore_index=True)
        st.toast(f"Registered: {p_name}", icon="✅")

# --- 🚀 STEP 5: MAIN INTERFACE ---
tab1, tab2, tab3, tab4 = st.tabs(["🔬 Analysis Hub", "📂 Patient Portal", "📈 Analytics", "🤖 AI Med-Chat"])

# --- TAB 1: NEURAL ANALYSIS ---
with tab1:
    l_col, r_col = st.columns([1, 1.2])
    with l_col:
        st.subheader("📸 Radiograph Input")
        mode = st.radio("Source", ["File Upload", "📸 Camera Scan"])
        file = st.file_uploader("Upload X-ray") if mode == "File Upload" else st.camera_input("Scan X-ray")
        
        if file and st.button("🔬 EXECUTE AI CLINICAL SCAN"):
            if not p_name:
                st.error("⚠ Please enter the Patient Name in the sidebar to proceed!")
            else:
                with st.spinner("AI Core Analyzing..."):
                    img = Image.open(file)
                    arr = prediction.preprocess(img)
                    
                    # --- FIXED LINE BELOW: Ensure heatmap is returned from your prediction function ---
                    label, conf, heatmap = prediction.predict_image(model, arr)
                    
                    st.write(f"📊 Heatmap Intensity Range: {heatmap.min()} to {heatmap.max()}")
                    localized_img = gradcam.overlay_heatmap(img, heatmap)
                    
                    # CDSS & Risk Logic
                    risk_lvl = "HIGH" if (label == "PNEUMONIA" and conf > 0.8) else "MEDIUM" if label == "PNEUMONIA" else "LOW"
                    advice = "Immediate Hospitalization" if risk_lvl == "HIGH" else "Home Isolation & Meds" if risk_lvl == "MEDIUM" else "Standard Observation"
                    conf_str = f"{conf*100:.1f}%"
                    cur_date = datetime.date.today().strftime("%d-%m-%Y")
                    
                    # --- 🚀 ATOMIC DATA CAPTURE ---
                    df = st.session_state['hospital_registry']
                    mask = (df['Patient Name'] == p_name) & (df['Diagnosis'] == "PENDING")
                    
                    new_entry = {
                        "Patient Name": p_name, "Age": p_age, "Gender": p_gender, 
                        "DOB": cur_date, "Phone": p_phone if p_phone else "N/A", 
                        "Diagnosis": label, "Confidence (%)": conf_str, 
                        "Risk Level": risk_lvl, "CDSS Advice": advice
                    }

                    if mask.any():
                        df.loc[mask, ["Phone", "Diagnosis", "Confidence (%)", "Risk Level", "CDSS Advice"]] = [p_phone, label, conf_str, risk_lvl, advice]
                    else:
                        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
                    
                    st.session_state['hospital_registry'] = df
                    db.save_result(p_name, p_age, p_gender, p_phone, p_id, cur_date, label, conf_str)
                    st.session_state.current_result = (label, conf, localized_img, risk_lvl, advice, img)
                    st.success(f"✅ Record Finalized for {p_name}")

    if 'current_result' in st.session_state and file:
        label, conf, localized_img, risk_lvl, advice, original_img = st.session_state.current_result
        with r_col:
            st.subheader(f"Finding: {label}")
            st.write(f"Recommendation: **{advice}**")
            
            img_3d = original_img.convert('L').resize((150, 150))
            fig_3d = go.Figure(data=[go.Surface(z=np.array(img_3d).tolist(), colorscale='Viridis')])
            fig_3d.update_layout(scene=dict(aspectratio=dict(x=1,y=1,z=0.4)), height=350, margin=dict(l=0, r=0, b=0, t=0), paper_bgcolor='rgba(0,0,0,0)', template='plotly_dark')
            st.plotly_chart(fig_3d, use_container_width=True, theme=None)
            
            st.image(localized_img, caption="Grad-CAM Anomaly Heatmap", use_column_width=True)
            
            pdf_buf = report.generate_medical_pdf(p_name, p_age, p_id, label, conf, label, localized_img, st.session_state['hospital_registry'])
            st.download_button("📥 DOWNLOAD SMART REPORT", data=pdf_buf, file_name=f"Report_{p_id}.pdf")

# --- TAB 2: PORTAL ---
with tab2:
    st.subheader("📂 Hospital Registry")
    st.data_editor(st.session_state['hospital_registry'], use_container_width=True)
    
    if not st.session_state['hospital_registry'].empty:
        xlsx = BytesIO()
        with pd.ExcelWriter(xlsx, engine='xlsxwriter') as wr:
            st.session_state['hospital_registry'].to_excel(wr, index=False)
        st.download_button("📥 EXPORT TO EXCEL", data=xlsx.getvalue(), file_name="Hospital_Database.xlsx")

# --- TAB 3: ANALYTICS ---
with tab3:
    df_an = st.session_state['hospital_registry']
    if not df_an.empty:
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.pie(df_an, names='Diagnosis', hole=0.5, title="Case Distribution"), use_container_width=True)
        c2.plotly_chart(px.histogram(df_an, x="Age", color="Diagnosis", barmode="group", title="Age Trend"), use_container_width=True)

# --- TAB 4: SMART CHATBOT ---
with tab4:
    st.header("🤖 Medical Intelligence Assistant")
    st.info("I am powered by SmartScan Semantic Logic. Ask me about symptoms, diets, or AI heatmap interpretation.")

    for m in st.session_state.msgs:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    
    if prompt := st.chat_input("Ask about treatment, diet, or how the AI works..."):
        st.session_state.msgs.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        query = prompt.lower()
        
        if any(word in query for word in ["symptom", "sign", "how to know", "identify"]):
            response = """
            **Clinical Diagnostic Indicators:**
            * **Radiographic:** Presence of focal or patchy opacities (see 'Analysis Hub').
            * **Physical:** High-grade fever, tachypnea (rapid breathing), and pleuritic chest pain.
            * **Auscultation:** Crackling sounds in the lungs during deep breaths.
            """
        elif any(word in query for word in ["diet", "food", "eat", "drink", "nutrition"]):
            response = """
            **Recovery Nutritional Protocol:**
            * **Hydration:** Warm saline fluids and herbal infusions to thin pulmonary secretions.
            * **Protein-Rich:** Lean proteins (eggs, pulses, fish) to facilitate tissue repair.
            * **Anti-Inflammatory:** Citrus fruits (Vitamin C) and Turmeric (Curcumin) to support the immune response.
            * **Avoid:** Extremely cold dairy products which may increase mucus viscosity.
            """
        elif any(word in query for word in ["step", "treatment", "cure", "medicine", "doctor"]):
            response = """
            **Standard Hospital Treatment Pathway:**
            1. **Stabilization:** Immediate monitoring of SpO2 (Oxygen Saturation) levels.
            2. **Pathology:** Sputum and blood culture to differentiate Bacterial vs Viral.
            3. **Pharmacy:** Initiation of targeted antibiotics (if Bacterial) or bronchodilators.
            4. **Observation:** Follow-up radiograph in 10-14 days to monitor resolution.
            """
        elif any(word in query for word in ["how work", "grad-cam", "heatmap", "cnn", "technology"]):
            response = """
            **System Technical Insight:**
            I utilize a **Convolutional Neural Network (CNN)**. The **Grad-CAM** (Class Activation Mapping) you see in the Analysis tab works by calculating the gradients of the target class with respect to the final convolutional layer. It highlights the specific pixels that led the AI to its diagnosis.
            """
        else:
            response = "I am the SmartScan Clinical Assistant. Please specify if you need information on **symptoms**, **dietary protocols**, **treatment steps**, or **AI technical logic**."

        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.msgs.append({"role": "assistant", "content": response})