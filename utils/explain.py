def explain_diagnosis(label, confidence):
    """Generates a human-readable clinical interpretation of the AI results."""
    
    conf_pct = round(confidence * 100, 2)
    
    if label == "PNEUMONIA":
        if confidence > 0.85:
            return (f"CRITICAL FINDING: The AI has identified significant pulmonary opacities "
                    f"with {conf_pct}% confidence. This pattern is highly consistent with "
                    f"Acute Bacterial Pneumonia. Immediate clinical intervention and "
                    f"antibiotic therapy are advised.")
        else:
            return (f"MODERATE FINDING: The neural scan detected markers of fluid or inflammation "
                    f"({conf_pct}% confidence). This suggests early-stage pneumonia or viral "
                    f"infiltration. Correlation with a blood count (WBC) is recommended.")
    
    else:
        # For NORMAL cases
        if confidence > 0.90:
            return (f"CLEAR SCAN: The AI found no significant markers of infection "
                    f"({conf_pct}% confidence). Lung parenchyma appears healthy and "
                    f"radiotranslucent. No acute treatment required.")
        else:
            return (f"OBSERVATION: Lungs appear mostly clear ({conf_pct}% confidence), "
                    f"but minor shadows were noted. While not pneumonia, the patient should "
                    f"be monitored if symptoms like a persistent cough continue.")

def get_treatment_suggestion(label, confidence):
    """Provides automated treatment protocols based on the diagnosis."""
    if label == "NORMAL":
        return "Rest, hydration, and routine follow-up if symptoms persist."
    
    # Treatment for Pneumonia based on severity (confidence)
    if confidence > 0.80:
        return ("1. Broad-spectrum antibiotics (e.g., Ceftriaxone).\n"
                "2. Supplementary Oxygen if SpO2 < 92%.\n"
                "3. Admission to Acute Care Unit.")
    else:
        return ("1. Oral antibiotics (e.g., Amoxicillin).\n"
                "2. Increased fluid intake and fever management (Paracetamol).\n"
                "3. Home isolation with 48-hour follow-up.")