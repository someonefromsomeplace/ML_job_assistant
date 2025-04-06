import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("salary_predictor.pkl")

# Function to extract average salary
def extract_avg_salary(s):
    try:
        s = s.replace('$', '').replace('K', '').replace(',', '').strip()
        parts = list(map(int, s.split('-')))
        return sum(parts) / len(parts)
    except:
        return None

# Predict and compare salary
def compare_salary(job_title, location, experience, skills, posted_salary):
    skill_count = len(skills.split(','))
    input_data = pd.DataFrame([{
        'Job Title': job_title.title().strip(),
        'location': location.title().strip(),
        'Experience': experience,
        'skill_count': skill_count
    }])
    predicted = model.predict(input_data)[0]
    actual_salary = extract_avg_salary(posted_salary)
    
    if actual_salary is None:
        comparison = "Unable to analyze offer."
    else:
        comparison = "Below Market" if actual_salary < predicted else \
                     "Above Market" if actual_salary > predicted else "At Market"
    
    return predicted, actual_salary, comparison

# Generate negotiation script
def generate_negotiation_script(role, current_offer, predicted_value):
    return f"""
Hi [Hiring Manager],

Thank you for the offer for the **{role}** position. I'm genuinely excited about the opportunity to contribute to your team.

Based on my skills, experience, and current market trends, I was expecting compensation closer to **${int(predicted_value)}**. 
Would there be flexibility to revisit the offer and align it more closely with industry standards?

Looking forward to your thoughts.

Best,  
[Your Name]
"""

# --- Streamlit UI ---

st.title("AI Salary Negotiation Assistant")
st.markdown("Let me help you assess your offer, predict your worth, and negotiate confidently! 💼💸")

with st.form("negotiation_form"):
    job_title = st.text_input("Job Title", "Software Engineer")
    location = st.text_input("Job Location", "San Francisco")
    experience = st.slider("Years of Experience", 0, 30, 3)
    skills = st.text_input("Your Skills (comma-separated)", "Python,React,Docker")
    posted_salary = st.text_input("Salary Offered (e.g. $85K-$95K)", "$85K-$95K")
    
    submitted = st.form_submit_button("🔍 Analyze Offer")

if submitted:
    predicted, actual, comparison = compare_salary(job_title, location, experience, skills, posted_salary)
    
    st.subheader("Analysis Result")
    st.markdown(f"**Predicted Market Salary:** ${int(predicted)}")
    st.markdown(f"**Offered Salary Range:** ${actual if actual else 'Invalid Format'}")
    st.markdown(f"**Market Comparison:** {comparison}")
    
    st.subheader("Suggested Negotiation Message")
    st.code(generate_negotiation_script(job_title, actual, predicted))

    st.success("Ready to negotiate with confidence!")

