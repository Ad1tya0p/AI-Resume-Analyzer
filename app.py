import streamlit as st
import pdfminer.high_level
import io, os
from fpdf import FPDF
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List

load_dotenv()

# --- STEP 1: Data Architecture ---
class OptimizedProfile(BaseModel):
    name: str = Field(description="Extracted full name")
    email: str = Field(description="Extracted email")
    phone: str = Field(description="Extracted phone")
    summary: str = Field(description="ATS-optimized professional summary")
    skills: List[str] = Field(description="Technical skills matching the JD")
    experience: List[str] = Field(description="Impactful bullet points with results")
    education: List[str] = Field(description="Extracted education history")
    ats_score: int = Field(description="ATS match score from 0-100")
    optimization_points: List[str] = Field(description="List of specific improvements made")

# --- STEP 2: PDF Logic ---
class ResumePDF(FPDF):
    def generate(self, data):
        self.add_page()
        self.set_font("helvetica", "B", 16)
        self.cell(0, 10, data['name'].upper(), ln=True, align="C")
        self.set_font("helvetica", "", 10)
        self.cell(0, 5, f"{data['email']} | {data['phone']}", ln=True, align="C")
        self.ln(10)
        
        sections = [("SUMMARY", data['summary']), ("SKILLS", ", ".join(data['skills'])), 
                    ("EXPERIENCE", data['experience']), ("EDUCATION", data['education'])]
        
        for title, content in sections:
            self.set_font("helvetica", "B", 12)
            self.cell(0, 10, title, ln=True)
            self.set_font("helvetica", "", 10)
            if isinstance(content, list):
                for item in content:
                    self.multi_cell(0, 6, f"- {item}")
            else:
                self.multi_cell(0, 6, content)
            self.ln(5)

# --- STEP 3: Streamlit UI ---
st.set_page_config(page_title="NLP Resume Optimizer", layout="wide")
st.title("🚀 Professional NLP Resume Engine")

col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("📤 Source Data")
    uploaded_file = st.file_uploader("Upload PDF Resume", type="pdf")
    job_desc = st.text_area("Target Job Description", height=200)

    if st.button("Parse & Optimize"):
        if uploaded_file and job_desc:
            with st.spinner("NLP Engine Analyzing..."):
                raw_text = pdfminer.high_level.extract_text(io.BytesIO(uploaded_file.read()))
                
                llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.1)
                parser = JsonOutputParser(pydantic_object=OptimizedProfile)
                prompt = ChatPromptTemplate.from_template(
                    "Act as an NLP Parser. Extract the candidate's name, email, phone, and education. "
                    "Then optimize the summary, skills, and experience for this JD.\n\n"
                    "Resume: {text}\nJD: {jd}\n{format_instructions}"
                )
                
                chain = prompt | llm | parser
                st.session_state.data = chain.invoke({"text": raw_text, "jd": job_desc, "format_instructions": parser.get_format_instructions()})

with col2:
    if "data" in st.session_state:
        data = st.session_state.data
        
        # --- NEW: NLP Extraction View ---
        st.subheader("🔍 NLP Extraction Breakdown")
        tabs = st.tabs(["👤 Profile", "🛠 Skills", "💼 Experience", "🎓 Education"])
        
        with tabs[0]:
            st.markdown(f"**Name:** {data['name']}")
            st.markdown(f"**Email:** {data['email']}")
            st.markdown(f"**Phone:** {data['phone']}")
        
        with tabs[1]:
            st.write(data['skills'])
            
        with tabs[2]:
            for exp in data['experience']:
                st.write(f"🔹 {exp}")
        
        with tabs[3]:
            for edu in data['education']:
                st.write(f"📖 {edu}")

        st.divider()
        
        # --- Optimization Section ---
        st.subheader(f"📊 ATS Score: {data['ats_score']}%")
        with st.expander("✨ Optimization Highlights", expanded=True):
            for point in data['optimization_points']:
                st.success(f"**Improvement:** {point}")

        # PDF Generation
        pdf = ResumePDF()
        pdf.generate(data)
        pdf.output("result.pdf")
        
        with open("result.pdf", "rb") as f:
            st.download_button(f"📥 Download Optimized PDF for {data['name']}", f, f"{data['name']}_Resume.pdf")
    else:
        st.info("Upload your resume and click 'Parse' to see the NLP breakdown.")
