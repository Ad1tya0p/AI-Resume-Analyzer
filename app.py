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

# --- STEP 1: Enhanced Data Architecture ---
class OptimizedProfile(BaseModel):
    name: str = Field(description="Extracted full name")
    email: str = Field(description="Extracted email")
    phone: str = Field(description="Extracted phone")
    summary: str = Field(description="ATS-optimized summary with high-impact keywords")
    skills: List[str] = Field(description="Technical skills grouped by domain (e.g., 'MLOps: ...')")
    experience: List[str] = Field(description="Bullet points using the Action-Result formula with percentages")
    education: List[str] = Field(description="Education history with relevant coursework")
    ats_score: int = Field(description="ATS match score from 0-100")
    optimization_points: List[str] = Field(description="Specific keywords and metrics added")

# --- STEP 2: Professional PDF Engine ---
class ResumePDF(FPDF):
    def header(self):
        self.set_font("helvetica", "B", 18)
        self.set_text_color(33, 37, 41)
        self.cell(0, 10, self.candidate_name.upper(), ln=True, align="C")
        self.set_font("helvetica", "", 9)
        self.cell(0, 5, f"{self.email} | {self.phone}", ln=True, align="C")
        self.ln(5)

    def section_header(self, title, width):
        self.set_font("helvetica", "B", 11)
        self.set_fill_color(240, 242, 246)
        self.cell(width, 8, f"  {title}", ln=True, fill=True)
        self.ln(2)

    def generate(self, data):
        self.candidate_name = data['name']
        self.email = data['email']
        self.phone = data['phone']
        self.set_margins(15, 15, 15)
        self.add_page()
        self.set_auto_page_break(auto=True, margin=15)
        eff_width = self.w - 30 
        
        sections = [
            ("PROFESSIONAL SUMMARY", data['summary']),
            ("TECHNICAL EXPERTISE", ", ".join(data['skills'])),
            ("SELECTED EXPERIENCE", data['experience']),
            ("EDUCATION", data['education'])
        ]

        for title, content in sections:
            self.section_header(title, eff_width)
            self.set_font("helvetica", "", 10)
            if isinstance(content, list):
                for item in content:
                    self.multi_cell(eff_width, 6, f"- {item}")
                    self.ln(1)
            else:
                self.multi_cell(eff_width, 6, content)
            self.ln(4)

# --- STEP 3: Streamlit UI ---
st.set_page_config(page_title="ATS Optimizer Pro", layout="wide")
st.title("🚀 High-Impact ATS Optimization Engine")

if "data" not in st.session_state:
    st.session_state.data = None

col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("📤 Source Data")
    uploaded_file = st.file_uploader("Upload PDF Resume", type="pdf")
    job_desc = st.text_area("Target Job Description", height=250, placeholder="Paste the full JD here for best results...")

    if st.button("Generate Optimized Resume"):
        if uploaded_file and job_desc:
            with st.spinner("AI is re-engineering your resume for maximum ATS match..."):
                raw_text = pdfminer.high_level.extract_text(io.BytesIO(uploaded_file.read()))
                llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.1)
                parser = JsonOutputParser(pydantic_object=OptimizedProfile)
                
                # REFINED PROMPT FOR HIGHER ATS SCORE
                prompt = ChatPromptTemplate.from_template(
                    "Act as a Senior Executive Recruiter. Your goal is to maximize the ATS score.\n"
                    "1. Use the 'Action + Result' framework for every experience bullet.\n"
                    "2. Include quantifiable metrics (%, $, time) for impact.\n"
                    "3. Semantically group skills (e.g., 'Cloud: AWS, Azure').\n"
                    "4. Extract real name and contact info from the resume.\n\n"
                    "Resume: {text}\nJD: {jd}\n{format_instructions}"
                )
                
                chain = prompt | llm | parser
                st.session_state.data = chain.invoke({"text": raw_text, "jd": job_desc, "format_instructions": parser.get_format_instructions()})

with col2:
    if st.session_state.data:
        data = st.session_state.data
        st.header(f"👤 {data['name']}")
        st.subheader(f"📊 Optimized ATS Score: {data['ats_score']}%")
        
        with st.expander("✨ Optimization Breakdown", expanded=True):
            for point in data['optimization_points']:
                st.info(f"**ATS Boost:** {point}")

        pdf = ResumePDF()
        pdf.generate(data)
        pdf.output("optimized.pdf")
        
        with open("optimized.pdf", "rb") as f:
            st.download_button(f"📥 Download Optimized Resume", f, f"{data['name']}_ATS_Optimized.pdf", use_container_width=True)
    else:
        st.info("Upload your resume and the target Job Description to begin.")
