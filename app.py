import streamlit as st
import pdfminer.high_level
import io, os
from fpdf import FPDF
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional

load_dotenv()

# --- STEP 1: Dynamic Data Architecture ---
class OptimizedProfile(BaseModel):
    name: str = Field(description="Extracted full name of the candidate")
    email: str = Field(description="Extracted email address")
    phone: str = Field(description="Extracted phone number")
    summary: str = Field(description="ATS-optimized professional summary")
    skills: List[str] = Field(description="Technical skills matching the JD")
    experience: List[str] = Field(description="Impactful bullet points with results")
    education: List[str] = Field(description="Extracted and formatted education history")
    ats_score: int = Field(description="ATS match score from 0-100")
    optimization_points: List[str] = Field(description="List of specific improvements made")

# --- STEP 2: Universal Professional PDF Engine ---
class ResumePDF(FPDF):
    def header(self):
        # Dynamically uses the extracted candidate name
        self.set_font("helvetica", "B", 20)
        self.set_text_color(33, 37, 41)
        self.cell(0, 12, self.candidate_name.upper(), ln=True, align="C")
        
        # Contact Information Line
        self.set_font("helvetica", "", 9)
        contact_info = f"{self.email} | {self.phone}"
        self.cell(0, 5, contact_info, ln=True, align="C")
        self.ln(5)

    def section_header(self, title):
        self.set_font("helvetica", "B", 11)
        self.set_fill_color(245, 245, 245)
        self.cell(0, 8, f"  {title}", ln=True, fill=True)
        self.ln(2)

    def generate(self, data):
        self.candidate_name = data['name']
        self.email = data['email']
        self.phone = data['phone']
        self.add_page()
        
        sections = [
            ("PROFESSIONAL SUMMARY", data['summary']),
            ("TECHNICAL EXPERTISE", ", ".join(data['skills'])),
            ("EXPERIENCE", data['experience']),
            ("EDUCATION", data['education'])
        ]

        for title, content in sections:
            self.section_header(title)
            self.set_font("helvetica", "", 10)
            if isinstance(content, list):
                for item in content:
                    self.multi_cell(0, 6, f"- {item}")
                    self.ln(1)
            else:
                self.multi_cell(0, 6, content)
            self.ln(4)

# --- STEP 3: Streamlit UI with NLP Analysis ---
st.set_page_config(page_title="AI Resume Parser & Optimizer", layout="wide")
st.title("🚀 Universal Professional Resume Engine")

if "optimized_data" not in st.session_state:
    st.session_state.optimized_data = None

col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("📤 Upload Resume")
    uploaded_file = st.file_uploader("Upload PDF Resume", type="pdf")
    job_desc = st.text_area("Target Job Description", height=200, placeholder="Paste JD here...")

    if st.button("Parse & Optimize"):
        if uploaded_file and job_desc:
            with st.spinner("NLP Engine extracting candidate data..."):
                # Raw Text Extraction
                raw_text = pdfminer.high_level.extract_text(io.BytesIO(uploaded_file.read()))
                
                llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.1)
                parser = JsonOutputParser(pydantic_object=OptimizedProfile)
                
                # Prompt specifically instructed to extract candidate info
                prompt = ChatPromptTemplate.from_template(
                    "Act as an NLP Resume Parser. Extract the candidate's real name, email, and phone from the text.\n"
                    "Then, optimize their resume for the JD.\n\n"
                    "Resume Text: {text}\nJD: {jd}\n{format_instructions}"
                )
                
                chain = prompt | llm | parser
                st.session_state.optimized_data = chain.invoke({
                    "text": raw_text, "jd": job_desc,
                    "format_instructions": parser.get_format_instructions()
                })

with col2:
    if st.session_state.optimized_data:
        data = st.session_state.optimized_data
        
        # Display Dynamic Extracted Name
        st.header(f"👤 Candidate: {data['name']}")
        st.subheader(f"📊 ATS Score: {data['ats_score']}%")
        
        # Optimization Insights
        with st.expander("✨ Analysis Highlights", expanded=True):
            for point in data['optimization_points']:
                st.write(f"✅ {point}")

        # PDF Generation
        pdf = ResumePDF()
        pdf.generate(data)
        pdf.output("result.pdf")
        
        with open("result.pdf", "rb") as f:
            st.download_button(
                label=f"📥 Download Optimized Resume for {data['name']}",
                data=f,
                file_name=f"{data['name']}_Optimized.pdf",
                mime="application/pdf"
            )