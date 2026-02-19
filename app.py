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

# --- STEP 2: PDF Logic (Fixed for Margin Errors) ---
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
        self.set_fill_color(245, 245, 245)
        self.cell(width, 8, f"  {title}", ln=True, fill=True)
        self.ln(2)

    def generate(self, data):
        self.candidate_name = data['name']
        self.email = data['email']
        self.phone = data['phone']
        
        # Explicitly set margins to prevent "Not enough horizontal space" error
        self.set_margins(15, 15, 15)
        self.add_page()
        self.set_auto_page_break(auto=True, margin=15)
        
        # Calculate effective width (Page width - margins)
        eff_width = self.w - 30 
        
        sections = [
            ("PROFESSIONAL SUMMARY", data['summary']),
            ("TECHNICAL EXPERTISE", ", ".join(data['skills'])),
            ("EXPERIENCE", data['experience']),
            ("EDUCATION", data['education'])
        ]

        for title, content in sections:
            self.section_header(title, eff_width)
            self.set_font("helvetica", "", 10)
            if isinstance(content, list):
                for item in content:
                    # Use eff_width instead of 0 to ensure wrapping space
                    self.multi_cell(eff_width, 6, f"- {item}")
                    self.ln(1)
            else:
                self.multi_cell(eff_width, 6, content)
            self.ln(4)

# --- STEP 3: Streamlit UI ---
st.set_page_config(page_title="AI Resume Parser", layout="wide")
st.title("🚀 Universal Professional Resume Engine")

if "data" not in st.session_state:
    st.session_state.data = None

col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("📤 Upload Source")
    uploaded_file = st.file_uploader("Upload PDF Resume", type="pdf")
    job_desc = st.text_area("Target Job Description", height=200)

    if st.button("Parse & Optimize"):
        if uploaded_file and job_desc:
            with st.spinner("Extracting & Optimizing..."):
                raw_text = pdfminer.high_level.extract_text(io.BytesIO(uploaded_file.read()))
                llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.1)
                parser = JsonOutputParser(pydantic_object=OptimizedProfile)
                prompt = ChatPromptTemplate.from_template(
                    "Extract candidate name, email, phone, and education. Optimize for JD.\n"
                    "Resume: {text}\nJD: {jd}\n{format_instructions}"
                )
                chain = prompt | llm | parser
                st.session_state.data = chain.invoke({"text": raw_text, "jd": job_desc, "format_instructions": parser.get_format_instructions()})

with col2:
    if st.session_state.data:
        data = st.session_state.data
        st.subheader(f"👤 Candidate: {data['name']}")
        
        # NLP Breakdown Tabs
        tabs = st.tabs(["📊 Score", "🛠 Skills", "💼 Experience", "🎓 Education"])
        with tabs[0]:
            st.metric("ATS Match", f"{data['ats_score']}%")
            for p in data['optimization_points']:
                st.success(f"Optimized: {p}")
        with tabs[1]:
            st.write(", ".join(data['skills']))
        with tabs[2]:
            for e in data['experience']:
                st.write(f"🔹 {e}")
        with tabs[3]:
            for ed in data['education']:
                st.write(f"📖 {ed}")

        # PDF Export
        pdf = ResumePDF()
        pdf.generate(data)
        pdf.output("result.pdf")
        
        with open("result.pdf", "rb") as f:
            st.download_button(f"📥 Download Optimized PDF", f, f"{data['name']}_Resume.pdf")
