import streamlit as st
import json
import os
import zipfile
import io
import numpy as np
from typing import Dict, List, Tuple, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

ResumeData = Dict[str, Any]
MatchResult = Tuple[str, float, ResumeData]

class ResumeProcessor:
    """Handles resume data processing and matching logic."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with specified sentence transformer model."""
        self.model_name = model_name
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the sentence transformer model."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def calculate_match_score(self, resume_data: ResumeData, query_embedding: np.ndarray, required_skills: set, required_certs: set) -> float:
        """
        Calculate a match score based on:
          1) Semantic similarity,
          2) Required skills,
          3) Required certifications.
        """
        scores = []
        weights = []

        # 1) Semantic similarity
        doc_embedding = self.model.encode(resume_data['full_text'])
        semantic_score = float(cosine_similarity([query_embedding], [doc_embedding])[0][0])
        scores.append(semantic_score)
        weights.append(0.4)

        # 2) Skills match
        resume_skills = set().union(*resume_data['skills'].values())
        skills_score = len(required_skills.intersection(resume_skills)) / len(required_skills) if required_skills else 1.0
        scores.append(skills_score)
        weights.append(0.3)

        # 3) Certifications match
        resume_certs = set(resume_data['certifications'])
        certs_score = len(required_certs.intersection(resume_certs)) / len(required_certs) if required_certs else 1.0
        scores.append(certs_score)
        weights.append(0.3)

        return sum(score * weight for score, weight in zip(scores, weights))

class ResumeMatcherUI:
    """Handles the Streamlit UI flow for uploading, searching, and displaying results."""

    def __init__(self):
        """Initialize the processor and Streamlit session state."""
        self.processor = ResumeProcessor()
        self.initialize_session_state()

    @staticmethod
    def initialize_session_state():
        """Set up Streamlit session variables."""
        if 'data' not in st.session_state:
            st.session_state.data = {}
        if 'current_matches' not in st.session_state:
            st.session_state.current_matches = None

    def process_multiple_files(self):
        """Allow users to upload multiple JSON files and combine their data."""
        st.title("Resume Qualification Matcher")
        st.write("Upload multiple JSON files containing resumes.")
        uploaded_files = st.file_uploader(
            "Upload Resume JSON Files (maximum size per file: 200MB)", 
            type=["json"], 
            accept_multiple_files=True
        )

        if uploaded_files:
            combined_data = {}
            for uploaded_file in uploaded_files:
                try:
                    data = json.load(uploaded_file)
                    combined_data.update(data.get('resumes', {}))
                    st.success(f"File '{uploaded_file.name}' uploaded successfully!")
                except json.JSONDecodeError:
                    st.error(f"Failed to parse file: {uploaded_file.name}")

            if combined_data:
                st.session_state.data = combined_data
                st.success("All files uploaded and combined successfully!")

    def display_search_interface(self) -> Tuple[set, set, str]:
        """Display UI elements for specifying search criteria."""
        st.sidebar.header("Search Criteria")

        required_skills = set(st.sidebar.text_area(
            "Enter Required Skills (comma-separated)",
            placeholder="e.g., Python, AWS, SQL"
        ).split(","))

        required_certs = set(st.sidebar.text_area(
            "Enter Required Certifications (comma-separated)",
            placeholder="e.g., PMP, AWS Certified Solutions Architect"
        ).split(","))

        search_query = st.sidebar.text_area(
            "Describe the ideal candidate",
            placeholder="e.g., Experienced data scientist with expertise in machine learning and cloud computing..."
        )

        return required_skills, required_certs, search_query

    def display_results(self, matches: List[MatchResult]):
        """Display the top 20 matching resumes in a table and allow saving full resumes as a zip file."""
        if not matches:
            st.warning("No matches found.")
            return

        st.header("Top 20 Matching Candidates")

        # Limit matches to top 20 results
        top_matches = matches[:20]

        results_df = pd.DataFrame([{
            "Candidate Name": name,
            "Match Score": f"{score:.2f}",
            "Skills": ', '.join(set().union(*resume_data['skills'].values())),
            "Certifications": ', '.join(resume_data['certifications'])
        } for name, score, resume_data in top_matches])

        st.dataframe(results_df)

        st.download_button(
            label="Download Results as CSV",
            data=results_df.to_csv(index=False),
            file_name="top_20_matching_candidates.csv",
            mime="text/csv"
        )

        # Allow saving full resumes as a zip file
        selected_candidates = st.multiselect(
            "Select candidates to download full resumes:", 
            options=[name for name, _, _ in top_matches]
        )

        if selected_candidates:
            if st.button("Download Resumes as ZIP"):
                with io.BytesIO() as zip_buffer:
                    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                        for name in selected_candidates:
                            resume_data = next(
                                resume for resume, _, _ in top_matches if resume == name
                            )
                            zip_file.writestr(f"{name}.txt", st.session_state.data[name]['full_text'])
                    zip_buffer.seek(0)
                    st.download_button(
                        label="Download ZIP",
                        data=zip_buffer.getvalue(),
                        file_name="selected_resumes.zip",
                        mime="application/zip"
                    )

    def run(self):
        """Main entry point for the Streamlit app."""
        self.process_multiple_files()

        if st.session_state.data:
            required_skills, required_certs, search_query = self.display_search_interface()

            if st.button("Search"):
                query_embedding = self.processor.model.encode(search_query) if search_query.strip() else None
                matches = []

                for name, resume_data in st.session_state.data.items():
                    try:
                        score = self.processor.calculate_match_score(
                            resume_data,
                            query_embedding,
                            required_skills,
                            required_certs
                        )
                        matches.append((name, score, resume_data))
                    except Exception as e:
                        st.warning(f"Failed to process resume for {name}: {str(e)}")

                matches.sort(key=lambda x: x[1], reverse=True)
                st.session_state.current_matches = matches

            if st.session_state.current_matches:
                self.display_results(st.session_state.current_matches)

if __name__ == "__main__":
    ui = ResumeMatcherUI()
    ui.run()
