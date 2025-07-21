import streamlit as st, uuid, json, datetime

def solution_form():
    st.subheader("🚀 Submit New Solution / Enhancement")
    with st.form("solution_form"):
        title = st.text_input("Title")
        description = st.text_area("Description", height=150)
        industry = st.text_input("Industry / Vertical")
        tech_stack = st.text_area("Tech Stack / Tools")
        submitted = st.form_submit_button("Submit")
        if submitted:
            payload = {
                "id": str(uuid.uuid4()),
                "timestamp": str(datetime.datetime.utcnow()),
                "title": title,
                "description": description,
                "industry": industry,
                "tech_stack": tech_stack
            }
            st.success("Submitted! (In real life POST to backend)")
            st.json(payload)
            return payload