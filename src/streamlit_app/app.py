import streamlit as st
import sys
import os, re

# Ensure we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils.common import query_fastapi_backend

st.set_page_config(page_title="Enterprise Knowledge Assistant", page_icon="🧠", layout="centered")

st.title("🧠 Enterprise Knowledge Assistant")
st.markdown("Ask questions about internal documentation, policies, and code.")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "citations" in message and message["citations"]:
            with st.expander("📚 View Citations"):
                for cit in message["citations"]:
                    st.caption(f"**{cit['source']}** ({cit.get('section', 'General')}): [Link]({cit.get('url', '#')})")
        if "confidence" in message and message["confidence"]:
            st.caption(f"🤖 Confidence Score: {message['confidence']:.2f}")

# User Input
if prompt := st.chat_input("How do I use structural pattern matching?"):
    # Add user message to UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call Backend
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            response_data = query_fastapi_backend(prompt)

        if "error" in response_data:
            st.error(f"⚠️ Error: {response_data['error']}")
        else:
            answer = response_data.get("answer", "No answer generated.")
            citations = response_data.get("citations", [])
            confidence = response_data.get("confidence_score", 0.0)

            think_match = re.search(r'<think>(.*?)</think>', answer, flags=re.DOTALL)

            if think_match:
                thinking_process = think_match.group(1).strip()
                # Remove the think block from the final answer
                clean_answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()

                # Show the reasoning in a collapsible box
                if thinking_process:
                    with st.expander("🧠 View Agent's Internal Reasoning"):
                        st.caption(thinking_process)

                st.markdown(clean_answer)
            else:
                # If no think tags are present, just show the answer
                st.markdown(answer)

            if citations:
                with st.expander("📚 View Citations"):
                    for cit in citations:
                        st.caption(
                            f"**{cit.get('source', 'Unknown')}** ({cit.get('section', 'General')}): [Link]({cit.get('url', '#')})")

            st.caption(f"🤖 Confidence Score: {confidence:.2f}")

            # Feedback mechanism
            col1, col2 = st.columns([1, 15])
            with col1:
                if st.button("👍", key=f"up_{len(st.session_state.messages)}"):
                    st.toast("Feedback recorded. Thank you!")
            with col2:
                if st.button("👎", key=f"down_{len(st.session_state.messages)}"):
                    st.toast("Feedback recorded. We will improve this answer.")

            # Save assistant response to state
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "citations": citations,
                "confidence": confidence
            })