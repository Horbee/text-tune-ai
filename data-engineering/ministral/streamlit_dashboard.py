import streamlit as st
import diff_match_patch as dmp_module
import textdistance

from ollama import chat
from ollama import ChatResponse

# --- CONFIGURATION ---
st.set_page_config(page_title="German GEC Visualizer", layout="wide")

st.title("🇩🇪 German GEC Performance Visualizer")
st.markdown("Compare **Original Input** vs **Model Correction** with precise diff highlighting.")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Settings")

    selected_model = st.selectbox("Model", ["ministral-3:8b", "ministral-3:3b"], index=0)

    st.markdown("### Test Examples")
    examples = [
        "Ich habe gestern in dem Park gegangen.",
        "Weil ich müde war ich bin nach Hause gegangen.",
        "Das ist ein schön Auto.",
        "Wir freuen uns auf Ihre Antwort."
    ]
    selected_example = st.selectbox("Load Example", [""] + examples)

# --- INPUT AREA ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("📝 Input Text")
    user_input = st.text_area("Enter German text with errors:", 
                              value=selected_example if selected_example else "", 
                              height=200)

# --- LOGIC ---
if st.button("Correct Text", type="primary"):
    if not user_input:
        st.warning("Please enter some text first.")
    if not selected_model:
        st.warning("Please select a model.")
    else:
        with st.spinner(f"Asking {selected_model} to correct..."):
            try:
                response: ChatResponse = chat(model=selected_model, messages=[
                        {
                            'role': 'system',
                            'content': 'Korrigiere die Grammatik im folgenden Text, aber behalte den ursprünglichen Stil und Ton bei. Verleihe dem Text keine formelle Note, wenn er diese nicht hat. Gib **nur** den korrigierten Satz zurück, ohne Anmerkungen.',
                        },
                        {
                            'role': 'user',
                            'content': user_input,
                        },
                ])
                
                corrected_text = response["message"]["content"]

                # 2. GENERATE DIFFS
                dmp = dmp_module.diff_match_patch()
                diffs = dmp.diff_main(user_input, corrected_text)
                dmp.diff_cleanupSemantic(diffs)
                
                html_diff = dmp.diff_prettyHtml(diffs)

                # 3. CALCULATE METRICS
                # Similarity: How close is the output to the input? (Lower = more changes)
                similarity = textdistance.levenshtein.normalized_similarity(user_input, corrected_text)
                changes_count = sum(1 for op, text in diffs if op != 0)

            except Exception as e:
                st.error(f"Error calling API: {e}")
                st.stop()

        # --- OUTPUT AREA ---
        with col2:
            st.subheader("✅ Correction")
            st.text_area("Raw Output:", value=corrected_text, height=200)

        # --- VISUALIZATION SECTION ---
        st.divider()
        st.subheader("🔍 Visual Diff Analysis")
        
        # Metrics Columns
        m1, m2, m3 = st.columns(3)
        m1.metric("Similarity Score", f"{similarity:.2%}", help="100% means no changes made")
        m2.metric("Edits Made", changes_count, help="Number of distinct insertion/deletion blocks")
        m3.metric("Length Delta", len(corrected_text) - len(user_input), help="Char count difference")

        # The HTML Diff View
        st.markdown("### Detailed Changes")
        st.markdown(html_diff, unsafe_allow_html=True)
        
        st.caption("Legend: <del style='background:#ffe6e6;color:black;'>Red</del> = Deleted, <ins style='background:#e6ffe6;color:black;'>Green</ins> = Inserted", unsafe_allow_html=True)