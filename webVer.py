
import streamlit as st
import cv2
import numpy as np
import pickle
from models import get_embedding


page_style = """
<style>
    .stApp {
        background: linear-gradient(135deg, #ff9ce3 0%, #b28dff 100%);
        padding: 20px;
    }
    .result-text {
    font-size: 23px;
    font-weight: 500;
    color: #fff;
    margin: 10px 0;
}

    .score-text {
        font-size: 22px;
        font-weight: 600;
        color: #fff;
        margin: 5px 0;
    }
</style>
"""
st.markdown(page_style, unsafe_allow_html=True)


with open("fingerData.pkl", "rb") as f:
    DB = pickle.load(f)



def verify(frame, threshold=0.85):
    emb_test = get_embedding(frame)
    best_score = -1
    match_user = None

    for user, emb_list in DB.items():
        for emb in emb_list:
            dist = np.dot(emb_test, emb) / (np.linalg.norm(emb_test) * np.linalg.norm(emb)) #cosine similarity
            if dist > best_score:
                best_score = dist
                match_user = user

    if best_score > threshold:
        return match_user, best_score
    else:
        return "Not Found", best_score


#display 
st.title("Fingerprint Verification App")
st.subheader("Upload fingerprint → Get match result")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if frame is not None:

        col1, col2 = st.columns(2)

    
        with col1:
            st.markdown("### Uploaded Image")
            resized = cv2.resize(frame, (350, 350), interpolation=cv2.INTER_AREA)
            st.image(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB), use_container_width=False)

        
        with col2:
            st.markdown("### Verification Result")
            user, score = verify(frame)
            st.markdown(f'<div class="result-text">User: {user}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="score-text">Score: {score:.3f}</div>', unsafe_allow_html=True)

    else:
        st.error("Error reading image!")
