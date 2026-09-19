import streamlit as st
import joblib

# Load Saved Models 
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

st.set_page_config(page_title="Spam Classifier", page_icon="📩")
st.title("📩 SMS & Email Spam Classifier")
st.write("Enter any message below to predict whether it is **Spam** or **Ham (Safe)**.")

user_input = st.text_area("Message Input", height=150, placeholder="Type your message here...")

if st.button("Predict"):
    if user_input.strip() != "":
        vectorized_text = vectorizer.transform([user_input])
        prediction = model.predict(vectorized_text)[0]
        confidence = model.predict_proba(vectorized_text)[0]

        st.subheader("Prediction Result:")
        if prediction == 1:
            st.error("🚨 **SPAM DETECTED**")
            st.write(f"Confidence: **{confidence[1]*100:.2f}%**")
        else:
            st.success("✅ **HAM (SAFE) MESSAGE**")
            st.write(f"Confidence: **{confidence[0]*100:.2f}%**")
    else:
        st.warning("Please enter a valid message to predict.")