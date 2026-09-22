import streamlit as st
import joblib
import datetime
from pymongo import MongoClient

# १. MongoDB Connection (Local)

MONGO_URI = "mongodb://localhost:27017/"



@st.cache_resource
def get_database():
    client = MongoClient(MONGO_URI)
    return client['spam_detection_db'] # Database 

db = get_database()
collection = db['predictions'] # Collection (Table) 


# 2. Saved Models Load

model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

st.set_page_config(page_title="Spam Classifier with DB", page_icon="📩")
st.title("📩 SMS & Email Spam Classifier")
st.write("Enter any message below to predict whether it is **Spam** or **Ham (Safe)**.")

user_input = st.text_area("Message Input", height=150, placeholder="Type your message here...")

if st.button("Predict"):
    if user_input.strip() != "":

        # Prediction 
        vectorized_text = vectorizer.transform([user_input])
        prediction = model.predict(vectorized_text)[0]
        confidence = model.predict_proba(vectorized_text)[0]

        result_text = "Spam" if prediction == 1 else "Ham"
        confidence_val = float(confidence[1] if prediction == 1 else confidence[0])

        st.subheader("Prediction Result:")
        if prediction == 1:
            st.error("🚨 **SPAM DETECTED**")
            st.write(f"Confidence: **{confidence_val*100:.2f}%**")
        else:
            st.success("✅ **HAM (SAFE) MESSAGE**")
            st.write(f"Confidence: **{confidence_val*100:.2f}%**")

       
        # 3. save prediction data in MongoDB 
    
        log_data = {
            "message": user_input,
            "prediction": result_text,
            "confidence": round(confidence_val * 100, 2),
            "timestamp": datetime.datetime.now()
        }
        
        try:
            collection.insert_one(log_data)
            st.toast("✅ Prediction result saved to MongoDB database!", icon="💾")
        except Exception as e:
            st.warning(f"Could not save to MongoDB: {e}")

    else:
        st.warning("Please enter a valid message to predict.")