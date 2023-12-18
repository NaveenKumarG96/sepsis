import streamlit as st
import requests

# Function to make a POST request to FastAPI
def predict_sepsis(json_data):
    url = "http://127.0.0.1:8000/predict"  # Replace with your actual FastAPI endpoint
    headers = {"Content-Type": "application/json"}

    response = requests.post(url, json=json_data, headers=headers)

    if response.status_code == 200:
        result = response.json()
        return result["sepsis"]
    else:
        return None

# Streamlit App
def main():
    st.title("Sepsis Prediction App")

    # Input JSON text area
    st.subheader("Enter JSON Input:")
    json_input = st.text_area("Paste your JSON input here:", "")

    # Convert the JSON input to a dictionary
    try:
        json_data = eval(json_input) if json_input else None
    except Exception as e:
        st.error(f"Error parsing JSON: {e}")
        return

    # Make prediction when the user clicks the "Predict" button
    if st.button("Predict"):
        if json_data:
            # Call the FastAPI endpoint for prediction
            prediction = predict_sepsis(json_data)

            if prediction is not None:
                st.success(f"Sepsis Prediction: {prediction}")
            else:
                st.error("Error in making the prediction. Please check your input.")
        else:
            st.warning("Please enter valid JSON input.")

if __name__ == "__main__":
    main()
