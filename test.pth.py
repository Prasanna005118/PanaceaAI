import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import cv2
import numpy as np
import os

# Define the class names and first aid knowledge
class_names = ["Abrasions", "Bruises", "Burns", "Cut", "Ingrown_nails", "Laceration", "Stab_wound"]
first_aid = {
    "Abrasions": "Clean the wound gently with water to remove dirt and debris. Pat dry with a clean cloth. Apply an antiseptic cream to prevent infection and cover with a sterile bandage. Avoid touching the wound unnecessarily and change the dressing daily.",
    "Bruises": "Apply a cold compress or ice wrapped in a cloth to the affected area for 10-15 minutes to reduce swelling. Keep the affected area elevated to minimize blood flow and swelling. Avoid massaging the bruised area to prevent further damage.",
    "Burns": "Cool the burn under running water for 10-20 minutes to reduce heat. Avoid using ice directly on the burn. Cover with a non-stick, sterile dressing or cling film. Do not burst blisters and seek medical help for severe burns.",
    "Cut": "Apply firm pressure with a clean cloth to stop bleeding. Rinse the wound under clean running water and gently pat dry. Apply an antiseptic ointment and cover with a sterile bandage. If the cut is deep or continues to bleed, seek medical attention.",
    "Ingrown_nails": "Soak the affected foot or hand in warm, soapy water for 15-20 minutes to reduce swelling. Avoid cutting the nail too short and wear comfortable footwear. If the condition worsens or becomes infected, consult a doctor.",
    "Laceration": "Clean the wound under running water to remove debris. Apply firm pressure to stop bleeding using a clean cloth. Apply an antiseptic solution and cover with a sterile dressing. Seek medical attention if the wound is deep or shows signs of infection.",
    "Stab_wound": "Do not remove any embedded object as it may worsen bleeding. Apply pressure around the wound to control bleeding. Cover with a clean cloth and keep the injured person calm. Seek immediate medical attention without delay."
}

# Load model function
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

# Preprocess image function
def preprocess_image(image, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)
    return input_tensor

# Predict wound type function
def predict_wound_type(model, input_tensor):
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted_class = torch.max(outputs, 1)
        return class_names[predicted_class.item()]

# Emergency numbers for Indian states and union territories
emergency_numbers = {
    "Andhra Pradesh": "<span style='color:red;'>112, 108</span>",
    "Arunachal Pradesh": "<span style='color:red;'>112, 108</span>",
    "Assam": "<span style='color:red;'>112, 108</span>",
    "Bihar": "<span style='color:red;'>112, 108</span>",
    "Chhattisgarh": "<span style='color:red;'>112, 108</span>",
    "Goa": "<span style='color:red;'>112, 108</span>",
    "Gujarat": "<span style='color:red;'>112, 108</span>",
    "Haryana": "<span style='color:red;'>112, 108</span>",
    "Himachal Pradesh": "<span style='color:red;'>112, 108</span>",
    "Jharkhand": "<span style='color:red;'>112, 108</span>",
    "Karnataka": "<span style='color:red;'>112, 108</span>",
    "Kerala": "<span style='color:red;'>112, 108</span>",
    "Madhya Pradesh": "<span style='color:red;'>112, 108</span>",
    "Maharashtra": "<span style='color:red;'>112, 108</span>",
    "Manipur": "<span style='color:red;'>112, 108</span>",
    "Meghalaya": "<span style='color:red;'>112, 108</span>",
    "Mizoram": "<span style='color:red;'>112, 108</span>",
    "Nagaland": "<span style='color:red;'>112, 108</span>",
    "Odisha": "<span style='color:red;'>112, 108</span>",
    "Punjab": "<span style='color:red;'>112, 108</span>",
    "Rajasthan": "<span style='color:red;'>112, 108</span>",
    "Sikkim": "<span style='color:red;'>112, 108</span>",
    "Tamil Nadu": "<span style='color:red;'>112, 108</span>",
    "Telangana": "<span style='color:red;'>112, 108</span>",
    "Tripura": "<span style='color:red;'>112, 108</span>",
    "Uttar Pradesh": "<span style='color:red;'>112, 108</span>",
    "Uttarakhand": "<span style='color:red;'>112, 108</span>",
    "West Bengal": "<span style='color:red;'>112, 108</span>",
    "Andaman and Nicobar Islands": "<span style='color:red;'>112, 108</span>",
    "Chandigarh": "<span style='color:red;'>112, 108</span>",
    "Dadra and Nagar Haveli and Daman and Diu": "<span style='color:red;'>112, 108</span>",
    "Delhi": "<span style='color:red;'>112</span>",
    "Jammu and Kashmir": "<span style='color:red;'>112, 108</span>",
    "Ladakh": "<span style='color:red;'>112, 108</span>",
    "Lakshadweep": "<span style='color:red;'>112, 108</span>",
    "Puducherry": "<span style='color:red;'>112, 108</span>"
}

# Load model
model_path = "wound_classification_model.pth"  # Update the path if necessary
model, device = load_model(model_path)

# Streamlit UI
def main():
    st.set_page_config(page_title="Panacea AI", layout="wide", page_icon=":hospital:")

    # Navigation bar
    menu = ["Home", "About", "Emergency"]
    choice = st.sidebar.selectbox("Navigation", menu)

    if choice == "Home":
        st.title("Wound Classifier and First Aid App")
        st.write("Upload an image or take a picture to classify the wound type and get first aid advice.")

        # Image upload option
        uploaded_file = st.file_uploader("Upload an image of the wound", type=["jpg", "jpeg", "png"])

        # Take a picture option
        camera_image = st.camera_input("Take a picture using your device")

        # Process the image
        if uploaded_file or camera_image:
            image_source = uploaded_file if uploaded_file else camera_image
            image = Image.open(image_source).convert("RGB")

            st.image(image, caption="Uploaded Image", use_container_width=True)
            input_tensor = preprocess_image(image, device)

            # Predict wound type
            predicted_label = predict_wound_type(model, input_tensor)

            st.subheader(f"Predicted Wound Type: {predicted_label}")
            st.write(f"**First Aid Advice:** {first_aid[predicted_label]}")
            st.warning("**Important:** Always seek medical attention for severe cases.")

    elif choice == "About":
        st.title("About the App")
        st.image("image.png", use_container_width=True)
        st.write("""
        This application is a state-of-the-art tool designed to assist users in identifying various wound types and providing initial first aid instructions. By leveraging a powerful deep learning model, the app classifies common wound categories, such as abrasions, burns, cuts, and more, based on uploaded or captured images. 

        The aim is to empower users to take immediate and informed action during emergencies by offering concise and accurate first aid advice. However, it is important to note that this tool is not a substitute for professional medical care. For severe injuries or if symptoms persist, seeking prompt consultation from healthcare professionals is imperative.

        With an intuitive interface, this app ensures a user-friendly experience. Whether you're at home, in the workplace, or traveling, it equips you with critical knowledge and guidance for handling minor injuries effectively.

        Your safety and well-being are our priorities. Use this app as a companion to enhance your emergency response capabilities, but always prioritize seeking proper medical attention.
        """)

    elif choice == "Emergency":
        st.title("Emergency Numbers")
        st.write("Below are emergency contact numbers for various states and union territories in India:")
        for state, numbers in emergency_numbers.items():
            st.markdown(f"<h3><strong>{state}:</strong> {numbers}</h3>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()