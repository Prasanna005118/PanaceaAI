# EIR AI - Wound Classification & First Aid Assistant

EIR AI is a cutting-edge application designed to help users identify different types of wounds and provide first aid guidance. Inspired by Eir, the Norse goddess of healing, this app leverages artificial intelligence to assist in making informed decisions during medical emergencies. Whether you're at home, work, or on the go, EIR AI empowers users with accurate first aid advice for minor injuries and wounds.

## Features

- **Wound Classification**: Upload an image or use your device's camera to capture a photo of a wound. The app will classify the wound into categories such as abrasions, bruises, cuts, burns, and more.
- **First Aid Advice**: After classifying the wound, EIR AI provides tailored first aid instructions based on the injury type.
- **Emergency Numbers**: Access emergency contact numbers for various states and union territories in India, helping you act quickly in critical situations.
- **User-Friendly Interface**: An intuitive and easy-to-navigate interface that guides users through the wound classification process and presents helpful information in a clear, concise manner.

## Technologies Used

- **Streamlit**: The app’s user interface is built using Streamlit, enabling easy image upload and interaction.
- **PyTorch**: The app uses a deep learning model built with PyTorch to classify wounds based on uploaded images.
- **TorchVision**: Preprocessing and image transformation for the deep learning model.
- **OpenCV**: For camera input functionality.

## Installation

To run the app locally, follow these steps:

### Prerequisites

1. **Python** 3.7 or higher
2. **PyTorch** with CUDA support (optional for GPU acceleration)
3. **Streamlit**
4. **Other Dependencies** listed in the `requirements.txt`

### Steps

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/eir-ai.git
    cd eir-ai
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Download or train the **wound classification model** (`wound_classification_model.pth`) and place it in the root directory of the project.

4. Run the app:

    ```bash
    streamlit run app.py
    ```

The app will be available at [http://localhost:8501](http://localhost:8501).

## How It Works

1. **Upload or capture an image** of a wound.
2. The app processes the image and classifies it into one of the following categories:
   - Abrasions
   - Bruises
   - Burns
   - Cuts
   - Ingrown Nails
   - Lacerations
   - Stab Wounds
3. **First Aid Advice** is displayed based on the classification, guiding you on how to handle the injury.
4. **Emergency Numbers** for your region are provided for immediate help in critical situations.

## First Aid Advice

Here are a few examples of the first aid advice provided:

- **Abrasions**: Clean gently with water, apply antiseptic, and cover with a sterile bandage.
- **Bruises**: Apply a cold compress, keep elevated, and avoid massaging.
- **Burns**: Cool under running water, cover with a sterile dressing, and seek medical help for severe burns.

## Contributing

We welcome contributions to improve the app! Here’s how you can help:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add new feature'`).
5. Push to your forked repository (`git push origin feature-branch`).
6. Open a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- **Eir**: Norse goddess of healing, the inspiration for the name and purpose of this app.
- **PyTorch** and **Streamlit** for providing powerful tools to build the app.
- **OpenCV** for enabling camera functionality.
- Thanks to all contributors and the open-source community for their invaluable resources.

## Contact

For any questions or feedback, feel free to open an issue on the repository or contact the developer.

---
Stay safe and take care! 💙