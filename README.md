# FingerPrint-Verification
This project is a Fingerprint Verification System designed to identify users by comparing captured fingerprint images against a pre-recorded database. It utilizes a high-performance MobileOne deep learning model to translate visual ridge patterns into mathematical vectors (embeddings) for rapid and accurate matching.

## System Description
The system operates as a Camera-Based Biometric Authenticator. Unlike traditional capacitive scanners found on smartphones, this system is designed to process standard image files (JPG/PNG), making it compatible with high-resolution cameras or webcams.

### Feature Extraction:
The system uses the mobileone_s0 architecture, which is optimized for mobile-speed performance while maintaining high accuracy.

### Vector Normalization:
Each fingerprint is converted into a normalized 512-dimensional embedding, ensuring that the "direction" of the fingerprint features is the primary point of comparison.

### Similarity: 
Matches are determined by calculating the dot product between a live "test" embedding and the stored database embeddings.

### Threshold Management: 
The system includes an evaluation tool (one.py) to test various sensitivity levels (from 0.6 to 0.9) to find the perfect balance between security and user convenience.

## Project Structure
* **models.py**: The "Brain" – Loads the MobileOne model and handles image-to-tensor conversion.
* **datasets.py**: The "Enrollment" – Processes raw images from the `/images` folder to create a searchable `fingerData.pkl` database.
* **test.py**: The "Analyst" – Runs accuracy tests on a `test_images` folder to determine the optimal similarity threshold.
* **webVer.py**: The "Interface" – A Streamlit web app that allows users to upload a fingerprint and see real-time match results and similarity scores.

##  Technical Workflow
1. **Preprocessing**: Images are normalized via `preprocess.py` (external dependency).
2. **Inference**: Torch-based inference extracts a unique 512-d signature.
3. **Verification**: The system compares the live signature against the Pickle database.
4. **Result**: If the score exceeds the defined threshold (e.g., 0.85), the user identity is displayed.

