# Document Heading Classification Solution

## Approach
This solution is designed to classify headings in documents using a machine learning model. The workflow involves extracting features from document data, training a model to recognize heading patterns, and then using the trained model to predict headings in new documents. The process is automated to handle multiple formats (CSV, JSON, PDF) and leverages preprocessed features for robust classification.

## Models and Libraries Used
- **Model:** The main model used is a scikit-learn based classifier, saved as `heading_model.joblib` in the `model/` directory.
- **Libraries:**
  - `scikit-learn` for machine learning
  - `pandas` for data manipulation
  - `joblib` for model serialization
  - `numpy` for numerical operations
  - Other dependencies as listed in `requirements.txt`

## How to Build and Run

### Local Run
1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Directory Setup:**
   Ensure the following directories exist in the project root:
   - `input/`
   - `output/`
   - `model/`
   - `training_csvs/`
   - `training_jsons/`
   - `training_pdfs/`

3. **Run Training:**
   ```bash
   python train.py
   ```
4. **Run Testing:**
   > **Note:** You need to update file paths in `test.py` to match your local directory structure if running outside Docker.
   ```bash
   python test.py
   ```

### Run as Docker Container
1. **Build the Docker Image:**
   ```bash
   docker build -t heading-classifier .
   ```
2. **Run the Container:**
   ```bash
   docker run --rm -it heading-classifier
   ```
   This will execute `test.py` by default. The model is trained during the build process.

## Additional Notes
- All required data and model files should be placed in their respective directories as shown in the workspace structure.
- For custom data or different directory layouts, update the paths in `test.py` and `train.py` accordingly.
- The solution is designed to be reproducible and portable via Docker.
