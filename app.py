from flask import Flask, request, jsonify, render_template, send_from_directory
from ultralytics import YOLO
import cv2
import os
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Inisialisasi Flask app
app = Flask(__name__)

# Konfigurasi logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Memastikan direktori kerja diatur dengan benar ke direktori skrip
script_dir = os.path.dirname(os.path.abspath(__file__))

# Path model YOLO (relative path, resolved to absolute)
model_path = os.path.join(script_dir, "runs/detect/train_finetune_v6/weights/best.pt")

try:
    model = YOLO(model_path)
    logging.info("YOLO model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load YOLO model: {e}")
    raise

# Direktori untuk menyimpan file
UPLOAD_FOLDER = os.path.join(script_dir, 'uploads')
RESULT_FOLDER = os.path.join(script_dir, 'results')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Validasi cek format
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route untuk halaman utama
@app.route('/')
def index():
    logging.info("Accessed index page.")
    return render_template('index.html')

# Route untuk prediksi
@app.route('/predict', methods=['POST'])
def predict():
    logging.info("Prediction request received.")

    if 'file' not in request.files:
        logging.warning("No file part in the request.")
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        logging.warning("No file selected.")
        return jsonify({'error': 'No file selected'}), 400

    # Validasi format file
    if not allowed_file(file.filename):
        logging.warning(f"Unsupported file format: {file.filename}")
        return jsonify({'error': 'Unsupported file format'}), 400

    # Simpan file yang diunggah
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    try:
        file.save(file_path)
        logging.info(f"File saved to {file_path}.")
    except Exception as e:
        logging.error(f"Error saving file: {e}")
        return jsonify({'error': 'Failed to save file'}), 500

    # Membaca gambar
    try:
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError("Invalid image format.")
        logging.info("Image loaded successfully.")
    except Exception as e:
        logging.error(f"Error reading image: {e}")
        return jsonify({'error': 'Failed to read image'}), 500

    # Deteksi objek
    try:
        results = model(image)
        result = results[0]  # Mengambil hasil pertama dari daftar

        # Konversi hasil dari BGR ke RGB sebelum memplot
        rgb_image = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)

        # Plot hasil deteksi dan simpan ke file
        plt.imshow(rgb_image)
        plt.axis('off')
        result_filename = f"result_{file.filename}"
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        plt.savefig(result_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        logging.info(f"Detection completed. Result saved to {result_path}.")
        return jsonify({'message': 'Detection completed', 'result_image': result_filename}), 200
    except Exception as e:
        logging.error(f"Error during YOLO detection: {e}")
        return jsonify({'error': 'Failed to detect objects'}), 500

# Route untuk melayani file statis di folder 'results'
@app.route('/results/<path:filename>')
def results_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    logging.info("Starting Flask app...")
    app.run(debug=True, host='0.0.0.0', port=5000)
