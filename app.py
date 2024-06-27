import os
import time
import hashlib
from functools import wraps
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from pymongo import MongoClient
from bson.binary import Binary
from cached_code_generation import process_paper
from dotenv import load_dotenv
import PyPDF2
import io
import logging

# Load environment variables
load_dotenv()

app = Flask(__name__, static_folder='static')

# MongoDB configuration
MONGO_URI = os.environ.get('MONGO_URI')
if not MONGO_URI:
    raise ValueError("MONGO_URI environment variable is not set")

# Extract database name from MONGO_URI or use a default
db_name = os.getenv("MONGO_DB_NAME") or "Cluster0"

client = MongoClient(MONGO_URI)
db = client[db_name]
files_collection = db.files
upload_history_collection = db.upload_history

# Set max content length
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max file size

# Allowed file extensions
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx'}

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/logic.html')
def logic():
    return send_from_directory('static', 'logic.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

RATE_LIMIT = 5  # Max uploads per IP
RATE_LIMIT_PERIOD = 60  # Rate limit period in seconds

def rate_limit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        ip = request.remote_addr
        current_time = time.time()

        # Check recent requests from this IP
        recent_uploads = upload_history_collection.count_documents({
            'ip': ip,
            'timestamp': {'$gt': current_time - RATE_LIMIT_PERIOD}
        })
        if recent_uploads >= RATE_LIMIT:
            return jsonify({'error': 'Rate limit exceeded. Please try again later.'}), 429

        # Record this request
        upload_history_collection.insert_one({
            'ip': ip,
            'timestamp': current_time
        })

        return func(*args, **kwargs)
    return wrapper

def extract_text_from_pdf(file_content):
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {str(e)}")
        return None
@app.route('/upload', methods=['POST'])
@rate_limit
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected for upload'}), 400
    if file and allowed_file(file.filename):
        if file.content_length > app.config['MAX_CONTENT_LENGTH']:
            return jsonify({'error': 'File size exceeds the maximum limit of 16MB'}), 413

        filename = secure_filename(file.filename)
        file_content = file.read()

        # Generate a hash of the file content
        content_hash = hashlib.md5(file_content).hexdigest()

        # Check if this file has been processed before
        existing_file = files_collection.find_one({'content_hash': content_hash})
        if existing_file:
            # If both steps and code exist, return them
            if existing_file.get('steps') and existing_file.get('code'):
                return jsonify({
                    'steps': existing_file['steps'],
                    'code': existing_file['code'],
                    'message': 'File has been processed before. Returning existing results.'
                })
            # If only steps exist, generate code
            elif existing_file.get('steps'):
                try:
                    code = process_paper(existing_file['steps'], generate_steps=False)
                    files_collection.update_one(
                        {'_id': existing_file['_id']},
                        {'$set': {'code': code}}
                    )
                    return jsonify({
                        'steps': existing_file['steps'],
                        'code': code,
                        'message': 'Steps existed. Generated new code.'
                    })
                except Exception as e:
                    return jsonify({'error': f'Error generating code: {str(e)}'}), 500

        # Process new file
        try:
            if filename.lower().endswith('.pdf'):
                paper_content = extract_text_from_pdf(file_content)
                if paper_content is None:
                    return jsonify({'error': 'Failed to extract text from PDF'}), 500
            else:
                paper_content = file_content.decode('utf-8')

            steps, code = process_paper(paper_content)

            if steps is None:
                return jsonify({'error': 'Failed to generate steps from the paper'}), 500

            # Store results in MongoDB
            files_collection.insert_one({
                'filename': filename,
                'content_hash': content_hash,
                'steps': steps,
                'code': code
            })

            return jsonify({'steps': steps, 'code': code})
        except UnicodeDecodeError:
            return jsonify({'error': 'The uploaded file is not a valid text file'}), 400
        except Exception as e:
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Allowed file types are txt, pdf, doc, docx'}), 400

# This line is needed for Vercel
if __name__ == '__main__':
    app.run()