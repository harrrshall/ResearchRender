import os
import time
import hashlib
import asyncio
import io
import logging
from functools import wraps
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from pymongo import MongoClient
from bson.binary import Binary
import PyPDF2
from dotenv import load_dotenv

from cached_code_generation import process_paper

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

# Rate limiting configuration
RATE_LIMIT = 5  # Max uploads per IP
RATE_LIMIT_PERIOD = 60  # Rate limit period in seconds

# Create a ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=4)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def run_async(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))
    return wrapper

def async_rate_limit(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        ip = request.remote_addr
        current_time = time.time()

        # Check recent requests from this IP
        recent_uploads = await asyncio.to_thread(
            upload_history_collection.count_documents,
            {
                'ip': ip,
                'timestamp': {'$gt': current_time - RATE_LIMIT_PERIOD}
            }
        )
        if recent_uploads >= RATE_LIMIT:
            return jsonify({'error': 'Rate limit exceeded. Please try again later.'}), 429

        # Record this request
        await asyncio.to_thread(
            upload_history_collection.insert_one,
            {
                'ip': ip,
                'timestamp': current_time
            }
        )

        return await func(*args, **kwargs)
    return wrapper

def extract_text_from_pdf(file_content):
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        return "".join(page.extract_text() for page in pdf_reader.pages)
    except Exception as e:
        app.logger.error(f"Error extracting text from PDF: {str(e)}")
        return None

async def process_file(file_content, filename):
    content_hash = hashlib.md5(file_content).hexdigest()

    # Check if this file has been processed before
    existing_file = await asyncio.to_thread(files_collection.find_one, {'content_hash': content_hash})
    if existing_file:
        if existing_file.get('steps') and existing_file.get('code'):
            return {
                'steps': existing_file['steps'],
                'code': existing_file['code'],
                'message': 'File has been processed before. Returning existing results.'
            }
        elif existing_file.get('steps'):
            try:
                code = await asyncio.to_thread(process_paper, existing_file['steps'], generate_steps=False)
                await asyncio.to_thread(
                    files_collection.update_one,
                    {'_id': existing_file['_id']},
                    {'$set': {'code': code}}
                )
                return {
                    'steps': existing_file['steps'],
                    'code': code,
                    'message': 'Steps existed. Generated new code.'
                }
            except Exception as e:
                return {'error': f'Error generating code: {str(e)}'}, 500

    # Process new file
    if filename.lower().endswith('.pdf'):
        paper_content = await asyncio.to_thread(extract_text_from_pdf, file_content)
        if paper_content is None:
            return {'error': 'Failed to extract text from PDF'}, 500
    else:
        paper_content = file_content.decode('utf-8')

    steps, code = await asyncio.to_thread(process_paper, paper_content)

    if steps is None:
        return {'error': 'Failed to generate steps from the paper'}, 500

    # Store results in MongoDB
    await asyncio.to_thread(
        files_collection.insert_one,
        {
            'filename': filename,
            'content_hash': content_hash,
            'steps': steps,
            'code': code
        }
    )

    return {'steps': steps, 'code': code}

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/logic.html')
def logic():
    return send_from_directory('static', 'logic.html')

@app.route('/upload', methods=['POST'])
@run_async
@async_rate_limit
async def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected for upload'}), 400

    if file and allowed_file(file.filename):
        if file.content_length > app.config['MAX_CONTENT_LENGTH']:
            return jsonify({'error': f'File size exceeds the maximum limit of {app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)}MB'}), 413

        filename = secure_filename(file.filename)
        file_content = file.read()

        try:
            result = await process_file(file_content, filename)
            return jsonify(result)
        except UnicodeDecodeError:
            return jsonify({'error': 'The uploaded file is not a valid text file'}), 400
        except Exception as e:
            app.logger.error(f"Error processing file: {str(e)}")
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Allowed file types are txt, pdf, doc, docx'}), 400

if __name__ == '__main__':
    app.run(debug=True)