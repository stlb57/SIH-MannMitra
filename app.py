# app.py
import certifi
import os
import json
import logging
import uuid
from datetime import datetime, timedelta
from bson import ObjectId
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from pymongo import MongoClient
from transformers import pipeline
from chat_agent import initialize_chat_agent
from twilio.rest import Client

# --- App Initialization & Configuration ---
logging.basicConfig(level=logging.INFO)
load_dotenv()
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY")
CORS(app, supports_credentials=True)
bcrypt = Bcrypt(app)

# --- Twilio Configuration ---
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# --- Database Connection ---
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client.mannmitra_db
users_collection = db.users
analyses_collection = db.analyses
goals_collection = db.goals
checkins_collection = db.checkins
doctors_collection = db.doctors
contacts_collection = db.contacts

# --- Mock Data and Geospatial Index for Doctors ---
def setup_doctors():
    """Ensures there is mock doctor data and a geospatial index."""
    if doctors_collection.count_documents({}) == 0:
        logging.info("No doctors found, inserting mock data...")
        mock_doctors = [
            {"name": "Dr. Anjali Sharma", "specialty": "Cognitive Behavioral Therapy", "price": 1500, "availability": "Mon-Fri, 9am-5pm", "location": {"type": "Point", "coordinates": [77.216721, 28.644800]}}, # Delhi
            {"name": "Dr. Rohan Mehta", "specialty": "Psychiatry & Mindfulness", "price": 2000, "availability": "Tue-Sat, 11am-7pm", "location": {"type": "Point", "coordinates": [72.8777, 19.0760]}}, # Mumbai
            {"name": "Dr. Priya Desai", "specialty": "Adolescent Psychology", "price": 1200, "availability": "Mon-Wed, 10am-6pm", "location": {"type": "Point", "coordinates": [77.5946, 12.9716]}}, # Bangalore
            {"name": "Dr. Vikram Singh", "specialty": "Stress & Anxiety Management", "price": 1800, "availability": "Wed-Sun, 8am-2pm", "location": {"type": "Point", "coordinates": [80.2707, 13.0827]}}, # Chennai
        ]
        doctors_collection.insert_many(mock_doctors)
        logging.info("Mock doctors inserted.")
    
    if "location_2dsphere" not in doctors_collection.index_information():
        doctors_collection.create_index([("location", "2dsphere")], name="location_2dsphere")
        logging.info("Created 2dsphere index on doctors collection.")

setup_doctors()

# --- User Login Management ---
login_manager = LoginManager()
login_manager.init_app(app)

class User(UserMixin):
    def __init__(self, user_data):
        self.id = str(user_data["_id"])
        self.username = user_data["username"]
        self.role = user_data.get("role", "student")

@login_manager.user_loader
def load_user(user_id):
    try:
        user_data = users_collection.find_one({"_id": ObjectId(user_id)})
    except Exception:
        return None
    return User(user_data) if user_data else None

# --- Load AI Models ---
chat_agent, system_prompt, type_classifier, intensity_classifier = None, None, None, None
try:
    logging.info("Initializing All Models...")
    chat_agent, system_prompt = initialize_chat_agent()
    type_classifier = pipeline("text-classification", model="./models/final_stress_classifier_model")
    intensity_classifier = pipeline("text-classification", model="./models/final_stress_classifier")
    logging.info("All models initialized successfully.")
except Exception as e:
    logging.error(f"Error initializing models: {e}")

# --- API Endpoints ---

# == AUTHENTICATION ROUTES ==
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json(); username = data.get('username'); password = data.get('password')
    if not username or not password: return jsonify({"error": "Username and password required"}), 400
    if users_collection.find_one({'username': username}): return jsonify({"error": "Username already exists"}), 409
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    users_collection.insert_one({'username': username, 'password': hashed_password, 'role': 'student'})
    return jsonify({"message": "User registered successfully"}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json(); username = data.get('username'); password = data.get('password')
    user_data = users_collection.find_one({'username': username})
    if user_data and bcrypt.check_password_hash(user_data['password'], password):
        user = User(user_data); login_user(user)
        return jsonify({"message": "Login successful", "username": user.username}), 200
    return jsonify({"error": "Invalid username or password"}), 401

@app.route('/logout')
@login_required
def logout():
    logout_user(); return jsonify({"message": "Logout successful"}), 200

@app.route('/@me')
@login_required
def get_current_user():
    return jsonify({"username": current_user.username})

# == STUDENT FEATURES ==
@app.route('/chat', methods=['POST'])
@login_required
def handle_chat():
    data = request.get_json(); user_query = data['text']
    full_query = f"{system_prompt}\nUser Query: {user_query}"
    response = chat_agent.run(full_query, reset=False)
    return jsonify({'reply': response})

@app.route('/analyze', methods=['POST'])
@login_required
def analyze_text():
    data = request.get_json(); text = data['text']
    positive_keywords = ['happy', 'great', 'amazing', 'wonderful', 'fantastic', 'excellent', 'joy', 'good', 'love']
    is_positive = any(word in text.lower() for word in positive_keywords)
    if is_positive: analysis_doc = {'type': 'Positive', 'intensity': 'Positive'}
    else: analysis_doc = {'type': type_classifier(text)[0]['label'], 'intensity': intensity_classifier(text)[0]['label']}
    analysis_doc.update({'user_id': current_user.id, 'username': current_user.username, 'text': text, 'timestamp': datetime.utcnow()})
    analyses_collection.insert_one(analysis_doc)
    return jsonify({"message": "Analysis saved"}), 200

@app.route('/checkin', methods=['POST'])
@login_required
def save_checkin():
    data = request.get_json()
    checkin_doc = {"user_id": current_user.id, "mood": data.get("mood"), "notes": data.get("notes", ""),"timestamp": datetime.utcnow()}
    checkins_collection.insert_one(checkin_doc)
    return jsonify({"message": "Check-in saved successfully"}), 201

@app.route('/get_my_report', methods=['GET'])
@login_required
def get_my_report():
    pipeline = [{"$match": {"user_id": current_user.id}}, {"$group": {"_id": "$intensity", "count": {"$sum": 1}}}]
    results = list(analyses_collection.aggregate(pipeline))
    frequency_report = {item['_id']: item['count'] for item in results}
    checkin_pipeline = [{"$match": {"user_id": current_user.id}}, {"$group": {"_id": "$mood", "count": {"$sum": 1}}}]
    checkin_results = list(checkins_collection.aggregate(checkin_pipeline))
    checkin_frequencies = {item['_id']: item['count'] for item in checkin_results}
    weights = {"high": -2, "mid": -1, "low": 0, "neutral": 1, "Neutral": 1, "Positive": 2, "bad": -2, "down": -1, "meh": 0, "okay": 1, "great": 2}
    total_score = sum(weights.get(k, 0) * v for k, v in frequency_report.items()) + sum(weights.get(k, 0) * v for k, v in checkin_frequencies.items())
    if total_score >= 5: final_status = "Feeling Great"
    elif total_score > 0: final_status = "Doing Okay"
    elif total_score > -5: final_status = "Feeling Down"
    else: final_status = "Under Significant Stress"
    return jsonify({"status": final_status, "score": total_score, "frequency": frequency_report, "checkin_frequency": checkin_frequencies})

@app.route('/get_my_activity_stats', methods=['GET'])
@login_required
def get_my_activity_stats():
    try:
        chat_count = analyses_collection.count_documents({"user_id": current_user.id})
        checkin_count = checkins_collection.count_documents({"user_id": current_user.id})
        return jsonify({"chat_count": chat_count, "checkin_count": checkin_count})
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/goals', methods=['POST'])
@login_required
def add_goal():
    data = request.get_json(); goal_text = data.get('text'); reminder_time = data.get('time')
    if not goal_text: return jsonify({"error": "Goal text is required"}), 400
    goal_doc = {"user_id": current_user.id, "text": goal_text, "progress": 0, "reminderTime": reminder_time, "created_at": datetime.utcnow()}
    goals_collection.insert_one(goal_doc)
    return jsonify({"message": "Goal added successfully"}), 201

@app.route('/goals', methods=['GET'])
@login_required
def get_goals():
    user_goals = list(goals_collection.find({"user_id": current_user.id}))
    for goal in user_goals: goal["_id"] = str(goal["_id"])
    return jsonify(user_goals)

# == ADMIN FEATURES ==
@app.route('/admin/reports', methods=['GET'])
@login_required
def get_admin_reports():
    if current_user.role != 'admin': return jsonify({"error": "Unauthorized"}), 403
    pipeline = [{"$sort": {"timestamp": -1}}, {"$group": {"_id": "$user_id", "username": {"$first": "$username"}, "last_intensity": {"$first": "$intensity"}, "last_type": {"$first": "$type"}, "last_timestamp": {"$first": "$timestamp"}}}, {"$sort": {"last_timestamp": -1}}]
    all_reports = list(analyses_collection.aggregate(pipeline))
    one_week_ago = datetime.utcnow() - timedelta(days=7)
    for report in all_reports:
        report["_id"] = str(report["_id"])
        high_stress_count = analyses_collection.count_documents({"user_id": report["_id"], "intensity": "high", "timestamp": {"$gte": one_week_ago}})
        report["needs_attention"] = high_stress_count >= 3
    return jsonify(all_reports)

# == DOCTOR FEATURES ==
@app.route('/doctors', methods=['GET'])
@login_required
def get_nearby_doctors():
    try:
        lat = float(request.args.get('lat')); lon = float(request.args.get('lon')); limit = int(request.args.get('limit', 10))
        pipeline = [{'$geoNear': {'near': { 'type': "Point",  'coordinates': [ lon, lat ] }, 'distanceField': "distance", 'distanceMultiplier': 1 / 1000, 'spherical': True}},{ '$limit': limit }]
        doctors = list(doctors_collection.aggregate(pipeline))
        for doc in doctors: doc['_id'] = str(doc['_id'])
        return jsonify(doctors)
    except Exception as e: logging.error(f"Error fetching doctors: {e}"); return jsonify({"error": "Could not fetch doctors"}), 500

@app.route('/start_video_call', methods=['POST'])
@login_required
def start_video_call():
    data = request.get_json(); doctor_id = data.get('doctor_id')
    if not doctor_id: return jsonify({"error": "Doctor ID is required"}), 400
    room_name = f"MannMitra-Call-{uuid.uuid4().hex[:12]}"; video_call_url = f"https://meet.jit.si/{room_name}"
    logging.info(f"User {current_user.id} starting call with doctor {doctor_id} at {video_call_url}")
    return jsonify({"url": video_call_url})

# == EMERGENCY CONTACTS & SOS FEATURES ==
@app.route('/contacts', methods=['POST'])
@login_required
def add_contact():
    data = request.get_json(); contact_name = data.get('name'); contact_phone = data.get('phone')
    if not contact_name or not contact_phone: return jsonify({"error": "Name and phone number are required"}), 400
    contact_doc = {"user_id": current_user.id, "name": contact_name, "phone": contact_phone}
    contacts_collection.insert_one(contact_doc)
    return jsonify({"message": "Contact added successfully"}), 201

@app.route('/contacts', methods=['GET'])
@login_required
def get_contacts():
    user_contacts = list(contacts_collection.find({"user_id": current_user.id}))
    for contact in user_contacts: contact["_id"] = str(contact["_id"])
    return jsonify(user_contacts)

@app.route('/contacts/<contact_id>', methods=['DELETE'])
@login_required
def delete_contact(contact_id):
    result = contacts_collection.delete_one({"_id": ObjectId(contact_id), "user_id": current_user.id})
    if result.deleted_count == 1: return jsonify({"message": "Contact deleted"}), 200
    return jsonify({"error": "Contact not found or unauthorized"}), 404

@app.route('/sos-alert', methods=['POST'])
@login_required
def trigger_sos_alert():
    user_contacts = list(contacts_collection.find({"user_id": current_user.id}))
    if not user_contacts: return jsonify({"error": "No emergency contacts found"}), 404
    message_body = (f"This is an automated alert from MannMitra. {current_user.username} has indicated they are "
                    f"in significant distress. Please check in with them as soon as you can. "
                    f"This is not a substitute for emergency services.")
    errors = []
    for contact in user_contacts:
        try:
            message = twilio_client.messages.create(body=message_body, from_=TWILIO_PHONE_NUMBER, to=contact['phone'])
            logging.info(f"Sent SOS message to {contact['phone']}, SID: {message.sid}")
        except Exception as e:
            logging.error(f"Failed to send SOS to {contact['phone']}: {e}")
            errors.append(contact['name'])
    if errors: return jsonify({"message": f"Alert sent to some contacts, but failed for: {', '.join(errors)}"}), 207
    return jsonify({"message": "SOS alerts sent successfully to all contacts."}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)