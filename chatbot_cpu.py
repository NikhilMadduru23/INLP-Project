import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, pipeline, T5Tokenizer, T5ForConditionalGeneration
from langchain_huggingface import HuggingFacePipeline
from peft import PeftModel
import spacy
import re
import json
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from datetime import datetime
import time

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Initialize spacy for NER
nlp = spacy.load("en_core_web_sm")

# Initialize device (CPU-only)
device = torch.device("cpu")
device_id = -1
print(f"Using device: {device}")

# Ensure logs directory exists
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "chatbot_metrics.json")

# Global history store, entity cache, and last known context
history_store = {}
entity_cache = {}  # Cache for extracted entities per message
last_known_context = {}  # Cache for the last known context per session

# Load the fine-tuned DistilBERT model and tokenizer
distilbert_model_path = "./saved_models/intent_classifier"
try:
    distilbert_tokenizer = DistilBertTokenizer.from_pretrained(distilbert_model_path)
    distilbert_model = DistilBertForSequenceClassification.from_pretrained(distilbert_model_path)
    distilbert_model.to(device)
    distilbert_model.eval()
except Exception as e:
    raise Exception(f"Failed to load DistilBERT model from {distilbert_model_path}: {e}")

# Load label encoder for intents
def load_label_encoder():
    try:
        df = pd.read_csv("./data/processed/atis_train.csv")
        label_encoder = LabelEncoder()
        label_encoder.fit(df['intent'])
        return label_encoder
    except Exception as e:
        raise Exception(f"Failed to load atis_train.csv: {e}")

label_encoder = load_label_encoder()

# Initialize HuggingFacePipeline with google/flan-t5-base for CPU compatibility
base_model_name = "google/flan-t5-base"
cpu_lora_adapter_path = "./cpu_lora_adapters"
print(f"Loading {base_model_name} for CPU compatibility...")
t5_model = T5ForConditionalGeneration.from_pretrained(base_model_name)
# Explicitly set legacy=True to suppress tokenizer warning
t5_tokenizer = T5Tokenizer.from_pretrained(base_model_name, legacy=True)
if os.path.exists(cpu_lora_adapter_path):
    print("Loading CPU-specific LoRA adapters...")
    try:
        t5_model = PeftModel.from_pretrained(t5_model, cpu_lora_adapter_path)
        t5_model.to(device)
        # Unwrap PeftModel to get the base T5ForConditionalGeneration for pipeline compatibility
        base_model = t5_model.get_base_model()
    except Exception as e:
        print(f"Failed to load CPU-specific LoRA adapters: {e}. Proceeding with base model...")
        base_model = t5_model
else:
    base_model = t5_model

llm = HuggingFacePipeline(pipeline=pipeline(
    "text2text-generation",
    model=base_model,
    tokenizer=t5_tokenizer,
    max_length=30,  # Reduced for CPU compatibility
    device=device_id
))

# Disable summarization pipeline for non-GPU laptop
summarizer = None

# Fare code explanations for atis_abbreviation
fare_code_explanations = {
    "q": "Fare code Q is an economy class fare, typically discounted with restrictions such as advance purchase or non-refundability.",
    "f": "Fare code F is a first-class fare, offering premium services and flexibility.",
    "y": "Fare code Y is a full-fare economy class ticket, often fully refundable and flexible.",
    "h": "Fare code H is an economy class fare with moderate restrictions, often used for mid-tier pricing.",
    "qw": "Fare code QW is a promotional economy fare, usually with strict conditions like limited availability.",
    "c": "Fare code C is a business class fare, offering enhanced services with flexibility."
}

# Intent-based response templates
intent_responses = {
    "atis_abbreviation": "Please provide the abbreviation or fare code you'd like explained.",
    "atis_aircraft": "Please specify the flight or route to check the aircraft type.",
    "atis_airfare": "I can check airfares for you. Please provide the departure city, destination, and travel date.",
    "atis_airline": "Which airline are you inquiring about, and for what route or service?",
    "atis_flight": "I can help you find flights. Please specify the departure city, destination, and travel date.",
    "atis_flight_time": "I can provide flight schedules. Please tell me the departure city, destination, and date.",
    "atis_ground_service": "I can provide information on ground transportation. Please tell me the city and airport.",
    "atis_quantity": "Please provide more details, such as the airline or route, to check quantities like flight counts."
}

# Preprocess text for DistilBERT
def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(tokens)

# Intent classification with enhanced post-processing
def classify_intent(text):
    start_time = time.time()
    processed_text = preprocess_text(text)
    inputs = distilbert_tokenizer(processed_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    with torch.no_grad():
        outputs = distilbert_model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()
    
    intent = label_encoder.inverse_transform([predicted_class])[0]
    text_lower = text.lower()
    if any(keyword in text_lower for keyword in ["how many", "number of", "daily", "weekly"]):
        intent = "atis_quantity"
    elif any(keyword in text_lower for keyword in ["fare code", "explain code"]):
        intent = "atis_abbreviation"
    elif any(keyword in text_lower for keyword in ["aircraft", "plane", "boeing", "airbus"]):
        intent = "atis_aircraft"
    elif any(keyword in text_lower for keyword in ["airfare", "cheapest", "fares", "cost", "deal", "economy"]):
        intent = "atis_airfare"
    elif any(keyword in text_lower for keyword in ["fly", "flight", "nonstop", "class", "morning", "evening", "asap"]):
        intent = "atis_flight"
    elif any(keyword in text_lower for keyword in ["schedule", "schedules", "earliest", "flight time"]):
        intent = "atis_flight_time"
    elif any(keyword in text_lower for keyword in ["ground", "transportation", "taxi", "shuttle", "airport"]):
        intent = "atis_ground_service"
    elif any(keyword in text_lower for keyword in ["airline", "delta", "united", "american", "southwest", "route"]):
        intent = "atis_airline"
    latency = time.time() - start_time
    return intent, latency

# Extract entities using spaCy NER with caching and fallback
def extract_entities(text, history, session_id):
    global entity_cache, last_known_context
    context = {"cities": [], "dates": [], "airlines": [], "times": [], "classes": []}
    
    # Process current input
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "GPE":
            city = ent.text.lower()
            if city not in context["cities"]:
                context["cities"].append(city)
        if ent.label_ == "DATE" and ent.text not in context["dates"]:
            context["dates"].append(ent.text)
        if ent.label_ == "ORG" and ent.text.lower() in ["delta", "united", "american", "southwest"] and ent.text not in context["airlines"]:
            context["airlines"].append(ent.text)
    for token in doc:
        if token.lower_ in ["morning", "afternoon", "evening", "before 9 am", "after 5 pm"] and token.text not in context["times"]:
            context["times"].append(token.text)
        if token.lower_ in ["economy", "business", "first class", "premium"] and token.text not in context["classes"]:
            context["classes"].append(token.text)
    
    # Explicitly check for airlines in text to improve extraction
    text_lower = text.lower()
    known_airlines = ["delta", "united", "american", "southwest"]
    for airline in known_airlines:
        if airline in text_lower and airline not in context["airlines"]:
            context["airlines"].append(airline)
    
    # Include entities from recent history (last 3 messages, prioritize current input)
    history_context = {"cities": [], "dates": [], "airlines": [], "times": [], "classes": []}
    for msg in history[-3:]:
        # Skip bot responses to avoid extracting entities from generated text
        if msg.startswith("Bot:"):
            continue
        # Remove "User:" prefix to improve spaCy entity extraction
        msg_cleaned = msg.replace("User: ", "")
        # Check if we've cached entities for this cleaned message
        if msg_cleaned in entity_cache:
            cached_entities = entity_cache[msg_cleaned]
            for key in history_context:
                history_context[key].extend(cached_entities[key])
            continue
        # Extract entities and cache them
        doc = nlp(msg_cleaned)
        entities = {"cities": [], "dates": [], "airlines": [], "times": [], "classes": []}
        for ent in doc.ents:
            if ent.label_ == "GPE":
                city = ent.text.lower()
                if city not in entities["cities"]:
                    entities["cities"].append(city)
            if ent.label_ == "DATE" and ent.text not in entities["dates"]:
                entities["dates"].append(ent.text)
            if ent.label_ == "ORG" and ent.text.lower() in ["delta", "united", "american", "southwest"] and ent.text not in entities["airlines"]:
                entities["airlines"].append(ent.text)
        for token in doc:
            if token.lower_ in ["morning", "afternoon", "evening", "before 9 am", "after 5 pm"] and token.text not in entities["times"]:
                entities["times"].append(token.text)
            if token.lower_ in ["economy", "business", "first class", "premium"] and token.text not in entities["classes"]:
                entities["classes"].append(token.text)
        # Explicitly check for airlines in history message
        msg_lower = msg_cleaned.lower()
        for airline in known_airlines:
            if airline in msg_lower and airline not in entities["airlines"]:
                entities["airlines"].append(airline)
        entity_cache[msg_cleaned] = entities
        for key in history_context:
            history_context[key].extend(entities[key])
    
    # Combine current and history context, prioritizing current input
    for key in context:
        if not context[key]:  # If current input doesn't have this entity, use history
            context[key] = history_context[key]
        elif history_context[key]:  # If both exist, append history entities not in current
            context[key].extend([item for item in history_context[key] if item not in context[key]])
    
    # Deduplicate entities to prevent repeats
    for key in context:
        context[key] = list(dict.fromkeys(context[key]))
    
    # Validate context (handle invalid cities)
    valid_cities = ["boston", "denver", "pittsburgh", "atlanta", "new york", "miami", "chicago", "dallas", "san francisco", "seattle"]
    context["cities"] = [city for city in context["cities"] if city in valid_cities]
    
    # Update last known context
    if context["cities"]:  # Only update if we have valid cities
        if session_id not in last_known_context:
            last_known_context[session_id] = {"cities": [], "dates": [], "airlines": [], "times": [], "classes": []}
        last_known_context[session_id]["cities"] = context["cities"]
        last_known_context[session_id]["dates"] = context["dates"]
        last_known_context[session_id]["airlines"] = context["airlines"]
        last_known_context[session_id]["times"] = context["times"]
        last_known_context[session_id]["classes"] = context["classes"]
    
    # Fallback to last known context if cities are missing and not a new flight request
    if not context["cities"] and "fly to" not in text.lower():
        if session_id in last_known_context and last_known_context[session_id]["cities"]:
            context["cities"] = last_known_context[session_id]["cities"]
    
    return context

# Format context for prompt
def format_context(context):
    parts = []
    if context["cities"]:
        parts.append(f"Cities: {', '.join(context['cities'][:2])}")
    if context["dates"]:
        parts.append(f"Dates: {', '.join(context['dates'][:1])}")
    if context["airlines"]:
        parts.append(f"Airlines: {', '.join(context['airlines'][:1])}")
    if context["times"]:
        # Prioritize the most recent time entity to avoid repetition
        parts.append(f"Times: {context['times'][-1] if context['times'] else ''}")
    if context["classes"]:
        parts.append(f"Classes: {context['classes'][-1] if context['classes'] else ''}")
    return "; ".join(parts) if parts else "No specific context identified"

# Log metrics to JSON file
def log_metrics(metrics):
    try:
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                logs = json.load(f)
        else:
            logs = []
        logs.append(metrics)
        with open(log_file, "w") as f:
            json.dump(logs, f, indent=4)
    except Exception as e:
        print(f"Failed to log metrics to {log_file}: {e}")

# Main chatbot function
def chatbot(user_input, session_id="default"):
    global history_store, last_known_context
    start_time = time.time()
    
    # Ensure user_input is a string
    user_input_str = str(user_input) if user_input is not None else ""
    
    # Initialize session history if not exists
    if session_id not in history_store:
        history_store[session_id] = []

    # Classify intent
    intent, intent_latency = classify_intent(user_input)
    print(f"Detected Intent: {intent}")

    # Get conversation history
    history = history_store[session_id]
    
    # Handle atis_abbreviation with fare code lookup
    if intent == "atis_abbreviation":
        fare_code = re.search(r'fare code\s+([a-zA-Z]+)', user_input.lower())
        fare_code = fare_code.group(1) if fare_code else user_input.lower().split("fare code")[-1].strip().rstrip('?')
        response = fare_code_explanations.get(fare_code, f"Fare code {fare_code.upper()} is not recognized. Please provide a known fare code for an explanation.")
        response_latency = time.time() - start_time
        metrics = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_input": user_input,
            "intent": intent,
            "intent_classification_latency": intent_latency,
            "response_latency": response_latency,
            "response_length": len(response)
        }
        log_metrics(metrics)
        return response
    
    # Extract entities
    context = extract_entities(user_input, history, session_id)
    
    # Validate context (handle invalid cities)
    valid_cities = ["boston", "denver", "pittsburgh", "atlanta", "new york", "miami", "chicago", "dallas", "san francisco", "seattle"]
    # Already validated in extract_entities
    
    # Do not reset history unless the input explicitly indicates a new flight request with invalid cities
    user_input_str = str(user_input) if user_input is not None else ""
    if not context["cities"] and "fly to" in user_input_str.lower():
        context = {"cities": [], "dates": [], "airlines": [], "times": [], "classes": []}
        if session_id in last_known_context:
            last_known_context[session_id] = {"cities": [], "dates": [], "airlines": [], "times": [], "classes": []}
    
    # Fallback to last known context if cities are missing and not a new flight request
    if not context["cities"] and "fly to" not in user_input_str.lower():
        if session_id in last_known_context and last_known_context[session_id]["cities"]:
            context["cities"] = last_known_context[session_id]["cities"]
    
    context_summary = format_context(context)
    
    # Get intent-specific response
    intent_response = intent_responses.get(intent, "I'm not sure how to assist with that. Could you provide more details?")
    
    # Generate response
    prompt = f"""
You are a travel assistant chatbot for the ATIS dataset. Provide a concise, accurate response based on the user's intent and the provided intent-specific response. Strictly use the entities from the conversation context summary (e.g., {context_summary}) to maintain continuity and avoid introducing unrelated entities (e.g., airlines, cities, or dates not mentioned in the current query or recent history). For example, if the context includes cities like Boston and Denver, ensure your response references them and avoids invalid locations like Narnia. Prioritize entities from the current query over historical ones. Incorporate all relevant context entities (cities, dates, airlines, times, classes) from the current query first, then recent history if needed. If the context lacks sufficient entities or includes invalid ones, use the intent-specific response and request clarification with valid options. Do not assume unavailable information (e.g., specific flight counts like '0 flights') or generate incoherent text. Ensure airline entities (e.g., United, Delta) match the query exactly.

Intent: {intent}
Intent Response: {intent_response}
Conversation Context Summary: {context_summary}
Conversation History:
{'\n'.join(history[-4:])}

User Input: {user_input}

Response:
"""
    try:
        response = llm.invoke(prompt).strip()
    except Exception as e:
        print(f"Error generating response with LLM: {e}")
        response = intent_response
    
    # Customize response with context to avoid repetition
    if intent in ["atis_flight", "atis_airfare", "atis_flight_time"]:
        cities = context["cities"][:2]
        dates = context["dates"][:1]
        times = context["times"][-1:]  # Use the most recent time
        classes = context["classes"][-1:]  # Use the most recent class
        if len(cities) >= 2:
            response = f"I can {'find flights' if intent == 'atis_flight' else 'check airfares' if intent == 'atis_airfare' else 'provide flight schedules'} from {cities[0]} to {cities[1]}{' on ' + dates[0] if dates else ''}{' ' + times[0].lower() if times else ''}{' in ' + classes[0].lower() if classes else ''}. Please provide additional details like {'airline or class' if intent == 'atis_flight' else 'travel dates or class' if intent == 'atis_airfare' else 'time or airline'}."
        else:
            response = f"{intent_response} Valid cities include: {', '.join(valid_cities)}."
    elif intent == "atis_airline":
        airlines = context["airlines"][:1]
        cities = context["cities"][:2]
        if airlines and len(cities) >= 2:
            response = f"I can check if {airlines[0]} flies from {cities[0]} to {cities[1]}. Please provide more details about the route or service."
        elif len(cities) >= 2:
            response = f"Which airline are you inquiring about for flights from {cities[0]} to {cities[1]}?"
        else:
            response = f"{intent_response} Please specify the cities for the route. Valid cities include: {', '.join(valid_cities)}."
    elif intent == "atis_ground_service":
        cities = context["cities"][:1]
        if cities:
            response = f"I can provide information on ground transportation in {cities[0]}. Please specify the airport or city details."
        else:
            response = f"{intent_response} Valid cities include: {', '.join(valid_cities)}."
    elif intent == "atis_quantity":
        airlines = context["airlines"][:1]
        cities = context["cities"][:2]
        if airlines and len(cities) >= 1:
            response = f"Please provide more details to check the number of {airlines[0]} flights from {cities[0]}{' to ' + cities[1] if len(cities) >= 2 else ''}."
        else:
            response = f"{intent_response} Please specify the airline and cities. Valid cities include: {', '.join(valid_cities)}."
    elif intent == "atis_aircraft":
        cities = context["cities"][:2]
        if len(cities) >= 2:
            response = f"Please specify the flight from {cities[0]} to {cities[1]} to check the aircraft type."
        else:
            response = f"{intent_response} Valid cities include: {', '.join(valid_cities)}."
    
    # Log metrics
    response_latency = time.time() - start_time
    metrics = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_input": user_input,
        "intent": intent,
        "intent_classification_latency": intent_latency,
        "response_latency": response_latency,
        "response_length": len(response)
    }
    log_metrics(metrics)
    
    return response