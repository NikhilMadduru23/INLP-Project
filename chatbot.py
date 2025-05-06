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

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_id = 0 if torch.cuda.is_available() else -1
print(f"Using device: {device}")

# Ensure logs directory exists
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "chatbot_metrics.json")

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

# Initialize HuggingFacePipeline with google/flan-t5-xl and LoRA adapters
lora_adapter_path = "./lora_adapters"
if os.path.exists(lora_adapter_path):
    print("Loading LoRA adapters...")
    try:
        t5_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl")
        t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
        t5_model = PeftModel.from_pretrained(t5_model, lora_adapter_path)
        t5_model.to(device)
        base_model = t5_model.get_base_model()
    except Exception as e:
        print(f"Failed to load LoRA adapters: {e}. Falling back to base google/flan-t5-xl...")
        t5_model = None
        t5_tokenizer = None
        base_model = None
else:
    print("No LoRA adapters found, using base google/flan-t5-xl...")
    t5_model = None
    t5_tokenizer = None
    base_model = None

llm = HuggingFacePipeline(pipeline=pipeline(
    "text2text-generation",
    model=base_model if base_model else "google/flan-t5-xl",
    tokenizer=t5_tokenizer if t5_tokenizer else "google/flan-t5-xl",
    max_length=200,
    device=device_id,
    model_kwargs={"torch_dtype": torch.float16} if torch.cuda.is_available() else {}
))

# Enable summarization pipeline
try:
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device_id)
except Exception as e:
    print(f"Failed to load summarization pipeline: {e}")
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
    elif any(keyword in text_lower for keyword in ["airline", "delta", "united", "american", "southwest", "route"]):
        intent = "atis_airline"
    elif any(keyword in text_lower for keyword in ["schedule", "schedules", "earliest", "flight time"]):
        intent = "atis_flight_time"
    elif any(keyword in text_lower for keyword in ["ground", "transportation", "taxi", "shuttle", "airport"]):
        intent = "atis_ground_service"
    elif any(keyword in text_lower for keyword in ["fly", "flight", "nonstop", "class", "morning", "evening", "asap"]):
        intent = "atis_flight"
    latency = time.time() - start_time
    return intent, latency

# Extract entities using spaCy NER
def extract_entities(text, history):
    context = {"cities": [], "dates": [], "airlines": [], "times": [], "classes": []}
    # Process current input
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "GPE":
            context["cities"].append(ent.text)
        if ent.label_ == "DATE":
            context["dates"].append(ent.text)
        if ent.label_ == "ORG" and ent.text.lower() in ["delta", "united", "american", "southwest"]:
            context["airlines"].append(ent.text)
    for token in doc:
        if token.lower_ in ["morning", "afternoon", "evening", "before 9 am", "after 5 pm"]:
            context["times"].append(token.text)
        if token.lower_ in ["economy", "business", "first class", "premium"]:
            context["classes"].append(token.text)
    
    # Include entities from recent history (last 3 messages, prioritize current input)
    for msg in history[-3:]:
        doc = nlp(msg)
        for ent in doc.ents:
            if ent.label_ == "GPE" and ent.text not in context["cities"]:
                context["cities"].append(ent.text)
            if ent.label_ == "DATE" and ent.text not in context["dates"]:
                context["dates"].append(ent.text)
            if ent.label_ == "ORG" and ent.text.lower() in ["delta", "united", "american", "southwest"] and ent.text not in context["airlines"]:
                context["airlines"].append(ent.text)
        for token in doc:
            if token.lower_ in ["morning", "afternoon", "evening", "before 9 am", "after 5 pm"] and token.text not in context["times"]:
                context["times"].append(token.text)
            if token.lower_ in ["economy", "business", "first class", "premium"] and token.text not in context["classes"]:
                context["classes"].append(token.text)
    
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
        parts.append(f"Times: {', '.join(context['times'][:1])}")
    if context["classes"]:
        parts.append(f"Classes: {', '.join(context['classes'][:1])}")
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
    global history_store
    start_time = time.time()
    
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
        history_store[session_id].append(f"User: {user_input}")
        history_store[session_id].append(f"Bot: {response}")
        return response
    
    # Extract entities
    context = extract_entities(user_input, history)
    
    # Validate context (handle invalid cities)
    valid_cities = ["boston", "denver", "pittsburgh", "atlanta", "new york", "miami", "chicago", "dallas", "san francisco", "seattle"]
    context["cities"] = [city for city in context["cities"] if city.lower() in valid_cities]
    
    # Retain history entries with relevant entities
    if context["cities"] or context["dates"] or context["airlines"]:
        history_entities = set()
        for msg in history:
            doc = nlp(msg)
            for ent in doc.ents:
                if ent.label_ in ["GPE", "DATE", "ORG"]:
                    history_entities.add(ent.text.lower())
        history_store[session_id] = [msg for msg in history_store[session_id][-4:] if "User: " not in msg or any(ent in msg.lower() for ent in history_entities)]
    else:
        history_store[session_id] = history_store[session_id][-4:]
    
    # Use summarization to improve context handling
    if summarizer and history:
        try:
            history_text = '\n'.join(history[-4:])
            history_summary = summarizer(history_text, max_length=50, min_length=10, do_sample=False)[0]['summary_text']
        except Exception as e:
            print(f"Error summarizing history: {e}")
            history_summary = '\n'.join(history[-4:])
    else:
        history_summary = '\n'.join(history[-4:])
    
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
    
    # Customize response with context
    if intent in ["atis_flight", "atis_airfare", "atis_flight_time"]:
        cities = context["cities"][:2]
        dates = context["dates"][:1]
        times = context["times"][:1]
        classes = context["classes"][:1]
        if len(cities) >= 2:
            response = f"I can {'find flights' if intent == 'atis_flight' else 'check airfares' if intent == 'atis_airfare' else 'provide flight schedules'} from {cities[0]} to {cities[1]}{' on ' + dates[0] if dates else ''}{' ' + times[0] if times else ''}{' in ' + classes[0] if classes else ''}. Please provide additional details like {'airline or class' if intent == 'atis_flight' else 'travel dates or class' if intent == 'atis_airfare' else 'time or airline'}."
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

    # Update history
    history_store[session_id].append(f"User: {user_input}")
    history_store[session_id].append(f"Bot: {response}")
    
    return response

# In-memory history store
history_store = {}

# Example interaction loop with test examples
def main():
    print("Welcome to the Travel Assistant Chatbot! Type 'exit' to quit.")
    session_id = "default"
    test_examples = [
        # atis_flight (multi-turn)
        {"input": "I want to fly from Boston to Denver tomorrow"},
        {"input": "Morning flight, before 9 AM"},
        {"input": "Any nonstop flights?"},
        {"input": "Business class available?"},
        # atis_abbreviation
        {"input": "What is fare code Q?"},
        {"input": "Explain fare code F"},
        {"input": "What’s fare code XYZ?"},
        # atis_airfare
        {"input": "Cheapest airfare from Pittsburgh to Atlanta next week"},
        {"input": "Round-trip fares under $300"},
        {"input": "Any deals for economy class?"},
        # atis_airline
        {"input": "Which airlines fly from New York to Miami?"},
        {"input": "Does Delta operate on this route?"},
        {"input": "Is United available?"},
        # atis_aircraft
        {"input": "What type of aircraft is used for flights from Chicago to Dallas?"},
        # atis_flight_time
        {"input": "Flight schedules from San Francisco to Seattle on Monday"},
        {"input": "Earliest flight available"},
        # atis_ground_service
        {"input": "Ground transportation options in Denver airport"},
        {"input": "Are there taxis available?"},
        # atis_quantity
        {"input": "How many flights does Delta operate from Atlanta?"},
        {"input": "Number of daily flights to Chicago"},
        # Edge cases
        {"input": "Fly to Narnia"},
        {"input": "What’s the cheapest way to get to Dallas?"},
        {"input": "I need a flight ASAP"}
    ]
    
    print("\nRunning test examples:")
    for example in test_examples:
        user_input = example["input"]
        print(f"\nYou: {user_input}")
        response = chatbot(user_input, session_id)
        print(f"Bot: {response}")
    
    print("\nInteractive mode:")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        response = chatbot(user_input, session_id)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()