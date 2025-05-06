ATIS Chatbot
============

Overview
--------

The ATIS Chatbot is a Flask-based web app that helps users query flight information using the ATIS dataset. It uses NLP models (google/flan-t5-base with LoRA adapters for responses and DistilBERT for intent classification) optimized for CPU usage. The chatbot supports multi-turn conversations, context retention, and a modern UI, ideal for presentations on non-GPU laptops.

### Features

-   Classifies user intents (e.g., flight requests, ground services, fare codes).

-   Generates natural language responses with context retention.

-   Modern UI with aligned chat bubbles (user on the right, bot on the left) and a "Clear Chat" icon.

Project Structure
-----------------

```
project_root/
├── app/
├── classification_logs/
├── cpu_lora_adapters/
├── cpu_lora_output/
├── data/
├── flask_session/
├── logs/
├── lora_adapters/
├── lora_output/
├── metrics/
├── plots/
├── results/
├── saved_models/
├── venv/
├── wandb/
├── base_classifier.py
├── chatbot.py
├── chatbot_cpu.py
├── data_loader.py
├── train_cpu_lora.py
├── train_lora.py
├── transfer_learning.py
├── requirements.txt
├── atis_dialogues.json
├── .gitignore
└── README.md
```

Setup Instructions
------------------

### Prerequisites

-   Python 3.8+

-   Non-GPU laptop

### Steps

1.  **Clone the Repository**:

    -   Clone the project from the [repository](https://github.com/NikhilMadduru23/INLP-Project) 

    -   Alternatively, download the project dependencies from [Google Drive Link](https://drive.google.com/file/d/1YBc75cnCu1V2LMTWBKLlcwz3qmrsxeGu/view?usp=sharing)

2.  **Download Models and Data**:

    -   From the Google Drive link above, download and unzip:

        ```
        unzip inlp_project_dependencies.zip -d ./
        ```

3.  **Set Up the Virtual Environment**:

    -   Create and activate a virtual environment:

        ```
        python -m venv venv
        source venv/bin/activate  # On Windows: venv\Scripts\activate
        ```

    -   Install dependencies (ensures torch is installed for CPU-based inference):

        ```
        pip install -r requirements.txt
        python -m spacy download en_core_web_sm
        ```

    -   **Note**: If torch is not installed correctly for CPU usage, you can install it explicitly:

        ```
        pip install torch --index-url https://download.pytorch.org/whl/cpu
        ```

4.  **Set the Flask Secret Key**:

    -   **On Unix/Linux/Mac**:

        ```
        export FLASK_SECRET_KEY='your-secret-key'
        ```

    -   **On Windows**:

        ```
        set FLASK_SECRET_KEY='your-secret-key'
        ```

5.  **Run the Application**:

    -   Start the Flask app:

        ```
        python app/app.py
        ```

    -   Open http://localhost:5000 in your browser.

Usage
-----

1.  **Interact with the Chatbot**:

    -   Type a query (e.g., "I want to fly from Boston to Denver on May 15th").

    -   Press the "Send" button (paper plane icon) or hit Enter.

    -   The chatbot will respond or ask for more details.

2.  **Clear the Chat**:

    -   Click the trash can icon in the top-right corner to clear the chat history.

3.  **Offline Mode**:

    -   Disconnect from the internet after setup to demo offline.

### Example Interactions

-   **Flight Booking**:

    -   User: "I want to fly from Boston to Denver on May 15th"

    -   Bot: "I can find flights from boston to denver on May 15th. Please provide additional details like airline or class."

-   **Ground Service Query**:

    -   User: "What ground transportation is available in Chicago?"

    -   Bot: "I can provide information on ground transportation in chicago. Please specify the airport or city details."