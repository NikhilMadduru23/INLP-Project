import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, TrainerCallback
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import json
import os
from multiprocessing import freeze_support
import wandb
import matplotlib.pyplot as plt
import numpy as np

# Check for SentencePiece
try:
    import sentencepiece
except ImportError:
    raise ImportError("SentencePiece is required for T5Tokenizer. Install it with: `pip install sentencepiece`")

# Expanded synthetic ATIS dialogue dataset (~100 examples)
dialogues = [
    {"input": "I want to fly from Boston to Denver tomorrow", "output": "I can find flights from Boston to Denver on tomorrow. Please provide additional details like time or airline."},
    {"input": "Morning flight, before 9 AM", "output": "I can find flights from Boston to Denver on tomorrow before 9 AM. Please provide additional details like airline or class."},
    {"input": "Any nonstop flights?", "output": "I can find nonstop flights from Boston to Denver on tomorrow. Please provide additional details like time or airline."},
    {"input": "Business class available?", "output": "I can find flights from Boston to Denver on tomorrow in business class. Please provide additional details like airline."},
    {"input": "What is fare code Q?", "output": "Fare code Q is an economy class fare, typically discounted with restrictions such as advance purchase or non-refundability."},
    {"input": "Explain fare code F", "output": "Fare code F is a first-class fare, offering premium services and flexibility."},
    {"input": "What’s fare code XYZ?", "output": "Fare code XYZ is not recognized. Please provide a known fare code for an explanation."},
    {"input": "Cheapest airfare from Pittsburgh to Atlanta next week", "output": "I can check airfares from Pittsburgh to Atlanta for next week. Please provide specific travel dates or class."},
    {"input": "Round-trip fares under $300", "output": "I can check round-trip airfares from Pittsburgh to Atlanta for next week under $300. Please provide specific travel dates."},
    {"input": "Any deals for economy class?", "output": "I can check economy class airfares from Pittsburgh to Atlanta for next week. Please provide specific travel dates."},
    {"input": "Which airlines fly from New York to Miami?", "output": "Which airline are you inquiring about for flights from New York to Miami? Options include Delta, United, American, or Southwest."},
    {"input": "Does Delta operate on this route?", "output": "I can check if Delta flies from New York to Miami. Please provide more details about the route or service."},
    {"input": "Is United available?", "output": "I can check if United flies from New York to Miami. Please provide more details about the route or service."},
    {"input": "What type of aircraft is used for flights from Chicago to Dallas?", "output": "Please specify the flight from Chicago to Dallas to check the aircraft type."},
    {"input": "Flight schedules from San Francisco to Seattle on Monday", "output": "I can provide flight schedules from San Francisco to Seattle on Monday. Please specify additional details like time or airline."},
    {"input": "Earliest flight available", "output": "I can provide the earliest flight schedules from San Francisco to Seattle on Monday. Please specify additional details like airline."},
    {"input": "Ground transportation options in Denver airport", "output": "I can provide information on ground transportation options in Denver, such as taxis or shuttles. Please specify the airport or city details."},
    {"input": "Are there taxis available?", "output": "I can provide information on taxi availability in Denver. Please specify the airport or city details."},
    {"input": "How many flights does Delta operate from Atlanta?", "output": "Please provide more details to check the number of Delta flights from Atlanta, such as the destination city or date."},
    {"input": "Number of daily flights to Chicago", "output": "Please provide more details to check the number of Delta flights from Atlanta to Chicago, such as the date."},
    {"input": "Fly to Narnia", "output": "I can help you find flights. Please specify a valid departure city, destination, and travel date. Valid cities include: boston, denver, pittsburgh, atlanta, new york, miami, chicago, dallas, san francisco, seattle."},
    {"input": "What’s the cheapest way to get to Dallas?", "output": "I can check airfares to Dallas. Please provide the departure city and travel date. Valid cities include: boston, denver, pittsburgh, atlanta, new york, miami, chicago, dallas, san francisco, seattle."},
    {"input": "I need a flight ASAP", "output": "I can help you find flights as soon as possible. Please specify the departure city, destination, and travel date. Valid cities include: boston, denver, pittsburgh, atlanta, new york, miami, chicago, dallas, san francisco, seattle."},
    {"input": "Does American Airlines fly from Chicago to Dallas?", "output": "I can check if American Airlines flies from Chicago to Dallas. Please provide more details about the route or service."},
    {"input": "What’s the earliest flight from Boston to Miami?", "output": "I can provide flight schedules from Boston to Miami. Please specify the date and additional details like time or airline."},
    {"input": "Is there a shuttle service in Atlanta airport?", "output": "I can provide information on shuttle services in Atlanta. Please specify the airport or city details."},
    {"input": "Number of weekly flights to Seattle", "output": "Please provide more details to check the number of flights to Seattle, such as the departure city and airline."},
    {"input": "Fly from Narnia to Hogwarts", "output": "I can help you find flights. Please specify a valid departure city and destination. Valid cities include: boston, denver, pittsburgh, atlanta, new york, miami, chicago, dallas, san francisco, seattle."},
    {"input": "I want a flight from Boston to Denver next week", "output": "I can find flights from Boston to Denver for next week. Please provide additional details like time or airline."},
    {"input": "Evening flight, after 5 PM", "output": "I can find flights from Boston to Denver for next week after 5 PM. Please provide additional details like airline or class."},
    {"input": "How many flights to Miami from Atlanta?", "output": "Please provide more details to check the number of flights from Atlanta to Miami, such as the airline or date."},
    {"input": "Is Southwest available for flights from Dallas to Chicago?", "output": "I can check if Southwest flies from Dallas to Chicago. Please provide more details about the route or service."},
    {"input": "Economy class flights from Miami to Seattle", "output": "I can find flights from Miami to Seattle in economy class. Please provide additional details like date or airline."},
    {"input": "Change to first class", "output": "I can find flights from Miami to Seattle in first class. Please provide additional details like date or airline."},
    {"input": "How many United flights from New York to Miami?", "output": "Please provide more details to check the number of United flights from New York to Miami, such as the date."},
    {"input": "Flight from Chicago to nowhere", "output": "I can help you find flights. Please specify a valid departure city and destination. Valid cities include: boston, denver, pittsburgh, atlanta, new york, miami, chicago, dallas, san francisco, seattle."},
    {"input": "I want a flight from Miami to Chicago on Friday", "output": "I can find flights from Miami to Chicago on Friday. Please provide additional details like time or airline."},
    {"input": "Any flights after 6 PM?", "output": "I can find flights from Miami to Chicago on Friday after 6 PM. Please provide additional details like airline or class."},
    {"input": "How many daily flights does Southwest operate from Dallas?", "output": "Please provide more details to check the number of Southwest flights from Dallas, such as the destination city."},
    {"input": "I want a flight from Chicago to Seattle on Wednesday", "output": "I can find flights from Chicago to Seattle on Wednesday. Please provide additional details like time or airline."},
    {"input": "Any flights tomorrow evening?", "output": "I can find flights from Chicago to Seattle on tomorrow evening. Please provide additional details like airline or class."},
    {"input": "Is Delta available for this route?", "output": "I can check if Delta flies from Chicago to Seattle. Please provide more details about the route or service."},
    {"input": "Flight from New York to Dallas on 2025-05-10", "output": "I can find flights from New York to Dallas on 2025-05-10. Please provide additional details like time or airline."},
    {"input": "Any flights before noon?", "output": "I can find flights from New York to Dallas on 2025-05-10 before noon. Please provide additional details like airline or class."},
    {"input": "How many American Airlines flights from Miami to Boston?", "output": "Please provide more details to check the number of American Airlines flights from Miami to Boston, such as the date."},
    {"input": "I want a flight from San Francisco to Chicago next Monday", "output": "I can find flights from San Francisco to Chicago on next Monday. Please provide additional details like time or airline."},
    {"input": "Any premium economy flights?", "output": "I can find premium economy flights from San Francisco to Chicago on next Monday. Please provide additional details like airline."},
    {"input": "Is United operating flights from Denver to Atlanta?", "output": "I can check if United flies from Denver to Atlanta. Please provide more details about the route or service."},
    {"input": "What’s the cheapest flight from Seattle to Pittsburgh next month?", "output": "I can check airfares from Seattle to Pittsburgh for next month. Please provide specific travel dates or class."},
    {"input": "Any nonstop flights under $200?", "output": "I can check nonstop airfares from Seattle to Pittsburgh for next month under $200. Please provide specific travel dates."},
    {"input": "How many weekly flights does Southwest operate from Chicago?", "output": "Please provide more details to check the number of Southwest flights from Chicago, such as the destination city."},
    {"input": "Flight from Atlanta to nowhere", "output": "I can help you find flights. Please specify a valid departure city and destination. Valid cities include: boston, denver, pittsburgh, atlanta, new york, miami, chicago, dallas, san francisco, seattle."},
    {"input": "I want a flight from Dallas to Miami tomorrow morning", "output": "I can find flights from Dallas to Miami on tomorrow morning. Please provide additional details like airline or class."},
    {"input": "Any flights with American Airlines?", "output": "I can find flights from Dallas to Miami on tomorrow morning with American Airlines. Please provide additional details like class."},
    {"input": "What’s the earliest flight from Pittsburgh to San Francisco on Tuesday?", "output": "I can provide flight schedules from Pittsburgh to San Francisco on Tuesday. Please specify additional details like airline."},
    {"input": "Any business class deals?", "output": "I can find business class flights from Pittsburgh to San Francisco on Tuesday. Please specify additional details like airline."},
    {"input": "How many flights to Denver from Boston next week?", "output": "Please provide more details to check the number of flights from Boston to Denver next week, such as the airline or date."},
    {"input": "Is there a flight from Miami to New York on 2025-05-15?", "output": "I can find flights from Miami to New York on 2025-05-15. Please provide additional details like time or airline."},
    {"input": "Any evening flights after 7 PM?", "output": "I can find flights from Miami to New York on 2025-05-15 after 7 PM. Please provide additional details like airline or class."},
    {"input": "Ground transportation options in Seattle airport", "output": "I can provide information on ground transportation options in Seattle, such as taxis or shuttles. Please specify the airport or city details."},
    {"input": "Are there rental cars available?", "output": "I can provide information on rental car availability in Seattle. Please specify the airport or city details."},
    {"input": "What is fare code C?", "output": "Fare code C is a business class fare, offering enhanced services with flexibility."},
    {"input": "Explain fare code Y", "output": "Fare code Y is a full-fare economy class ticket, often fully refundable and flexible."},
    {"input": "I want a flight from Denver to Chicago next Friday", "output": "I can find flights from Denver to Chicago on next Friday. Please provide additional details like time or airline."},
    {"input": "Any flights after 3 PM?", "output": "I can find flights from Denver to Chicago on next Friday after 3 PM. Please provide additional details like airline or class."},
    {"input": "How many Delta flights from Dallas to Atlanta daily?", "output": "Please provide more details to check the number of Delta flights from Dallas to Atlanta daily, such as the date."},
    {"input": "Flight from Boston to nowhere", "output": "I can help you find flights. Please specify a valid departure city and destination. Valid cities include: boston, denver, pittsburgh, atlanta, new york, miami, chicago, dallas, san francisco, seattle."},
    {"input": "Cheapest flight from Chicago to Miami next month", "output": "I can check airfares from Chicago to Miami for next month. Please provide specific travel dates or class."},
    {"input": "Any economy deals under $150?", "output": "I can check economy airfares from Chicago to Miami for next month under $150. Please provide specific travel dates."},
    {"input": "Is American Airlines available for flights from Seattle to Dallas?", "output": "I can check if American Airlines flies from Seattle to Dallas. Please provide more details about the route or service."},
    {"input": "What’s the earliest flight from Atlanta to Denver on 2025-05-20?", "output": "I can provide flight schedules from Atlanta to Denver on 2025-05-20. Please specify additional details like time or airline."},
    {"input": "Any nonstop flights before 10 AM?", "output": "I can find nonstop flights from Atlanta to Denver on 2025-05-20 before 10 AM. Please provide additional details like airline or class."},
    {"input": "How many United flights from Chicago to San Francisco weekly?", "output": "Please provide more details to check the number of United flights from Chicago to San Francisco weekly, such as the date."},
    {"input": "I want a flight from New York to Seattle on next Tuesday", "output": "I can find flights from New York to Seattle on next Tuesday. Please provide additional details like time or airline."},
    {"input": "Any first class flights?", "output": "I can find first class flights from New York to Seattle on next Tuesday. Please provide additional details like airline."},
    {"input": "Is Southwest operating flights from Miami to Atlanta?", "output": "I can check if Southwest flies from Miami to Atlanta. Please provide more details about the route or service."},
    {"input": "Ground transportation options in Chicago airport", "output": "I can provide information on ground transportation options in Chicago, such as taxis or shuttles. Please specify the airport or city details."},
    {"input": "Are there buses available?", "output": "I can provide information on bus availability in Chicago. Please specify the airport or city details."},
    {"input": "What is fare code H?", "output": "Fare code H is an economy class fare with moderate restrictions, often used for mid-tier pricing."},
    {"input": "Explain fare code QW", "output": "Fare code QW is a promotional economy fare, usually with strict conditions like limited availability."}
]

# Save dataset
dataset_path = "atis_dialogues.json"
with open(dataset_path, "w") as f:
    json.dump(dialogues, f, indent=4)

def main():
    # Initialize WandB
    wandb.init(project="atis_chatbot_lora", name="flan-t5-base-cpu-lora-finetuning", config={
        "model": "google/flan-t5-base",
        "dataset": "atis_dialogues",
        "num_epochs": 5,
        "batch_size": 1,
        "lora_rank": 32
    })

    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        # Load dataset
        with open(dataset_path, "r") as f:
            dialogues = json.load(f)
        dataset = Dataset.from_list(dialogues)

        # Load model and tokenizer
        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
        model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base").to(device)

        # Apply LoRA
        lora_config = LoraConfig(
            r=32,
            lora_alpha=32,
            target_modules=["q", "v", "k", "o"],
            lora_dropout=0.1
        )
        model = get_peft_model(model, lora_config)

        # Tokenize dataset
        def tokenize_function(examples):
            inputs = tokenizer(examples["input"], padding="max_length", truncation=True, max_length=128)
            outputs = tokenizer(examples["output"], padding="max_length", truncation=True, max_length=128)
            inputs["labels"] = outputs["input_ids"]
            return inputs

        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        # Custom callback for loss logging
        class LossLoggingCallback(TrainerCallback):
            def __init__(self):
                self.train_losses = []
                self.steps = []

            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs.get("loss") is not None:
                    self.train_losses.append(logs["loss"])
                    self.steps.append(state.global_step)
                    wandb.log({"train/loss": logs["loss"], "step": state.global_step})

            def on_train_end(self, args, state, control, **kwargs):
                # Plot and save training loss
                plt.figure(figsize=(10, 6))
                plt.plot(self.steps, self.train_losses, label="Training Loss", color='blue')
                plt.xlabel("Training Steps")
                plt.ylabel("Loss")
                plt.title("Training Loss Over Time (Flan-T5-Base CPU)")
                plt.legend()
                plt.grid(True)
                plot_path = "./plots/lora_training_loss_plot_cpu.png"
                plt.savefig(plot_path)
                plt.close()
                wandb.log({"training_loss_plot": wandb.Image(plot_path)})
                print(f"Training loss plot saved to {plot_path}")

        # Training arguments
        training_args = TrainingArguments(
            output_dir="./cpu_lora_output",
            per_device_train_batch_size=1,
            num_train_epochs=5,
            save_steps=500,
            save_total_limit=2,
            logging_steps=10,
            learning_rate=1e-4,
            fp16=False,  # Disable FP16 for CPU compatibility during inference
            report_to="wandb",
            dataloader_num_workers=2 if torch.cuda.is_available() else 0
        )

        # Train
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            callbacks=[LossLoggingCallback()]
        )

        print("Starting LoRA fine-tuning for Flan-T5-Base (CPU)...")
        trainer.train()
        print("Fine-tuning completed.")

        # Save LoRA adapters
        model.save_pretrained("./cpu_lora_adapters")
        tokenizer.save_pretrained("./cpu_lora_adapters")
        print(f"LoRA adapters saved to ./cpu_lora_adapters")

        # Finish WandB run
        wandb.finish()

    except Exception as e:
        print(f"Error during LoRA fine-tuning: {e}")
        print("Ensure SentencePiece is installed (`pip install sentencepiece`) and sufficient resources are available.")
        print("For GPU usage, verify CUDA with: `python -c \"import torch; print(torch.cuda.is_available(), torch.version.cuda)\"`")
        wandb.finish()

if __name__ == '__main__':
    freeze_support()
    main()