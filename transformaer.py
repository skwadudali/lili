
!pip install transformers

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the tokenizer and model directly
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ar")

# Function to translate English text (can be a single string or a list of strings) to Arabic
def translate_en_to_ar(texts):
    if isinstance(texts, str):
        texts = [texts] # Convert single string to a list for consistent processing

    # Prepare the input for the model
    batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

    # Generate translations
    translated_tokens = model.generate(**batch)

    # Decode the generated tokens back to text
    translated_texts = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
    return translated_texts

# Example usage with a single sentence:
english_text_single = "Hello, how are you?"
arabic_text_single = translate_en_to_ar(english_text_single)
print(f"Single English: {english_text_single}")
print(f"Single Arabic: {arabic_text_single[0]}\n")

# Example usage with a list of sentences:
english_texts_list = [
    "This is a test sentence for translation.",
    "How are you doing today?",
    "The weather is beautiful."
]
arabic_texts_list = translate_en_to_ar(english_texts_list)

for i, (en_text, ar_text) in enumerate(zip(english_texts_list, arabic_texts_list)):
    print(f"English {i+1}: {en_text}")
    print(f"Arabic {i+1}: {ar_text}\n")
