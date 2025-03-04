# English-Vietnamese Translator

This project is an English-Vietnamese Translator that utilizes a machine learning model to translate text from English to Vietnamese.

## Features

- Train a translation model with customizable epochs.
- Save the trained model for future use.
- Translate English text to Vietnamese.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```bash
   cd <project-directory>
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Initialize the translator:
   ```python
   from translator import EnglishVietnameseTranslator
   translator = EnglishVietnameseTranslator()
   ```

2. Train the model:
   ```python
   translator.train(epochs=4)
   ```

3. Save the model:
   ```python
   translator.save_model()
   ```

4. Translate a sentence:
   ```python
   text = "Hello, how are you?"
   translation = translator.translate(text)
   print(f"Input: {text}")
   print(f"Translation: {translation}")
   ```
