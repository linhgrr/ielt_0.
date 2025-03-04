# Khởi tạo translator
from translator import EnglishVietnameseTranslator

translator = EnglishVietnameseTranslator()

# Huấn luyện mô hình
translator.train(epochs=4)

# Lưu mô hình
translator.save_model()

# Dịch một câu
text = "Hello, how are you?"
translation = translator.translate(text)
print(f"Input: {text}")
print(f"Translation: {translation}")