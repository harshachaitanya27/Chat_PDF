# 📄 AI-Powered PDF Q&A

This project allows users to **upload multiple PDFs, extract text, and ask AI-powered questions** using **LangChain, ChromaDB, and OpenAI embeddings**. The UI is built with **Streamlit**, making it easy to use.

---

## 🚀 Features
- 📂 **Upload multiple PDFs**  
- 🧠 **Automatically extracts & indexes text** into **ChromaDB**  
- 🔍 **Ask AI-based questions** and get responses from uploaded PDFs  
- 🗑 **Clear database** and delete all stored PDFs when done  
- 💡 **Uses LangChain's `Ollama` model** for AI-powered responses  

---

## 🛠 Installation

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/yourusername/pdf-ai-chatbot.git
cd pdf-ai-chatbot
```

### **2️⃣ Clone the Repository**
```
python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows
```
### **3️⃣ Install Dependencies**
```
pip install -r requirements.txt
```

### **4️⃣ Set Up API Key**
```
touch .env

(Add your API key)
OPENAI_API_KEY=your-api-key-here
```

### **Start the Streamlit App**
```
streamlit run app.py 
```

Upload PDFs
Click "Process PDFs" to extract and store text
Ask AI-powered questions in the chatbox
