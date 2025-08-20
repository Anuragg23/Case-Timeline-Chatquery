import os
from pdf2image import convert_from_path
import pytesseract
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from tqdm import tqdm
import re
from datetime import datetime
from typing import Optional, Dict, List
import json

# Load .env
load_dotenv()

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Poppler path and document folder
POPLER_PATH = r"C:\Users\vidit\AppData\Local\Microsoft\WinGet\Packages\oschwartz10612.Poppler_Microsoft.Winget.Source_8wekyb3d8bbwe\poppler-24.08.0\Library\bin"
PDF_FOLDER = "./Documents"

# Language configurations
LANGUAGES = {
    "en": {
        "name": "English",
        "code": "en",
        "date_patterns": [
            r"\b([A-Za-z]+ \d{1,2}, \d{4})\b",
            r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{4})\b",
            r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b"
        ],
        "date_formats": ["%B %d, %Y", "%m/%d/%Y", "%Y/%m/%d", "%d/%m/%Y"]
    },
    "hi": {
        "name": "हिंदी (Hindi)",
        "code": "hi",
        "date_patterns": [
            r"\b([A-Za-z]+ \d{1,2}, \d{4})\b",
            r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{4})\b",
            r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b"
        ],
        "date_formats": ["%B %d, %Y", "%m/%d/%Y", "%Y/%m/%d", "%d/%m/%Y"]
    },
    "es": {
        "name": "Español (Spanish)",
        "code": "es",
        "date_patterns": [
            r"\b([A-Za-z]+ \d{1,2}, \d{4})\b",
            r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{4})\b",
            r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b"
        ],
        "date_formats": ["%B %d, %Y", "%m/%d/%Y", "%Y/%m/%d", "%d/%m/%Y"]
    },
    "fr": {
        "name": "Français (French)",
        "code": "fr",
        "date_patterns": [
            r"\b([A-Za-z]+ \d{1,2}, \d{4})\b",
            r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{4})\b",
            r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b"
        ],
        "date_formats": ["%B %d, %Y", "%m/%d/%Y", "%Y/%m/%d", "%d/%m/%Y"]
    },
    "de": {
        "name": "Deutsch (German)",
        "code": "de",
        "date_patterns": [
            r"\b([A-Za-z]+ \d{1,2}, \d{4})\b",
            r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{4})\b",
            r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b"
        ],
        "date_formats": ["%B %d, %Y", "%m/%d/%Y", "%Y/%m/%d", "%d/%m/%Y"]
    },
    "zh": {
        "name": "中文 (Chinese)",
        "code": "zh",
        "date_patterns": [
            r"\b([A-Za-z]+ \d{1,2}, \d{4})\b",
            r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{4})\b",
            r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b"
        ],
        "date_formats": ["%B %d, %Y", "%m/%d/%Y", "%Y/%m/%d", "%d/%m/%Y"]
    },
    "ja": {
        "name": "日本語 (Japanese)",
        "code": "ja",
        "date_patterns": [
            r"\b([A-Za-z]+ \d{1,2}, \d{4})\b",
            r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{4})\b",
            r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b"
        ],
        "date_formats": ["%B %d, %Y", "%m/%d/%Y", "%Y/%m/%d", "%d/%m/%Y"]
    }
}

# UI Messages in different languages
UI_MESSAGES = {
    "en": {
        "extracting": "🔍 Extracting text and creating vector store...",
        "choose_option": "📘 Choose an option:",
        "generate_timeline": "1. Generate Timeline",
        "ask_question": "2. Ask a Question (Chat with Memory)",
        "change_language": "3. Change Language",
        "exit": "4. Exit",
        "timeline_output": "📅 TIMELINE OUTPUT:",
        "question_prompt": "\nAsk your legal question (or type 'back' to return to menu): ",
        "exiting": "👋 Exiting.",
        "invalid_choice": "❌ Invalid choice. Try again.",
        "language_selection": "🌐 Select your preferred language:",
        "current_language": "Current language",
        "no_info": "I don't have sufficient information to answer this based on the available documents.",
        "bot_prefix": "🤖"
    },
    "hi": {
        "extracting": "🔍 टेक्स्ट निकाल रहे हैं और वेक्टर स्टोर बना रहे हैं...",
        "choose_option": "📘 एक विकल्प चुनें:",
        "generate_timeline": "1. समयरेखा बनाएं",
        "ask_question": "2. प्रश्न पूछें (मेमोरी के साथ चैट)",
        "change_language": "3. भाषा बदलें",
        "exit": "4. बाहर निकलें",
        "timeline_output": "📅 समयरेखा आउटपुट:",
        "question_prompt": "\nअपना कानूनी प्रश्न पूछें (या मेनू पर वापस जाने के लिए 'back' टाइप करें): ",
        "exiting": "👋 बाहर निकल रहे हैं।",
        "invalid_choice": "❌ गलत विकल्प। फिर से कोशिश करें।",
        "language_selection": "🌐 अपनी पसंदीदा भाषा चुनें:",
        "current_language": "वर्तमान भाषा",
        "no_info": "उपलब्ध दस्तावेजों के आधार पर इसका उत्तर देने के लिए मेरे पास पर्याप्त जानकारी नहीं है।",
        "bot_prefix": "🤖"
    },
    "es": {
        "extracting": "🔍 Extrayendo texto y creando almacén vectorial...",
        "choose_option": "📘 Elige una opción:",
        "generate_timeline": "1. Generar Cronología",
        "ask_question": "2. Hacer una Pregunta (Chat con Memoria)",
        "change_language": "3. Cambiar Idioma",
        "exit": "4. Salir",
        "timeline_output": "📅 SALIDA DE CRONOLOGÍA:",
        "question_prompt": "\nHaz tu pregunta legal (o escribe 'back' para volver al menú): ",
        "exiting": "👋 Saliendo.",
        "invalid_choice": "❌ Opción inválida. Inténtalo de nuevo.",
        "language_selection": "🌐 Selecciona tu idioma preferido:",
        "current_language": "Idioma actual",
        "no_info": "No tengo suficiente información para responder esto basándome en los documentos disponibles.",
        "bot_prefix": "🤖"
    },
    "fr": {
        "extracting": "🔍 Extraction de texte et création du magasin vectoriel...",
        "choose_option": "📘 Choisissez une option:",
        "generate_timeline": "1. Générer la Chronologie",
        "ask_question": "2. Poser une Question (Chat avec Mémoire)",
        "change_language": "3. Changer de Langue",
        "exit": "4. Quitter",
        "timeline_output": "📅 SORTIE DE CHRONOLOGIE:",
        "question_prompt": "\nPosez votre question juridique (ou tapez 'back' pour revenir au menu): ",
        "exiting": "👋 Sortie.",
        "invalid_choice": "❌ Choix invalide. Réessayez.",
        "language_selection": "🌐 Sélectionnez votre langue préférée:",
        "current_language": "Langue actuelle",
        "no_info": "Je n'ai pas suffisamment d'informations pour répondre à cela basé sur les documents disponibles.",
        "bot_prefix": "🤖"
    },
    "de": {
        "extracting": "🔍 Text extrahieren und Vektorspeicher erstellen...",
        "choose_option": "📘 Wählen Sie eine Option:",
        "generate_timeline": "1. Zeitleiste generieren",
        "ask_question": "2. Eine Frage stellen (Chat mit Gedächtnis)",
        "change_language": "3. Sprache ändern",
        "exit": "4. Beenden",
        "timeline_output": "📅 ZEITLEISTEN-AUSGABE:",
        "question_prompt": "\nStellen Sie Ihre rechtliche Frage (oder geben Sie 'back' ein, um zum Menü zurückzukehren): ",
        "exiting": "👋 Beenden.",
        "invalid_choice": "❌ Ungültige Wahl. Versuchen Sie es erneut.",
        "language_selection": "🌐 Wählen Sie Ihre bevorzugte Sprache:",
        "current_language": "Aktuelle Sprache",
        "no_info": "Ich habe nicht genügend Informationen, um dies basierend auf den verfügbaren Dokumenten zu beantworten.",
        "bot_prefix": "🤖"
    },
    "zh": {
        "extracting": "🔍 正在提取文本并创建向量存储...",
        "choose_option": "📘 选择一个选项:",
        "generate_timeline": "1. 生成时间线",
        "ask_question": "2. 提问 (带记忆的聊天)",
        "change_language": "3. 更改语言",
        "exit": "4. 退出",
        "timeline_output": "📅 时间线输出:",
        "question_prompt": "\n提出您的法律问题 (或输入 'back' 返回菜单): ",
        "exiting": "👋 正在退出。",
        "invalid_choice": "❌ 无效选择。请重试。",
        "language_selection": "🌐 选择您的首选语言:",
        "current_language": "当前语言",
        "no_info": "根据可用文档，我没有足够的信息来回答这个问题。",
        "bot_prefix": "🤖"
    },
    "ja": {
        "extracting": "🔍 テキストを抽出してベクターストアを作成中...",
        "choose_option": "📘 オプションを選択してください:",
        "generate_timeline": "1. タイムラインを生成",
        "ask_question": "2. 質問する (メモリ付きチャット)",
        "change_language": "3. 言語を変更",
        "exit": "4. 終了",
        "timeline_output": "📅 タイムライン出力:",
        "question_prompt": "\n法的質問をしてください ('back'と入力してメニューに戻る): ",
        "exiting": "👋 終了中。",
        "invalid_choice": "❌ 無効な選択です。もう一度試してください。",
        "language_selection": "🌐 お好みの言語を選択してください:",
        "current_language": "現在の言語",
        "no_info": "利用可能な文書に基づいて、これに答えるのに十分な情報がありません。",
        "bot_prefix": "🤖"
    }
}

class MultilingualRAGSystem:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vector_store: Optional[FAISS] = None
        self.model = ChatOpenAI(model="gpt-4o", temperature=0.7)
        self.memory: Optional[ConversationSummaryBufferMemory] = None
        self.qa_chain: Optional[ConversationalRetrievalChain] = None
        self.current_language = "en"  # Default language
        self.timeline_cache = None  # Cache timeline across language switches
        
    def set_language(self, language_code: str):
        """Set the current language for responses"""
        if language_code in LANGUAGES:
            self.current_language = language_code
            # Recreate QA chain with new language if it exists
            if self.qa_chain is not None:
                self.setup_conversational_query_system()
        else:
            raise ValueError(f"Language {language_code} not supported")

    def get_ui_message(self, key: str) -> str:
        """Get UI message in current language"""
        return UI_MESSAGES.get(self.current_language, UI_MESSAGES["en"]).get(key, key)

    def extract_text_from_pdfs(self, folder_path):
        all_docs = []
        for file in tqdm(os.listdir(folder_path)):
            if file.endswith(".pdf"):
                pdf_path = os.path.join(folder_path, file)
                try:
                    images = convert_from_path(pdf_path, poppler_path=POPLER_PATH)
                except Exception as e:
                    # Fallback: try without poppler_path if it fails
                    try:
                        images = convert_from_path(pdf_path)
                    except Exception as e2:
                        print(f"Error processing {file}: {e2}")
                        continue

                full_text = ""
                for img in images:
                    text = pytesseract.image_to_string(img, lang="hin+eng")
                    full_text += text.strip() + "\n"

                doc = Document(page_content=full_text, metadata={"source": file})
                # print("document", file)
                # print("content", full_text)
                all_docs.append(doc)
        return all_docs

    def chunk_documents(self, docs):
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        return splitter.split_documents(docs)

    def create_vector_store(self, chunks):
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        self.vector_store.save_local("faiss_index")

    def load_vector_store(self):
        self.vector_store = FAISS.load_local(
            "faiss_index", self.embeddings, allow_dangerous_deserialization=True
        )

    def extract_date_advanced(self, entry: str) -> datetime:
        """Advanced date extraction supporting multiple patterns and languages"""
        lang_config = LANGUAGES.get(self.current_language, LANGUAGES["en"])
        
        for pattern in lang_config["date_patterns"]:
            match = re.search(pattern, entry)
            if match:
                date_str = match.group(1)
                for date_format in lang_config["date_formats"]:
                    try:
                        return datetime.strptime(date_str, date_format)
                    except ValueError:
                        continue
        
        # Additional fallback patterns
        fallback_patterns = [
            r"\b(\d{1,2}[./]\d{1,2}[./]\d{2,4})\b",
            r"\b(\d{4}[./]\d{1,2}[./]\d{1,2})\b",
            r"\b(\d{1,2}\s+[A-Za-z]+\s+\d{4})\b"
        ]
        
        for pattern in fallback_patterns:
            match = re.search(pattern, entry)
            if match:
                date_str = match.group(1)
                for fmt in ["%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d", "%d.%m.%Y", "%d %B %Y", "%d %b %Y"]:
                    try:
                        return datetime.strptime(date_str, fmt)
                    except ValueError:
                        continue
        
        return datetime.max  # Put undated entries at the end

    def generate_timeline(self, force_refresh: bool = False):
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Load or create it first.")

        # Use cached timeline if available and not forcing refresh
        if self.timeline_cache and not force_refresh:
            # Just translate the cached timeline to current language
            return self.translate_timeline_to_language(self.timeline_cache)

        results = self.vector_store.similarity_search(
            "timeline of legal events, orders, applications, case filings, affidavits, wills, replies, and judgments", k=130
        )
        context = "\n\n".join([doc.page_content[:1500] for doc in results])

        language_name = LANGUAGES[self.current_language]["name"]
        
        prompt = f"""
You are a legal assistant reviewing scanned documents (translated to text via OCR). 

Your task is to generate a **detailed, chronological timeline in {language_name}** of all events relevant to the documents.

✅ For each entry:
- Extract the **full date** (day, month, year) if available
- Mention **the action taken**
- Mention **who did what**, against whom, and why
- Include any **case numbers**, **legal sections**, or **documents referenced**
- Respond ONLY in {language_name}

📌 Format strictly like this:
1. [Full Date] - [Actor] [Action] [Details and Legal References]

If responding in {language_name}, make sure ALL text including dates, names, and legal terms are properly expressed in {language_name}.

Documents:
{context}

Now, generate the full timeline in {language_name}:
"""
        print("PROMPT IS HERE")
        print(prompt)
        response = self.model.invoke(prompt)
        if hasattr(response, "content") and isinstance(response.content, str):
            text = response.content
        elif isinstance(response, str):
            text = response
        else:
            raise TypeError("Unexpected response type from model.invoke()")

        entries = [line.strip() for line in text.strip().split("\n") if re.match(r"^\d+\.\s", line)]

        # Sort entries chronologically using advanced date extraction
        sorted_entries = sorted(entries, key=self.extract_date_advanced)
        
        timeline_result = "\n".join(sorted_entries)
        
        # Cache the timeline
        self.timeline_cache = {
            "entries": sorted_entries,
            "context": context,
            "language": self.current_language,
            "full_text": timeline_result
        }
        
        return timeline_result, context

    def translate_timeline_to_language(self, cached_timeline):
        """Translate cached timeline to current language"""
        if cached_timeline["language"] == self.current_language:
            return cached_timeline["full_text"], cached_timeline["context"]
        
        language_name = LANGUAGES[self.current_language]["name"]
        
        prompt = f"""
Translate the following legal timeline to {language_name}. Maintain the exact same structure and chronological order. 
Make sure ALL content including dates, legal terms, and names are properly expressed in {language_name}.

Original Timeline:
{cached_timeline["full_text"]}

Translated Timeline in {language_name}:
"""
        
        response = self.model.invoke(prompt)
        if hasattr(response, "content") and isinstance(response.content, str):
            translated_text = response.content
        elif isinstance(response, str):
            translated_text = response
        else:
            translated_text = cached_timeline["full_text"]
        
        return translated_text, cached_timeline["context"]

    def setup_conversational_query_system(self):
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Load or create it first.")

        retriever = self.vector_store.as_retriever(search_kwargs={"k": 15})
        
        # Preserve existing memory if it exists
        if self.memory is None:
            self.memory = ConversationSummaryBufferMemory(
                llm=self.model,
                memory_key="chat_history",
                return_messages=True
            )

        language_name = LANGUAGES[self.current_language]["name"]
        no_info_message = self.get_ui_message("no_info")

        prompt_template = """You are a helpful legal assistant. You answer user queries about a civil property case based on documents.

IMPORTANT: Respond ONLY in {language_name}. All your responses must be in {language_name}.

✅ Use the provided context to answer the user's question, even if the documents are in different languages (Hindi, English, etc.).
✅ Extract relevant information from the context and provide a comprehensive response in **{language_name}**.
✅ If you can find ANY relevant information in the context about the legal case, use it to answer the question.
✅ Only say "{no_info_message}" if the context contains absolutely NO relevant information about the user's question.
✅ For general questions not about the legal case, use your general knowledge and respond in {language_name}.

Context from legal documents:
{{context}}

Chat History:
{{chat_history}}

User Question:
{{question}}

Detailed Answer (in {language_name}):""".format(language_name=language_name, no_info_message=no_info_message)

        PROMPT = PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template=prompt_template
        )

        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.model,
            retriever=retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": PROMPT}
        )

    def ask_question(self, user_question):
        if self.qa_chain is None:
            raise ValueError("Conversational query system not set up. Call setup_conversational_query_system() first.")
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Load or create it first.")
        
        # Test retrieval first to debug
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        docs = retriever.get_relevant_documents(user_question)
        # print(f"\n🔍 Found {len(docs)} relevant documents for your question.")
        
        if len(docs) == 0:
            return self.get_ui_message("no_info")
            
        # Use invoke instead of deprecated run method
        try:
            response = self.qa_chain.invoke({"question": user_question})
            return response.get("answer", "No response generated")
        except Exception as e:
            # Fallback: simple retrieval + direct model response
            return self.fallback_answer(user_question, docs)
    
    def fallback_answer(self, user_question, docs):
        """Fallback method if ConversationalRetrievalChain fails"""
        context = "\n\n".join([doc.page_content[:1000] for doc in docs])
        language_name = LANGUAGES[self.current_language]["name"]
        
        prompt = f"""You are a helpful legal assistant. Answer the user's question based on the provided context from legal documents.

IMPORTANT: Respond ONLY in {language_name}.

The context may contain text in different languages (Hindi, English, etc.). Extract and use ALL relevant information to provide a comprehensive answer.

Context from legal documents:
{context}

User Question: {user_question}

Based on the legal documents provided above, here is my detailed answer in {language_name}:"""

        response = self.model.invoke(prompt)
        if hasattr(response, "content"):
            return response.content
        return str(response)

def select_language() -> str:
    """Language selection interface"""
    print("\n🌐 Select your preferred language / अपनी पसंदीदा भाषा चुनें / Selecciona tu idioma preferido:")
    print("=" * 60)
    
    for i, (code, lang_info) in enumerate(LANGUAGES.items(), 1):
        print(f"{i}. {lang_info['name']}")
    
    while True:
        try:
            choice = input(f"\nEnter choice (1-{len(LANGUAGES)}): ").strip()
            choice_num = int(choice)
            if 1 <= choice_num <= len(LANGUAGES):
                return list(LANGUAGES.keys())[choice_num - 1]
            else:
                print(f"Please enter a number between 1 and {len(LANGUAGES)}")
        except ValueError:
            print("Please enter a valid number")

def change_language_menu(rag_system: MultilingualRAGSystem) -> bool:
    """Language change interface"""
    current_lang = LANGUAGES[rag_system.current_language]["name"]
    print(f"\n{rag_system.get_ui_message('current_language')}: {current_lang}")
    print(rag_system.get_ui_message('language_selection'))
    print("=" * 60)
    
    for i, (code, lang_info) in enumerate(LANGUAGES.items(), 1):
        marker = " ✓" if code == rag_system.current_language else ""
        print(f"{i}. {lang_info['name']}{marker}")
    
    print(f"{len(LANGUAGES) + 1}. Back to main menu")
    
    while True:
        try:
            choice = input(f"\nEnter choice (1-{len(LANGUAGES) + 1}): ").strip()
            choice_num = int(choice)
            if 1 <= choice_num <= len(LANGUAGES):
                new_lang = list(LANGUAGES.keys())[choice_num - 1]
                rag_system.set_language(new_lang)
                print(f"✅ Language changed to {LANGUAGES[new_lang]['name']}")
                return True
            elif choice_num == len(LANGUAGES) + 1:
                return False
            else:
                print(f"Please enter a number between 1 and {len(LANGUAGES) + 1}")
        except ValueError:
            print("Please enter a valid number")

def main():
    # Language selection at startup
    print("=" * 60)
    print("🏛️  MULTILINGUAL LEGAL RAG SYSTEM  🏛️")
    print("=" * 60)
    
    selected_language = select_language()
    rag = MultilingualRAGSystem()
    # rag.extract_text_from_pdfs(PDF_FOLDER)
    # return
    rag.set_language(selected_language)
    
    print(f"\n✅ Language set to: {LANGUAGES[selected_language]['name']}")

    if not os.path.exists("faiss_index"):
        print(rag.get_ui_message("extracting"))
        docs = rag.extract_text_from_pdfs(PDF_FOLDER)
        chunks = rag.chunk_documents(docs)
        rag.create_vector_store(chunks)
    else:
        rag.load_vector_store()

    while True:
        print(f"\n{rag.get_ui_message('choose_option')}")
        print(rag.get_ui_message("generate_timeline"))
        print(rag.get_ui_message("ask_question"))
        print(rag.get_ui_message("change_language"))
        print(rag.get_ui_message("exit"))
        
        choice = input("Enter your choice: ").strip()

        if choice == "1":
            print(f"\n{rag.get_ui_message('timeline_output')}\n")
            try:
                timeline, timeline_context = rag.generate_timeline()
                print(timeline)

                if rag.qa_chain is None:
                    rag.setup_conversational_query_system()

                if rag.memory and hasattr(rag.memory, "chat_memory"):
                    rag.memory.chat_memory.add_user_message("Generate timeline of the case.")
                    rag.memory.chat_memory.add_ai_message(timeline)
            except Exception as e:
                print(f"Error generating timeline: {e}")

        elif choice == "2":
            if rag.qa_chain is None:
                rag.setup_conversational_query_system()

            while True:
                question = input(rag.get_ui_message("question_prompt"))
                if question.lower() == "back":
                    break
                try:
                    response = rag.ask_question(question)
                    print(f"\n{rag.get_ui_message('bot_prefix')} {response}")
                except Exception as e:
                    print(f"Error: {e}")

        elif choice == "3":
            change_language_menu(rag)

        elif choice == "4":
            print(rag.get_ui_message("exiting"))
            break

        else:
            print(rag.get_ui_message("invalid_choice"))

if __name__ == "__main__":
    main()
