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
from neo4j import GraphDatabase
from dotenv import load_dotenv
from tqdm import tqdm
import re
from datetime import datetime
from typing import Optional, List, Dict
import hashlib

# Load .env
load_dotenv()

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# Poppler path and default case folder
POPLER_PATH = r"C:\\Users\\anura\\Downloads\\Release-24.08.0-0\\poppler-24.08.0\\Library\\bin"
DEFAULT_CASE_FOLDER = "./Documents"

class GraphHandler:
    def _init_(self, uri="neo4j://127.0.0.1:7687", user="neo4j", password="neo4j123"):
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            # Test the connection
            with self.driver.session() as session:
                result = session.run("RETURN 1")
                print("✅ Neo4j connection successful!")
        except Exception as e:
            print(f"❌ Neo4j connection failed: {e}")
            print("\n🔧 Troubleshooting steps:")
            print("1. Make sure Neo4j is running")
            print("2. Check your credentials in Neo4j Desktop")
            print("3. Try these connection strings:")
            print("   - neo4j://127.0.0.1:7687")
            print("   - bolt://127.0.0.1:7687")
            print("   - neo4j://localhost:7687")
            print("4. Default credentials are usually:")
            print("   - Username: neo4j")
            print("   - Password: (what you set during installation)")
            raise e

    def push_triplets(self, triplets):
        with self.driver.session() as session:
            for subj, rel, obj in triplets:
                # Determine entity types
                subj_type = self._determine_entity_type(subj.strip())
                obj_type = self._determine_entity_type(obj.strip())
                
                session.run("""
                    MERGE (a:Entity {name: $subj, type: $subj_type})
                    MERGE (b:Entity {name: $obj, type: $obj_type})
                    MERGE (a)-[:RELATION {type: $rel}]->(b)
                """, subj=subj.strip(), obj=obj.strip(), rel=rel.strip(), 
                     subj_type=subj_type, obj_type=obj_type)

    def _determine_entity_type(self, entity_name):
        """Determine the type of entity based on its name/content"""
        name = entity_name.lower()
        
        # Date patterns
        if re.search(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}', name) or re.search(r'\d{4}', name):
            return "Date"
        
        # Case number patterns
        if re.search(r'case|petition|suit|appeal|writ', name) and re.search(r'\d', name):
            return "Case"
        
        # Court patterns
        if any(word in name for word in ['court', 'tribunal', 'commission', 'high court', 'supreme court']):
            return "Court"
        
        # Document patterns
        if any(word in name for word in ['order', 'application', 'affidavit', 'reply', 'petition', 'judgment']):
            return "Document"
        
        # Location patterns
        if any(word in name for word in ['village', 'city', 'district', 'state', 'address', 'location']):
            return "Location"
        
        # Organization patterns
        if any(word in name for word in ['trust', 'company', 'corporation', 'society', 'association']):
            return "Organization"
        
        # Person patterns (names with titles or common name patterns)
        if re.search(r'^(mr\.|mrs\.|ms\.|dr\.|smt\.|shri|sri)', name) or len(name.split()) >= 2:
            return "Person"
        
        # Default to Entity for unknown types
        return "Entity"

    def query_relation(self, person1, person2):
        with self.driver.session() as session:
            result = session.run("""
                MATCH (a:Entity {name: $p1})-[r:RELATION]-(b:Entity {name: $p2})
                RETURN a.name AS source, r.type AS rel, b.name AS target
            """, p1=person1, p2=person2)
            return [f"{record['source']} -[{record['rel']}]-> {record['target']}" for record in result]
    
    def get_graph_statistics(self) -> Dict:
        """Get comprehensive graph statistics"""
        with self.driver.session() as session:
            result = session.run("MATCH (n) RETURN count(n) as node_count")
            node_count = result.single()["node_count"]
            
            result = session.run("MATCH ()-[r]->() RETURN count(r) as rel_count")
            rel_count = result.single()["rel_count"]
            
            result = session.run("MATCH ()-[r:RELATION]->() RETURN count(DISTINCT r.type) as rel_types")
            rel_types = result.single()["rel_types"]
            
            return {
                "total_nodes": node_count,
                "total_relationships": rel_count,
                "unique_relationship_types": rel_types
            }
    
    def find_entities_by_pattern(self, pattern: str) -> List[str]:
        """Search entities by name pattern"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n:Entity)
                WHERE n.name CONTAINS $pattern
                RETURN n.name as name
                LIMIT 20
            """, pattern=pattern)
            return [record["name"] for record in result]
    
    def get_relationship_types(self) -> List[str]:
        """Get all unique relationship types"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH ()-[r:RELATION]->()
                RETURN DISTINCT r.type as rel_type
                ORDER BY rel_type
            """)
            return [record["rel_type"] for record in result]
    
    def find_most_connected_entities(self, limit: int = 5) -> List[Dict]:
        """Find entities with most connections"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n:Entity)-[r:RELATION]-()
                RETURN n.name as entity, count(r) as connection_count
                ORDER BY connection_count DESC
                LIMIT $limit
            """, limit=limit)
            return [dict(record) for record in result]

class LegalRAGSystem:
    def _init_(self):
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        self.model = ChatOpenAI(model="gpt-4o", temperature=0)
        self.memory = None
        self.qa_chain = None
        self.current_case_folder = DEFAULT_CASE_FOLDER
        self.timeline_cache = None
        
        # Neo4j connection - Using your working credentials
        try:
            self.graph = GraphHandler(uri="neo4j://localhost:7687", user="neo4j", password="neo4j123")
        except Exception as e:
            print(f"⚠ Neo4j connection failed: {e}")
            print("💡 To fix this:")
            print("1. Make sure Neo4j Desktop is running")
            print("2. Check if your database is started")
            print("3. Try running test_neo4j_connection.py to verify credentials")
            self.graph = None

    def extract_triplets_from_text(self, text):
        prompt = f"""
You are a legal relationship extractor.

From the text below, extract all *family and legal relationships* in the format:
<Subject> -[Relation]-> <Object>

Examples: 
- Ramesh Kumar -[father]-> Sita Devi
- Sita Devi -[petitioner]-> Case_06_2023

Text:
{text}

Output:
"""
        response = self.model.invoke(prompt)
        return re.findall(r"(.+?)\s*-\[(.+?)\]->\s*(.+)", response.content.strip())

    def get_case_index_name(self, case_folder):
        folder_hash = hashlib.md5(case_folder.encode()).hexdigest()[:8]
        return f"faiss_index_{folder_hash}"

    def switch_case(self, new_case_folder):
        if not os.path.exists(new_case_folder):
            raise ValueError(f"Case folder '{new_case_folder}' does not exist")
        self.current_case_folder = new_case_folder
        self.timeline_cache = None
        self.qa_chain = None
        self.memory = None
        index_name = self.get_case_index_name(new_case_folder)
        if os.path.exists(index_name):
            self.load_vector_store(index_name)
        else:
            self.create_index_for_case(new_case_folder, index_name)

    def extract_text_from_pdfs(self, folder_path):
        all_docs = []
        for file in os.listdir(folder_path):
            if file.endswith(".pdf"):
                try:
                    images = convert_from_path(os.path.join(folder_path, file), poppler_path=POPLER_PATH)
                    text = "\n".join(pytesseract.image_to_string(img, lang="hin+eng") for img in images)
                    doc = Document(page_content=text, metadata={"source": file})
                    all_docs.append(doc)
                except Exception as e:
                    print(f"Error reading {file}: {e}")
        return all_docs

    def chunk_documents(self, docs):
        return RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)

    def create_vector_store(self, chunks, index_name):
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        self.vector_store.save_local(index_name)

    def load_vector_store(self, index_name):
        self.vector_store = FAISS.load_local(index_name, self.embeddings, allow_dangerous_deserialization=True)

    def create_index_for_case(self, folder, index_name):
        docs = self.extract_text_from_pdfs(folder)
        if self.graph:
            for doc in docs:
                triplets = self.extract_triplets_from_text(doc.page_content)
                self.graph.push_triplets(triplets)
        chunks = self.chunk_documents(docs)
        self.create_vector_store(chunks, index_name)

    def mmr_retrieve(self, query, k=10, fetch_k=40, lambda_mult=0.5):
        return self.vector_store.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult)

    def generate_timeline(self):
        results = self.mmr_retrieve("timeline of legal events, orders, applications, case filings, affidavits, wills, replies, and judgments", k=130)
        context = "\n\n".join([doc.page_content[:1500] for doc in results])
        if self.graph:
            triplets = self.extract_triplets_from_text(context)
            self.graph.push_triplets(triplets)
        prompt = f"""
You are a legal assistant reviewing scanned documents (translated to text via OCR). 

Your task is to generate a *detailed, chronological timeline in English* of all events relevant to the documents.

📌 Format strictly like this:
1. *DD-MM-YYYY* - [Actor] [Action] [Details and Legal References]

Documents:
{context}

Now, generate the full timeline in English:
"""
        response = self.model.invoke(prompt)
        return response.content

    def setup_conversational_query_system(self):
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 15})
        if self.memory is None:
            self.memory = ConversationSummaryBufferMemory(llm=self.model, memory_key="chat_history", return_messages=True)
        prompt_template = PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template="""
You are a helpful legal assistant. Respond ONLY in English.

Context:
{context}

Chat History:
{chat_history}

User Question:
{question}

Answer in English:
"""
        )
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.model,
            retriever=retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": prompt_template}
        )

    def ask_question(self, user_question):
        if self.qa_chain is None:
            self.setup_conversational_query_system()
        if self.graph and ("relation" in user_question.lower() or "related" in user_question.lower()):
            names = re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', user_question)
            if len(names) >= 2:
                rels = self.graph.query_relation(names[0], names[1])
                if rels:
                    return "\n".join(rels)
        docs = self.mmr_retrieve(user_question, k=10)
        try:
            response = self.qa_chain.invoke({"question": user_question})
            return response.get("answer", "No response generated")
        except Exception as e:
            print(f"❌ Error: {e}")
            context = "\n\n".join([doc.page_content[:1000] for doc in docs])
            prompt = f"""
You are a helpful legal assistant. Answer in English based on the context:

Context:
{context}

User Question: {user_question}

Answer:
"""
            return self.model.invoke(prompt).content

def main():
    rag = LegalRAGSystem()
    rag.switch_case(DEFAULT_CASE_FOLDER)
    
    while True:
        print("\n" + "="*50)
        print("🏛  LEGAL RAG SYSTEM  🏛")
        print("="*50)
        print("1. Generate Timeline")
        print("2. Chat with Memory")
        print("3. Graph Analytics")
        print("4. Exit")
        print("-"*50)
        
        choice = input("Enter your choice (1-4): ").strip()
        
        if choice == "1":
            print("\n📅 Generating Timeline...")
            try:
                timeline = rag.generate_timeline()
                print("\n" + "="*50)
                print("📅 TIMELINE OUTPUT:")
                print("="*50)
                print(timeline)
            except Exception as e:
                print(f"❌ Error generating timeline: {e}")
                
        elif choice == "2":
            print("\n💬 Chat Mode - Ask questions about the case")
            print("(Type 'back' to return to main menu)")
            while True:
                question = input("\nAsk your legal question: ").strip()
                if question.lower() == 'back':
                    break
                if question:
                    try:
                        response = rag.ask_question(question)
                        print(f"\n🤖 {response}")
                    except Exception as e:
                        print(f"❌ Error: {e}")
                        
        elif choice == "3":
            print("\n🔍 Graph Analytics Mode")
            if not rag.graph:
                print("❌ Neo4j not connected. Cannot perform graph analytics.")
                continue
                
            while True:
                print("\n📊 Graph Analytics Options:")
                print("1. Show Graph Statistics")
                print("2. Search Entities")
                print("3. Show Relationship Types")
                print("4. Find Most Connected Entities")
                print("5. Back to Main Menu")
                
                graph_choice = input("Enter choice (1-5): ").strip()
                
                if graph_choice == "1":
                    try:
                        stats = rag.graph.get_graph_statistics()
                        print(f"\n📊 Graph Statistics:")
                        print(f"   Total Nodes: {stats['total_nodes']}")
                        print(f"   Total Relationships: {stats['total_relationships']}")
                        print(f"   Unique Relationship Types: {stats['unique_relationship_types']}")
                    except Exception as e:
                        print(f"❌ Error: {e}")
                        
                elif graph_choice == "2":
                    pattern = input("Enter search pattern: ").strip()
                    if pattern:
                        try:
                            entities = rag.graph.find_entities_by_pattern(pattern)
                            print(f"\n🔍 Found Entities:")
                            for entity in entities:
                                print(f"   • {entity}")
                        except Exception as e:
                            print(f"❌ Error: {e}")
                            
                elif graph_choice == "3":
                    try:
                        rel_types = rag.graph.get_relationship_types()
                        print(f"\n🔗 Relationship Types:")
                        for rel_type in rel_types:
                            print(f"   • {rel_type}")
                    except Exception as e:
                        print(f"❌ Error: {e}")
                        
                elif graph_choice == "4":
                    try:
                        connected = rag.graph.find_most_connected_entities()
                        print(f"\n🌟 Most Connected Entities:")
                        for entity in connected:
                            print(f"   • {entity['entity']} ({entity['connection_count']} connections)")
                    except Exception as e:
                        print(f"❌ Error: {e}")
                        
                elif graph_choice == "5":
                    break
                else:
                    print("❌ Invalid choice. Please enter 1-5.")
                        
        elif choice == "4":
            print("\n👋 Exiting. Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")

if _name_ == "_main_":
    main()