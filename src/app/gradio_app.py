import os
import json
import gradio as gr
from dotenv import load_dotenv
from src.retrieval.indexing import load_indexes
from src.retrieval.retrieval import HybridRetriever, RerankPipeline, EnhancedRetriever
from src.generation.generation import ask

# Load environment variables
load_dotenv()

# Global variables for indexes and retriever
CHUNK_PATH = "./data/processed/chunks.json"
PERSIST_DIR = "./vectorstore/chroma_db"
COLLECTION_NAME = "parishiksha_v2"

retriever_chain = None

def initialize_system():
    global retriever_chain
    if not os.path.exists(CHUNK_PATH):
        return "Error: Chunks file not found. Please run indexing first."
    
    try:
        with open(CHUNK_PATH, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        vs, bm25 = load_indexes(PERSIST_DIR, COLLECTION_NAME, chunks)
        hybrid = HybridRetriever(vs, bm25, chunks)
        reranker = RerankPipeline(use_cohere=False)
        retriever_chain = EnhancedRetriever(hybrid, reranker, use_multiquery=True)
        return "System Initialized Successfully"
    except Exception as e:
        return f"Initialization Error: {str(e)}"

# Initialize on startup
init_status = initialize_system()

def predict(message, history):
    if retriever_chain is None:
        history.append({"role": "assistant", "content": "System not initialized. Please check backend logs."})
        yield history
        return

    # history is a list of dicts: [{"role": "user", "content": "..."}, ...]
    history.append({"role": "user", "content": message})
    yield history

    # Use the ask function from src.generation
    result = ask(message, retriever_chain)
    
    answer = result["answer"]
    sources = result["sources"]
    
    response = answer
    if sources:
        response += "\n\n**Sources:**\n- " + "\n- ".join(sources)
    
    history.append({"role": "assistant", "content": response})
    yield history

# CSS for Premium Aesthetics
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

body { 
    font-family: 'Outfit', sans-serif !important; 
    background: #0f172a; 
    color: #f1f5f9;
}

.gradio-container { 
    max-width: 1100px !important; 
    margin: 20px auto !important; 
}

.main-header { 
    text-align: center; 
    margin-bottom: 2rem; 
    padding: 3rem 2rem; 
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%); 
    border-radius: 24px; 
    color: white; 
    box-shadow: 0 20px 50px -12px rgba(0, 0, 0, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.main-header h1 { 
    font-size: 3rem !important; 
    font-weight: 800 !important; 
    margin-bottom: 0.5rem !important; 
    letter-spacing: -0.025em;
}

.main-header p { 
    font-size: 1.25rem !important; 
    opacity: 0.9; 
    max-width: 700px; 
    margin: auto;
}

#chatbot { 
    border-radius: 20px !important; 
    border: 1px solid rgba(255, 255, 255, 0.1) !important; 
    background: rgba(30, 41, 59, 0.7) !important; 
    backdrop-filter: blur(10px);
    height: 650px !important; 
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
}

.message { 
    padding: 12px 16px !important; 
    margin-bottom: 12px !important;
}

.message.user { 
    background: #4f46e5 !important; 
    color: white !important; 
    border-radius: 18px 18px 4px 18px !important; 
    align-self: flex-end !important;
}

.message.bot { 
    background: #334155 !important; 
    color: #f1f5f9 !important;
    border-radius: 18px 18px 18px 4px !important; 
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
}

.primary-btn { 
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important; 
    border: none !important; 
    border-radius: 12px !important;
    color: white !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}

.primary-btn:hover { 
    transform: translateY(-2px); 
    box-shadow: 0 8px 20px rgba(79, 70, 229, 0.4) !important; 
}

.footer { 
    text-align: center; 
    margin-top: 3rem; 
    font-size: 0.95rem; 
    color: #94a3b8; 
    padding-bottom: 2rem;
}

.status-badge {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 9999px;
    font-weight: 600;
    font-size: 0.85rem;
    margin-top: 1rem;
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
}
"""

# Build Interface
with gr.Blocks() as demo:
    with gr.Column(elem_classes="main-header"):
        gr.Markdown("## 📖 PariShiksha Assistant v2.0")
        gr.Markdown("Your Intelligent NCERT Science Tutor powered by Advanced RAG")
        if "Error" in init_status:
            gr.Markdown(f"⚠️ **Status:** {init_status}", elem_classes="status-badge")
        else:
            gr.Markdown(f"✨ **System Status:** Knowledge Base Ready", elem_classes="status-badge")

    chatbot = gr.Chatbot(
        elem_id="chatbot",
        show_label=False,
        avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=PariShiksha")
    )
    
    with gr.Row():
        msg = gr.Textbox(
            placeholder="Ask me about Motion, Force, or Gravitation...",
            container=False,
            scale=7
        )
        submit = gr.Button("Send Question", variant="primary", scale=1, elem_classes="primary-btn")

    with gr.Row():
        gr.Examples(
            examples=[
                "What is Newton's second law of motion?",
                "Explain the law of conservation of momentum.",
                "Why is the weight of an object on the moon 1/6th of its weight on Earth?",
                "An object travels 16m in 4s and 16m in 2s. What is average speed?"
            ],
            inputs=msg
        )

    gr.Markdown("---")
    gr.Markdown("Created for educational excellence using NCERT textbooks.", elem_classes="footer")

    msg.submit(predict, [msg, chatbot], [chatbot])
    submit.click(predict, [msg, chatbot], [chatbot])
    msg.submit(lambda: "", None, [msg])
    submit.click(lambda: "", None, [msg])

if __name__ == "__main__":
    demo.queue().launch(
        share=False,
        css=custom_css,
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="slate")
    )
