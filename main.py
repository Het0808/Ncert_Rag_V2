import os
import argparse
from dotenv import load_dotenv

def run_cli_qa():
    print("Initializing PariShiksha-NCERT-RAG Backend...")
    from src.app.gradio_app import initialize_system, retriever_chain
    from src.generation.generation import ask

    status = initialize_system()
    if "Error" in status:
        print(f"Failed to initialize: {status}")
        return

    print("System ready. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            user_input = input("\nQuestion: ")
            if not user_input.strip():
                continue
            if user_input.lower() in ['exit', 'quit']:
                break
            
            result = ask(user_input, retriever_chain)
            print(f"\nAnswer:\n{result['answer']}")
            if result['sources']:
                print("\nSources:")
                for source in result['sources']:
                    print(f"- {source}")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

def run_app():
    print("Starting Gradio App...")
    from src.app.gradio_app import demo
    demo.launch(share=False)

def main():
    parser = argparse.ArgumentParser(description="PariShiksha-NCERT-RAG")
    parser.add_argument("--app", action="store_true", help="Launch the Gradio Web App")
    parser.add_argument("--cli", action="store_true", help="Launch the CLI QA interface")
    args = parser.parse_args()

    if args.app:
        run_app()
    elif args.cli:
        run_cli_qa()
    else:
        # Default to CLI QA
        print("No mode specified. Defaulting to CLI QA. Use --app to run Gradio Web App.")
        run_cli_qa()

if __name__ == "__main__":
    load_dotenv()
    main()
