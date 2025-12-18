import gradio as gr
import asyncio
import fitz
import os
from pathlib import Path

from knowledge_graph import neo_connection, run_graph_population, populate_with_uploads
from rag import GraphRetrieval
from user_uploads import build_metadata, save_to_json

kg = neo_connection()
rag = GraphRetrieval(kg=kg)

async def answer_query(query):
    try:
        final_answer = await rag.generate_response(
            query, 
            top_k=3, 
            expand=True, 
            hops=1
        )

        return final_answer

    except Exception as e:
        return f"Error: {e}"

def handle_upload(file):
    if not file:
        return "No file uploaded."
    
    try:
        # Extract text from PDF
        text = ""
        with fitz.open(file.name) as doc:
            for page in doc:
                text += page.get_text()
        
        # Build and save metadata
        pdf_path = Path(file.name)
        metadata = build_metadata(pdf_path, text)
        save_to_json(metadata)
        
        return f"Successfully uploaded and processed: {os.path.basename(file.name)}"
    except Exception as e:
        return f"Error processing file: {str(e)}"

def process_and_populate(file):
    if not file:
        return "No file uploaded."
    
    try:
        # Extract text from PDF
        text = ""
        with fitz.open(file.name) as doc:
            for page in doc:
                text += page.get_text()
        
        # Build and save metadata
        pdf_path = Path(file.name)
        metadata = build_metadata(pdf_path, text)
        save_to_json(metadata)

        # Populate graph
        populate_with_uploads(metadata, text)
        
        return f"Successfully uploaded, processed, and added to graph: {os.path.basename(file.name)}"
    except Exception as e:
        return f"Error processing file: {str(e)}"
def run_gradio():
    with gr.Blocks(title="RAG System Demo") as demo:
        gr.Markdown("# Graph Augmented Generation Question Answering Demo")
        gr.Markdown("Ask a question about any paper in your knowledge graph.")

        query = gr.Textbox(label="Enter your question", lines=2)
        answer = gr.Textbox(label="Final Answer", lines=5)
        
        with gr.Row():
            upload = gr.File(label="Upload a paper as PDF", file_count="single", file_types=[".pdf"])
            upload_status = gr.Textbox(label="Upload Status", interactive=False)

        submit = gr.Button("Run Explanation")

        submit.click(
            answer_query,
            inputs=[query],
            outputs=[answer]
        )

        upload.upload(
            process_and_populate,
            inputs=[upload],
            outputs=[upload_status]
        )

    demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    asyncio.run(run_gradio())

