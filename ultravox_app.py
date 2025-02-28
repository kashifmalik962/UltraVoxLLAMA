import gradio as gr
import torch
import transformers
import speech_recognition as sr
import os
from transformers import AutoModel, AutoTokenizer, pipeline


ultravox_model_path = "./ultravox"

# ✅ Function to load models safely
def load_model(model_class, path, **kwargs):
    try:
        return model_class.from_pretrained(path, **kwargs)
    except Exception as e:
        print(f"❌ Error loading model from {path}: {e}")
        exit()


class UltravoxInterface:
    def __init__(self):
        """Initialize the Ultravox model and settings"""
        print("Loading Ultravox model... This may take a few minutes...")
        self.model = load_model(AutoModel, ultravox_model_path, trust_remote_code=True, torch_dtype=torch.float32)
        self.tokenizer = load_model(AutoTokenizer, ultravox_model_path, trust_remote_code=True)
        
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        print("Model loaded successfully!")
        
        # Default system prompt
        self.default_prompt = "You are a friendly and helpful character. You love to answer questions for people."
        
    def process_audio(self, audio_path, custom_prompt=None):
        """
        Process audio input and return model response
        
        Args:
            audio_path: Path to the audio file
            custom_prompt: Optional custom system prompt
        """
        try:
            recognizer = sr.Recognizer()
            with sr.AudioFile(audio_path) as source:
                print("🔄 Converting Speech to Text...")
                audio_data = recognizer.record(source)
                transcription = recognizer.recognize_google(audio_data)
                print("🎤 Transcription:", transcription)

            # Prepare prompt for text generation
            prompt = (custom_prompt if custom_prompt else self.default_prompt) + f"\nUser: {transcription}\nAI:"
            print("prompt==>", prompt)
            # Generate response
            print("🔄 Generating AI response...")
            response = self.pipe(prompt, max_new_tokens=50, pad_token_id=self.tokenizer.eos_token_id)
            print("response ==>", response)
            
            generated_text = response[0]["generated_text"]
            print("generated_text ==>", generated_text)
            
            return generated_text

        except sr.UnknownValueError:
            return "Sorry, I couldn't understand the audio."
        except sr.RequestError as e:
            return f"Speech Recognition service error: {e}"
        except Exception as e:
            return f"Error processing audio: {str(e)}"

    def create_interface(self):
        """Create and configure the Gradio interface"""
        
        with gr.Blocks(title="Ultravox Voice Interface") as interface:
            gr.Markdown("# 🎙️ Ultravox Voice Assistant")
            gr.Markdown("Speak into the microphone and get AI-generated responses!")
            
            with gr.Row():
                with gr.Column():
                    # Updated Audio input component
                    audio_input = gr.Audio(
                        label="Speak here",
                        sources=["microphone"],  # Changed from source to sources
                        type="filepath"
                    )
                    
                    # Optional system prompt
                    system_prompt = gr.Textbox(
                        label="System Prompt (Optional)",
                        placeholder="Enter custom system prompt or leave empty for default",
                        value=self.default_prompt
                    )
                    
                    # Submit button
                    submit_btn = gr.Button("Process Audio", variant="primary")
                
                with gr.Column():
                    # Output text area
                    output_text = gr.Textbox(
                        label="AI Response",
                        lines=5,
                        placeholder="AI response will appear here..."
                    )
            
            # Handle submission
            submit_btn.click(
                fn=self.process_audio,
                inputs=[audio_input, system_prompt],
                outputs=output_text
            )
            
            # Example usage instructions
            gr.Markdown("""
            ## How to use:
            1. Click the microphone icon and allow browser access
            2. Speak your question or prompt
            3. Click 'Stop' when finished
            4. Click 'Process Audio' to get the AI response
            
            ## Requirements:
            - GPU with 24GB+ VRAM recommended
            - Working microphone
            - Stable internet connection
            
            ## Note:
            First-time loading may take a few minutes as the model is downloaded.
            """)
            
        return interface

def main():
    # Create instance of UltravoxInterface
    app = UltravoxInterface()
    
    # Launch the interface
    interface = app.create_interface()
    interface.launch(
        share=True  # Enable sharing via Gradio
    )

if __name__ == "__main__":
    main()
