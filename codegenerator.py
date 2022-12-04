import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, pipeline



            


# change model to the finetuned one
tokenizer = AutoTokenizer.from_pretrained("codeparrot/codeparrot-small-text-to-code")
model = AutoModelForCausalLM.from_pretrained("codeparrot/codeparrot-small-text-to-code")

def make_doctring(gen_prompt):
    return "\"\"\"\n" + gen_prompt + "\n\"\"\"\n\n"

def code_generation(gen_prompt, max_tokens, temperature=0.6, seed=42):
    set_seed(seed)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    prompt = make_doctring(gen_prompt)
    generated_text = pipe(prompt, do_sample=True, top_p=0.95, temperature=temperature, max_new_tokens=max_tokens)[0]['generated_text']
    return generated_text


iface = gr.Interface(
    fn=code_generation, 
    inputs=[
        gr.Textbox(lines=10, label="Text"),
        gr.inputs.Slider(
            minimum=8,
            maximum=1000,
            step=1,
            default=8,
            label="Number of tokens to generate",
        ),
        gr.inputs.Slider(
            minimum=0,
            maximum=2.5,
            step=0.1,
            default=0.6,
            label="Temperature",
        ),
        gr.inputs.Slider(
            minimum=0,
            maximum=1000,
            step=1,
            default=42,
            label="Random seed to use for the generation"
        )
    ],
    outputs=gr.Textbox(label="Python code", lines=10),
    examples=example,
    layout="horizontal",
    theme="peach",
    description=description,
    title=title
)
iface.launch()