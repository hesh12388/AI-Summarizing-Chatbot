import gradio as gr

from openai import OpenAI

import google.generativeai as genai



def docToString(document):
    res = "-"

    if document is None:
        raise gr.Error("Document not attached")
    with open(document.name, "r") as f:
        text=f.read()
    res += text + "-"

    return res

def summarize(prompt, document, model2, temperature2):
    user_prompt=  docToString(document) + "\n"  + prompt
    completion = client.chat.completions.create(
    model=model2,
    messages=[
        {"role": "system", "content": "You are a helpful assistant whose role is to summarize information  about a specific individual from a given text. You will be given text as input. Within this text, there will be text delimitted by hyphens. Following this, there will also be a question that you will have to answer. Use the following steps to answer: 1- Derive from the question what individual or topic you have to provide information about 2- Use the text delimitted by hyphens to summarize information about the individual or topic that you identified in step 1. You should respond with only one of the following options:1- If the question is unrelated to the text delimited by hypens or there is no question, then write \" Your question is unrelated to the text provided\". Do not respond with anything else. 2- However, if the question is related to the text delimited by hyphens in some way but the text delimited by hyphens does not provide sufficient information to answer the question including if the text delimited by hyphens is empty, then instead, write \"I do not know\". Do not respond with anything else. 3- If the question is related and the text does provide sufficient information, then summarize the main points about the individual or topic stated in the question using the text delimited by hyphens. STRICTLY, do not respond with any information that is not provided in the text delimted by hyphens."},
        {"role": "user", "content": user_prompt}
    ],
    temperature=temperature2
    )

    return completion.choices[0].message.content



def summarize2(user_prompt, document, modelName, temperature2):
    model = genai.GenerativeModel(modelName)
    system_prompt="You are a helpful assistant whose role is to summarize information  about a specific individual from a given text. You will be given text below. Within this text, there will be text delimitted by hyphens. Following this, there will also be a question that you will have to answer. Use the following steps to answer: 1- Derive from the question what individual or topic you have to provide information about 2- Use the text delimitted by hyphens to summarize information about the individual or topic that you identified in step 1. You should respond with only one of the following options:1- If the question is unrelated to the text delimited by hypens or there is no question, then write \" Your question is unrelated to the text provided\". Do not respond with anything else. 2- However, if the question is related to the text delimited by hyphens in some way but the text delimited by hyphens does not provide sufficient information to answer the question including if the text delimited by hyphens is empty, then instead, write \"I do not know\". Do not respond with anything else. 3- If the question is related and the text does provide sufficient information, then summarize the main points about the individual or topic stated in the question using the text delimited by hyphens. STRICTLY, do not respond with any information that is not provided in the text delimted by hyphens. STRICTLY, only return your response. Do not return the text or question provided."
    input = system_prompt +"\n" +docToString(document) + "\n"  + user_prompt
    generation_config = genai.GenerationConfig(temperature=temperature2)
    response=model.generate_content(
        contents=input,
        generation_config=generation_config
    )
    return response.text

def pickModel(user_prompt, document, modelChoice, temperature2):
    if modelChoice == "gemini-pro":
        return summarize2(user_prompt, document, modelChoice, temperature2)
    else:
        return summarize(user_prompt, document, modelChoice, temperature2)

css="""
.image{
height=300px;
width=300px;
background:transparent;
}

.Download{
background:	#AD1B02;
}

.Download:hover{
background:	#E88D14;
}

"""

with gr.Blocks(css=css, theme='snehilsanyal/scikit-learn') as demo:
    gr.Image("/Users/heshamahmed/Downloads/PricewaterhouseCoopers_Logo.svg-removebg-preview (1).png",height="120px", width="120px",elem_classes="image",container=False, show_download_button=False)
    with gr.Row():
        with gr.Column(scale=1, min_width=200):
            with gr.Accordion("Attach document", open=False):
                document=gr.File(label="Document", type="filepath")
            with gr.Accordion("Select prompt", open=False):
                prompt=gr.Textbox(label="prompt")
            with gr.Row():
                modelSelection= gr.Dropdown(choices=["gemini-pro", "gpt-3.5-turbo"], label="Select Model")
            with gr.Row():
                temperature =gr.Slider(0, 2, value=0, label="Temperature")
            btn=gr.Button("Generate")
        with gr.Column(scale=1, min_width=400):
            output =gr.Textbox(label="Result", lines=18)
            gr.Button("Download", elem_classes="Download")
    btn.click(pickModel, inputs=[prompt, document,  modelSelection, temperature], outputs= output)
demo.launch()
