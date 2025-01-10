'''
Introduction
Images, rich with untapped information, often come under the radar of search engines and data systems. Transforming this visual 
data into machine-readable language is no easy task, but it's where image captioning AI is useful. Here's how image captioning AI
can make a difference:
Improves accessibility: Helps visually impaired individuals understand visual content.
Enhances SEO: Assists search engines in identifying the content of images.

Learning objectives:
Implement an image captioning tool using the BLIP model from Hugging Face's Transformers

Use Gradio to provide a user-friendly interface for your image captioning application

Adapt the tool for real-world business scenarios, demonstrating its practical applications

In this project, to build an AI app, you will use Gradio interface provided by Hugging Face.

Let's set up the environment and dependencies for this project.

STEPS : Create a Python virtual environment and install Gradio
pip3 install virtualenv 
virtualenv my_env # create a virtual environment my_env
source my_env/bin/activate # activate my_env
# installing required libraries in my_env
pip install langchain==0.1.11 gradio==4.44.0 transformers==4.38.2 bs4==0.0.2 requests==2.31.0 torch==2.2.1



Generating image captions with the BLIP model
Introducing: Hugging Face, Tranformers, and BLIP
Hugging Face is an organization that focuses on natural language processing (NLP) and artificial intelligence (AI).The organization 
is widely known for its open-source library called "Transformers" which provides thousands of pre-trained models to the community. 
The library supports a wide range of NLP tasks, such as translation, summarization, text generation, and more. 
Transformers has contributed significantly to the recent advancements in NLP, as it has made state-of-the-art models, 
such as BERT, GPT-2, and GPT-3, accessible to researchers and developers worldwide.

Tranformers library includes a model that can be used to capture information from images. The BLIP, or Bootstrapping 
Language-Image Pre-training, model is a tool that helps computers understand and generate language based on images. 
It's like teaching a computer to look at a picture and describe it, or answer questions about it.

You will be using AutoProcessor and BlipForConditionalGeneration from the transformers library.

"Blip2Processor" and "Blip2ForConditionalGeneration" are components of the BLIP model, which is a vision-language model available in the 
Hugging Face Transformers library.

AutoProcessor : This is a processor class that is used for preprocessing data for the BLIP model. 
It wraps a BLIP image processor and an OPT/T5 tokenizer into a single processor. This means it can handle both image and text data, 
preparing it for input into the BLIP model.

A tokenizer is a tool in natural language processing that breaks down text into smaller, manageable units (tokens), such as words or phrases, 
enabling models to analyze and understand the text.

BlipForConditionalGeneration : This is a model class that is used for conditional text generation given an image and an optional text prompt. 
In other words, it can generate text based on an input image and an optional piece of text.

'''
import requests
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

# Load the pretrained processor and model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
# Load your image, DONT FORGET TO WRITE YOUR IMAGE NAME
img_path = "asma_cherifa.png"
# convert it into an RGB format 
image = Image.open(img_path).convert('RGB')
# You do not need a question for image captioning
text = "the image of"
inputs = processor(images=image, text=text, return_tensors="pt")
# Generate a caption for the image
outputs = model.generate(**inputs, max_length=50)
# Decode the generated tokens to text
caption = processor.decode(outputs[0], skip_special_tokens=True)
# Print the caption
print(caption)

#to run : python3 image_cap.py

'''
Image captioning app with Gradio

'''


