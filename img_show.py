import tkinter as tk
from tkinter import scrolledtext, ttk
from PIL import Image, ImageTk
import json
import base64
from openai import OpenAI
import logging

# Set up logging
logging.basicConfig(filename='gpt_vision_log.txt', level=logging.INFO, 
                    format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Initialize OpenAI client
client = OpenAI()

class ScrollableImage(ttk.Frame):
    def __init__(self, master=None, **kw):
        ttk.Frame.__init__(self, master=master, **kw)
        self.canvas = tk.Canvas(self)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scrollbar_y = ttk.Scrollbar(self, orient=tk.VERTICAL)
        self.scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)

        self.scrollbar_x = ttk.Scrollbar(self, orient=tk.HORIZONTAL)
        self.scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)

        self.canvas.configure(yscrollcommand=self.scrollbar_y.set, xscrollcommand=self.scrollbar_x.set)
        self.scrollbar_y.configure(command=self.canvas.yview)
        self.scrollbar_x.configure(command=self.canvas.xview)

        self.canvas.bind('<Configure>', self.update_scrollregion)

    def update_scrollregion(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

def query_gpt4v(image_path, prompt):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Find all instances of '{prompt}' in the image. Return a list of bounding boxes in the format [{{\"x1\": int, \"y1\": int, \"x2\": int, \"y2\": int}}]."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
            ]
        }
    ]
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=300
    )
    
    result = response.choices[0].message.content
    
    # Log the GPT Vision output
    logging.info(f"Prompt: {prompt}")
    logging.info(f"GPT Vision Output: {result}")
    
    return result

def load_and_display_image():
    global img, photo
    
    image_path = image_entry.get()
    
    try:
        img = Image.open(image_path)
        photo = ImageTk.PhotoImage(img)
        
        scrollable_image.canvas.config(width=img.width, height=img.height)
        scrollable_image.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        
        text_entry.config(state='normal')
        query_button.config(state='normal')
        clear_button.config(state='normal')
        
    except Exception as e:
        result_label.config(text=f"Error: {str(e)}")

def query_and_display():
    image_path = image_entry.get()
    prompt = text_entry.get()
    
    try:
        gpt_output = query_gpt4v(image_path, prompt)
        
        result_textbox.delete(1.0, tk.END)
        result_textbox.insert(tk.END, f"GPT-4 Vision output for '{prompt}':\n\n{gpt_output}")
        
        result_label.config(text="GPT-4 Vision query completed.")
    
    except Exception as e:
        result_label.config(text=f"Error: {str(e)}")

def clear_results():
    result_label.config(text="Results cleared.")
    result_textbox.delete(1.0, tk.END)

# Create main window
root = tk.Tk()
root.title("GPT-4V Image Text Locator")

# Create and pack widgets
tk.Label(root, text="Enter PNG image path:").pack()
image_entry = tk.Entry(root, width=50)
image_entry.pack()

load_button = tk.Button(root, text="Load Image", command=load_and_display_image)
load_button.pack()

tk.Label(root, text="Enter text to locate:").pack()
text_entry = tk.Entry(root, width=50, state='disabled')
text_entry.pack()

query_button = tk.Button(root, text="Query GPT-4 Vision", command=query_and_display, state='disabled')
query_button.pack()

clear_button = tk.Button(root, text="Clear Results", command=clear_results, state='disabled')
clear_button.pack()

result_label = tk.Label(root, text="")
result_label.pack()

# Create scrolled text widget for results
result_textbox = scrolledtext.ScrolledText(root, height=10, width=50)
result_textbox.pack()

# Create scrollable image frame
scrollable_image = ScrollableImage(root)
scrollable_image.pack(fill=tk.BOTH, expand=True)

root.mainloop()