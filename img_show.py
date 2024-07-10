import tkinter as tk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

def load_and_display_image():
    global img, photo, canvas
    
    # Get the image path from the entry
    image_path = image_entry.get()
    
    try:
        # Open the image
        img = Image.open(image_path)
        
        # Convert to PhotoImage for Tkinter
        photo = ImageTk.PhotoImage(img)
        
        # Update canvas with new image
        canvas.config(width=img.width, height=img.height)
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        
        # Enable coordinate input
        x_entry.config(state='normal')
        y_entry.config(state='normal')
        plot_button.config(state='normal')
        
    except Exception as e:
        result_label.config(text=f"Error: {str(e)}")

def plot_coordinates():
    try:
        x = int(x_entry.get())
        y = int(y_entry.get())
        
        # Check if coordinates are within image bounds
        if 0 <= x < img.width and 0 <= y < img.height:
            # Draw a red dot at the specified coordinates
            canvas.create_oval(x-3, y-3, x+3, y+3, fill='red', outline='red')
            result_label.config(text=f"Dot plotted at ({x}, {y})")
        else:
            result_label.config(text="Coordinates out of image bounds")
    
    except ValueError:
        result_label.config(text="Invalid coordinate input")

# Create main window
root = tk.Tk()
root.title("Image Coordinate Plotter")

# Create and pack widgets
tk.Label(root, text="Enter PNG image path:").pack()
image_entry = tk.Entry(root, width=50)
image_entry.pack()

load_button = tk.Button(root, text="Load Image", command=load_and_display_image)
load_button.pack()

tk.Label(root, text="X coordinate:").pack()
x_entry = tk.Entry(root, state='disabled')
x_entry.pack()

tk.Label(root, text="Y coordinate:").pack()
y_entry = tk.Entry(root, state='disabled')
y_entry.pack()

plot_button = tk.Button(root, text="Plot Coordinates", command=plot_coordinates, state='disabled')
plot_button.pack()

result_label = tk.Label(root, text="")
result_label.pack()

# Create canvas for image display
canvas = tk.Canvas(root)
canvas.pack()

root.mainloop()