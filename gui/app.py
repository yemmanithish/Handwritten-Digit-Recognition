import tkinter as tk
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from model.cnn_model import build_model

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Digit Recognizer")
        self.geometry("300x400")
        self.canvas = tk.Canvas(self, width=280, height=280, bg="white")
        self.canvas.pack(pady=20)
        self.button_frame = tk.Frame(self)
        self.button_frame.pack()
        self.predict_btn = tk.Button(self.button_frame, text="Predict", command=self.predict_digit)
        self.predict_btn.pack(side="left", padx=10)
        self.clear_btn = tk.Button(self.button_frame, text="Clear", command=self.clear_canvas)
        self.clear_btn.pack(side="left", padx=10)
        self.label = tk.Label(self, text="Draw a digit", font=("Helvetica", 16))
        self.label.pack(pady=10)

        self.canvas.bind("<B1-Motion>", self.draw_lines)
        self.model = build_model()
        self.model.load_weights("trained_model.h5")

    def draw_lines(self, event):
        x, y = event.x, event.y
        r = 8
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.label.config(text="Draw a digit")

    def predict_digit(self):
        self.canvas.update()
        self.canvas.postscript(file="canvas.eps")
        img = Image.open("canvas.eps").convert("L")
        img = ImageOps.invert(img)
        img = img.resize((28, 28))
        img = np.array(img).astype("float32") / 255.0
        img = np.expand_dims(img, axis=(0, -1))
        pred = self.model.predict(img)
        digit = np.argmax(pred)
        self.label.config(text=f"Predicted Digit: {digit}")

if __name__ == "__main__":
    app = App()
    app.mainloop()