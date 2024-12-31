import tkinter as tk
from tkinter import messagebox
import random
from PIL import Image, ImageTk

class DiceSimulatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dice Simulator")

        self.bg_image = Image.open("dice.jpg")
        self.bg_image = self.bg_image.resize((375, 375), Image.Resampling.LANCZOS)
        self.bg_photo = ImageTk.PhotoImage(self.bg_image)

        self.bg_label = tk.Label(self.root, image=self.bg_photo)
        self.bg_label.place(relwidth=1, relheight=1)

        self.dice_images = {
            1: tk.PhotoImage(file="dice_1.png"),
            2: tk.PhotoImage(file="dice_2.png"),
            3: tk.PhotoImage(file="dice_3.png"),
            4: tk.PhotoImage(file="dice_4.png"),
            5: tk.PhotoImage(file="dice_5.png"),
            6: tk.PhotoImage(file="dice_6.png")
        }

        self.dice_label1 = tk.Label(self.root)
        self.dice_label2 = tk.Label(self.root)

        self.roll_button = tk.Button(self.root, text="Roll Dice", command=self.roll_dice)

        self.sum_label = tk.Label(self.root, text="Sum: 0", font=("Times", 14), bg="white")

        self.turn_label = tk.Label(self.root, text="Person 1's turn", font=("Arial", 12), bg="white")

        self.dice_label1.grid(row=0, column=0, padx=10, pady=10)
        self.dice_label2.grid(row=0, column=1, padx=10, pady=10)

        self.turn_label.grid(row=1, column=0, columnspan=2, pady=10)

        self.roll_button.grid(row=2, column=0, columnspan=2, pady=20)

        self.sum_label.grid(row=3, column=0, columnspan=2)

        self.num_players = 0 
        self.current_player = 1
        self.scores = []

        self.root.geometry("300x400")

        self.ask_for_players()

    def ask_for_players(self):
        self.input_window = tk.Toplevel(self.root)
        self.input_window.title("Enter Number of Players")
        
        self.num_players_label = tk.Label(self.input_window, text="Enter number of players:")
        self.num_players_label.pack(pady=10)

        self.num_players_entry = tk.Entry(self.input_window)
        self.num_players_entry.pack(pady=10)

        self.submit_button = tk.Button(self.input_window, text="OK", command=self.start_game)
        self.submit_button.pack(pady=10)

        self.input_window.geometry("300x150")

    def start_game(self):
        try:
            self.num_players = int(self.num_players_entry.get())
            if self.num_players < 1:
                raise ValueError("Number of players must be at least 1.")
            
            self.scores = [0] * self.num_players
            self.current_player = 1
            self.turn_label.config(text=f"Person {self.current_player}'s turn")

            self.input_window.destroy()

            self.roll_button.config(state="normal")
        
        except ValueError as e:
            messagebox.showerror("Invalid Input", str(e))

    def roll_dice(self):
        try:
            dice1_roll = random.randint(1, 6)
            dice2_roll = random.randint(1, 6)

            dice1_image = self.dice_images[dice1_roll]
            dice2_image = self.dice_images[dice2_roll]

            self.dice_label1.config(image=dice1_image)
            self.dice_label2.config(image=dice2_image)

            dice_sum = dice1_roll + dice2_roll
            self.scores[self.current_player - 1] += dice_sum
            self.sum_label.config(text=f"Sum: {dice_sum}")

            self.turn_label.config(text=f"Person {self.current_player % self.num_players + 1}'s turn")

            self.current_player += 1
            if self.current_player > self.num_players:
                self.current_player = 1

            self.root.update_idletasks()

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = DiceSimulatorApp(root)
    root.mainloop()
