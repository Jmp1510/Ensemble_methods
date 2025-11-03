import tkinter as tk
import subprocess
import os

# Function to run a Python script
def run_script(script_name):
    """
    Function to execute a Python script.
    """
    try:
        # Execute the Python script using subprocess
        subprocess.run(['python', script_name], check=True)
        print(f"Successfully ran {script_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error executing {script_name}: {e}")

# Function to create the GUI
def create_gui():
    # Create the main window
    root = tk.Tk()
    root.title("Project 3")

    # Set the window size (Width x Height)
    root.geometry("400x200")

    btn_width = 30

    # Add labels and buttons for scripts
    label = tk.Label(root, text="Select a dataset to run.", font=("Arial", 14))
    label.pack(pady=15)

    # Buttons for each model
    btn1 = tk.Button(root, text="Customer Segmentation Dataset", command=lambda: run_script('customer_segmentation.py'), width=btn_width)
    btn1.pack(pady=5)

    btn2 = tk.Button(root, text="Patient Survival Dataset", command=lambda: run_script('patientsurvival.py'), width=btn_width)
    btn2.pack(pady=5)


    # Button to exit the application
    btn_exit = tk.Button(root, text="Exit", command=root.quit)
    btn_exit.pack(pady=10)

    # Start the GUI loop
    root.mainloop()

# Main function to run the GUI
if __name__ == '__main__':
    create_gui()
