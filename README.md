# Spanish Sign Language Detection

Over the past decade, **computer vision** has made significant advances, enabling the development of systems capable of recognizing objects, detecting faces, and interpreting complex scenes. These advancements have been applied in various fields, such as autonomous vehicle systems, which can identify pedestrians, other vehicles, and even traffic signs to improve road safety.

However, these technologies also have enormous potential for social impact, particularly in the inclusion of people with disabilities. The objective of this project is to explore the capabilities of computer vision for the detection of signs in **Spanish Sign Language (LSE)**, aiming to facilitate communication and provide tools that help reduce accessibility barriers for the deaf community.

# Analysis Document

I prepared a **comprehensive and detailed document** covering all the insights obtained during the development of this project. You can access it through this link: [Analysis Document](https://docs.google.com/document/d/1sZoczUxFTcYERhrcVcfkPUARvrgqhNzgREj5DFiRui0/edit?usp=sharing)

# Project Structure

The project consists of a main file (*main_principal.py*) from which everything is executed. It features a main menu that allows users to navigate and run each of the developed tasks: **classic menu (shallow learning)** and **MediaPipe menu (deep learning)**.

For better organization and code clarity, the project has been divided into two main modules, each with a specific purpose:
- *clasico/*: Contains files for performing sign detection using classical methods, without neural networks.
- *pipeline_mediapipe/*: Contains files for performing sign detection using the MediaPipe library and deep learning techniques.


# Dependencies / Libraries

Various Python libraries were used to develop this project and ensure correct execution. To facilitate installation and reproducibility, a *requirements.txt* file is provided with all required dependencies.

It is recommended to create a virtual environment before running the project to keep dependencies isolated and avoid conflicts with other environments. Follow these steps:

1. Create a virtual environment
> python3 -m venv env_name

2. Activate the virtual environment
> source env_name/bin/activate   # macOS or Linux

> env_name\Scripts\activate      # Windows


3. Install required dependencies
> pip install -r requirements.txt
