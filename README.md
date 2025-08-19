Elyx Hackathon Project

🚀 Overview
This project simulates personalized health journeys for Elyx Life members.
It has two major parts:
Conversation Simulator → Generates WhatsApp-style chats between the member (Rohan Patel) and the Elyx team (Ruby, Dr. Warren, Advik, Carla, Rachel, Neel).
Journey Dashboard → A web app to visualize the member’s 8-month journey, interventions, and reasoning behind each decision.

The goal is to show how AI/LLMs can support proactive health management by generating realistic interactions and transparent “why” explanations for medical/lifestyle interventions.

✨ Features

🔄 Automated conversation generator (Python)
Member-initiated and Elyx-initiated chats.
Plan updates, travel adaptations, diagnostic panels, and weekly summaries.
Role-specific tones for each Elyx team member.

📊 Journey timeline visualization (React + Tailwind + Recharts)
Event timeline with plan updates, reports, and interventions.
Click an event → see decision + reasoning + linked chat messages.
Filter by event type (tests, plan updates, travel, reports).

🔍 “Why?” Drill-down
For every intervention, trace the trigger, reasoning, and outcome metrics.

🧑 Persona snapshot
Member’s goals, travel frequency, adherence level, and upcoming tests.


HOW TO RUN THE CODE
Step 1: Clone the repo to your local machine.
Step 2: Change the path to the Python environment and import the given modules in the environment.
Run the command: pip install fastapi uvicorn google-generativeai python-dotenv
Step 3: Run the command: $env: GEMINI_API_KEY="your_api_key_here"  (You need to keep your Gemini API key handy)
Step 4: Change the directory of the env to the scripts folder. (cd scripts)
Step 5: Run the command: uvicorn main: app --reload 
Step 6: Run the index.html script in your local env.
Step 7: On the webpage, run the conversation, and it will start generating the conversation.

