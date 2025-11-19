# 🏢 Room Booking Assistant

An AI-powered assistant for booking rooms, built with FastAPI (backend), React (frontend), and LLM-based natural language understanding.

---

## 🚀 Features

- **Natural Language Booking:** Users can book, check, or cancel rooms using plain English.
- **LLM Integration:** Uses DeepSeek/OpenAI for intent and entity extraction.
- **Entity Extraction Fallback:** spaCy + dateutil for robust parameter parsing.
- **Multi-turn Dialog:** Handles missing info and guides users to complete bookings.
- **Availability Suggestions:** Offers alternative slots if requested time is unavailable.
- **Database Support:** SQLite/MySQL via SQLAlchemy.

---

## 🛠️ Tech Stack

| Layer      | Technology                |
|------------|--------------------------|
| Frontend   | React, Zustand, MUI      |
| Backend    | FastAPI                  |
| LLM Engine | DeepSeek / OpenAI        |
| NLP        | spaCy, dateutil          |
| DB         | SQLite/MySQL             |

---

## ⚡ Quickstart

1. **Clone the repo**  
   `git clone <your-repo-url>`

2. **Backend Setup**
   ```bash
   cd backend
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   uvicorn main:app --reload
   ```

3. **Frontend Setup**
   ```bash
   cd frontend
   npm install
   npm start
   ```

4. **Environment Variables**  
   Add your OpenAI/DeepSeek API keys to `.env`.

---

## 📦 API Endpoints

- `/fetch_bookings?room_name=...` – Get bookings for a room
- `/check_availability/` – Check if a room is available (requires room_name, date, start_time, end_time)

---

## 💬 Example Conversation

> **User:** Book LT1 on 2025-06-25 from 3PM to 4PM  
> **Bot:** Booking confirmed for LT1 on 2025-06-25 from 15:00 to 16:00

---

## 📝 Documentation

See [RoomBookingAssistant_Documentation.md](c:\Users\eanus\Downloads\RoomBookingAssistant_Documentation.md) for detailed architecture, prompt examples, and conversation flows.

---

## 🧩 Future Improvements

- Session persistence
- User profile preferences
- Calendar integration

---

## 📄 License