
# 🏦 RBI Legal Chatbot

The **RBI Legal Chatbot** is an AI-powered assistant for regulatory compliance and legal research. It allows users to query RBI-related Acts, Rules, and Case Laws across courts such as the Supreme Court, High Courts, NCLT, and NCLAT.

This chatbot is ideal for legal professionals, compliance teams, and researchers seeking concise and relevant insights from large legal corpora.

---

## ⚙️ How It Works

### 1. 🗂️ Data & Embeddings

Prepare Chroma vector databases from uploaded PDFs (already in the repo):

```bash
python backend/diffchroma.py
```

* Documents are embedded using HuggingFace embeddings.
* Existing vectorstores are deleted and rebuilt for consistency.

---

### 2. 🧠 FastAPI Backend

The backend is defined in `main.py` and is managed by a Node.js wrapper to start/stop automatically.

> Do **not** run FastAPI manually – use `server.js` instead (explained below).

---

### 3. 🔁 Node.js Proxy + Backend Auto Control

The `server.js` file:

* Proxies `/api/*` requests to the FastAPI backend
* Launches `start_fastapi.bat` or `.sh` depending on OS
* Auto-shuts down the backend after 2 minutes of inactivity

Run the proxy:

```bash
npm install
node server.js
```

---

### 4. 💬 React Frontend (Chat UI)

Navigate to the `frontend/` folder and run:

```bash
npm install
npm start
```

Visit: `http://localhost:3000`

The frontend supports:

* Chat input for RBI legal queries
* Real-time status (backend up/down)
* Copyable answers and notes

---

## 🧾 API Overview

FastAPI exposes (via Node.js proxy):

| Method | Route                   | Description                      |
| ------ | ----------------------- | -------------------------------- |
| POST   | `/api/answer`           | Get legal answer with context    |
| GET    | `/api/backend-status`   | Check if FastAPI is live         |
| POST   | `/api/start-backend`    | Start backend (optional trigger) |
| POST   | `/api/shutdown-backend` | Force backend shutdown           |

---

## 📁 Project Layout

```
RBI-catbot-main/
├── backend/
│   ├── main.py              # FastAPI backend
│   ├── diffchroma.py        # Vector DB loader
│   └── start_fastapi.bat    # Script to launch backend
├── data/                    # All legal PDFs (Acts, Rules, Cases)
├── frontend/                # React-based chatbot UI
├── server.js                # Node.js proxy with auto backend management
└── README.md
```

---

## 📌 Requirements

### Backend:

* Python 3.9+
* FastAPI, Uvicorn
* LangChain, Chroma
* PyMuPDF, pdfminer

### Frontend:

* Node.js 18+
* React 18+

---

## ✅ Features

* Full-text legal question answering from:

  * Acts
  * Rules
  * Supreme Court
  * High Court
  * NCLAT
  * NCLT
* Automatic backend lifecycle management
* Interactive legal search interface
* Backend health monitoring and recovery
