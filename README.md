# 🚀 Smart Blueprint Chat - Backend

Welcome to the backend of the Smart Blueprint Chat, engineered for performance, scalability, and maintainability. This system seamlessly integrates AI with robust cloud infrastructure to allow deep interaction with architectural blueprints.

---

## 🏗️ Architecture

This backend follows a clean, service-oriented architecture:

-   **/app/api**: Contains all the FastAPI routers and endpoint definitions.
-   **/app/core**: Holds core logic like configuration management (`config.py`).
-   **/app/models**: Defines data structures and schemas (`schemas.py`).
-   **/app/services**: Contains the business logic (e.g., `ai_service.py`, `storage_service.py`).
-   **main.py**: The main application entry point that ties everything together.
-   **Dockerfile**: Defines the container for deployment.
-   **requirements.txt**: Lists all Python dependencies.

---

## 🛠️ Getting Started

### Prerequisites

-   Python 3.11+
-   An OpenAI API Key
-   An Azure Storage Account

### Local Setup

1.  **Clone the repository** (if applicable)

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: .\venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure your environment:**
    -   Copy `.env.example` to a new file named `.env`.
    -   Open `.env` and fill in your `OPENAI_API_KEY` and `AZURE_STORAGE_CONNECTION_STRING`.

5.  **Run the server:**
    ```bash
    uvicorn main:app --reload
    ```

The API will be available at `http://127.0.0.1:8000` and the interactive documentation at `http://127.0.0.1:8000/docs`.

---

## ✨ Adding New Features

To add a new feature, such as "User Management":

1.  Define new data models in `app/models/schemas.py`.
2.  Create a new service `app/services/user_service.py` with the business logic.
3.  Create new API routes in `app/api/routes/user_routes.py`.
4.  Import and include the new router in `main.py`.# test
