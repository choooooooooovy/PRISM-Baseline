# CASVE Decision Support System

Web application based on the CASVE (Communication, Analysis, Synthesis, Valuing, Execution) decision-making model

## Project Structure

```
baseline/
├── FE/                 # Frontend (Next.js)
│   ├── app/           # Next.js App Router
│   ├── components/    # UI Components (Radix UI)
│   ├── features/      # Feature modules (worksheet)
│   └── lib/           # Utilities, types
│
└── BE/                # Backend (FastAPI)
    ├── main.py        # FastAPI entry point
    ├── routers/       # API endpoints
    ├── services/      # OpenAI integration
    ├── utils/         # Logger utilities
    └── logs/          # Activity & LLM logs
```

## Tech Stack

### Frontend (FE/)
- **Framework**: Next.js 14+ (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS v4
- **UI Components**: Radix UI
- **State Management**: Context API + useReducer
- **Storage**: localStorage (auto-save)
- **Icons**: lucide-react

### Backend (BE/)
- **Framework**: FastAPI
- **Language**: Python 3.13
- **LLM**: OpenAI GPT-4o
- **Logging**: JSON-based file logging
- **Server**: Uvicorn

## Key Features

### 1. CASVE Decision-Making Worksheet (5 Steps)
- **Step 0**: Self Profile (values, interests, strengths, constraints, concerns)
- **Step 1**: Communication (problem definition, internal/external cues, key questions)
- **Step 2**: Analysis (evaluation criteria, constraints, information template)
- **Step 3**: Synthesis (option generation & profiles, LLM-assisted)
- **Step 4**: Valuing (priority ranking, final choice, tradeoff statement)

### 2. AI-Powered Option Generation
- Automatically generates personalized career/decision alternatives based on Steps 0-2 input
- Uses OpenAI GPT-4o model
- Structured prompt engineering

### 3. Real-time Preview & Validation
- Live preview of inputs in the right panel
- Step-by-step required field validation
- Missing items checklist

### 4. Final Report Generation
- Comprehensive report of Steps 0-4 results
- Highlighted final decision
- Print/save functionality

### 5. Logging System
- User activity logs (per session, daily)
- LLM generation history (prompts, responses, token usage)
- JSON file-based storage

## Getting Started

### 1. Running the Frontend

```bash
cd FE
npm install
npm run dev
```

Frontend: http://localhost:3000

### 2. Running the Backend

```bash
cd BE
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configure .env file (OPENAI_API_KEY required)
cp .env.example .env
# Set OPENAI_API_KEY in .env file

# Start server
uvicorn main:app --reload --port 8000
```

Backend API: http://localhost:8000
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### POST /api/generate-options
Generate AI alternatives based on Steps 0-2 data

**Request:**
```json
{
  "sessionId": "session-123",
  "step0": {
    "values": ["autonomy", "creativity"],
    "interests": ["development", "design"],
    "strengths": ["problem solving"],
    "mustHaveConstraints": ["remote work"],
    "niceToHaveConstraints": [],
    "concerns": "Struggling with career direction..."
  },
  "step1": {
    "problemDefinition": "Career choice after graduation",
    "internalCues": ["changing interests"],
    "externalCues": ["graduation timeline"],
    "keyQuestions": ["Which field is suitable?"]
  },
  "step2": {
    "evaluationCriteria": ["salary", "growth potential"],
    "constraints": ["location constraints"],
    "informationTemplate": []
  }
}
```

**Response:**
```json
{
  "success": true,
  "options": [
    {
      "title": "Frontend Developer",
      "description": "...",
      "profile": {
        "coreRole": "...",
        "requiredSkills": "...",
        "environment": "...",
        "growth": "..."
      },
      "matchReason": "..."
    }
  ],
  "tokensUsed": {
    "prompt_tokens": 500,
    "completion_tokens": 800,
    "total_tokens": 1300
  }
}
```

## Environment Variables

### Frontend (FE/.env.local)
```env
# Currently no environment variables needed (uses localStorage)
# Will be added when API integration is implemented
```

### Backend (BE/.env)
```env
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_MODEL=gpt-4o
FRONTEND_URL=http://localhost:3000
```

## Development Features

### Frontend
- **Korean IME Bug Fix**: Prevents duplicate saves during Korean input (`e.nativeEvent.isComposing`)
- **Auto-save**: All inputs are saved to localStorage in real-time
- **Drag & Drop**: Drag-to-rank option priorities in Step 4
- **Save Button UX**: Save button pattern for long text inputs
- **Step Validation**: Cannot proceed to next step without completing required fields

### Backend
- **No Database**: Simplified with JSON file-based logging
- **Structured Prompts**: Steps 0-2 data systematically structured for LLM
- **Detailed Logging**: All API calls and LLM generation history recorded
- **CORS Configuration**: Secure communication with frontend

## Log Files

Daily log files created in BE/logs/ directory:

- `app.log`: General application logs
- `user_activity_YYYYMMDD.json`: User activity logs
- `llm_generations_YYYYMMDD.json`: LLM generation history

## Production Deployment

### Frontend
```bash
cd FE
npm run build
npm start
```

### Backend
```bash
cd BE
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## License

This project is developed for educational/research purposes.

## Contributing

Bug reports and feature suggestions are welcome via GitHub Issues.

---

**Development Period**: January 2026
**Key Technologies**: Next.js 14, TypeScript, FastAPI, OpenAI GPT-4o
