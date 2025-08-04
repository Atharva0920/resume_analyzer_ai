
### =====================================================
# Development Setup Instructions
### =====================================================


## Quick Start Guide:

1. Backend Setup:
   ```bash
   pip install -r requirements.txt
   python app.py  # Runs on http://localhost:5000
   ```

2. Frontend Setup:
   ```bash
   npm start  # Runs on http://localhost:3000
   ```

3. Test Integration:
   - Upload a resume file
   - Check browser network tab for API calls
   - Verify responses in browser console

## File Structure:
```
project/
├── server/
│   ├── main.py (Flask API)
│   ├── api
│   │   └── routes.py
│   ├── core
│   │   └── config.py
│   ├── models
│   │   └── schemas.py
│   ├── .gitignore
│   └── requirements.txt
├── client/
│   ├── src/
│   │   ├── App.js
│   │   ├── App.js
│   │   ├── App.test.js
│   │   ├── index.css
│   │   ├── index.js
│   │   ├── reportWebVitals.js
│   │   └── setupTests.js
│   ├── package.json
│   ├── package.lock.json
│   ├── public/
│   └── node_modules/
└── README.md
```
