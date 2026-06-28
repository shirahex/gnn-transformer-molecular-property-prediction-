#!/bin/bash
# Molecular Property Prediction - One-Command Startup Script
# Usage: chmod +x run.sh && ./run.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Molecular Property Prediction${NC}"
echo -e "${BLUE}  Drug Discovery with Explainable AI${NC}"
echo -e "${BLUE}========================================${NC}"

# Detect Python command
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo -e "${RED}Error: Python is not installed${NC}"
    exit 1
fi

echo -e "${BLUE}Using Python: $PYTHON${NC}"

# Check Node.js
if ! command -v node &> /dev/null; then
    echo -e "${RED}Error: Node.js is not installed. Please install Node.js 18+.${NC}"
    exit 1
fi

NODE_VERSION=$(node --version | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    echo -e "${YELLOW}Warning: Node.js 18+ required. Current: $(node --version)${NC}"
fi

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Function to cleanup processes on exit
cleanup() {
    echo -e "\n${YELLOW}Shutting down services...${NC}"
    if [ -n "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null || true
    fi
    if [ -n "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null || true
    fi
    echo -e "${GREEN}All services stopped.${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM

# ============================================
# BACKEND
# ============================================
echo -e "\n${BLUE}[1/4] Starting Backend...${NC}"
cd "$PROJECT_ROOT/backend"

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    $PYTHON -m venv venv
fi

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
else
    echo -e "${RED}Error: Could not activate virtual environment${NC}"
    exit 1
fi

# Install dependencies
echo -e "${YELLOW}Installing Python dependencies...${NC}"
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo -e "${GREEN}Backend dependencies installed.${NC}"

# Start backend
export MOLECULAR_DEVICE="auto"
export MOLECULAR_EPOCHS="50"
export MOLECULAR_BATCH_SIZE="32"

echo -e "${YELLOW}Starting FastAPI backend on port 8000...${NC}"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for backend to start
sleep 3
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo -e "${RED}Backend failed to start!${NC}"
    exit 1
fi
echo -e "${GREEN}Backend running at http://localhost:8000${NC}"
echo -e "${GREEN}API docs at http://localhost:8000/docs${NC}"

# ============================================
# FRONTEND
# ============================================
echo -e "\n${BLUE}[2/4] Starting Frontend...${NC}"
cd "$PROJECT_ROOT/frontend"

# Install dependencies
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}Installing npm dependencies...${NC}"
    npm install
fi

echo -e "${YELLOW}Starting React frontend on port 3000...${NC}"
npm start &
FRONTEND_PID=$!

# Wait for frontend
sleep 5
if ! kill -0 $FRONTEND_PID 2>/dev/null; then
    echo -e "${YELLOW}Frontend dev server may have issues. Check above for errors.${NC}"
fi

echo -e "${GREEN}Frontend running at http://localhost:3000${NC}"

# ============================================
# STATUS
# ============================================
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}  All services are running!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${BLUE}Frontend:${NC}  http://localhost:3000"
echo -e "${BLUE}Backend:${NC}   http://localhost:8000"
echo -e "${BLUE}API Docs:${NC}  http://localhost:8000/docs"
echo -e ""
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"

# Wait for interrupt
wait
