#!/bin/bash

# Find the process ID (PID) of the process listening on port 5000 and kill it
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    # Windows
    PID=$(netstat -aon | grep ":5000" | findstr "LISTENING" | awk '{ print $5 }')
    if [ -z "$PID" ]; then
        echo "No process found running on port 5000"
    else
        echo "Killing process with PID $PID"
        taskkill /PID $PID /F
    fi
else
    # Unix-like systems
    PID=$(lsof -i :5000 -sTCP:LISTEN | grep 'mlflow' | awk '{ print $2 }')
    if [ -z "$PID" ]; then
        echo "No process found running on port 5000"
    else
        echo "Killing process with PID $PID"
        kill $PID
    fi
fi
