# `show_logs` target: Run the MLflow server to visualize experiment logs
# Start the MLflow server with the specified configuration
# Set the URI for the backend store (where MLflow metadata is stored)
# Set the default root directory for storing artifacts (e.g., models, plots)
# Set the host for the MLflow server to bind to (localhost in this case)
show_logs:
	mlflow server \
		--backend-store-uri sqlite:///mlflow.db \
		--host 127.0.0.1 \
		--port 5000

# `stop_server` target: Check if an MLflow server is running on port 5000 and shut it down if it is
# Find the process listening on port 5000, filter by 'mlflow', extract its process ID, and terminate it
stop_server:
	@-lsof -i :5000 -sTCP:LISTEN | grep 'mlflow' | awk '{ print $$2 }' | xargs -I {} kill {}
