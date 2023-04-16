from codecarbon import EmissionsTracker

# Initialize the tracker
tracker = EmissionsTracker()

# Start tracking emissions
tracker.start()

# Run your compute-intensive code here
# e.g., training a machine learning model

# Stop tracking emissions
tracker.stop()

# Print the emissions data
emissions_data = tracker.get_emissions_data()
print(emissions_data)