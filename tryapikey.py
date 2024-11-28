import time

import requests
import pandas as pd

##############################################################################################################
#       GET DISTANCE MATRIX
#
###############################################################################################################


API_KEY = 'your api key'




import pandas as pd
import requests
import numpy as np
import time

# Load the Excel file
read_df = pd.read_excel('novi_podaci/MLP lokacije.xlsx')

# Drop rows with missing Lat/Lon
filtered = read_df.dropna(subset=['Lat', 'Lon'])

# Ensure Lat/Lon columns are float
filtered['Lat'] = filtered['Lat'].astype(float)
filtered['Lon'] = filtered['Lon'].astype(float)

# Prepare locations and metadata
locations = filtered[["Lon", "Lat"]].values.tolist()
location_ids = filtered["Šifra"].tolist()

# API Key and endpoint
  # Replace with your valid OpenRouteService API key
headers = {
    "Accept": "application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8",
    "Authorization": API_KEY,
    "Content-Type": "application/json; charset=utf-8"
}

# Matrix for durations
duration_matrix = pd.DataFrame(np.nan, index=location_ids, columns=location_ids)

# Function to batch the locations
def batch_data(locations, batch_size):
    for i in range(0, len(locations), batch_size):
        yield locations[i:i + batch_size]

# Function to fetch matrix data for given source and destination batches
def fetch_matrix(source_batch, dest_batch):
    body = {
        "locations": source_batch + dest_batch,
        "sources": list(range(len(source_batch))),
        "destinations": list(range(len(source_batch), len(source_batch) + len(dest_batch))),
        "metrics": ["duration"]
    }
    response = requests.post(
        "https://api.openrouteservice.org/v2/matrix/driving-car",
        json=body,
        headers=headers
    )
    if response.status_code == 200:
        return response.json()["durations"]
    else:
        print(f"Error: {response.status_code} - {response.reason}")
        print(response.text)
        return None

# Process locations in smaller batches (e.g., 59 locations per batch)
batch_size = 59
for i, source_batch in enumerate(batch_data(locations, batch_size)):
    for j, dest_batch in enumerate(batch_data(locations, batch_size)):
        matrix_data = fetch_matrix(source_batch, dest_batch)
        if matrix_data is not None:
            source_ids = location_ids[i * batch_size:(i + 1) * batch_size]
            dest_ids = location_ids[j * batch_size:(j + 1) * batch_size]
            batch_df = pd.DataFrame(matrix_data, index=source_ids, columns=dest_ids)
            duration_matrix.update(batch_df)
        # Sleep between requests to respect rate limits
        time.sleep(15)

# Retry to fill NaN values for missing entries
missing_pairs = np.where(duration_matrix.isna())
for src_idx, dest_idx in zip(*missing_pairs):
    source = locations[src_idx:src_idx + 1]
    dest = locations[dest_idx:dest_idx + 1]
    matrix_data = fetch_matrix(source, dest)
    if matrix_data is not None:
        duration_matrix.iloc[src_idx, dest_idx] = matrix_data[0][0]
    time.sleep(2)  # Small delay between single requests

# Save final matrix to CSV
duration_matrix.to_csv("travel_duration_matrix_complete__yasd.csv")

# Show the final matrix
print("Final Travel Duration Matrix:")
print(duration_matrix)

##############################################################################################################
#       GET DURATION MATRIX
#
###############################################################################################################

# Load the Excel file
read_df = pd.read_excel('novi_podaci/MLP lokacije.xlsx')  # Load location data from an Excel file.

# Drop rows with missing Lat/Lon
filtered = read_df.dropna(subset=['Lat', 'Lon'])  # Remove rows where 'Lat' or 'Lon' values are missing.

# Ensure Lat/Lon columns are float
filtered['Lat'] = filtered['Lat'].astype(float)  # Ensure latitude values are floats.
filtered['Lon'] = filtered['Lon'].astype(float)  # Ensure longitude values are floats.

# Prepare locations and metadata
locations = filtered[["Lon", "Lat"]].values.tolist()  # Convert 'Lon' and 'Lat' columns to a list of [longitude, latitude] pairs.
location_ids = filtered["Šifra"].tolist()  # Extract unique identifiers (e.g., location codes) from the 'Šifra' column.

# API Key and endpoint
 # Replace with your actual OpenRouteService API key.
headers = {
    "Accept": "application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8",
    "Authorization": API_KEY,  # Include API key in the Authorization header.
    "Content-Type": "application/json; charset=utf-8"  # Specify the content type as JSON.
}

# Matrix for distances
distance_matrix = pd.DataFrame(np.nan, index=location_ids, columns=location_ids)  # Initialize a DataFrame with NaN values for distances.

# Function to batch the locations
def batch_data(locations, batch_size):
    """
    Splits the list of locations into smaller batches of a given size.
    """
    for i in range(0, len(locations), batch_size):  # Iterate in steps of batch_size.
        yield locations[i:i + batch_size]  # Yield the current batch of locations.

# Function to fetch matrix data for given source and destination batches
def fetch_matrix(source_batch, dest_batch):
    """
    Sends a POST request to the OpenRouteService API to calculate the distance matrix
    for a source and destination batch of locations.
    """
    body = {
        "locations": source_batch + dest_batch,  # Combine source and destination locations.
        "sources": list(range(len(source_batch))),  # Define indices of source locations.
        "destinations": list(range(len(source_batch), len(source_batch) + len(dest_batch))),  # Define indices of destination locations.
        "metrics": ["distance"]  # Request the distance metric instead of duration.
    }
    response = requests.post(
        "https://api.openrouteservice.org/v2/matrix/driving-car",  # API endpoint for driving-car distance matrix.
        json=body,  # Send the request body as JSON.
        headers=headers  # Include headers with the API key.
    )
    if response.status_code == 200:  # Check if the request was successful.
        return response.json()["distances"]  # Return the distances matrix.
    else:
        print(f"Error: {response.status_code} - {response.reason}")  # Print error information if the request fails.
        print(response.text)  # Print the response body for debugging.
        return None  # Return None if the request fails.

# Process locations in smaller batches (e.g., 59 locations per batch)
batch_size = 59  # Define the maximum number of locations per batch (API constraint).
for i, source_batch in enumerate(batch_data(locations, batch_size)):  # Iterate over batches of source locations.
    for j, dest_batch in enumerate(batch_data(locations, batch_size)):  # Iterate over batches of destination locations.
        matrix_data = fetch_matrix(source_batch, dest_batch)  # Fetch the distance matrix for the current batch pair.
        if matrix_data is not None:  # If the request was successful:
            source_ids = location_ids[i * batch_size:(i + 1) * batch_size]  # Get the source IDs for the batch.
            dest_ids = location_ids[j * batch_size:(j + 1) * batch_size]  # Get the destination IDs for the batch.
            batch_df = pd.DataFrame(matrix_data, index=source_ids, columns=dest_ids)  # Create a DataFrame for the batch data.
            distance_matrix.update(batch_df)  # Update the main distance matrix with the batch data.
        # Sleep between requests to respect rate limits
        time.sleep(15)  # Pause for 15 seconds to avoid exceeding the API rate limit.

# Retry to fill NaN values for missing entries
missing_pairs = np.where(distance_matrix.isna())  # Identify rows and columns where the matrix has NaN values.
for src_idx, dest_idx in zip(*missing_pairs):  # Iterate over each missing (NaN) pair.
    source = locations[src_idx:src_idx + 1]  # Get the source location for the missing entry.
    dest = locations[dest_idx:dest_idx + 1]  # Get the destination location for the missing entry.
    matrix_data = fetch_matrix(source, dest)  # Fetch the distance for the missing pair.
    if matrix_data is not None:  # If the request was successful:
        distance_matrix.iloc[src_idx, dest_idx] = matrix_data[0][0]  # Update the matrix with the fetched value.
    time.sleep(2)  # Small delay between single requests to avoid rate-limiting.

# Save final matrix to CSV
distance_matrix.to_csv("travel_distance_matrix_complete.csv")  # Save the final matrix to a CSV file.

# Show the final matrix
print("Final Travel Distance Matrix:")
print(distance_matrix)  # Print the completed distance matrix.



if __name__ == '__main__':
    import pandas as pd

    df = pd.read_csv('travel_duration_matrix_batched.csv')
    df.head(3)
    print
