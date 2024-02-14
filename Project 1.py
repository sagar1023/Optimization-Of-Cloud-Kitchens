# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 04:05:34 2023

@author: SAGAR
"""

#-----FINAl----

#Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import geopy.distance
from geopy.geocoders import Nominatim
import random
from tabulate import tabulate
import pulp
from scipy.sparse import csr_matrix, save_npz

#------ TASK 1 ----->
#Using the below code to ead our csv file into the python library
file_path = 'C:/Users/SAGAR/OneDrive/Desktop/Sagar/NCSU/MEM/ISE 535 Python for ISE/Project-28/TestData.csv' #Specify the systems file location
data = pd.read_csv(file_path)


#Using this to initialize geolocator library
geolocator = Nominatim(user_agent="Project Team 28")

#Obtaining the latitude and longitude coordinates for each address stored.
latitudes = []
longitudes = []
zip_codes = []


#Retreieving the zip codes , latitudes and longitudes.
for address in data['Address']:
    location = geolocator.geocode(address)
    latitudes.append(location.latitude)
    longitudes.append(location.longitude)
    zip_codes.append(address[-5:])

    
data['Latitude'] = latitudes
data['Longitude'] = longitudes
data['Zip Code'] = zip_codes

#Minimu distance between cloud kitchens should be atleast 0.5 miles.
min_distance_miles = 0.5
for i in range(len(latitudes)):
    for j in range(i+1, len(latitudes)):
        coords_1 = (latitudes[i], longitudes[i])
        coords_2 = (latitudes[j], longitudes[j])
        distance = geopy.distance.distance(coords_1, coords_2).miles
        if distance < min_distance_miles:
           print(f"{data['Name'][i]} under 0.5 miles")

#Saving the updated sheet back into the file to register the zipcodes.
data.to_csv("C:/Users/SAGAR/OneDrive/Desktop/Sagar/NCSU/MEM/ISE 535 Python for ISE/Project-28/TestData.csv", index=False) #Specify the systems file location

#Plotting the locations on the graph using scatter plot
plt.scatter(longitudes, latitudes, color='blue', label='Cloud Kitchen Location')
for lon, lat, label in zip(longitudes, latitudes, data['Name']):
    plt.text(lon, lat, label, fontsize=9, ha='right')

#Making sure there is atleast 2.5 miles boundary in the graph in all directions from the furthest location.
buffer_miles = 2.5  # 2.5 miles buffer
east_most_point = max(longitudes)
west_most_point = min(longitudes)
north_most_point = max(latitudes)
south_most_point = min(latitudes)

east_bound = geopy.distance.distance(miles=buffer_miles).destination((north_most_point, east_most_point), 90)[1]
west_bound = geopy.distance.distance(miles=buffer_miles).destination((north_most_point, west_most_point), 270)[1]
north_bound = geopy.distance.distance(miles=buffer_miles).destination((north_most_point, east_most_point), 0)[0]
south_bound = geopy.distance.distance(miles=buffer_miles).destination((south_most_point, east_most_point), 180)[0]

plt.xlim(west_bound, east_bound)
plt.ylim(south_bound, north_bound)

#Setting a random seed for reproducibility
random.seed(28)

#Generating 50 random service station's latitudes and longitudes which will be used to plot their locations on the map.
service_station_latitudes = [random.uniform(south_bound, north_bound) for _ in range(50)]
service_station_longitudes = [random.uniform(west_bound, east_bound) for _ in range(50)]

#Plotting the service station locations on the graph in red for better visibility against cloud kitchens which are plotted in blue.
plt.scatter(service_station_longitudes, service_station_latitudes, color='red', label='Service Station')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('25 Cloud Kitchens and 50 Service Stations')
plt.legend()
plt.grid(True)
plt.savefig("C:/Users/SAGAR/OneDrive/Desktop/Sagar/NCSU/MEM/ISE 535 Python for ISE/Project-28/Locations.jpeg", format='jpeg')
plt.show()
#Assigning legend , grids and labels in the above code.

#Initialiazing a table
table_data = []

#Adding Cloud Kitchen data to the table
for i, (name, address, zip_code, lat, lon) in enumerate(zip(data['Name'], data['Address'], data['Zip Code'], latitudes, longitudes)):
    table_data.append([i, name, address, zip_code, (lat, lon)])

#Adding Service Station data to the table
for i, (lat, lon) in enumerate(zip(service_station_latitudes, service_station_longitudes)):
    table_data.append([len(data) + i, "Service Station " + str(i+1), "N/A", "N/A", (lat, lon)])


#Creating a python table of collected data using tabulate library
print(tabulate(table_data, headers=['Index', 'Name', 'Street Address', 'Zip Code', 'Coordinates']))


#Below is a function to calculate distance in miles between two points given their latitudes and longitudes
def distance(lat1, lon1, lat2, lon2):
    return geopy.distance.distance((lat1, lon1), (lat2, lon2)).miles

#Below is a function to calculate distances between every cloud kitchen and every service station
def calculate_distances():
    dij = []

    for i in range(len(latitudes)):
        row = []
        for j in range(len(service_station_latitudes)):
            dist = distance(latitudes[i], longitudes[i], service_station_latitudes[j], service_station_longitudes[j])
            row.append(dist)
        dij.append(row)

    return dij

#Creating a a distance matrix
distances_matrix = calculate_distances()

#Defining the row labels (cloud kitchen names) and column labels (service station names)
row_labels = data['Name'].tolist()
column_labels = ["Service Station " + str(i+1) for i in range(len(service_station_latitudes))]

#Converting the matrix to a pandas DataFrame with the specified labels
distances_df = pd.DataFrame(distances_matrix, index=row_labels, columns=column_labels)

#Saving the df as a CSV file
distances_df.to_csv("C:/Users/SAGAR/OneDrive/Desktop/Sagar/NCSU/MEM/ISE 535 Python for ISE/Project-28/Distances.csv") #Specify the systems file location

#Saving the locations table to a .txt file using the tabulate library with a simple format
with open('C:/Users/SAGAR/OneDrive/Desktop/Sagar/NCSU/MEM/ISE 535 Python for ISE/Project-28/Locations.txt', 'w') as file: #Specify the systems file location
    file.write(tabulate(table_data, headers=['Index', 'Name', 'Street Address', 'Zip Code', 'Coordinates'], tablefmt='simple'))


#------- END OF TASK 1 --------->



#-------- TASK 2 --------------->


#Initializing the problem
prob = pulp.LpProblem("CloudKitchenAssignment", pulp.LpMinimize)

#Caluclating the number of cloud kitchens and service stations
num_kitchens = len(latitudes)
num_service_stations = len(service_station_latitudes)

#Creating the decision variable x_ij
x = pulp.LpVariable.dicts("x", (range(num_kitchens), range(num_service_stations)), 0, 1, pulp.LpBinary)

#Writing the objective function.
prob += pulp.lpSum([distances_matrix[i][j] * x[i][j] for i in range(num_kitchens) for j in range(num_service_stations)])

#Constraint:Each service station should receive exactly 1 delivery
for j in range(num_service_stations):
    prob += pulp.lpSum([x[i][j] for i in range(num_kitchens)]) == 1

#Constraint:Each cloud kitchen should deliver to 2 service stations
for i in range(num_kitchens):
    prob += pulp.lpSum([x[i][j] for j in range(num_service_stations)]) == 2

#Saving as Ap.mps
prob.writeMPS('C:/Users/SAGAR/OneDrive/Desktop/Sagar/NCSU/MEM/ISE 535 Python for ISE/Project-28/AP.mps') #Specify the systems file location

prob.solve()

#Storing the assignments in a sparse data structure
data = []
rows = []
cols = []
for i in range(num_kitchens):
    for j in range(num_service_stations):
        if x[i][j].value() == 1:
            data.append(1)
            rows.append(i)
            cols.append(j)

#Creating a sparse matrix using the data,rows,and columns
solution_matrix = csr_matrix((data, (rows, cols)), shape=(num_kitchens, num_service_stations))

#Saving teh solution as a .csr file
save_npz('C:/Users/SAGAR/OneDrive/Desktop/Sagar/NCSU/MEM/ISE 535 Python for ISE/Project-28/Solution.csr', solution_matrix) #Specify the systems file location

# --------- END OF TASK 2 --------->


#------------ TASK 3 --------------->

#Building an OD (Origin and destination) table
od_data = []

for i in range(solution_matrix.shape[0]):
    for j in range(solution_matrix.shape[1]):
        if solution_matrix[i, j] == 1:
            od_data.append([i, j, distances_matrix[i][j]])

od_df = pd.DataFrame(od_data, columns=['Cloud Kitchen Index (Origin)', 'Service Station Index (Destination)', 'Distance (miles)'])
print(od_df)

#Formatting the OD table using the tabulate library
formatted_od_data = tabulate(od_df, headers=['Service Station Index (Destination)', 'Cloud Kitchen Index (Origin)', 'Distance (miles)', 'Distance Range'], tablefmt='simple')

#Saving it as a .txt file
with open('C:/Users/SAGAR/OneDrive/Desktop/Sagar/NCSU/MEM/ISE 535 Python for ISE/Project-28/OD.txt', 'w') as file: #Specify the systems file location
    file.write(formatted_od_data)

#Plotting the graph to show the which cloud kitchens deliver to which service stations
for i, j in od_df[["Cloud Kitchen Index (Origin)", "Service Station Index (Destination)"]].values:
    kitchen = (latitudes[i], longitudes[i])
    station = (service_station_latitudes[j], service_station_longitudes[j]) 
   
    plt.scatter(longitudes[i], latitudes[i], color='blue')
    plt.annotate(i, (longitudes[i], latitudes[i]), fontsize=8, ha='center', va='bottom')
    
    
    plt.scatter(service_station_longitudes[j], service_station_latitudes[j], color='red')
    plt.annotate(j, (service_station_longitudes[j], service_station_latitudes[j]), fontsize=8, ha='center', va='bottom')
    
    #---
    
    plt.plot([kitchen[1], station[1]], [kitchen[0], station[0]], 'k-', lw=0.5)



plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Assignment Solution Visualization')
plt.legend()
plt.grid(True)
#Saving the plot as a jpeg file
plt.savefig('C:/Users/SAGAR/OneDrive/Desktop/Sagar/NCSU/MEM/ISE 535 Python for ISE/Project-28/Solution.jpeg', format='jpeg') #Specify the systems file location
plt.show()



#Defining distance constraints
bins = [0, 3, 6, float('inf')]
labels = ['< 3 miles', '3-6 miles', '> 6 miles']

#Categorizing the distances
od_df['Distance Range'] = pd.cut(od_df['Distance (miles)'], bins=bins, labels=labels, right=False)

#Calculating the frequency percentages
frequency_percentages = od_df['Distance Range'].value_counts(normalize=True) * 100

#Plotting the frequncy graph
plt.figure(figsize=(10, 6))
frequency_percentages.sort_index().plot(kind='bar', color='skyblue')
plt.ylabel('% of Origin-Destination assignments')
plt.xlabel('Distance Range')
plt.title('Frequency Distribution of Origin-Destination Distances')
plt.xticks(rotation=0)
plt.tight_layout()
plt.grid(axis='y')
plt.legend()
#Saving the plot as a jpeg file
plt.savefig('C:/Users/SAGAR/OneDrive/Desktop/Sagar/NCSU/MEM/ISE 535 Python for ISE/Project-28/Frequency.jpeg', format='jpeg') #Specify the systems file location
plt.show()




#------- End of Task 3 ------->