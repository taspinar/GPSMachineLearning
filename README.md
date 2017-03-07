# Synopsis

This repository contains code and jupyter notebooks with machine learning algorithms for working with GPS trajectories. It will be used during the Machine Learning hackathon of [IotTechDay2017](http://iottechday.nl/sessions/machine-learning-hackathon-2/).

The dataset used is the popular [GeoLife GPS Trajectories](https://www.microsoft.com/en-us/download/details.aspx?id=52367) 

# Tasks
1.	How can we load GPS trajectories in a proper way so that it will be easier to work with in the future.
2.	What is classification, and how can we apply classification to GPS data to automatically detect the modality of the transport (walking, bicycle, car, train).
3.	What are popular routes taken and does the popularity of the route of the route affect the traffic?
4.	Can driver behaviour be deduced from the GPS trajectories, and can we tell how driver behaviour affects traffic?
5.	What are the points of interest e.g., restaurants, stores, hotels, etc. 


# Notebooks 
We have provided some notebooks, which should give you a flying start, but feel free to do everything your own way. 
+ [1. loading the GPS data, enriching it with velocity and acc information, adding correct labels to each trajectory](https://github.com/taspinar/GPSMachineLearning/blob/master/notebooks/load_gps_data.ipynb)


# relevant datasets
+ [Bus routes and stops of Beijing](https://www.dropbox.com/s/ryk0197wnr145rv/DT18.rar) -> contains 1.543 bus routes and 42.161 stops 

+ [Beijing check-in records from Sina](https://www.beijingcitylab.com/app/download/10965785399/DT24.zip?t=1460518522) -> contains 868 m check-ins for all 143,576 venues.

+ [Road junction density of China in 2011](https://www.dropbox.com/s/eppkm3p1jfyx6ae/DT21.zip)

+ [Points of interest in Beijing](http://www.datatang.com/data/44484)

+ [Global cities shapefile](http://download.bbbike.org/osm/bbbike/)

+ [Beijing bus smart card data](http://www.datatang.com/data/42120)