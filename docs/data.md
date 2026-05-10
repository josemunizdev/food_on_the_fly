# Data Card - Zomato Delivery Operations Dataset

**Project**: Food on the Fly | SE 489 MLOps   
**Source**: [Kaggle - saurabhbadole/zomato-delivery-operations-analytics-dataset](https://www.kaggle.com/datasets/saurabhbadole/zomato-delivery-operations-analytics-dataset)

## Overview

45,584 food delivery records from Zomato over a 54-day window (Feb 11 - Apr 6, 2022). Target is `Time_taken (min)`.

## Columns

| Column | Type | Description |
|--------|------|-------------|
| ID | string | Order ID |
| Delivery_person_ID | string | Driver ID |
| Delivery_person_Age | float | Driver age |
| Delivery_person_Ratings | float | Driver rating (1-6) |
| Restaurant_latitude | float | Restaurant lat |
| Restaurant_longitude | float | Restaurant long |
| Delivery_location_latitude | float | Drop-off lat |
| Delivery_location_longitude | float | Drop-off long |
| Order_Date | string | Date (DD-MM-YYYY) |
| Time_Orderd | string | Time ordered |
| Time_Order_picked | string | Time picked up |
| Weather_conditions | string | Weather |
| Road_traffic_density | string | Traffic level |
| Vehicle_condition | int | Vehicle condition (0-3) |
| Type_of_order | string | Food category |
| Type_of_vehicle | string | Vehicle used |
| multiple_deliveries | float | Simultaneous deliveries |
| Festival | string | Festival occurring (Yes/No) |
| City | string | City type |
| Time_taken (min) | int | **Target** |

## Target Variable

- Range: 10-54 min
- Mean: 26.3, Median: 26.0
- Slight right skew (0.486)
- No zero-minute outliers

## Missing Values

| Column | Count | % |
|--------|-------|---|
| Delivery_person_Ratings | 1,908 | 4.2% |
| Delivery_person_Age | 1,854 | 4.1% |
| Time_Orderd | 1,731 | 3.8% |
| City | 1,200 | 2.6% |
| multiple_deliveries | 993 | 2.2% |
| Weather_conditions | 616 | 1.4% |
| Road_traffic_density | 601 | 1.3% |
| Festival | 228 | 0.5% |

Nothing over 5%. Target column has no missing values.

## Key Findings from EDA

- Jam and high traffic = longer delivery times
- Low traffic has noticeably shorter times and tighter spread
- Vehicle type differences are modest, but electric scooters trend slightly faster
- Haversine distance correlates 0.32 with delivery time (after filtering bad lat/long values over 50 km)
- Multiple deliveries has the strongest correlation with target (0.39)
- Higher rated drivers tend to deliver faster (-0.34)
- Some lat/long coordinates produce distances of 5,000-20,000 km, clearly bad data, filtered for analysis

## Outlier Handling

- No zero-minute deliveries found (min is 10)
- Bad lat/long values filtered at 50 km cutoff for distance analysis
- Delivery_person_Ratings max of 6.0 likely data entry error but kept since few rows affected

## Data Split (Temporal)

Per team discussion, data is split by date rather than randomly:

| Split | Date Range | Rows | Purpose |
|-------|-----------|------|---------|
| Train | Feb 11 - Mar 23 (70% of first 40 days) | ~21,513 | Model training |
| Val | Feb 11 - Mar 23 (15% of first 40 days) | ~4,610 | Hyperparameter tuning |
| Test | Feb 11 - Mar 23 (15% of first 40 days) | ~4,610 | Final evaluation |
| Drift | Mar 24 - Apr 6 (last 14 days) | ~14,851 | Simulate new data for pipeline monitoring |

Splits tracked via DVC with MD5 hashes for reproducibility. Random state = 42.