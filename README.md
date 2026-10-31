# Flight Ticket Price Prediction

## Project Overview

This project analyzes domestic Indian flight ticket prices and predicts the price of a ticket from its airline, journey date, route, stops, timings, and duration, using data science and machine learning techniques.

The project was completed as part of the **DataMites AI Engineer Program (Internship Project)**.

The primary objective was to perform exploratory data analysis on **10,683 tickets across 11 columns**, build a regression model that helps customers estimate flight prices and plan their journey, and give travellers practical suggestions on how to get a cheaper ticket.

## Project Objectives

- Perform data cleaning and exploratory data analysis (EDA).
- Convert text-based dates, times, and durations into usable numeric features.
- Handle duplicates, inconsistent labels, and impossible values.
- Analyze how airline, route, stops, journey date, and timing affect ticket price.
- Build and compare multiple regression models.
- Tune hyperparameters and evaluate advanced ensembles.
- Provide data-driven suggestions to travellers.

## Technologies & Libraries

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy

## Project Workflow

### 1. Data Cleaning and Preprocessing
- Read the Excel file with `pd.read_excel` (requires `openpyxl`).
- Removed 220 duplicate rows, 1 row with missing `Route` and `Total_Stops`, and 1 impossible 2-stop flight with a 5-minute duration.
- Merged inconsistent labels: `'New Delhi'` into `'Delhi'` and `'No Info'` into `'No info'`.
- Parsed `Date_of_Journey` with `dayfirst=True` (44 date strings were really 40 journey dates, 1 March to 27 June 2019) and extracted journey day, month, and weekday.
- Extracted departure and arrival hour, created a `NEXT DAY ARRIVAL` flag, converted `Duration` text (e.g. `'2h 50m'`) to minutes, and mapped `Total_Stops` to 0-4 by hand.
- Final cleaned data: **10,461 rows**, no missing values and no duplicates.

### 2. Exploratory Data Analysis (EDA)
- Analyzed price, duration, stops, departure hour, and journey day distributions. Price is right-skewed (skew 1.86).
- Examined median price by airline, stops, route, month, weekday, and additional info.
- Studied how price changes by journey date and departure hour.
- Used histograms, count plots, relational plots, pair plots, boxplots, line plots, bar charts, and correlation heatmaps.

### 3. Outlier Analysis
Used the Interquartile Range (IQR) method and boxplots to identify extreme values.

Key findings:
- 94 tickets were above the price upper limit of Rs 23,016.5 (max Rs 79,512). Most were Jet Airways (62), plus all 6 Jet Airways Business tickets. They were kept because they are real expensive tickets.
- 75 flights had durations above 2,020 minutes (max 2,860 minutes). They were kept as real flights with long overnight stopovers.
- Only the 5-minute flight was a clear data error, so no other rows were removed.

### 4. Feature Selection
- `STOPS` (0.60) and `DURATION MINS` (0.50) had the strongest correlation with price. They are related (0.73) but both were kept.
- **Removed `Destination`** because each source city always flies to the same destination, so it duplicates `Source`.
- Kept `Route` (128 values) because it shows which city the flight stops in.
- Categorical features were converted with `LabelEncoder` and scaled with `StandardScaler` for the linear and distance-based models.
- Final model used 12 features: `Airline`, `Source`, `Route`, `Additional_Info`, `JOURNEY DAY`, `JOURNEY MONTH`, `WEEKDAY`, `DEP HOUR`, `ARR HOUR`, `NEXT DAY ARRIVAL`, `DURATION MINS`, `STOPS`.

### 5. Machine Learning Model Development

This project uses supervised regression to predict the ticket price in rupees.

The following algorithms were implemented and evaluated:

1. Linear Regression
2. Decision Tree Regressor
3. K-Nearest Neighbors (KNN) Regressor
4. Random Forest Regressor
5. Gradient Boosting Regressor

The data was split 80/20 into training (8,368 rows) and test (2,093 rows) sets.

### 6. Model Evaluation and Comparison

Models were evaluated using:
- R²
- MAE, RMSE, and MAPE
- 5-fold cross-validation

| Model | Train R² | Test R² | MAE (Rs) | RMSE (Rs) | MAPE |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.979 | **0.891** | **704** | **1,503** | **8.1%** |
| Gradient Boosting | 0.841 | 0.830 | 1,230 | 1,875 | 15.1% |
| Decision Tree | 0.911 | 0.811 | 915 | 1,976 | 10.3% |
| KNN | 0.838 | 0.746 | 1,390 | 2,291 | 17.9% |
| Linear Regression | 0.431 | 0.454 | 2,484 | 3,361 | 32.9% |

In 5-fold cross-validation, a **tuned Gradient Boosting** (500 trees, depth 6) scored the best CV R² (0.897), ahead of Random Forest (0.878). A Random Forest trained on log(price) gave a similar CV R² (0.876) with the lowest CV MAE (Rs 659).

### 7. Hyperparameter Tuning and Advanced Ensembles
- **GridSearchCV** on Random Forest (`n_estimators` x `max_depth`) showed that depth matters a lot (depth 6: 0.729, depth 15 or more: about 0.838) and that more than 100 trees adds almost nothing.
- **RandomizedSearchCV** on Random Forest and HistGradientBoosting.
- **Voting** (average of tuned Random Forest, HistGradientBoosting, and Gradient Boosting) and **Stacking** (Ridge meta-model) ensembles.

| Model | Test R² | Test MAE (Rs) | % within 10% of price | CV R² |
|---|---:|---:|---:|---:|
| Stacking | 0.912 | 703 | 74.1% | **0.907** |
| Voting | **0.915** | **675** | 75.5% | 0.905 |
| HistGradientBoosting (random search) | 0.906 | 782 | 68.1% | 0.900 |
| Random Forest (hand-picked) | 0.891 | 704 | 75.5% | 0.878 |
| Random Forest (grid search) | 0.891 | 704 | 75.7% | 0.877 |
| Random Forest (random search) | 0.891 | 704 | **75.8%** | 0.877 |

Tuning Random Forest did not improve it, but boosting and the two ensembles raised CV R² by about 0.03, about twice the variation between folds.

**Best Performing Model: Voting Regressor**

The Voting model achieved the highest test R² (**0.915**) and the lowest test MAE (**Rs 675**, MAPE 7.9%), and tied with Stacking on cross-validation. It was chosen over Stacking because it scores the same and is simpler, with no meta-model to train. 75.5% of its predictions are within 10% of the real price and 91.0% within 20%, with a median error per route between 3.3% and 6.4%.

## Key Findings

- **Non-stop flights were the cheapest** (median Rs 4,667). On the same route, flights with stops cost **1.6 to 5 times more**.
- **SpiceJet was the cheapest or almost the cheapest on every route**, while Jet Airways usually cost 2 to 3 times more on the same route.
- On the same route, journeys on **1-12 March cost 63.6% more** than usual and **April was 28.1% cheaper**.
- **Sunday (+12.8%) and Friday (+12.0%)** were the most expensive days; Monday (-6.6%) and Wednesday (-5.0%) were the cheapest.
- Night flights looked much cheaper overall, but on the same route they were only **2.2% cheaper**.
- **1,602 flights appeared with more than one price** (median gap Rs 2,031), which limits how accurate any model can be because the booking date is not in the data.

## Suggestions for Travellers

- Choose a low-cost airline (SpiceJet, IndiGo, GoAir, Air Asia).
- Take a non-stop flight, which is both faster and cheaper.
- Avoid last-minute bookings and travel in off-peak periods like April.
- Fly on Monday or Wednesday instead of Friday or Sunday.
- Choose a "no check-in baggage" fare when possible (1-10% cheaper on SpiceJet).
- Check the model's predicted price before booking.

## Challenges and Solutions

Some of the main challenges encountered included:

- **Text-based features:** Converted dates, times, and durations (e.g. `'2h 50m'`) into numeric columns.
- **Inconsistent values:** Merged `'Delhi'`/`'New Delhi'` and `'No info'`/`'No Info'`, and parsed mixed date formats with `dayfirst=True`.
- **Data errors and duplicates:** Removed 220 duplicates, 1 missing row, and 1 impossible 5-minute flight.
- **Ordered category:** Mapped `Total_Stops` by hand so `'non-stop'` keeps its natural order.
- **Skewed target with outliers:** Kept real high-priced tickets and tested a log(price) Random Forest.
- **Overfitting:** Used a train/test split and 5-fold cross-validation to measure real performance.
- **Misleading group comparisons:** Compared groups within the same route using a `VS ROUTE MEDIAN %` column before making suggestions.

## Disclaimer

This project was completed for educational and portfolio purposes as part of the DataMites AI Engineer Program. The model results are based on historical ticket data from March to June 2019 and should not be treated as a production price prediction system.

## Author

**Suchitra Thanumalayan**

Data Science | Machine Learning | Artificial Intelligence

