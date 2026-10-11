# freight_rate_predictions
MACHINE LEARNING ENGINEER ASSESSMENT

Freight Rate Prediction
This report summarizes data exploration, data cleaning, validation strategy, modeling experiments, feature
engineering, and findings from the freight rate prediction assessment.
1. Objective and Dataset
The objective was to train a regression model to predict the posted freight rate (posted_rate) for each load in the
supplied validation dataset. The development data contained load attributes such as pickup and delivery locations,
geographic coordinates, distance, equipment type, weight, date, market_index, quote_signal, and the target
posted_rate. The validation file contains 12,000 loads requiring predictions.
The load identifier (load_id) was treated as an identifier rather than a predictive feature. The target variable was
posted_rate.
2. Data Exploration and Quality Checks
Initial descriptive statistics indicated that distance and posted_rate have right-tailed distributions. The target had a
mean of approximately 2,374 and a median of approximately 2,031, with a maximum around 25,533. This suggests
that a relatively small number of high-rate observations may contribute substantially to RMSE.
Data Cleaning
The following data-cleaning steps were performed:
• Missing values: Missing numerical values were filled using the median of the corresponding feature.
Median imputation was selected because the median is less sensitive to extreme values than the mean.
• Outlier detection and removal: Outliers were detected, and the identified records were removed. The
removed records represented approximately 7.35% of the dataset.
• Duplicate check: The dataset was checked for duplicated rows, and no duplicated rows were found.
The weight column contained missing values and negative values, including a minimum near -47,500. Negative
weights require careful interpretation against the dataset specification to determine whether they represent invalid
records or another encoding.
CatBoost does not require normally distributed input features. Therefore, transformations were evaluated based on
validation performance rather than an objective of forcing the features or target to follow a normal distribution.
For a production-quality evaluation, imputation values and outlier thresholds should be calculated using the
training data only and then applied consistently to the relevant validation data to prevent data leakage.
MACHINE LEARNING ENGINEER ASSESSMENT
Technical Assessment Report | 2
3. Validation Strategy
A time-aware expanding-window validation strategy was used to evaluate how well the model could predict future
freight rates. This approach preserves chronological order and avoids randomly mixing earlier and later
observations.
Two temporal validation approaches were considered.
3.1 Monthly Expanding-Window Validation
The monthly approach uses earlier months for training and the next month for testing. After each test period, the
training window expands to include the newly available earlier data.
Fold Training Data Testing Data
1 Months 1 Months 2
2 Months 1-2 Months 3
3 Months 1-2-3 Months 4
4 Months 1-2-3-4 Months 5
This process continues chronologically for the available months.
Why this approach was used: It simulates a realistic forecasting scenario in which historical data is available for
training, while the model must predict freight rates in a future month.
3.2 Daily Expanding-Window Validation
The daily approach follows the same principle but evaluates performance at a daily level.
For each validation day, the model is trained using all eligible data from earlier days and evaluated on the next day.
For example:
Fold Training Data Testing Data
1 Day 1 Day 2
2 Day 1-2 Day 3
3 Day 1-2-3 Day 4
4 Day 1-2-3-4 Day 5
The process continues, with the training dataset expanding as more historical observations become available.
The quick daily experiments used the first 30 available dates, yielding 29 daily validation steps. This allowed
practical comparisons of modeling choices while preserving temporal order.
The reported RMSE and MAE results below are approximate averages from the daily temporal validation
experiments. A separate monthly performance score should only be reported after calculating the metrics across
the monthly test folds.
MACHINE LEARNING ENGINEER ASSESSMENT
Technical Assessment Report | 3
4. Baseline Model and Results
CatBoostRegressor was selected as the primary model because the dataset combines numerical variables with
categorical variables such as pickup, delivery, and equipment. CatBoost can handle categorical features directly
and can learn nonlinear relationships and interactions.
Metric Approx. result Interpretation
RMSE 256 Penalizes large errors heavily; a few
extreme misses have a strong effect.
MAE 100 Average absolute prediction error is
approximately 100 rate units.
These are approximate results from the daily temporal validation experiments and should be interpreted as
experimental estimates, not as a guarantee of performance on the final hidden validation set.
5. Feature Engineering and Experiments
Experiment Validation result Conclusion
Baseline CatBoost Approx. RMSE 256; MAE 100 Used as the reference model.
Calendar features (day of week, day of
month, month) No meaningful improvement
Retained only if useful for the final
pipeline; not a demonstrated
performance gain.
Route feature (pickup → delivery) No meaningful improvement Added as a categorical representation of
the route.
Coordinate differences
(latitude/longitude deltas) No meaningful improvement Tested as directional/geographic features.
Weight per mile and distance scaling No meaningful improvement Tested simple interactions involving
weight and distance.
Historical route mean rate Avg. RMSE approx. 257.17; MAE approx.
98.50
Slight MAE improvement but no RMSE
improvement; limited value for the current
objective.
Log distance Avg. RMSE approx. 257.31; MAE approx.
100.57 Slightly worse than baseline.
Log-transformed target (log1p) Avg. RMSE approx. 272.43; MAE approx.
105.89
Worse than baseline after reversing the
transformation.
Distance-only polynomial models RMSE approx. 298 Underperformed CatBoost, indicating that
distance alone is insufficient.
6. Feature Importance and Observations
CatBoost feature importance indicated that distance was the dominant feature (approximately 88.8% of total
importance in the reported run), followed by equipment type (approximately 4.8%) and weight (approximately
1.7%). Market_index and quote_signal contributed smaller amounts. Geographic coordinates had comparatively
low importance in that run.
Exploratory analysis showed that rate per mile generally decreases as trip distance increases, while equipment type
affects typical rate per mile. However, these variables did not fully explain the largest prediction errors. The
rate_per_mile variable was used for analysis only and was not used as a model input because it is calculated from
the target and would cause target leakage.
MACHINE LEARNING ENGINEER ASSESSMENT
Technical Assessment Report | 4
7. Error Analysis
The largest errors included cases where the actual posted rate was several thousand units higher than the model
prediction, as well as some cases where the model substantially overpredicted the actual rate. For example,
several loads with actual rates around 5,000–6,400 were predicted around 1,000–1,900, while some long-distance
loads had predictions much higher than their actual rates. These cases indicate that the model captures the
dominant distance relationship but does not explain every exceptional price.
Potential next steps are to inspect the worst 20–50 errors across all validation days, check whether those records
share equipment, route, date, weight, market, or data-quality patterns, and confirm whether the extreme target
values are valid business observations. Additional features should only be added when supported by this analysis.
8. Conclusions and Recommended Next Steps
• CatBoost was the strongest tested approach among the experiments documented here, with approximate
temporal validation performance of RMSE 256 and MAE 100.
• Additional simple feature engineering and target/distance log transformations did not improve the overall result
meaningfully.
• RMSE is strongly affected by a small number of large prediction errors; error analysis is therefore a higher-value
next step than continuing arbitrary feature additions.
• Investigate negative and missing weight values using the dataset specification before applying cleaning rules.
• Before submission, train the chosen pipeline on all eligible development data, generate predictions for every
row in validation.csv, and verify that the output contains exactly 12,000 rows, all required load_id values, and
no missing or invalid predictions.
• If time permits, run a broader temporal backtest across more dates to check that performance is stable over
time.
9. Scope and Limitations
This report documents the experiments completed and the approximate results available during development.
Some feature experiments showed little or no improvement, and the quick daily temporal validation run was limited
to the first 30 available dates.
The final validation-set performance is unknown until predictions are evaluated against ground truth, which may not
be available.
The RMSE and MAE figures reported in this document refer to the daily temporal validation experiments; they should
not be interpreted as monthly validation results unless those results are calculated separately.
The 7.35% outlier-removal figure describes the data-cleaning step. The outlier detection method, thresholds, and
treatment of extreme target values should be documented in the implementation to ensure reproducibility and
avoid removing legitimate high-value freight loads.
Prepared by Amr Ashor
