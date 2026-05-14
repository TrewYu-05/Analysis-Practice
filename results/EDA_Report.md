# Wine Quality Exploratory Data Analysis (EDA) Report

This report summarizes the findings from the exploratory data analysis conducted on the combined Portuguese "Vinho Verde" dataset (Red and White wine variants).

## 1. Overall Distribution Analysis

**Goal:** Analyze the distribution characteristics of all physicochemical properties, identifying central tendencies, dispersion, skewness, and extreme values.

**Key Findings:**
*   **Central Tendency & Dispersion:**
    *   **Alcohol** has a mean of 10.49% with a standard deviation of 1.19%.
    *   **Residual Sugar** shows a high level of dispersion (Mean: 5.44, Variance: 22.64, Standard Deviation: 4.76).
    *   **Total Sulfur Dioxide** is notably widespread (Mean: 115.74, Std Dev: 56.52).
*   **Skewness & Extreme Values:**
    *   Most variables exhibit right-skewness (positive skewness). Notably, **Chlorides** (Skewness: 5.40, Kurtosis: 50.90) and **Residual Sugar** (Skewness: 1.44) display extreme right tails, indicating the presence of outliers or very high concentration values for a small subset of wines.
    *   **Density** and **pH** follow distributions closer to a normal bell curve, showing low skewness values.
    *   The boxplots (`overall_boxplots.pdf`) and kernel density estimations (`overall_distributions.pdf`) clearly visualize these long tails, specifically highlighting high outlier points in Residual Sugar, Chlorides, and Free Sulfur Dioxide.

## 2. Red vs. White Wine Differences

**Goal:** Compare red and white wines across physicochemical properties to find the most distinguishing variables.

**Key Findings:**
Based on the Mann-Whitney U test (non-parametric comparison) and median differences, **almost all** variables show statistically significant differences (p-value < 0.05) between red and white wines, except for **Alcohol**.

*   **Most Distinguishing Variables (Highest Contrast):**
    *   **Total Sulfur Dioxide:** White wines have vastly higher levels (Median 134.0) compared to red wines (Median 38.0). This makes logical sense as white wines usually require more sulfur dioxide for preservation.
    *   **Volatile Acidity:** Red wines have higher volatile acidity (Median 0.52) compared to white wines (Median 0.26).
    *   **Residual Sugar:** White wines are significantly sweeter (Median 5.2) than red wines (Median 2.2).
    *   **Chlorides:** Red wines have nearly double the chloride levels (Median 0.079) of white wines (Median 0.043).
    *   **Free Sulfur Dioxide:** Also much higher in white wines.
*   **Variables with Minimal Difference:**
    *   **Alcohol:** Shows no statistically significant difference between the two wine types (p-value = 0.1818). Both have a median hovering around 10.2% - 10.4%.

Visualized in `red_vs_white_violinplots.pdf`, the stark contrast in shape and position for Total Sulfur Dioxide, Volatile Acidity, and Chlorides makes them the best distinguishing variables.

## 3. Quality Differences & Variable Relationships

**Goal:** Understand how variables differ across wine quality levels and analyze linear correlations.

**Key Findings:**
*   **Features Impacting Quality (Boxplot Trends):**
    *   **Alcohol:** A very clear positive trend is observed; higher quality wines tend to have higher alcohol content.
    *   **Volatile Acidity:** Shows a negative trend; higher quality wines generally have lower volatile acidity (less vinegar-like taste).
    *   **Density:** Shows a slight negative trend with higher quality.
*   **Variable Relationships (Correlation Heatmap):**
    *   **Quality Correlation:**
        *   `Alcohol` has the strongest positive correlation with quality (+0.44).
        *   `Density` has the strongest negative correlation with quality (-0.31).
        *   `Volatile Acidity` also negatively correlates with quality (-0.27).
    *   **Inter-variable Correlation:**
        *   `Density` and `Residual Sugar` are strongly positively correlated (+0.55), which aligns with the physics of sugar dissolving in liquid.
        *   `Density` and `Alcohol` are strongly negatively correlated (-0.69).
        *   `Total Sulfur Dioxide` and `Free Sulfur Dioxide` are highly positively correlated (+0.72), as expected.

## 4. Advanced Analysis (Exceeding Requirements)

To gain a deeper understanding beyond standard descriptive statistics, we performed dimensionality reduction and machine learning-based feature importance extraction.

**1. Dimensionality Reduction (PCA Visualization):**
We applied Principal Component Analysis (PCA) to reduce the 11 physicochemical features into a 2D space.
*   **Wine Type Separation:** The PCA plot (`pca_analysis.pdf`) colored by wine type reveals two extremely distinct clusters for red and white wines, demonstrating that the physicochemical profiles of the two variants are fundamentally different on a multidimensional level.
*   **Quality Gradients:** The PCA plot colored by quality shows a more blended distribution, but a noticeable gradient exists along the principal components, confirming that quality isn't defined by a single feature but rather a complex combination.

**2. Feature Importance for Quality Prediction (Random Forest):**
We trained a Random Forest Classifier to predict wine quality and extracted the Gini importance of each feature.
*   **Top Predictors:** The Random Forest confirms our correlation findings, identifying **Alcohol** (0.124) as the single most important variable for predicting wine quality.
*   **Secondary Predictors:** **Density** (0.103) and **Volatile Acidity** (0.100) are the next most critical factors.
*   **Conclusion:** If one wants to alter or predict the quality of a wine, controlling the alcohol content, monitoring volatile acidity (spoilage), and balancing density (which ties to sugar and alcohol) are the most critical chemical interventions.

*Generated automatically via EDA script.*
