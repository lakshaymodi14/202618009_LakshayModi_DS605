# README

## Assignment Title

**Data Scraping and Preprocessing using Python and Scrapy**

## Student Details

* **Name:** Lakshay Modi
* **Student ID:** 202618009

## Project Overview

This project analyses a dataset of books collected from an online book catalogue. The aim is to demonstrate the complete data analysis process, including data cleaning, exploratory analysis, feature engineering, visualisation, and interpretation of the results.

The project uses Python and libraries such as **Pandas, Matplotlib, Seaborn, and WordCloud** to analyse the book data.

## Dataset

The dataset contains information about books, including features such as:

* Book title
* Price
* Rating
* Category
* Product description
* Number of reviews

The cleaned dataset is stored as `books_cleaned.csv`.

## Data Analysis

The project includes summary statistics to understand the basic characteristics of the dataset, including:

* Average price
* Minimum and maximum price
* Average rating
* Distribution of numerical features
* Variation within the dataset

Several visualisations were created to identify patterns in the data:

1. **Price Distribution** – shows how book prices are distributed.
2. **Rating Distribution** – shows the number of books for each rating.
3. **Average Price by Category** – compares the average price of books across categories.
4. **Price vs Rating** – compares book prices across different rating levels.
5. **Word Cloud** – shows the most frequently occurring words in book descriptions.

## Value Score

A `value_score` feature was created to investigate whether the most expensive books are necessarily the best-rated books.

The purpose of this feature is to consider **rating in relation to price**, rather than looking at either feature independently. The analysis showed that the **Moderate price bracket had a better average value score than the Expensive bracket**.

This suggests that higher prices do not necessarily mean better ratings or better value. In a real e-commerce environment, a value-based metric could potentially be useful for book recommendations, first-page rankings, or "Top Sellers" sections alongside traditional rating-based metrics.

However, the dataset is synthetic and relatively small, so these findings should not be treated as predictions of real-world customer behaviour.

## Category Analysis

The analysis also compares pricing and value across book categories.

Some large differences were observed between categories. However, these differences should be interpreted carefully because the dataset contains only 100 books spread across many categories. Some categories therefore contain only a small number of books, meaning that one unusually expensive book can significantly affect the category average.

With a larger dataset containing hundreds or thousands of books per category, these category-level differences could change substantially.

## Word Cloud and Further Analysis

The word cloud provides a visual way to identify common keywords used in book descriptions.

This analysis could be extended by calculating the frequency of individual words and comparing those words with features such as:

* Rating
* Price
* Value score
* Description word count

This could help investigate whether particular words or themes are associated with higher ratings or better value scores.

## Limitations

There are several limitations to this dataset:

* The dataset contains only 100 books.
* The data is synthetic and may not accurately represent a real online bookstore.
* `number_of_reviews` is zero for all books, so it does not provide useful variation for analysis.
* Some categories contain very few books.
* The relationships found in the dataset may not hold in a larger real-world dataset.

These limitations are important when interpreting the visualisations and conclusions.

## Tools and Libraries

The project was developed using Python with the following libraries:

* `pandas` – data loading, cleaning and analysis
* `matplotlib` – data visualisation
* `seaborn` – statistical visualisation
* `wordcloud` – visualisation of common words
* `IPython` – displaying analysis tables in the notebook

## Project Files

```text
project/
│
├── books_cleaned.csv
├── scraper_project_updated.ipynb
└── README.md
```

## Conclusion

Overall, the analysis shows that **price alone is not necessarily a good indicator of a book's rating or value**. The `value_score` provides a more useful way of identifying books that combine a strong rating with a reasonable price, with the Moderate price bracket performing better than the Expensive bracket in this dataset.

The analysis also demonstrates how visualisation and engineered features can reveal patterns that may not be obvious from the original columns alone. With a larger and more realistic dataset, the same approach could be extended into a more advanced recommendation system using price, ratings, review counts, description features, and word-level analysis.

