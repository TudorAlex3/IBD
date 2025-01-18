from flask import Flask, render_template
import os
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/beginner")
def beginner():
    return render_template("beginner.html")

@app.route("/intermediate")
def intermediate():
    return render_template("intermediate.html")

@app.route("/advanced")
def advanced():
    return render_template("advanced.html")

@app.route('/iris')
def iris():
    return render_template('iris.html')

@app.route('/iris/description')
def iris_description():
    description = {
        "title": "Iris Dataset Description",
        "text": (
            "The Iris dataset is a well-known dataset in machine learning and statistics. "
            "It consists of 150 samples, with each sample representing an iris flower from one of three species: "
            "Setosa, Versicolor, or Virginica. The dataset contains 4 numerical features: "
            "sepal length (cm), sepal width (cm), petal length (cm), and petal width (cm). These features describe the dimensions of the flowers. "
            "The 'species' column is the target variable, represented as 0 (Setosa), 1 (Versicolor), and 2 (Virginica)."
        ),
        "data_head": [
            [5.1, 3.5, 1.4, 0.2, 0],
            [4.9, 3.0, 1.4, 0.2, 0],
            [4.7, 3.2, 1.3, 0.2, 0],
            [4.6, 3.1, 1.5, 0.2, 0],
            [5.0, 3.6, 1.4, 0.2, 0],
        ],
        "columns": ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)", "species"],
    }
    return render_template('iris_description.html', description=description)

@app.route('/iris/histograms')
def iris_histograms():
    return render_template('iris_histograms.html')

@app.route('/iris/scatter')
def iris_scatter():
    return render_template('iris_scatter.html')

@app.route('/iris/pairplot')
def iris_pairplot():
    return render_template('iris_pairplot.html')

@app.route('/iris/metrics')
def iris_metrics():
    return render_template('iris_metrics.html')

@app.route('/iris/confusion_matrix')
def iris_confusion_matrix():
    return render_template('iris_confusion_matrix.html')

@app.route('/loan')
def loan():
    return render_template('loan.html')

@app.route('/loan/description')
def loan_description():
    description = {
        "title": "Loan Dataset Description",
        "text": (
            "The Loan dataset contains information about applicants for a loan. It includes 615 entries, each representing "
            "an individual applicant's details, including personal information, income, loan details, and loan status. "
            "The dataset includes both categorical (e.g., Gender, Education, Loan Status) and numerical (e.g., ApplicantIncome, LoanAmount) features. "
            "The target variable, 'Loan_Status', indicates whether a loan was approved (Y) or not (N)."
        ),
        "data_head": [
            ['LP001002', 'Male', 'No', 0, 'Graduate', 'No', 5849, 0, None, 360, 1, 'Urban', 'Y'],
            ['LP001003', 'Male', 'Yes', 1, 'Graduate', 'No', 4583, 1508, 128, 360, 1, 'Rural', 'N'],
            ['LP001005', 'Male', 'Yes', 0, 'Graduate', 'Yes', 3000, 0, 66, 360, 1, 'Urban', 'Y'],
            ['LP001006', 'Male', 'Yes', 0, 'Not Graduate', 'No', 2583, 2358, 120, 360, 1, 'Urban', 'Y'],
            ['LP001008', 'Male', 'No', 0, 'Graduate', 'No', 6000, 0, 141, 360, 1, 'Urban', 'Y']
        ],
        "columns": [
            "Loan_ID", "Gender", "Married", "Dependents", "Education", "Self_Employed", "ApplicantIncome", 
            "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History", "Property_Area", "Loan_Status"
        ]
    }
    return render_template('loan_description.html', description=description)

@app.route('/loan/histograms')
def loan_histograms():
    return render_template('loan_histograms.html')

@app.route('/loan/scatter')
def loan_scatter():
    # List of scatter plot images with explanations
    plots = [
        {"image": "static/images/scatter1.png", 
         "description": "This scatter plot displays the relationship between Applicant Income and Loan Amount. Higher incomes generally correlate with higher loan amounts."},
        {"image": "static/images/scatter2.png", 
         "description": "This scatter plot shows the relationship between Coapplicant Income and Loan Amount. Coapplicant contributions may increase the loan amount approved."},
        {"image": "static/images/scatter3.png", 
         "description": "This scatter plot visualizes the dependency of Loan Amount on Loan Term. Longer loan terms could imply larger loan amounts."},
        {"image": "static/images/scatter4.png", 
         "description": "This scatter plot depicts the impact of Credit History on Loan Amount. A clean credit history tends to result in larger approved loans."}
    ]
    return render_template('loan_scatter.html', plots=plots)

@app.route('/loan/pairplots_including')
def loan_pairplots_including():
    return render_template('loan_pairplots_including.html')

@app.route('/loan/pairplots_after')
def loan_pairplots_after():
    return render_template('loan_pairplots_after.html')

@app.route('/loan/metrics')
def loan_metrics():
    return render_template('loan_metrics.html')

@app.route('/loan/matrix')
def loan_matrix():
    return render_template('loan_confusion_matrix.html')

@app.route('/bigmart')
def bigmart():
    return render_template('bigmart.html')

@app.route('/bigmart/description')
def bigmart_description():
    description = {
        "title": "BigMart Sales Dataset Description",
        "text": (
            "The BigMart Sales dataset contains transactional data for products sold across multiple stores of a retail chain. "
            "It includes 8,523 entries, each representing a unique product at a particular store. "
            "The dataset provides information about product characteristics, store details, and sales data. "
            "It contains both categorical features (e.g., Item_Fat_Content, Outlet_Type) and numerical features (e.g., Item_Visibility, Item_MRP). "
            "The target variable, 'Item_Outlet_Sales', represents the total sales of a product in a store and is the variable to be predicted."
        ),
        "data_head": [
            ['FDA15', 'Dairy', 'Low Fat', 9.3, 0.016047, 249.8092, 'OUT049', 'Supermarket Type1', 'Tier 1', 1999, 3735.138],
            ['DRC01', 'Soft Drinks', 'Regular', 5.92, 0.019278, 48.2692, 'OUT018', 'Supermarket Type2', 'Tier 3', 2009, 443.4228],
            ['FDN15', 'Meat', 'Low Fat', 17.5, 0.016760, 141.618, 'OUT049', 'Supermarket Type1', 'Tier 1', 1999, 2097.27],
            ['FDX20', 'Household', 'Regular', 19.2, 0.000000, 182.095, 'OUT010', 'Grocery Store', 'Tier 3', 1998, 732.38],
            ['NCD19', 'Household', 'Low Fat', 8.93, 0.000000, 53.8614, 'OUT013', 'Supermarket Type1', 'Tier 3', 1987, 994.7052]
        ],
        "columns": [
            "Item_Identifier", "Item_Type", "Item_Fat_Content", "Item_Weight", "Item_Visibility", "Item_MRP", 
            "Outlet_Identifier", "Outlet_Type", "Outlet_Location_Type", "Outlet_Establishment_Year", "Item_Outlet_Sales"
        ]
    }
    return render_template('bigmart_description.html', description=description)

@app.route('/bigmart/histograms')
def bigmart_histograms():
    return render_template('bigmart_histograms.html')

@app.route('/bigmart/scatter')
def bigmart_scatter():
    return render_template('bigmart_scatter.html')

@app.route('/bigmart/pairplot_before')
def bigmart_including():
    return render_template('bigmart_pairplots_including.html')

@app.route('/bigmart/pairplot_after')
def bigmart_removed():
    return render_template('bigmart_pairplots_after.html')

@app.route('/bigmart/metrics')
def bigmart_metrics():
    return render_template('bigmart_metrics.html')

@app.route('/wine')
def wine():
    return render_template('wine.html')

@app.route('/wine/description')
def wine_description():
    return render_template('wine_description.html')

@app.route('/wine/histograms')
def wine_histograms():
    return render_template('wine_histograms.html')

@app.route('/wine/scatter')
def wine_scatter():
    return render_template('wine_scatter.html')

@app.route('/wine/pairplots_including')
def wine_pairplots_including():
    return render_template('wine_pairplots_including.html')

@app.route('/wine/pairplots_after')
def wine_pairplots_after():
    return render_template('wine_pairplots_after.html')

@app.route('/wine/metrics')
def wine_metrics():
    return render_template('wine_metrics.html')

@app.route('/wine/matrix')
def wine_matrix():
    return render_template('wine_confusion_matrix.html')

@app.route('/turkiye')
def turkiye():
    return render_template('turkiye.html')

@app.route('/turkiye/description')
def turkiye_description():
    return render_template('turkiye_description.html')

@app.route('/turkiye/histograms')
def turkiye_histograms():
    return render_template('turkiye_histograms.html')

@app.route('/turkiye/scatter')
def turkiye_scatter():
    return render_template('turkiye_scatter.html')

@app.route('/turkiye/pairplots_including')
def turkiye_pairplots_including():
    return render_template('turkiye_pairplots_including.html')

@app.route('/turkiye/pairplots_after')
def turkiye_pairplots_after():
    return render_template('turkiye_pairplots_after.html')

@app.route('/turkiye/metrics')
def turkiye_metrics():
    return render_template('turkiye_metrics.html')

@app.route('/turkiye/matrix')
def turkiye_matrix():
    return render_template('turkiye_confusion_matrix.html')

@app.route('/heights')
def heights():
    return render_template('heights.html')

@app.route('/heights/description')
def heights_description():
    return render_template('heights_description.html')

@app.route('/heights/histograms')
def heights_histograms():
    return render_template('heights_histograms.html')

@app.route('/heights/scatter')
def heights_scatter():
    return render_template('heights_scatter.html')

@app.route('/heights/pairplots_including')
def heights_pairplots_including():
    return render_template('heights_pairplots_including.html')

@app.route('/heights/pairplots_after')
def heights_pairplots_after():
    return render_template('heights_pairplots_after.html')

@app.route('/heights/metrics')
def heights_metrics():
    return render_template('heights_metrics.html')

@app.route('/heights/matrix')
def heights_matrix():
    return render_template('heights_confusion_matrix.html')


@app.route('/boston')
def boston():
    return render_template('boston.html')

@app.route('/boston/description')
def boston_description():
    return render_template('boston_description.html')

@app.route('/boston/histograms')
def boston_histograms():
    return render_template('boston_histograms.html')

@app.route('/boston/scatter')
def boston_scatter():
    return render_template('boston_scatter.html')

@app.route('/boston/pairplots_including')
def boston_pairplots_including():
    return render_template('boston_pairplots_including.html')

@app.route('/boston/pairplots_after')
def boston_pairplots_after():
    return render_template('boston_pairplots_after.html')

@app.route('/boston/metrics')
def boston_metrics():
    return render_template('boston_metrics.html')

@app.route('/boston/matrix')
def boston_matrix():
    return render_template('boston_confusion_matrix.html')



@app.route('/time')
def time():
    return render_template('time.html')

@app.route('/time/description')
def time_description():
    return render_template('time_description.html')

@app.route('/time/histograms')
def time_histograms():
    return render_template('time_histograms.html')

@app.route('/time/scatter')
def time_scatter():
    return render_template('time_scatter.html')

@app.route('/time/pairplots_including')
def time_pairplots_including():
    return render_template('time_pairplots_including.html')

@app.route('/time/pairplots_after')
def time_pairplots_after():
    return render_template('time_pairplots_after.html')

@app.route('/time/metrics')
def time_metrics():
    return render_template('time_metrics.html')

@app.route('/time/matrix')
def time_matrix():
    return render_template('time_confusion_matrix.html')




@app.route('/black')
def black():
    return render_template('black.html')

@app.route('/black/description')
def black_description():
    return render_template('black_description.html')

@app.route('/black/histograms')
def black_histograms():
    return render_template('black_histograms.html')

@app.route('/black/scatter')
def black_scatter():
    return render_template('black_scatter.html')

@app.route('/black/pairplot_before')
def black_pairplots_including():
    return render_template('black_pairplots_including.html')

@app.route('/black/pairplot_after')
def black_pairplots_after():
    return render_template('black_pairplots_after.html')

@app.route('/black/metrics')
def black_metrics():
    return render_template('black_metrics.html')

@app.route('/black/matrix')
def black_matrix():
    return render_template('black_confusion_matrix.html')



@app.route('/human')
def human():
    return render_template('human.html')

@app.route('/human/description')
def human_description():
    description = {
        "columns": [
            "tBodyAcc-mean()-X", "tBodyAcc-mean()-Y", "tBodyAcc-mean()-Z", 
            "...", "tBodyGyroMag-std()", "activity", "subject_id"
        ]
    }
    return render_template('human_description.html', description=description)


@app.route('/human/histograms')
def human_histograms():
    return render_template('human_histograms.html')

@app.route('/human/scatter')
def human_scatter():
    return render_template('human_scatter.html', os=os)

@app.route('/human/pairplots_including')
def human_pairplots_including():
    return render_template('human_pairplots_including.html')

@app.route('/human/pairplots_after')
def human_pairplots_after():
    return render_template('human_pairplots_after.html')

@app.route('/human/metrics')
def human_metrics():
    return render_template('human_metrics.html')

@app.route('/human/matrix')
def human_matrix():
    return render_template('human_confusion_matrix.html')



@app.route('/trip')
def trip():
    return render_template('trip.html')

@app.route('/trip/description')
def trip_description():
    return render_template('trip_description.html')

@app.route('/trip/histograms')
def trip_histograms():
    return render_template('trip_histograms.html')

@app.route('/trip/scatter')
def trip_scatter():
    return render_template('trip_scatter.html')

@app.route('/trip/pairplot_before')
def trip_pairplots_including():
    return render_template('trip_pairplots_including.html')

@app.route('/trip/pairplot_after')
def trip_pairplots_after():
    return render_template('trip_pairplots_after.html')

@app.route('/trip/metrics')
def trip_metrics():
    return render_template('trip_metrics.html')

@app.route('/trip/matrix')
def trip_matrix():
    return render_template('trip_confusion_matrix.html')

@app.route('/census')
def census():
    return render_template('census.html')

@app.route('/census/description')
def census_description():
    return render_template('census_description.html')

@app.route('/census/histograms')
def census_histograms():
    return render_template('census_histograms.html')

@app.route('/census/scatter')
def census_scatter():
    return render_template('census_scatter.html')

@app.route('/census/pairplot_before')
def census_pairplots_including():
    return render_template('census_pairplots_including.html')

@app.route('/census/pairplot_after')
def census_pairplots_after():
    return render_template('census_pairplots_after.html')

@app.route('/census/metrics')
def census_metrics():
    return render_template('census_metrics.html')

@app.route('/census/matrix')
def census_matrix():
    return render_template('census_confusion_matrix.html')

@app.route('/coupon')
def coupon():
    return render_template('coupon.html')

@app.route('/coupon/description')
def coupon_description():
    return render_template('coupon_description.html')

@app.route('/coupon/histograms')
def coupon_histograms():
    return render_template('coupon_histograms.html')

@app.route('/coupon/scatter')
def coupon_scatter():
    return render_template('coupon_scatter.html')

@app.route('/coupon/pairplot_before')
def coupon_pairplots_including():
    return render_template('coupon_pairplots_including.html')

@app.route('/coupon/pairplot_after')
def coupon_pairplots_after():
    return render_template('coupon_pairplots_after.html')

@app.route('/coupon/metrics')
def coupon_metrics():
    return render_template('coupon_metrics.html')

@app.route('/coupon/confusion_matrix')
def coupon_matrix():
    return render_template('coupon_confusion_matrix.html')




@app.route('/churn')
def churn():
    return render_template('churn.html')

@app.route('/churn/description')
def churn_description():
    return render_template('churn_description.html')

@app.route('/churn/histograms')
def churn_histograms():
    return render_template('churn_histograms.html')

@app.route('/churn/scatter')
def churn_scatter():
    return render_template('churn_scatter.html')

@app.route('/churn/pairplot_before')
def churn_pairplots_including():
    return render_template('churn_pairplots_including.html')

@app.route('/churn/pairplot_after')
def churn_pairplots_after():
    return render_template('churn_pairplots_after.html')

@app.route('/churn/metrics')
def churn_metrics():
    return render_template('churn_metrics.html')

@app.route('/churn/confusion_matrix')
def churn_matrix():
    return render_template('churn_confusion_matrix.html')



@app.route('/movie')
def movie():
    return render_template('movie.html')

@app.route('/movie/description')
def movie_description():
    return render_template('movie_description.html')

@app.route('/movie/histograms')
def movie_histograms():
    return render_template('movie_histograms.html')

@app.route('/movie/scatter')
def movie_scatter():
    return render_template('movie_scatter.html')

@app.route('/movie/pairplot_before')
def movie_pairplots_including():
    return render_template('movie_pairplots_including.html')

@app.route('/movie/pairplot_after')
def movie_pairplots_after():
    return render_template('movie_pairplots_after.html')

@app.route('/movie/metrics')
def movie_metrics():
    return render_template('movie_metrics.html')

@app.route('/movie/matrix')
def movie_matrix():
    return render_template('movie_confusion_matrix.html')



@app.route('/song')
def song():
    return render_template('song.html')

@app.route('/song/description')
def song_description():
    return render_template('song_description.html')

@app.route('/song/histograms')
def song_histograms():
    return render_template('song_histograms.html')

@app.route('/song/scatter')
def song_scatter():
    return render_template('song_scatter.html')

@app.route('/song/pairplot_before')
def song_pairplots_including():
    return render_template('song_pairplots_including.html')

@app.route('/song/pairplot_after')
def song_pairplots_after():
    return render_template('song_pairplots_after.html')

@app.route('/song/metrics')
def song_metrics():
    return render_template('song_metrics.html')

@app.route('/song/matrix')
def song_matrix():
    return render_template('song_confusion_matrix.html')


@app.route('/digits')
def digits():
    return render_template('digits.html')

@app.route('/digits/description')
def digits_description():
    return render_template('digits_description.html')

@app.route('/digits/histograms')
def digits_histograms():
    return render_template('digits_histograms.html')

@app.route('/digits/scatter')
def digits_scatter():
    return render_template('digits_scatter.html', os=os)

@app.route('/digits/pairplot_before')
def digits_pairplots_including():
    return render_template('digits_pairplots_including.html')

@app.route('/digits/pairplot_after')
def digits_pairplots_after():
    return render_template('digits_pairplots_after.html')

@app.route('/digits/metrics')
def digits_metrics():
    return render_template('digits_metrics.html')

@app.route('/digits/matrix')
def digits_matrix():
    return render_template('digits_confusion_matrix.html')



@app.route('/urban')
def urban():
    return render_template('urban.html')

@app.route('/urban/description')
def urban_description():
    return render_template('urban_description.html')

@app.route('/urban/histograms')
def urban_histograms():
    return render_template('urban_histograms.html')

@app.route('/urban/scatter')
def urban_scatter():
    return render_template('urban_scatter.html')

@app.route('/urban/pairplot_before')
def urban_pairplots_including():
    return render_template('urban_pairplots_including.html')

@app.route('/urban/pairplot_after')
def urban_pairplots_after():
    return render_template('urban_pairplots_after.html')

@app.route('/urban/metrics')
def urban_metrics():
    return render_template('urban_metrics.html')

@app.route('/urban/matrix')
def urban_matrix():
    return render_template('urban_confusion_matrix.html')


@app.route('/market')
def market():
    return render_template('market.html')

@app.route('/market/description')
def market_description():
    return render_template('market_description.html')

@app.route('/market/histograms')
def market_histograms():
    return render_template('market_histograms.html')

@app.route('/market/scatter')
def market_scatter():
    return render_template('market_scatter.html')

@app.route('/market/pairplot_before')
def market_pairplots_including():
    return render_template('market_pairplots_including.html')

@app.route('/market/pairplot_after')
def market_pairplots_after():
    return render_template('market_pairplots_after.html')

@app.route('/market/metrics')
def market_metrics():
    return render_template('market_metrics.html')

@app.route('/market/matrix')
def market_matrix():
    return render_template('market_confusion_matrix.html')




@app.route('/imagenet')
def imagenet():
    return render_template('imagenet.html')

@app.route('/imagenet/description')
def imagenet_description():
    return render_template('imagenet_description.html')

@app.route('/imagenet/histograms')
def imagenet_histograms():
    return render_template('imagenet_histograms.html')

@app.route('/imagenet/scatter')
def imagenet_scatter():
    return render_template('imagenet_scatter.html')

@app.route('/imagenet/pairplot_before')
def imagenet_pairplots_including():
    return render_template('imagenet_pairplots_including.html')

@app.route('/imagenet/pairplot_after')
def imagenet_pairplots_after():
    return render_template('imagenet_pairplots_after.html')

@app.route('/imagenet/metrics')
def imagenet_metrics():
    return render_template('imagenet_metrics.html')

@app.route('/imagenet/matrix')
def imagenet_matrix():
    return render_template('imagenet_confusion_matrix.html')





@app.route('/chicago')
def chicago():
    return render_template('chicago.html')

@app.route('/chicago/description')
def chicago_description():
    return render_template('chicago_description.html')

@app.route('/chicago/histograms')
def chicago_histograms():
    return render_template('chicago_histograms.html')

@app.route('/chicago/scatter')
def chicago_scatter():
    return render_template('chicago_scatter.html')

@app.route('/chicago/pairplot_before')
def chicago_pairplots_including():
    return render_template('chicago_pairplots_including.html')

@app.route('/chicago/pairplot_after')
def chicago_pairplots_after():
    return render_template('chicago_pairplots_after.html')

@app.route('/chicago/metrics')
def chicago_metrics():
    return render_template('chicago_metrics.html')

@app.route('/chicago/matrix')
def chicago_matrix():
    return render_template('chicago_confusion_matrix.html')



@app.route('/covid')
def covid():
    return render_template('covid.html')

@app.route('/covid/description')
def covid_description():
    return render_template('covid_description.html')

@app.route('/covid/histograms')
def covid_histograms():
    return render_template('covid_histograms.html')

@app.route('/covid/scatter')
def covid_scatter():
    return render_template('covid_scatter.html')

@app.route('/covid/pairplot_before')
def covid_pairplots_including():
    return render_template('covid_pairplots_including.html')

@app.route('/covid/pairplot_after')
def covid_pairplots_after():
    return render_template('covid_pairplots_after.html')

@app.route('/covid/metrics')
def covid_metrics():
    return render_template('covid_metrics.html')

@app.route('/covid/matrix')
def covid_matrix():
    return render_template('covid_confusion_matrix.html')



@app.route('/recommendation')
def recommendation():
    return render_template('recommendation.html')

@app.route('/recommendation/description')
def recommendation_description():
    return render_template('recommendation_description.html')

@app.route('/recommendation/histograms')
def recommendation_histograms():
    return render_template('recommendation_histograms.html')

@app.route('/recommendation/scatter')
def recommendation_scatter():
    return render_template('recommendation_scatter.html')

@app.route('/recommendation/pairplot_before')
def recommendation_pairplots_including():
    return render_template('recommendation_pairplots_including.html')

@app.route('/recommendation/pairplot_after')
def recommendation_pairplots_after():
    return render_template('recommendation_pairplots_after.html')

@app.route('/recommendation/metrics')
def recommendation_metrics():
    return render_template('recommendation_metrics.html')

@app.route('/recommendation/matrix')
def recommendation_matrix():
    return render_template('recommendation_confusion_matrix.html')



@app.route('/license')
def license():
    return render_template('license.html')

@app.route('/license/description')
def license_description():
    return render_template('license_description.html')

@app.route('/license/histograms')
def license_histograms():
    return render_template('license_histograms.html')

@app.route('/license/scatter')
def license_scatter():
    return render_template('license_scatter.html')

@app.route('/license/pairplot_before')
def license_pairplots_including():
    return render_template('license_pairplots_including.html')

@app.route('/license/pairplot_after')
def license_pairplots_after():
    return render_template('license_pairplots_after.html')

@app.route('/license/metrics')
def license_metrics():
    return render_template('license_metrics.html')

@app.route('/license/matrix')
def license_matrix():
    return render_template('license_confusion_matrix.html')


@app.route('/objectives')
def objectives():
    return render_template('objectives.html')

@app.route('/history')
def history():
    return render_template('history.html')

if __name__ == "__main__":
    app.run(debug=True)
