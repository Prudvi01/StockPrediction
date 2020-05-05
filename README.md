# Stock Prediction

Stock prediction using sentiment analysis

## Steps
1. **Clone this repo**
 `git clone https://github.com/Prudvi01/stockprediction.git`
2. **Go to the directory**
3. **Activate virtual environment (you can skip this but it's recommended)**
	`virtualenv env;source env/bin/activate`

		Note: install virtualenv using `pip install virtualenv`
4. **Install requirements**
	`pip install -r requirements.txt`
5. **Create API keys**
	Using 2 APIs in this project
	
	*a. Twitter API* 
	- Get this API from [Twitter](developer.twitter.com). You need to create a developer account.
	- Get 4 keys after following [these steps](https://developer.twitter.com/en/docs/basics/authentication/guides/access-tokens).
	- Update line 12-17 in code.py
	
	*b. Alphavantage*
	- Get API from [AlphaVantage](https://www.alphavantage.co/support/#api-key).
	- Update line 18 in code.py

6.	**Run python file**
`python code.py`
7.	Enter any stock quote to see prediction


*Note: If this code isn't working locally for some reason, try it on Google Colab* 
