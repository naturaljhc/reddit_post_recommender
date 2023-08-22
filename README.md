# DEPRECATED
This app is no longer functional due to changes with Reddit's API policies

## Description
This app regularly gathered posts from news subreddits and performed topic modeling to generate post recommendations.

### Process
* Posts with over 1000 upvotes are saved into a sqlite database
* Data is tokenized via a custom tokenizer created using SpaCy
* NMF is used for topic modeling to generate 20 topics
* Model is pickled for use in web app
* Web app is created using Flask which prompts users to input a Reddit URL
* Post title from the URL is vectorized and ran through the trained model
* The posts relevancy in each topic is used to determine similar posts using cosine similarity
* Top 5 recommended posts are returned to the user
