"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np
from PIL import Image
import requests
from streamlit_lottie import st_lottie
# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

movie_image = Image.open(r'Movies-AI.jpg')
news_image = Image.open(r'news_img.jpg')
logo_a= Image.open(r'forefront_AI.png')

def load_lottieurl(url):
	r = requests.get(url)
	if r.status_code != 200:
		return None
	return r.json()

# Load your raw data
raw_m = pd.read_csv("resources/data/movies.csv")
raw_r= pd.read_csv("resources/data/ratings.csv")

#load lottie urls
data_lottie = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_8gmx5ktv.json")
info_lottie = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_HhOOsG.json")

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')

# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["About us", "Recommender System","Solution Overview", "Explore the Dataset", "Movie Information", "Our Team", "News"]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "Solution Overview":
        st.title("Solution Overview")
        st.title("The movie recommendation engine")
        st.write('##')
	

        st.image(movie_image, caption='https://thinkml.ai/top-10-series-and-movies-about-ai/',
		)
        with st.container():
            left_column, right_column = st.columns((4,1))
            with left_column:
				
                st.write('---')
                st.info("General Information")
			
                st.write('##')
                st.write("""
				Use an unsupervised machine learning algorithm to recommend movies to users. 
                In the world we live in, there are enormous data and a large collections of 
                resources. In the movie ecosytem, most times, knowing exactly what to go for can be
                very streinous to avoid loss of time and resources. It is therefore a game changing 
                action to have a such system as a recommender engine that suggest what movie to
                watch to very good level of accuracy. 
				""")
            with right_column:
                st_lottie(info_lottie)
        st.write('#')
        st.write('---')
		# You can read a markdown file from supporting resources folder
        st.markdown(
		"""
## About Recommendation System

<p align ="justify"> There are two major types of recommender system. Content-Based Filtering and Collaborative Filtering. In 
Content-based Filtering, we seek to make recommendations based on how similar the properties or features 
of an item are to other items. Say for example, Mr A has read book A and he is looking for another book to 
read. The algorithm will compute similarity between book A and other books in storage, after then, books 
with close similarity values to book A will be recommended to Mr A. </p>

<p align ="justify"> In collaborative, the engine makes recommendtations 
based on similarity between users. If Mr A has read book A loved it by giving a good rating, Mr A has also 
read book B and loved it but He didnt like book C. Here too, Mr B has read book A and loved, he has also read 
book C and didnt love it. The algorithm will understand that Mr A is similar to Mr B and recommend other books 
that Mr A loves (or books that are similar)  to Mr B. In this movie recommender project, user can select 
either of the two systems to generate predicitons for movies they'd love to watch. </p>

## Resources;

 <p align ="justify"> Movielens and IMDB <a href= "https://www.kaggle.com/competitions/edsa-movie-recommendation-predict/data">data</a> 
 relating to movie user ratings and general information from 1970 to recently were collected, analyzed
 and fed into machine learning algorithms to generate predictions. The data have been cleaned and processed
 thoroughly to format that are usable for algorithm based learnings. </p>
 
##
 <p align ="justify"> <a href= "https://streamlit.io/">Streamlit</a> was used to create a web application that hosts the model and project information.
 The app was then deployed to allow acces by organisations and individuals who might need them. </p>


##
---
### objectives


    - To build a recommendation engine capable of recommending movies to users 
      with selected machine learning models.

    - To help movie companies align their production efforts with viewers' interest.

    - To help streaming media engage users with effective recommendation algorithm

    - To build a system that enables movie companies and media generate more revenue
      from their activities """, unsafe_allow_html= True)

        st.subheader("Raw movies data and label")
        if st.checkbox('Show raw data'): # data is hidden if box is unchecked
            st.write(raw_m[['movieId', 'title', 'genres']]) # will write the df to the pag
        
        st.subheader("Raw ratings data and label")
        if st.checkbox('Show raw data', key= 2): # data is hidden if box is unchecked
            st.write(raw_r[['movieId', 'rating']]) # will write the df to the pag
    
    if page_selection == "Movie Information":
        # Header Contents
        st.write("# Movie Information")
        images = ['resources/imgs/info.png']
        for i in images:
            st.image(i, use_column_width=True)
        filters = ["Top rated Movies", "High Budget Movies"]
        filter_selection = st.selectbox("Fact Check", filters)
        if filter_selection == "Top rated Movies":
            movie_list = pd.read_csv('resources/data/movies.csv')
            ratings = pd.read_csv('resources/data/ratings.csv')
            df = pd.merge(movie_list, ratings, on='movieId', how='left')
            movie_ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
            movie_ratings["Number_Of_Ratings"] = pd.DataFrame(df.groupby('title')['rating'].count())
            indes = movie_ratings.index
            new_list = []
            for movie in indes:
                i = ' '.join(movie.split(' ')[-1])
                new_list.append(i)
            new_lists = []
            for i in new_list:
                if len(i) < 2:
                    empty = i
                    new_lists.append(empty)
                elif i[0] == "(" and i[-1] == ")" and len(i) == 11:
                    R_strip = i.rstrip(i[-1])
                    L_strip = R_strip.lstrip(R_strip[0])
                    spaces = ''.join(L_strip.split())
                    data_type_int = int(spaces)
                    new_lists.append(data_type_int)
                else:
                    new_lists.append(i)
            cnn = []
            for i in new_lists:
                if type(i) != int:
                    i = 0
                    cnn.append(i)
                else:
                    cnn.append(i)
            movie_ratings["Year"] = cnn

            def user_interaction(Year, n):
                list_movies = movie_ratings[movie_ratings['Year'] == Year].sort_values('Number_Of_Ratings',
                                                                                       ascending=False).index
                return list_movies[:n]

            selected_year = st.selectbox("Year released", range(1970, 2020))
            no_of_outputs = st.radio("Movies to view", (5, 10, 20, 50))
            output_list = user_interaction(selected_year, no_of_outputs)
            new_list = []
            for movie in output_list:
                updated_line = ' '.join(movie.split(' ')[:-1])
                updated_line = "+".join(updated_line.split())
                new_list.append(updated_line)
            url = "https://www.imdb.com/search/title/?title="
            movie_links = []
            for i in new_list:
                links = url + i
                movie_links.append(links)
            dict_from_list = dict(zip(output_list, movie_links))
            for items in dict_from_list:
                st.subheader(items)

    if page_selection == "Explore the Dataset":

        st.title('EXPLORING THE MOVIE DATASET')
        if st.checkbox("Ratings Overview"):
            st.subheader("Overview of rating counts given by users")
            st.image('resources/imgs/ratings.png', use_column_width=True,
                    caption='The majority of ratings lie between 3.0 and 4.0')

            
        if st.checkbox("Most Rated Movies", key= 3):
            st.subheader("List of movies rated the most")
            st.image('resources/imgs/ratingspermovie.png', use_column_width=True,
                    caption='Ratings per movie')

        # if st.checkbox("correlation"):
        #     st.subheader("Correlation between features")
        #     st.image('resources/imgs/correlation.png',use_column_width=ie)

        if st.checkbox("Keyword wordcloud"):
            st.subheader("Keyword WordCloud")
            st.image('resources/imgs/wordclud.png', use_column_width=True,
                     caption='The most important plot keywords can be observed')

        if st.checkbox("Popular Actors"):
            st.subheader("Popular Actors")
            st.image('resources/imgs/popactors.png', use_column_width=True,
                     caption='Samuel. L. Jackson seems to be the most popular actorm appearing in over 80 films')

        if st.checkbox("Directors"):
            st.subheader("Number of Directors per movie")
            st.image('resources/imgs/popactors.png', use_column_width=True,
                     caption='The data shows that Woody Allen is the most prolific directors. With more than 25 movies under his belt')

        if st.checkbox("Genres"):
            st.subheader("Top Genres")
            st.image('resources/imgs/popgenres.png', use_column_width=True,
                     caption='Drama is the most popular genre among the movies, showing up in over 25000 movies. Comedy and Thillers are next. About 5000 movies were not allocated a specific genre.')

    if page_selection == "Our Team":
        st.subheader("Meet the fabulous team behind the App!")




        # Two equal columns:
        col1, col2 = st.columns(2)
        col1.write("**Nonokazi Cele**")
        with col1:
            st.image("resources/imgs/nono.jpg", width=200)

        col2.write("**Data Scientist**")
        with col2:
            st.markdown("""

               “I am a Data Scientist with a strong background in software engineering; and used to handling a variety of data pipelines and databases, included unstructured ones. I have prototyped four products, and I am looking for product oriented role (also consulting), possibly building from scratch.”

               """)

        # Two equal columns:
        col3, col4 = st.columns(2)
        col3.write("**Data Analyst**")
        with col3:
            st.markdown("""

            “Data Analyst and process engineer contributing to the sustainability of energy production and consumption. Open to new opportunities, holding German and -Nigerian- nationalities.”

               """)

        col4.write("**John Chukwuebuka**")
        with col4:
            st.image("resources/imgs/john.jpg", width=200)


        # Two equal columns:
        col5, col6 = st.columns(2)
        with col5:
            col5.write("**Business Analyst**")
            st.image("resources/imgs/o2.jpg")

        col6.write("**Ololade Ogunleye**")
        with col6:
            st.markdown("""

           "Passionate utilizing data to further your business needs, having 1+ years of experience in predictive modeling and data mining. Excited to implement statistical machine learning solutions for Atlas Intelligence. At Stack Intellect, implemented demand forecasting models improving forecast accuracy by 34%"

               """)

        # Two equal columns:
        col7, col8 = st.columns(2)
        col7.write("**Data Engineer**")
        with col7:
            st.markdown("""

           "Professional Big Data Engineer with 2 years industry experience including 1 year in Big Data technologies. Expertise in Hadoop/Spark developtment experiene, automation tools and E2C life cycle of 
           software design process. Outstanding communication skills, dedicated to maintain up-to-date IT skills and industry knowledge"

               """)

        col8.write("**Omolayo Ipinsanmi**")
        with col8:
            st.image("resources/imgs/omolayo.jpg")

        # Two equal columns:
        col9, col10 = st.columns(2)
        with col9:
            col9.write("**Machine Learning Specialist**")
            st.image("resources/imgs/mandy.jpg")

        col10.write("**Mandy Rasemphe**")
        with col10:
            st.markdown("""

           "Experienced Machine Learning specialist with proven track record of leading major projects. Implemented organization-wide standardization of invoicing systems. Managed launch of streamlined payment systems, increasing productivity and customer satisfaction. Proficient at identifying user needs and finding optimal solutions. Effective team leader who excels in motivating and organizing team for optimal performance. Track record of coming up with innovative solutions to reduce costs, streamline operations, and improve security.
               """)

    
    if page_selection == "News":
        st.title("Get The Latest Movie news")
        st.write('---')
		
        st.write("""Stay in the know on matters relating to movie releases, actors,
        awards, financing. never miss an update on upcoming movies and a chance to 
        be one of the first people to watch on big screen. Know more about the movie
        industry around the world with a single click.

		"""
		)
        st.write('##')
        st.image(news_image, width=600, caption=" Source: https://www.freepik.com/")
        st.write('---')
        st.write("""
		Click the button below to to get a round up of the latest news in chimate change and global warming from the web.
		 You can proceed to the news source by clicking the provided link to the article
		""")
        btn = st.button("Click to get latest movies related news")

        if btn:
            url ="https://newsapi.org/v2/everything?" 
            request_params = {
		    	"q": 'hollywood OR upcomming movies OR new movies OR hollywood actors',
				"sort by": "latest",
				"language": 'en',
				"apikey": "950fae5906d4465cb25932f4c5e1202c"
			}
            r = requests.get(url, request_params)
            r = r.json()
            articles = r['articles']

            for article in articles:
                st.header(article['title'])
                if article['author']:
                    st.write(f"Author: {article['author']}")
                st.write(f"Source: {article['source']['name']}")
                st.write(article['description'])
                st.write(f"link to article: {article['url']}")
                st.image(article['urlToImage'])

    if page_selection == 'About us':
		
		
		#define a function to access lottiee files

        def load_lottieurl(url):
            r = requests.get(url)
            if r.status_code != 200:
                return None
            return r.json()
		
        lottie_coding = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_dsxct2el.json")
        phonecall_lottie = load_lottieurl("https://assets2.lottiefiles.com/private_files/lf30_rvyzng8q.json")
		# header section
        with st.container():
                
            st.subheader("Hi :wave:, we are Forefront AI")
            st.image(logo_a, use_column_width=True)
            st.write('---')
            st.title('An AI/Data Science based team focused on creating real-world solutions')
            st.write(""" \n We are passionate about the use of data to help
			companies to make informed decisions""")
		
		#what do we do?
        with st.container():
            st.write('---')
            left_column, right_column = st.columns(2)

            with left_column:
                st.header("What do we do?")
                st.write('##')
                st.markdown(
					"""
					<ul> We create viable AI solutions to clients to activate better level of productivity while reducing cost
					 <li> - Leveraging available data to analyse trends and usage. </li>
					 <li> - Building Machine learning models to generate predictions on users' behavior. </li>
					 <li> - Buildiin recommender engines to recommend products to users.
                     <li> - Building classification models to classify and categorise users, items and services. </li>
					 <li> - Building ready to use web applications for solution deployment. </li>
                     </ul>
					    """
				, unsafe_allow_html= True)
            with right_column:
                st_lottie(lottie_coding, height = 300, key = "coding" )
			
		# This project
        with st.container():
            st.write("---")
            st.header(" This Projects")
            st.write("##")

            image_column, text_column = st.columns((1,2))

            with image_column:
			# import the image
                st.image(movie_image)
            with text_column:
                st.subheader("Generate movie predictions on our streamlit web application")
                st.write(
					"""
					Select from list of movies you love and get recommendations on movies we think you will like;
					- You will first have to select the filtering method you'd prefer to use from a list of either
                    - Content based or
                    - Collaborative
                    - Then click on recommend
				""")
			
            with st.container():
                st.write('---')
                st.header("Get In Touch With Us")
                st.write("##")
                contact_form = """
				<form action="https://formsubmit.co/ipinsanmitimothy@gmail.com" method="POST">
     <input type="text" name="message" placeholder = "enter a message" required>
     <input type="email" name="email" placeholder = "enter your email" required>
     <button type="submit">Send</button>
</form>
					"""
                info_column, phonecall_column = st.columns((2,1))

                with info_column:
                    st.markdown(contact_form, unsafe_allow_html=True)   
                
                
                with phonecall_column:
                    st_lottie(phonecall_lottie)

				#styling the contact form
            
            def locall_css(filename):
                with open(filename) as f:
                    st.markdown(f"<style>{f.read()}</style", unsafe_allow_html=True)
					
                locall_css("style/style.css")

    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.


if __name__ == '__main__':
    main()
