from flask import Flask, render_template, request
import config
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from bs4 import BeautifulSoup
import requests
import re
from googletrans import Translator
import openai
import blog
from emailpy import *
import config
openai.api_key = config.OPENAI_API_KEY


translator = Translator()

def summarizer(url,seed=1.2):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    results = soup.find_all([ 'p'])
    text = [result.text for result in results]
    ARTICLE = ' '.join(text)

    # Tokenizing the text
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(ARTICLE)

    # Creating a frequency table to keep the
    # score of each word

    freqTable = dict()
    for word in words:
        word = word.lower()
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    # Creating a dictionary to keep the score
    # of each sentence
    sentences = sent_tokenize(ARTICLE)
    sentenceValue = dict()

    for sentence in sentences:
        for word, freq in freqTable.items():
            if word in sentence.lower():
                if sentence in sentenceValue:
                    sentenceValue[sentence] += freq
                else:
                    sentenceValue[sentence] = freq



    sumValues = 0
    for sentence in sentenceValue:
        sumValues += sentenceValue[sentence]

    # Average value of a sentence from the original text

    average = int(sumValues / len(sentenceValue))

    # Storing sentences into our summary.
    summary = ''
    for sentence in sentences:
        if (sentence in sentenceValue) and (sentenceValue[sentence] > (seed * average)):
            summary += " " + sentence
    print(summary)
    return ARTICLE , summary


def summarize_Text(ARTICLE,seed=1.2):


    # Tokenizing the text
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(ARTICLE)

    # Creating a frequency table to keep the
    # score of each word

    freqTable = dict()
    for word in words:
        word = word.lower()
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    # Creating a dictionary to keep the score
    # of each sentence
    sentences = sent_tokenize(ARTICLE)
    sentenceValue = dict()

    for sentence in sentences:
        for word, freq in freqTable.items():
            if word in sentence.lower():
                if sentence in sentenceValue:
                    sentenceValue[sentence] += freq
                else:
                    sentenceValue[sentence] = freq



    sumValues = 0
    for sentence in sentenceValue:
        sumValues += sentenceValue[sentence]

    # Average value of a sentence from the original text

    average = int(sumValues / len(sentenceValue))

    # Storing sentences into our summary.
    summary = ''
    for sentence in sentences:
        if (sentence in sentenceValue) and (sentenceValue[sentence] > (seed * average)):
            summary += " " + sentence
    return ARTICLE , summary


def emailextractro(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    results = soup.find_all(['h1','h2','h3','h4','h5','h6','pre', 'p','a','li','span','ul','ol','li'])
    text = [result.text for result in results]
    ARTICLE = ' '.join(text)
    reg = re.findall(r"[A-Za-z0-9_%+-.]+"
				r"@[A-Za-z0-9.-]+"
				r"\.[A-Za-z]{2,5}",ARTICLE)
    return reg

def transulate_Link(url,lang):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    results = soup.find_all([ 'h1','h2','p'])
    text = [result.text for result in results]
    ARTICLE = ' '.join(text)
    translator = Translator(service_urls=['translate.googleapis.com'])
    translated_text = translator.translate(ARTICLE,dest=lang)
    return translated_text.text


def transulate_text(text,lang):
    translator = Translator(service_urls=['translate.googleapis.com'])
    translated_text = translator.translate(text,dest=lang)
    return translated_text.text


def openai_grammer(text):
    response = openai.Completion.create(
    model="davinci-instruct-beta-v3",
    prompt=f"Correct this to standard English:\n\n{text}",
    temperature=0,
    max_tokens=60,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
    )

    return response.choices[0].text


def openai_quray(pompt):
    response = openai.Completion.create(model="text-davinci-003",
    prompt=pompt,
    temperature=0.7,
    max_tokens=2000,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
    )

    return response.choices[0].text

def page_not_found(e):
  return render_template('404.html'), 404


app = Flask(__name__)
app.config.from_object(config.config['development'])
app.register_error_handler(404, page_not_found)


@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html', **locals())


@app.route('/summarize', methods=["GET", "POST"])
def summarize():

    if request.method == 'POST':
        query = request.form['Article-Link']
        
        text1 , sumg = summarizer(query,seed=float(request.form['seed']))

        prompt = 'The summary of the article  {} is:'.format(query)
        openAIAnswer = sumg

    return render_template('summarize.html', **locals())

@app.route('/Email-extractor', methods=["GET", "POST"])
def Emailextractor():

    if request.method == 'POST':
        query = request.form['website-Link']
        
        email = emailextractro(query)

        prompt = 'Emails found in the following website  {} are:'.format(query)
        openAIAnswer = email

    return render_template('Emailextractor.html', **locals())


@app.route('/summarizeText', methods=["GET", "POST"])
def summarizeText():

    if request.method == 'POST':
        query = request.form['Text']
        
        text1 , sumg = summarize_Text(query,float(request.form['seed']))

        # prompt = 'The summary of the article  {} is:'.format(query)
        openAIAnswer = sumg

    return render_template('summarizeText.html', **locals())



@app.route('/transulate', methods=["GET", "POST"])
def transulate():
    
        if request.method == 'POST' :
            query = request.form['url']
            lang = request.form['lang']
            text = transulate_Link(query,lang)
            
        
    
            # prompt = 'The summary of the article  {} is:'.format(query)
            openAIAnswer = text
    
        return render_template('transulate.html', **locals())

@app.route('/transulateText', methods=["GET", "POST"])
def transulateText():
        
            if request.method == 'POST' :
                query = request.form['text']
                lang = request.form['lang']
                text = transulate_text(query,lang)
                openAIAnswer = text
            return render_template('transulatetext.html', **locals())

@app.route('/grammer', methods=["GET", "POST"])
def grammer():
    if request.method == 'POST' :
        query = request.form['text']
        text = openai_grammer(text=query)
        openAIAnswer = text
    return render_template('grammercorrection.html', **locals())

@app.route('/product-description', methods=["GET", "POST"])
def productDescription():

    if request.method == 'POST':
        query = request.form['productDescription']
        print(query)

        pmpt = f'''
        write a seo friendly Product sales Description for:
        productname:{request.form['productname']},
        productDescription:{request.form['productDescription']},
        productFeatures:{request.form['productFeatures']},
        productBenefits:{request.form['productBenefits']},
        tone:{request.form['tone']},
        productType:{request.form['productType']},
        productBrand:{request.form['productBrand']},
        seed words:{request.form['seedword']}.
        
        '''
        openAIAnswer = openai_quray(pmpt)

        
        

    return render_template('product-description.html', **locals())


@app.route('/product-salesadd', methods=["GET", "POST"])
def salesadd():

    if request.method == 'POST':
        query = request.form['productDescription']
        print(query)

        pmpt = f'''
        write a seo friendly Product {request.form['platform']} ADD for:
        platform:{request.form['platform']},
        productname:{request.form['productname']},
        productDescription:{request.form['productDescription']},
        productFeatures:{request.form['productFeatures']},
        productBenefits:{request.form['productBenefits']},
        tone:{request.form['tone']},
        productType:{request.form['productType']},
        productBrand:{request.form['productBrand']},
        seed words:{request.form['seedword']}.
        
        '''
        openAIAnswer = openai_quray(pmpt)

        
        

    return render_template('productsocialmediaadd.html', **locals())


@app.route('/salesadd', methods=["GET", "POST"])
def salesad_d():

    if request.method == 'POST':
        query = request.form['productDescription']
        print(query)

        pmpt = f"Write a creative ad for the following product to run on {request.form['platform']} aimed at parents:\n\nProduct: {query}"
        openAIAnswer = openai_quray(pmpt)

        
        

    return render_template('socialmediaadd.html', **locals())

@app.route('/blog', methods=["GET", "POST"])
def blogen():

    if request.method == 'POST':
        if 'form1' in request.form:
            prompt = request.form['blogTopic']
            blogT = blog.generateBlogTopics(prompt)
            blogTopicIdeas = blogT.replace('\n', '<br>')

        if 'form2' in request.form:
            prompt = request.form['blogSection']
            blogT = blog.generateBlogSections(prompt)
            blogSectionIdeas = blogT.replace('\n', '<br>')

        if 'form3' in request.form:
            prompt = request.form['blogExpander']
            blogT = blog.blogSectionExpander(prompt)
            blogExpanded = blogT.replace('\n', '<br>')


    return render_template('blog.html', **locals())

@app.route('/job-description', methods=["GET", "POST"])
def jobDescription():

    if request.method == 'POST':
        query = request.form['jobDescription']
        

        print(query)
        pompt = f'''
        write a detailed Job Description in active voice for the following :
        name:{request.form['name']},
        jobTitle:{request.form['jobDescription']},
        experince:{request.form['experince']},
        projects done:{request.form['projects']},
        skills:{request.form['skills']},
        tools and technology:{request.form['tools']},
        education:{request.form['education']},
        certification:{request.form['certification']},
        company:{request.form['company']},
        location:{request.form['location']},
        keypoints:{request.form['keypoints']}

        '''
        openAIAnswer = openai_quray(pompt)


    return render_template('job-description.html', **locals())



@app.route('/tweet-ideas', methods=["GET", "POST"])
def tweetIdeas():

    if request.method == 'POST':
        query = request.form['tweetIdeas']
        print(query)

        pompt = f''' 
        Expand the tweet section in to a detailed professional , witty and clever explanation. {request.form['tweetIdeas']}:
        '''
        openAIAnswer = openai_quray(pompt)

    return render_template('tweet-ideas.html', **locals())



@app.route('/cold-emails', methods=["GET", "POST"])
def coldEmails():

    if request.method == 'POST':
        query = request.form['coldEmails']
        print(query)
        pompt = f'''
        write a cold email :{query}'''

        openAIAnswer = openai_quray(pompt)

    return render_template('cold-emails.html', **locals())



@app.route('/social-media', methods=["GET", "POST"])
def socialMedia():

    if request.method == 'POST':
        query = request.form['socialMedia']
        print(query)

        prompt = 'AI Suggestions for {} are:'.format(query)
        openAIAnswer = 'Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.'

    return render_template('social-media.html', **locals())


@app.route('/business-pitch', methods=["GET", "POST"])
def businessPitch():

    if request.method == 'POST':
        query = request.form['businessPitch']
        print(query)

        pmpt = f'''
        write a high level business pitch for the following bussiness idea:{query}'''
        openAIAnswer = openai_quray(pmpt)

    return render_template('business-pitch.html', **locals())


@app.route('/video-ideas', methods=["GET", "POST"])
def videoIdeas():

    if request.method == 'POST':
        query = request.form['videoIdeas']
        print(query)

        pmpt = f''' Generate video ideas based on {query}
        '''
        openAIAnswer = openai_quray(pmpt)

    return render_template('video-ideas.html', **locals())


@app.route('/video-description', methods=["GET", "POST"])
def videoDescription():

    if request.method == 'POST':
        query = request.form['videoDescription']
        print(query)
        pmpt = f''' Write SEO friendly short description for your YouTube video:{query}
        '''
        openAIAnswer = openai_quray(pmpt)


    return render_template('video-description.html', **locals())

@app.route('/video-script', methods=["GET", "POST"])
def videoScript():
    
        if request.method == 'POST':
            query = request.form['text']
            print(query)
            pmpt = f''' Write a detailed professional explanation youtube video script in 1500 words for the following:{query}
            '''
            openAIAnswer = openai_quray(pmpt)
    
        return render_template('video-script.html', **locals())

@app.route('/Bussiness', methods=["GET", "POST"])
def Bussiness():
        
            if request.method == 'POST':
                query = request.form['text']
                print('requesting.......')
                pmpt = f''' Write the business case study of {query} company  and explain all the strategies in detailed also explain all the marketing strategies followed by the food {query} company to increase their sales. 
                '''
                openAIAnswer = openai_quray(pmpt)
                
                print(openAIAnswer)

                print('for the result ...')
        
            return render_template('Bussiness.html', **locals())
@app.route('/Bussiness-marketing', methods=["GET", "POST"])
def BussinessMarketing():
            
            if request.method == 'POST':
                query = request.form['text']
                print('requesting.......')
                pmpt = f''' Write a detailed professional explanation of the  marketing strategies followed by the {query}business to increase their sales. 
                '''
                openAIAnswer = openai_quray(pmpt)
                
                print(openAIAnswer)

                print('for the result ...')
        
            return render_template('Bussiness-marketing.html', **locals())
@app.route('/Linkedinpost', methods=["GET", "POST"])
def Linkedinposts():
            
            if request.method == 'POST':
                query = request.form['Text']
                print('requesting.......')
                pmpt = f''' Write a detailed and viral Linkedin post on{query} . 
                '''
                openAIAnswer = openai_quray(pmpt)
                
                print(openAIAnswer)

                print('for the result ...')
        
            return render_template('Linkedin.html', **locals())
@app.route('/Speach', methods=["GET", "POST"])
def Speach():
                
                if request.method == 'POST':
                    query = request.form['Text']
                    print('requesting.......')
                    pmpt = f''' Write a high level  speech on:{query} . 
                    '''
                    openAIAnswer = openai_quray(pmpt)
                    
                    print(openAIAnswer)
    
                    print('for the result ...')
            
                return render_template('Speach.html', **locals())
@app.route('/Bussiness-plan', methods=["GET", "POST"])
def BussinessPlan():
                        
                        if request.method == 'POST':
                            query = request.form['text']
                            print('requesting.......')
                            pmpt = f''' Write a high level  business plan for:{query} . 
                            '''
                            openAIAnswer = openai_quray(pmpt)
                            
                            print(openAIAnswer)
            
                            print('for the result ...')
                    
                        return render_template('Bussinessplan.html', **locals())
@app.route('/emailvalidate', methods=["GET", "POST"])
def emailvalidate():
                        
                        if request.method == 'POST':
                            query = request.form['Text']
                            print('requesting.......')
                            pmpt = f''' Write a high level  business plan for:{query} . 
                            '''
                            openAIAnswer = openai_quray(pmpt)
                            
                            print(openAIAnswer)
            
                            print('for the result ...')
                    
                        return render_template('emailvalidate.html', **locals())
@app.route('/validate', methods=['POST'])
def validate():
    firstname = request.form['firstname']
    lastname = request.form['lastname']
    domin = request.form['domin']
    email = run(firstname, lastname, domin)
    send = f'The delivarible Email Address of the person is {email}'
    return render_template('emailvalidate.html', email=send)






if __name__ == '__main__':
    app.run(debug=True)
