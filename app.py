from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
import random
import string
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__)

# Download NLTK data
nltk.download('punkt', quiet=True)

# Define product catalog with links
product_catalog = {
    "Adidas white sneakers": {
        "price": "Rs.1000",
        "description": "This is the description for Adidas white sneakers.",
        "file": "product1.html",
        "image": "https://th.bing.com/th/id/OIP.AHH8YiHiF25e6a2zctLkfgAAAA?rs=1&pid=ImgDetMain"
    },
    "Joyfay Teddybear": {
        "price": "Rs.550",
        "description": "This is the description for Joyfay Teddybear.",
        "file": "product2.html",
        "image": "https://i5.walmartimages.com/asr/8baa3751-0658-4aa5-8b68-89c73841e754_1.119af20f942cfb2e8e9d024424b41dba.jpeg"
    },
    "Gildan Unisex T-shirt": {
        "price": "Rs.499",
        "description": "This is the description for Gildan Unisex T-shirt.",
        "file": "product3.html",
        "image": "https://cdn.shopify.com/s/files/1/1650/5551/products/men-s-round-neck-plain-t-shirt-navy-blue-regular-fit-t-shirt-wolfattire-2549451063341.jpg?v=1561008965"
    },
    "Head Phone": {
        "price": "Rs.3000",
        "description": "This is the description for Head Phone.",
        "file": "product4.html",
        "image": "https://th.bing.com/th/id/OIP.RLf_snDFaUIz5k4-td1eZQHaHa?rs=1&pid=ImgDetMain"
    },
    "Iphone": {
        "price": "Rs.99,999",
        "description": "This is the description for Iphone.",
        "file": "product5.html",
        "image": "https://m.xcite.com/media/catalog/product//i/p/iphone_14_5g_-_red_1_3.jpg"
    },
    "Refrigerator": {
        "price": "Rs.78,999",
        "description": "This is the description for Refrigerator.",
        "file": "product6.html",
        "image": "https://4.bp.blogspot.com/-4uPZdAxTNN0/UnhNWuF44WI/AAAAAAAAA1g/Ixdk1KIRbaI/s1600/whirlpool-gi6farxxb-gold-refrigerator.jpg"
    },
    "Book": {
        "price": "Rs.199",
        "description": "This is the description for Book.",
        "file": "product7.html",
        "image": "http://4.bp.blogspot.com/-wcnWhKCkyeE/UOKqyVyOzkI/AAAAAAAAAEE/WWy5N3b_krk/s1600/375.jpg"
    },
    "BiCycle": {
        "price": "Rs.5,999",
        "description": "This is the description for BiCycle.",
        "file": "product8.html",
        "image": "https://th.bing.com/th/id/OIP.6Nx6tQx289YtLO9kqHpoAwHaG7?rs=1&pid=ImgDetMain"
    },
    "Tripod": {
        "price": "Rs.7,999",
        "description": "This is the description for Tripod.",
        "file": "product9.html",
        "image": "https://www.zomei.com/u_file/1811/products/01/fcc0ae96e8.jpg"
    },
    "Earbud": {
        "price": "Rs.999",
        "description": "This is the description for Earbud.",
        "file": "product10.html",
        "image": "https://m.media-amazon.com/images/I/715CLGC8OML.jpg"
    },
    "Spectacle": {
        "price": "Rs.899",
        "description": "This is the description for Spectacle.",
        "file": "product11.html",
        "image": "https://m.media-amazon.com/images/I/41jRQKdc0OL._AC_UY1100_.jpg"
    },
    "Table": {
        "price": "Rs.1499",
        "description": "This is the description for Table.",
        "file": "product12.html",
        "image": "https://th.bing.com/th/id/OIP.46zoV9CwJcfZBaV5kOq5-gAAAA?rs=1&pid=ImgDetMain"
    },
    "Floral Kurthi": {
        "price": "Rs.1299",
        "description": "This is the description for Floral Kurthi.",
        "file": "product13.html",
        "image": "https://th.bing.com/th/id/OIP.IAJV3rzsll0VU9C4JjN_iwHaLH?w=575&h=863&rs=1&pid=ImgDetMain"
    },
    "Sofa": {
        "price": "Rs.39,999",
        "description": "This is the description for Sofa.",
        "file": "product14.html",
        "image": "https://th.bing.com/th/id/OIP.TizgmrV4EQ0bqINFk5gCAAAAAA?rs=1&pid=ImgDetMain"
    }
}

# Generate a simple corpus for chatbot
corpus = " ".join(product_catalog.keys())

# Tokenize corpus into sentences
sent_tokens = nltk.sent_tokenize(corpus)

# Prepare punctuation removal dictionary
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return nltk.word_tokenize(text.lower().translate(remove_punct_dict))

# Greeting inputs and responses
greeting_input = ["hi", "hello", "hey", "hola", "namaste"]
greeting_response = ["howdy", "hey there", "hi", "hello :)"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in greeting_input:
            return random.choice(greeting_response)
    return None

def response(user_response):
    user_response = user_response.lower()
    robo_response = ''
    sent_tokens.append(user_response)
    tfidfvec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = tfidfvec.fit_transform(sent_tokens)
    val = cosine_similarity(tfidf[-1], tfidf[:-1])
    idx = val.argsort()[0][-1]
    flat = val.flatten()
    flat.sort()
    score = flat[-1]
    if score == 0:
        robo_response = "Sorry, I don't understand"
    else:
        robo_response = sent_tokens[idx]
    sent_tokens.pop()
    return robo_response

def clean_price(price_str):
    # Remove any characters that are not digits or decimal points
    cleaned_price_str = re.sub(r'[^\d.]', '', price_str)
    return float(cleaned_price_str)    

def get_product_info(product_name):
    for key in product_catalog:
        if product_name.lower() in key.lower():
            product = product_catalog[key]
            return (f"{key}: {product['description']} Price: {product['price']}. "
                    f"You can view more details <a href='{url_for('serve_file', filename=product['file'])}' target='_blank'>here</a>.")
    return "Sorry, I couldn't find that product."

def find_products_in_price_range(product_name, min_price, max_price):
    product = product_catalog.get(product_name, None)
    if product:
        # Convert the catalog price to a float
        price = clean_price(product['price'])
        
        # Check if the price is within the specified range
        if min_price <= price <= max_price:
            return get_product_info(product_name)
        else:
            return (f"The {product_name} is priced at Rs.{price:.2f}, which is outside "
                    f"the range of Rs.{min_price:.2f} to Rs.{max_price:.2f}.")
    return f"Sorry, I couldn't find {product_name}."


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    user_input = request.form["user_input"].strip()
    
    # Check for greeting
    if greeting(user_input) is not None:
        return jsonify({"response": greeting(user_input)})
    
    # Extract product name and price range
    product_name = None
    min_price = None
    max_price = None
    
    # Check if user input contains a price range
    prices = re.findall(r'\d+', user_input)
    if len(prices) == 2:
        min_price = float(prices[0])
        max_price = float(prices[1])
    else:
        # Handle case where no or invalid price range is found
        min_price = 0
        max_price = float('inf')    
    
    # Extract product name from user input
    for product in product_catalog:
        if any(part.lower() in user_input.lower() for part in product.split()):
            product_name = product
            break
    
    if product_name and min_price is not None and max_price is not None:
        return jsonify({"response": find_products_in_price_range(product_name, min_price, max_price)})
    
    if product_name:
        return jsonify({"response": get_product_info(product_name)})
    
    return jsonify({"response": response(user_input)})

@app.route("/place_order", methods=["POST"])
def place_order():
    product_name = request.form["product"].strip()
    for product in product_catalog:
        if product.lower() == product_name.lower():
            return jsonify({"response": f"Your order for {product} has been placed successfully!"})
    return jsonify({"response": "Sorry, we couldn't process your order. Please check the product name."})

@app.route('/files/<filename>')
def serve_file(filename):
    return send_from_directory('.', filename)

if __name__ == "__main__":
    app.run(debug=True)
