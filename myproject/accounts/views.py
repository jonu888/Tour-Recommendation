from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required, user_passes_test
from .forms import UserRegisterForm, UserProfileForm
from .models import UserProfile
from django.contrib.auth.models import User
import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import requests

# Configure Google Generative AI with your API key
genai.configure(api_key="api")

def gencon(place_name, user_input):
    # Use Google Gemini AI for generating content
    ai_model = genai.GenerativeModel("gemini-pro")
    response = ai_model.generate_content(f"Description of {place_name} maintaining meaning {user_input} in 2-3 lines")
    if response and hasattr(response, 'candidates') and response.candidates:
        candidate = response.candidates[0]
        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts') and candidate.content.parts:
            return candidate.content.parts[0].text
    return "No information available."

def generate_google_maps_link(lat, lon):
    return f"https://www.google.com/maps?q={lat},{lon}"

def get_location_osm(place_name):
    url = f"https://nominatim.openstreetmap.org/search?q={place_name}&format=json"
    headers = {
        "User-Agent": "TourRecommendationSystem/1.0 (your_email@example.com)"
    }
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return None

        data = response.json()
        if data:
            # Convert coordinates to float and return as a tuple
            lat, lon = float(data[0]["lat"]), float(data[0]["lon"])
            print(f"Coordinates for {place_name}: ({lat}, {lon})")
            return (lat, lon)
        else:
            return None

    except requests.exceptions.RequestException as e:
        return None
    except ValueError as e:
        return None

def register(request):
    if request.method == 'POST':
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            if not UserProfile.objects.filter(user=user).exists():
                UserProfile.objects.create(user=user)
            return redirect('login')
    else:
        form = UserRegisterForm()
    return render(request, 'accounts/register.html', {'form': form})

def user_login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            if user.is_superuser:
                return redirect('admin_dashboard')
            else:
                return redirect('user_profile')
        else:
            return render(request, 'accounts/login.html', {'error': 'Invalid credentials'})
    return render(request, 'accounts/login.html')

def nlp_chech(descriptions, user_input):
    # Remove non-alphabet characters
    descrip = descriptions.replace('[^a-zA-Z]', ' ', regex=True)
    ps = PorterStemmer()
    nltk.download('punkt')  # Use the standard 'punkt' package
    descrip = descrip.apply(lambda row: [ps.stem(token.lower()) for token in nltk.word_tokenize(row)]
                            ).apply(lambda tokens: " ".join(tokens))
    nltk.download('stopwords')
    sw = stopwords.words('english')
    descrip = descrip.apply(lambda row: [token for token in nltk.word_tokenize(row) if token not in sw]
                            ).apply(lambda tokens: " ".join(tokens))
    tfidf = TfidfVectorizer()
    train_data = tfidf.fit_transform(descrip)
    user_input_tfidf = tfidf.transform(user_input)
    user_similarity = cosine_similarity(user_input_tfidf, train_data)
    recommended_indices = user_similarity[0].argsort()[-5:][::-1]
    return recommended_indices

def re_search_Europe(user_input):
    df = pd.read_csv(r"C:\Users\josan\OneDrive\Desktop\Project\p3\myproject\archive (7)\destinations.csv", encoding='ISO-8859-1')
    df = df.drop(columns=['Latitude', 'Destination', 'Longitude', 'Country', 'Category',
                          'Approximate Annual Tourists', 'Currency', 'Majority Religion',
                          'Famous Foods', 'Language', 'Cost of Living', 'Safety', 'Cultural Significance'])
    df = df.dropna().drop_duplicates()
    descriptions = df['Description']
    recommended_indices = nlp_chech(descriptions, user_input)
    place = {}
    for i in recommended_indices:
        place_details = dict(df.iloc[i][['Region', 'Best Time to Visit']])
        info = gencon(place_details['Region'], user_input)
        # Optionally, you can add map links here as well if available
        place[place_details['Region']] = [place_details['Best Time to Visit'], info]
    place = dict(list(place.items())[:3])
    return place

def re_search_india(user_input):
    # Use a raw string for the file path
    df1 = pd.read_csv(r"C:\Users\josan\OneDrive\Desktop\Project\p3\myproject\archive (7)\holidify (1).csv", encoding='ISO-8859-1')
    df1 = df1.dropna().drop_duplicates(subset=['City'])
    descrip = df1['About the city (long Description)']
    descrip = descrip.replace('[^a-zA-Z]', ' ', regex=True)
    ps = PorterStemmer()
    nltk.download('punkt')
    descrip = descrip.apply(lambda row: [ps.stem(token.lower()) for token in nltk.word_tokenize(row)]
                            ).apply(lambda tokens: " ".join(tokens))
    recommended_indices = nlp_chech(descrip, user_input)
    place = {}
    for i in recommended_indices:
        place_details = dict(df1.iloc[i][['City', 'Best Time to visit']])
        info = gencon(place_details['City'], user_input)
        location = get_location_osm(place_details['City'])
        if location:
            lat, lon = location
            map_link = generate_google_maps_link(lat, lon)
        else:
            map_link = ""
        # Store Best Time, info, and the map link
        place[place_details['City']] = [place_details['Best Time to visit'], info, map_link]
    place = dict(list(place.items())[:3])
    return place

@login_required
def user_profile(request):
    user_profile_obj, created = UserProfile.objects.get_or_create(user=request.user)
    if request.method == 'POST':
        des = request.POST.get('description')
        pre_country = request.POST.get('preferred_country')
        form = UserProfileForm(request.POST, instance=user_profile_obj)
        if form.is_valid():
            if pre_country == 'Europe':
                rec_place = re_search_Europe([des])
            else:
                rec_place = re_search_india([des])
            form.save()
            return render(request, 'accounts/user_profile.html', {'form': form, 'search_results': rec_place})
    else:
        form = UserProfileForm(instance=user_profile_obj)
    return render(request, 'accounts/user_profile.html', {'form': form})

@user_passes_test(lambda u: u.is_superuser)
def admin_dashboard(request):
    user_profiles = UserProfile.objects.all()
    if request.method == 'POST' and request.POST.get('search_username'):
        username = request.POST['search_username'].strip()
        user = get_object_or_404(User, username=username)
        user_profile_obj = UserProfile.objects.get(user=user)
        return render(request, 'accounts/admin_dashboard.html', {
            'user_profiles': user_profiles,
            'searched_profile': user_profile_obj,
            'search_username': username,
        })
    if request.method == 'POST' and 'update_profile' in request.POST:
        profile_id = request.POST['profile_id']
        user_profile_obj = get_object_or_404(UserProfile, id=profile_id)
        form = UserProfileForm(request.POST, instance=user_profile_obj)
        if form.is_valid():
            form.save()
            return redirect('admin_dashboard')
    else:
        form = UserProfileForm()
    return render(request, 'accounts/admin_dashboard.html', {
        'user_profiles': user_profiles,
        'form': form,
    })

def user_logout(request):
    logout(request)
    return redirect('login')
