from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError  # Import this
from .models import UserProfile

class UserRegisterForm(UserCreationForm):
    # Additional fields from UserProfile
    description = forms.CharField(widget=forms.Textarea, required=False)
    location = forms.CharField(max_length=30, required=False)
    budget = forms.IntegerField(required=False)  # Renamed from 'budjet' to 'budget'
    preferred_country = forms.CharField(max_length=30, required=False)

    class Meta:
        model = User
        fields = ['username', 'password1', 'password2', 'description', 'location', 'budget', 'preferred_country']

    def clean_password1(self):
        password1 = self.cleaned_data.get('password1')
        
        # Add custom validation rules
        if not any(char.isdigit() for char in password1):
            raise ValidationError("Password must contain at least one number.")
        if not any(char.islower() for char in password1):
            raise ValidationError("Password must contain at least one lowercase letter.")
        
        return password1

    def clean_password2(self):
        password2 = self.cleaned_data.get('password2')
        password1 = self.cleaned_data.get('password1')
        
        # Check if the two passwords match
        if password1 != password2:
            raise ValidationError("Passwords do not match.")
        
        return password2

    def save(self, commit=True):
        # Save the basic user information
        user = super().save(commit=False)  # Don't save yet if commit=False

        if commit:
            user.save()  # Save user before creating UserProfile

        # Create UserProfile instance
        UserProfile.objects.create(
            user=user,
            description=self.cleaned_data.get('description'),
            location=self.cleaned_data.get('location'),
            budjet=self.cleaned_data.get('budjet'),
            preferred_country=self.cleaned_data.get('preferred_country')
        )
        
        return user


class UserProfileForm(forms.ModelForm):
    COUNTRY_CHOICES = [
        ('India', 'India'),
        ('Europe', 'Europe'),
    ]

    preferred_country = forms.ChoiceField(
        choices=COUNTRY_CHOICES, 
        widget=forms.Select(attrs={'class': 'form-control'})
    )

    class Meta:
        model = UserProfile
        fields = ['description', 'preferred_country', 'location', 'budjet']
        widgets = {
            'description': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
        }
