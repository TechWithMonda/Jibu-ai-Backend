from django.contrib.sitemaps import Sitemap
from django.urls import reverse_lazy

class VueStaticViewSitemap(Sitemap):
    changefreq = "weekly"
    priority = 0.8

    def items(self):
        # These are all Vue frontend paths
        return [
            '',  # '/'
            'about',
            'login',
            'signup',
            'dashboard',
            'notes',
            'timetable',
            'aispeech',
            'BasicPackage',
            'PremiumPackage',
            'school',
            'users',
            'quizai',
            'features',
            'aistudy',
            'resources',
            'tutor',
            'timer',
            'uploadpaper',
        ]

    def location(self, item):
        return f'/{item}' if item else '/'
