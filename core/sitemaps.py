from django.contrib.sitemaps import Sitemap

class VueStaticViewSitemap(Sitemap):
    changefreq = "weekly"
    priority = 0.8

    def items(self):
        return [
            '',  # home
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
            'Privacypolicy',
            'Termsofservice',
        ]

    def location(self, item):
        return f"https://jibuai.app/{item}" if item else "https://jibuai.app/"
