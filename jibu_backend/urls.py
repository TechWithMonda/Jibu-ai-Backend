from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from core.sitemaps import VueStaticViewSitemap
from django.contrib.sitemaps.views import sitemap  
sitemaps = {
    'vue': VueStaticViewSitemap,
}


urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('core.urls')),
    path('sitemap.xml', sitemap, {'sitemaps': sitemaps}, name='sitemap'),  # ← Note the trailing slash
]

# Add static/media URLs only if in DEBUG mode
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
