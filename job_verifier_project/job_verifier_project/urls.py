from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),

    # 🔴 Disable API routes (ML related) for free-tier stability
    # path('api/', include('main_app.api_urls')),

    # ✅ Enable normal web pages only
    path('', include('main_app.urls')),
]
