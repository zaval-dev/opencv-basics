# from django.template import engines
# from django.http import HttpResponse

# def index(request):
#     # Obtener las rutas de búsqueda de plantillas
#     template_dirs = engines['django'].engine.dirs

#     # Crear una respuesta con las rutas
    # return HttpResponse(f"Rutas de búsqueda de plantillas: {template_dirs}")

from django.shortcuts import render
import opencvBasics.utils.file_util as utils

def index(request):
    utils.delete_all_media()
    return render(request, 'main.html')