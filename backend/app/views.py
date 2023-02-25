from django.shortcuts import render

from posturedetect import Model

# Create your views here.
def index(request):
    return render(request, "pages/index.html")

def detect(request):
    image_dir = ""
    model_inst= Model()
    # sample response {'text': 'Not Correct Posture'}
    return model_inst.predictions(image_dir)
