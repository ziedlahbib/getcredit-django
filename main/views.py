from django.shortcuts import render
import easyocr
import PIL
from PIL import ImageDraw
from django.shortcuts import render
from django.http import HttpResponse
import tempfile
from django.http import JsonResponse
from PIL import Image




def ocr(request):
    #download the model
    reader = easyocr.Reader(['fr'], gpu = False)
    if request.method == 'POST':
        uploaded_file = request.FILES['file']  # Assuming the file input field has the name 'file'

        # Get the original file name
        file_name = uploaded_file.name

        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())

        file_path = temp_file.name

        im = PIL.Image.open(file_path)
        bounds = reader.readtext(im)
        my_list = []
        for i in bounds:
            print(i[1])
            my_list.append(i[1])

        return JsonResponse(my_list, safe=False)

    return HttpResponse("Invalid request method.")


def draw_boxes(image, bounds, color='red', width=2):
    draw = ImageDraw.Draw(image)
    for bound in bounds:
        p0, p1, p2, p3 = bound[0]
        draw.line([*p0, *p1, *p2, *p3, *p0], fill=color, width=width)
    return image

